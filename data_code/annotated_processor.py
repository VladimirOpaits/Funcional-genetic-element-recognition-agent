import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fsspec
import torch
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.nt_model import NucleotideTransformer


CLASSES = ["GAG", "INT", "RT", "RN", "LTR"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

DEFAULT_CHUNK_BP = 6000
DEFAULT_STRIDE_BP = 6000


def open_text(uri: str, mode: str = "r"):
    return fsspec.open(uri, mode)


def torch_save_uri(obj, uri: str) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    with fs.open(path, "wb") as f:
        f.write(buf.read())


def read_fasta_uri(uri: str):
    with fsspec.open(uri, "r") as f:
        text = f.read()
    return next(SeqIO.parse(io.StringIO(text), "fasta"))


class AnnotatedTokenLabeler:

    def __init__(
        self,
        nt: Optional[NucleotideTransformer] = None,
        chunk_bp: int = DEFAULT_CHUNK_BP,
        stride_bp: int = DEFAULT_STRIDE_BP,
        max_chunks_per_seq: Optional[int] = None,
    ):
        self.nt = nt or NucleotideTransformer()
        self.chunk_bp = chunk_bp
        self.stride_bp = stride_bp
        self.max_chunks_per_seq = max_chunks_per_seq

    @staticmethod
    def _multi_label_for_span(
        global_start: int,
        global_end: int,
        domains: Dict[str, List[Tuple[int, int]]],
    ) -> torch.Tensor:
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        if global_end <= global_start:
            return label
        for class_name, intervals in domains.items():
            idx = CLASS_TO_IDX.get(class_name)
            if idx is None:
                continue
            for s, e in intervals:
                if s < global_end and e > global_start:
                    label[idx] = 1.0
                    break
        return label

    def process_sequence(
        self,
        full_seq: str,
        domains: Dict[str, List[Tuple[int, int]]],
    ) -> List[Dict]:
        chunks: List[Dict] = []
        seq_len = len(full_seq)
        chunk_count = 0

        for chunk_start in range(0, seq_len, self.stride_bp):
            chunk_end = min(chunk_start + self.chunk_bp, seq_len)
            chunk_seq = full_seq[chunk_start:chunk_end]
            if len(chunk_seq) < 50:
                break

            embeddings, offsets = self.nt.embed_with_offsets(chunk_seq)
            num_tokens = embeddings.shape[1]

            token_labels = torch.zeros(num_tokens, NUM_CLASSES, dtype=torch.float32)
            global_offsets: List[Tuple[int, int]] = []
            for i, (lstart, lend) in enumerate(offsets):
                gstart = chunk_start + lstart
                gend = chunk_start + lend
                global_offsets.append((gstart, gend))
                token_labels[i] = self._multi_label_for_span(gstart, gend, domains)

            chunks.append({
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "embeddings": embeddings.squeeze(0),
                "labels": token_labels,
                "token_offsets": global_offsets,
            })

            chunk_count += 1
            if self.max_chunks_per_seq is not None and chunk_count >= self.max_chunks_per_seq:
                break
            if chunk_end == seq_len:
                break

        return chunks

    def process_dataset(
        self,
        annotations_uri: str,
        sequences_uri: str,
    ) -> Dict[str, Dict]:
        with open_text(annotations_uri, "r") as f:
            annotations = json.load(f)

        sequences_uri = sequences_uri.rstrip("/")
        ann_parent = annotations_uri.rsplit("/", 1)[0] if "/" in annotations_uri else "."

        result: Dict[str, Dict] = {}
        for ann in annotations:
            accession = ann["accession"]
            candidates = [
                f"{sequences_uri}/{accession}.fasta",
                f"{ann_parent}/{ann.get('file', '')}",
            ]
            record = None
            for uri in candidates:
                if not uri or uri.endswith("/"):
                    continue
                try:
                    record = read_fasta_uri(uri)
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"Error reading {uri}: {e}")
            if record is None:
                print(f"Missing FASTA: {accession}, skipping")
                continue

            full_seq = str(record.seq).upper()
            domains = {k: [tuple(x) for x in v] for k, v in ann["domains"].items() if k in CLASS_TO_IDX}

            print(f"Processing {accession} ({len(full_seq)} bp)...", flush=True)
            chunks = self.process_sequence(full_seq, domains)
            pos_counts = torch.zeros(NUM_CLASSES)
            for c in chunks:
                pos_counts += c["labels"].sum(dim=0)
            stats = ", ".join(f"{name}={int(pos_counts[i])}" for i, name in enumerate(CLASSES))
            print(f"  -> {len(chunks)} chunks; positive tokens per class: {stats}")

            result[accession] = {
                "length": len(full_seq),
                "domains": domains,
                "chunks": chunks,
            }

        return result


def flatten_chunks(
    dataset: Dict[str, Dict],
    drop_specials: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for entry in dataset.values():
        for chunk in entry["chunks"]:
            emb = chunk["embeddings"]
            lab = chunk["labels"]
            if drop_specials:
                keep = torch.tensor(
                    [s != e for s, e in chunk["token_offsets"]],
                    dtype=torch.bool,
                )
                emb = emb[keep]
                lab = lab[keep]
            xs.append(emb)
            ys.append(lab)
    if not xs:
        return torch.empty(0), torch.empty(0, NUM_CLASSES)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def default_input_uri() -> str:
    bucket = os.environ.get("GCS_BUCKET")
    prefix = os.environ.get("GCS_PREFIX", "annotated_sequences")
    if bucket:
        return f"gs://{bucket}/{prefix}"
    return str((Path(__file__).resolve().parent.parent / "data" / "annotated_sequences").as_posix())


def default_output_uri() -> str:
    bucket = os.environ.get("GCS_BUCKET")
    prefix = os.environ.get("GCS_OUTPUT_PREFIX", "annotated_eval")
    if bucket:
        return f"gs://{bucket}/{prefix}"
    return str(Path(__file__).resolve().parent.as_posix())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=default_input_uri(), help="dir with annotations.json and sequences/")
    parser.add_argument("--output", default=default_output_uri(), help="dir to write .pt files (local or gs://)")
    parser.add_argument("--chunk-bp", type=int, default=DEFAULT_CHUNK_BP)
    parser.add_argument("--stride-bp", type=int, default=DEFAULT_STRIDE_BP)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()

    base = args.input.rstrip("/")
    annotations_uri = f"{base}/annotations.json"
    sequences_uri = f"{base}/sequences"
    out_base = args.output.rstrip("/")

    print(f"Input:  {base}")
    print(f"Output: {out_base}")

    labeler = AnnotatedTokenLabeler(
        chunk_bp=args.chunk_bp,
        stride_bp=args.stride_bp,
        max_chunks_per_seq=args.max_chunks,
    )
    dataset = labeler.process_dataset(annotations_uri, sequences_uri)

    out_dataset = f"{out_base}/annotated_eval_dataset.pt"
    torch_save_uri(dataset, out_dataset)
    print(f"\nSaved per-sequence dataset -> {out_dataset}")

    X, Y = flatten_chunks(dataset)
    out_flat = f"{out_base}/annotated_eval_flat.pt"
    torch_save_uri({"X": X, "Y": Y, "classes": CLASSES}, out_flat)
    print(f"Saved flat (X, Y) -> {out_flat}")
    print(f"Shapes: X={tuple(X.shape)}, Y={tuple(Y.shape)}")
    if Y.numel() > 0:
        per_class = Y.sum(dim=0).int().tolist()
        print(f"Positive tokens per class: {dict(zip(CLASSES, per_class))}")


if __name__ == "__main__":
    main()