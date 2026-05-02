import gc
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.nt_model import NucleotideTransformer


CLASSES = ["INT", "RT", "RN", "GAG"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

DEFAULT_CHUNK_BP = 6000
DEFAULT_STRIDE_BP = 6000
EMB_DTYPE = torch.float16
DEFAULT_CHUNKS_PER_SHARD = 256


class AnnotatedTokenLabeler:
    def __init__(
        self,
        nt: Optional[NucleotideTransformer] = None,
        chunk_bp: int = DEFAULT_CHUNK_BP,
        stride_bp: int = DEFAULT_STRIDE_BP,
        max_chunks_per_seq: Optional[int] = None,
        emb_dtype: torch.dtype = EMB_DTYPE,
    ):
        self.nt = nt or NucleotideTransformer()
        self.chunk_bp = chunk_bp
        self.stride_bp = stride_bp
        self.max_chunks_per_seq = max_chunks_per_seq
        self.emb_dtype = emb_dtype

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

    def iter_chunks(
        self,
        full_seq: str,
        domains: Dict[str, List[Tuple[int, int]]],
    ) -> Iterator[Dict]:
        seq_len = len(full_seq)
        chunk_count = 0

        for chunk_start in range(0, seq_len, self.stride_bp):
            chunk_end = min(chunk_start + self.chunk_bp, seq_len)
            chunk_seq = full_seq[chunk_start:chunk_end]
            if len(chunk_seq) < 50:
                break

            with torch.no_grad():
                embeddings, offsets = self.nt.embed_with_offsets(chunk_seq)
            embeddings = embeddings.squeeze(0).to(self.emb_dtype).contiguous()
            num_tokens = embeddings.shape[0]

            token_labels = torch.zeros(num_tokens, NUM_CLASSES, dtype=torch.float32)
            global_offsets: List[Tuple[int, int]] = []
            for i, (lstart, lend) in enumerate(offsets):
                gstart = chunk_start + lstart
                gend = chunk_start + lend
                global_offsets.append((gstart, gend))
                token_labels[i] = self._multi_label_for_span(gstart, gend, domains)

            yield {
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "embeddings": embeddings,
                "labels": token_labels,
                "token_offsets": global_offsets,
            }

            chunk_count += 1
            if self.max_chunks_per_seq is not None and chunk_count >= self.max_chunks_per_seq:
                break
            if chunk_end == seq_len:
                break

    def process_dataset_to_shards(
        self,
        annotations_path: Path,
        sequences_dir: Path,
        shard_dir: Path,
        chunks_per_shard: int = DEFAULT_CHUNKS_PER_SHARD,
    ) -> List[Dict]:
        annotations_path = Path(annotations_path)
        sequences_dir = Path(sequences_dir)
        shard_dir = Path(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

        manifest: List[Dict] = []

        with open(annotations_path) as f:
            annotations = json.load(f)

        for ann in annotations:
            accession = ann["accession"]
            fasta_path = sequences_dir / f"{accession}.fasta"
            if not fasta_path.exists():
                fasta_path = annotations_path.parent / ann["file"]
            if not fasta_path.exists():
                print(f"Missing FASTA: {accession}, skipping")
                continue

            try:
                record = next(SeqIO.parse(str(fasta_path), "fasta"))
            except Exception as e:
                print(f"Error reading {accession}: {e}")
                continue

            full_seq = str(record.seq).upper()
            domains = {k: [tuple(x) for x in v] for k, v in ann["domains"].items()}

            print(f"Processing {accession} ({len(full_seq)} bp)...", flush=True)

            buffer: List[Dict] = []
            shard_idx = 0
            total_chunks = 0
            pos_counts = torch.zeros(NUM_CLASSES)
            shard_files: List[str] = []

            def flush():
                nonlocal shard_idx, buffer
                if not buffer:
                    return
                shard_path = shard_dir / f"{accession}_{shard_idx:04d}.pt"
                torch.save(
                    {
                        "accession": accession,
                        "shard_idx": shard_idx,
                        "domains": domains,
                        "chunks": buffer,
                    },
                    shard_path,
                )
                shard_files.append(str(shard_path.relative_to(shard_dir.parent)))
                print(f"  wrote {shard_path.name} ({len(buffer)} chunks)", flush=True)
                shard_idx += 1
                buffer = []
                gc.collect()

            for chunk in self.iter_chunks(full_seq, domains):
                pos_counts += chunk["labels"].sum(dim=0)
                buffer.append(chunk)
                total_chunks += 1
                if len(buffer) >= chunks_per_shard:
                    flush()
            flush()

            stats = ", ".join(f"{name}={int(pos_counts[i])}" for i, name in enumerate(CLASSES))
            print(f"  -> {total_chunks} chunks total; positive tokens per class: {stats}")

            manifest.append({
                "accession": accession,
                "length": len(full_seq),
                "num_chunks": total_chunks,
                "shards": shard_files,
            })

            with open(shard_dir.parent / "annotated_eval_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return manifest


def iter_shard_chunks(shard_dir: Path) -> Iterator[Dict]:
    shard_dir = Path(shard_dir)
    for shard_path in sorted(shard_dir.glob("*.pt")):
        entry = torch.load(shard_path, map_location="cpu", weights_only=False)
        for chunk in entry["chunks"]:
            yield chunk
        del entry
        gc.collect()


def write_flat_streaming(
    shard_dir: Path,
    out_dir: Path,
    drop_specials: bool = True,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Two-pass flatten using numpy memmaps so the full (X, Y) never
    has to fit in RAM. Writes ``X.npy``-style memmap and ``Y.npy``
    memmap plus a small ``flat_meta.json`` describing shapes/dtypes.
    """
    shard_dir = Path(shard_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    hidden_dim = None
    emb_np_dtype = None
    for chunk in iter_shard_chunks(shard_dir):
        emb = chunk["embeddings"]
        if drop_specials:
            keep_n = sum(1 for s, e in chunk["token_offsets"] if s != e)
        else:
            keep_n = emb.shape[0]
        total_tokens += keep_n
        if hidden_dim is None:
            hidden_dim = emb.shape[1]
            emb_np_dtype = np.dtype("float16") if emb.dtype == torch.float16 else np.dtype("float32")

    x_path = out_dir / "X.f16.dat"
    y_path = out_dir / "Y.f32.dat"
    meta_path = out_dir / "flat_meta.json"

    if total_tokens == 0:
        meta = {"x_shape": [0, 0], "y_shape": [0, NUM_CLASSES], "x_dtype": "float16",
                "y_dtype": "float32", "classes": CLASSES,
                "x_path": x_path.name, "y_path": y_path.name}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return (0, 0), (0, NUM_CLASSES)

    X = np.memmap(x_path, dtype=emb_np_dtype, mode="w+", shape=(total_tokens, hidden_dim))
    Y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(total_tokens, NUM_CLASSES))

    cursor = 0
    for chunk in iter_shard_chunks(shard_dir):
        emb = chunk["embeddings"]
        lab = chunk["labels"]
        if drop_specials:
            keep = torch.tensor(
                [s != e for s, e in chunk["token_offsets"]],
                dtype=torch.bool,
            )
            emb = emb[keep]
            lab = lab[keep]
        n = emb.shape[0]
        X[cursor:cursor + n] = emb.numpy()
        Y[cursor:cursor + n] = lab.numpy()
        cursor += n

    X.flush()
    Y.flush()
    del X, Y

    meta = {
        "x_shape": [total_tokens, hidden_dim],
        "y_shape": [total_tokens, NUM_CLASSES],
        "x_dtype": str(emb_np_dtype),
        "y_dtype": "float32",
        "classes": CLASSES,
        "x_path": x_path.name,
        "y_path": y_path.name,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return (total_tokens, hidden_dim), (total_tokens, NUM_CLASSES)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    base = project_root / "data" / "annotated_sequences"
    shard_dir = Path(__file__).parent / "annotated_shards"

    labeler = AnnotatedTokenLabeler()
    manifest = labeler.process_dataset_to_shards(
        annotations_path=base / "annotations.json",
        sequences_dir=base / "sequences",
        shard_dir=shard_dir,
    )

    manifest_path = Path(__file__).parent / "annotated_eval_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest -> {manifest_path}")
    print(f"Per-sequence shards -> {shard_dir}")

    flat_dir = Path(__file__).parent / "annotated_eval_flat"
    x_shape, y_shape = write_flat_streaming(shard_dir, flat_dir)
    print(f"Saved flat memmap (X, Y) -> {flat_dir}")
    print(f"Shapes: X={x_shape}, Y={y_shape}")
