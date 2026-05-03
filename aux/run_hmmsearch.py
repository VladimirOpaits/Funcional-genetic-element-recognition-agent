import argparse
import gzip
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fsspec
import requests
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


CLASSES = ["GAG", "INT", "RT", "RN", "LTR"]

PFAM_HMMS = {
    "GAG": ["PF00540", "PF00607", "PF06875", "PF02093"],
    "INT": ["PF00665", "PF02022", "PF13683"],
    "RT": ["PF00078", "PF07727", "PF13456"],
    "RN": ["PF00075"],
}

EVALUE_CUTOFF = 0.01


def download_pfam_hmm(pfam_id: str, dest: Path) -> bool:
    urls = [
        f"https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{pfam_id}?annotation=hmm",
        f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfam_id}?annotation=hmm",
    ]
    for url in urls:
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=60)
                if r.status_code != 200:
                    time.sleep(2)
                    continue
                data = r.content
                if data[:2] == b"\x1f\x8b":
                    data = gzip.decompress(data)
                if b"HMMER" not in data[:200]:
                    time.sleep(2)
                    continue
                dest.write_bytes(data)
                return True
            except Exception as e:
                print(f"  attempt {attempt+1} {pfam_id}: {e}")
                time.sleep(2)
    return False


def fetch_all_hmms(cache_dir: Path) -> Dict[str, List[Path]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, List[Path]] = {c: [] for c in PFAM_HMMS}
    for cls, ids in PFAM_HMMS.items():
        for pfam_id in ids:
            dest = cache_dir / f"{pfam_id}.hmm"
            if not dest.exists() or dest.stat().st_size < 200:
                print(f"  downloading {pfam_id} -> {cls}")
                if not download_pfam_hmm(pfam_id, dest):
                    print(f"  FAILED {pfam_id}")
                    continue
            out[cls].append(dest)
    return out


def six_frame_translate(record: SeqRecord) -> List[Tuple[SeqRecord, str, int]]:
    frames = []
    seq = record.seq
    try:
        rc = seq.reverse_complement()
    except Exception:
        return frames
    for strand_name, s in [("F", seq), ("R", rc)]:
        for frame in range(3):
            sub = s[frame:]
            sub = sub[: len(sub) - (len(sub) % 3)]
            if len(sub) < 3:
                continue
            try:
                protein = sub.translate(to_stop=False)
            except Exception:
                continue
            new_rec = SeqRecord(protein, id=f"{record.id}__{strand_name}{frame}", description="")
            frames.append((new_rec, strand_name, frame))
    return frames


def run_hmmsearch(hmm_path: Path, protein_fasta: Path, domtable: Path) -> bool:
    try:
        result = subprocess.run(
            ["hmmsearch", "--domtblout", str(domtable), "-E", "1.0", str(hmm_path), str(protein_fasta)],
            capture_output=True,
            timeout=600,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  hmmsearch error: {e}")
        return False


def parse_domtable(domtable: Path) -> List[Dict]:
    hits = []
    if not domtable.exists():
        return hits
    with open(domtable) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 19:
                continue
            try:
                target = parts[0]
                ievalue = float(parts[12])
                ali_from = int(parts[17])
                ali_to = int(parts[18])
            except (ValueError, IndexError):
                continue
            if ievalue > EVALUE_CUTOFF:
                continue
            hits.append({"target": target, "ali_from": ali_from, "ali_to": ali_to})
    return hits


def protein_to_nucleotide(target: str, ali_from: int, ali_to: int, seq_len: int) -> Optional[Tuple[int, int]]:
    if "__" not in target:
        return None
    flag = target.rsplit("__", 1)[1]
    if len(flag) < 2:
        return None
    strand = flag[0]
    try:
        frame = int(flag[1])
    except ValueError:
        return None
    nt_start = frame + (ali_from - 1) * 3
    nt_end = frame + ali_to * 3
    if strand == "R":
        nt_start, nt_end = seq_len - nt_end, seq_len - nt_start
    nt_start = max(0, nt_start)
    nt_end = min(seq_len, nt_end)
    if nt_end <= nt_start:
        return None
    return (nt_start, nt_end)


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def annotate_one_sequence(
    record: SeqRecord,
    hmm_paths_by_class: Dict[str, List[Path]],
    workdir: Path,
) -> Dict[str, List[Tuple[int, int]]]:
    seq_len = len(record.seq)
    domains: Dict[str, List[Tuple[int, int]]] = {c: [] for c in CLASSES}

    frames = six_frame_translate(record)
    if not frames:
        return domains

    protein_fasta = workdir / "proteins.fasta"
    SeqIO.write([f[0] for f in frames], str(protein_fasta), "fasta")

    for cls, hmm_paths in hmm_paths_by_class.items():
        for hmm_path in hmm_paths:
            domtable = workdir / f"{cls}_{hmm_path.stem}.tbl"
            if not run_hmmsearch(hmm_path, protein_fasta, domtable):
                continue
            for hit in parse_domtable(domtable):
                coords = protein_to_nucleotide(hit["target"], hit["ali_from"], hit["ali_to"], seq_len)
                if coords:
                    domains[cls].append(coords)

    domains = {k: merge_intervals(v) for k, v in domains.items()}
    return domains


def list_fasta_uris(sequences_uri: str) -> List[str]:
    fs, path = fsspec.core.url_to_fs(sequences_uri)
    proto = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    listing = fs.ls(path)
    out = []
    for entry in listing:
        if isinstance(entry, dict):
            entry = entry.get("name", "")
        if not (entry.endswith(".fasta") or entry.endswith(".fa")):
            continue
        if proto == "file":
            out.append(entry)
        else:
            out.append(f"{proto}://{entry}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="dir with sequences/ subdir")
    parser.add_argument("--output", default=None, help="output dir; defaults to --input")
    parser.add_argument("--hmm-cache", default="./pfam_hmms")
    parser.add_argument("--keep-existing-ltr", action="store_true", help="merge LTR coords from existing annotations.json")
    parser.add_argument("--min-len", type=int, default=500)
    args = parser.parse_args()

    input_uri = args.input.rstrip("/")
    output_uri = (args.output or args.input).rstrip("/")
    sequences_uri = f"{input_uri}/sequences"

    print(f"Input:  {input_uri}")
    print(f"Output: {output_uri}")

    cache_dir = Path(args.hmm_cache)
    print(f"\nDownloading HMMs to {cache_dir}...")
    hmm_paths_by_class = fetch_all_hmms(cache_dir)
    for cls, paths in hmm_paths_by_class.items():
        print(f"  {cls}: {len(paths)} profiles")

    fasta_uris = list_fasta_uris(sequences_uri)
    print(f"\nFound {len(fasta_uris)} FASTA files in {sequences_uri}")

    existing_ltr: Dict[str, List[Tuple[int, int]]] = {}
    if args.keep_existing_ltr:
        ann_uri = f"{input_uri}/annotations.json"
        try:
            with fsspec.open(ann_uri, "r") as f:
                existing = json.load(f)
            for ann in existing:
                ltr = ann.get("domains", {}).get("LTR", [])
                if ltr:
                    existing_ltr[ann["accession"]] = [tuple(x) for x in ltr]
            print(f"Loaded LTR coords for {len(existing_ltr)} sequences from existing annotations")
        except Exception as e:
            print(f"Could not load existing LTR: {e}")

    annotations: List[dict] = []
    success = 0

    with tempfile.TemporaryDirectory() as workdir_str:
        workdir = Path(workdir_str)

        for i, fasta_uri in enumerate(fasta_uris, 1):
            try:
                with fsspec.open(fasta_uri, "r") as f:
                    text = f.read()
                record = next(SeqIO.parse(io.StringIO(text), "fasta"))
            except Exception as e:
                print(f"[{i}] read error {fasta_uri}: {e}")
                continue

            try:
                seq_len = len(record.seq)
                _ = str(record.seq[:10])
            except Exception:
                print(f"[{i}] {record.id}: undefined sequence, skip")
                continue

            if seq_len < args.min_len:
                continue

            print(f"[{i}/{len(fasta_uris)}] {record.id} ({seq_len}bp)...", flush=True)
            domains = annotate_one_sequence(record, hmm_paths_by_class, workdir)

            if record.id in existing_ltr:
                domains["LTR"] = merge_intervals(domains.get("LTR", []) + existing_ltr[record.id])

            present = {k: len(v) for k, v in domains.items() if v}
            print(f"  -> {present}")

            if not any(domains.values()):
                continue

            annotations.append({
                "accession": record.id,
                "description": record.description,
                "length": seq_len,
                "domains": domains,
                "file": f"sequences/{record.id}.fasta",
            })
            success += 1

            if i % 10 == 0:
                ann_path = f"{output_uri}/annotations.json"
                with fsspec.open(ann_path, "w") as f:
                    f.write(json.dumps(annotations, indent=2))
                print(f"  [checkpoint] {success} annotated so far")

    ann_path = f"{output_uri}/annotations.json"
    with fsspec.open(ann_path, "w") as f:
        f.write(json.dumps(annotations, indent=2))

    print(f"\nWrote {success} annotations -> {ann_path}")
    counts = {c: 0 for c in CLASSES}
    total_intervals = {c: 0 for c in CLASSES}
    for ann in annotations:
        for d, v in ann["domains"].items():
            if v:
                counts[d] = counts.get(d, 0) + 1
                total_intervals[d] += len(v)
    print("Sequences with each domain class (intervals total):")
    for c in CLASSES:
        print(f"  {c}: {counts.get(c, 0)} seqs ({total_intervals.get(c, 0)} intervals)")


if __name__ == "__main__":
    main()
