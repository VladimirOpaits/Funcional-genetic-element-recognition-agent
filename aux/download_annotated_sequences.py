import argparse
import io
import json
import os
import subprocess
import tempfile
import time
from typing import Dict, List, Tuple

import fsspec
from Bio import Entrez, SeqIO

Entrez.email = os.environ.get("ENTREZ_EMAIL", "663vova@gmail.com")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY

CLASSES = ["GAG", "INT", "RT", "RN", "LTR"]


def classify_feature(feature) -> List[str]:
    qualifiers = feature.qualifiers
    desc_parts = []
    for key in ("product", "note", "gene", "function", "standard_name"):
        if key in qualifiers:
            desc_parts.append(str(qualifiers[key][0]))
    desc = " ".join(desc_parts).upper()
    rpt_type = str(qualifiers.get("rpt_type", [""])[0]).upper()

    hits: List[str] = []

    if feature.type == "repeat_region":
        if "LONG_TERMINAL_REPEAT" in rpt_type or "LTR" in desc or "LONG TERMINAL REPEAT" in desc:
            hits.append("LTR")
        return hits

    if feature.type == "LTR":
        hits.append("LTR")
        return hits

    if "LONG TERMINAL REPEAT" in desc or " LTR" in f" {desc}" or desc.startswith("LTR") or "5'LTR" in desc or "3'LTR" in desc:
        hits.append("LTR")

    if "INTEGRASE" in desc:
        hits.append("INT")
    if "REVERSE TRANSCRIPTASE" in desc or "RVT" in desc:
        hits.append("RT")
    if "RNASE H" in desc or "RNASEH" in desc or "RIBONUCLEASE H" in desc:
        hits.append("RN")
    if "GAG" in desc or "CAPSID" in desc or "NUCLEOCAPSID" in desc:
        hits.append("GAG")

    return hits


def parse_features(seq_record) -> Dict[str, List[Tuple[int, int]]]:
    domains: Dict[str, List[Tuple[int, int]]] = {c: [] for c in CLASSES}
    if not hasattr(seq_record, "features"):
        return domains
    allowed = {"CDS", "mat_peptide", "misc_feature", "repeat_region", "LTR", "protein_bind", "gene"}
    for feature in seq_record.features:
        if feature.type not in allowed:
            continue
        try:
            start = int(feature.location.start)
            end = int(feature.location.end)
        except Exception:
            continue
        if end <= start:
            continue
        for cls in classify_feature(feature):
            domains[cls].append((start, end))
    return domains


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


def search_ids(query: str, max_results: int) -> List[str]:
    print(f"Searching: {query}")
    try:
        handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_results, sort="length")
        record = Entrez.read(handle)
        handle.close()
        ids = record["IdList"]
        print(f"  found {len(ids)}")
        return ids
    except Exception as e:
        print(f"  error: {e}")
        return []


def fetch_record(accession: str, attempt: int = 0):
    if attempt > 3:
        return None
    try:
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        return record
    except Exception as e:
        print(f"  fetch error {accession}: {e}")
        time.sleep(1.5 * (attempt + 1))
        return fetch_record(accession, attempt + 1)


def write_text(fs, path: str, text: str) -> None:
    with fs.open(path, "w") as f:
        f.write(text)


class AnnotatedSequenceDownloader:

    def __init__(self, output_uri: str, min_len: int = 5000, require_domains: bool = True):
        self.output_uri = output_uri.rstrip("/")
        self.sequences_uri = f"{self.output_uri}/sequences"
        self.fs, _ = fsspec.core.url_to_fs(self.output_uri)
        self.fs.makedirs(self.output_uri, exist_ok=True)
        self.fs.makedirs(self.sequences_uri, exist_ok=True)
        self.min_len = min_len
        self.require_domains = require_domains
        self.annotations: List[dict] = []
        self.seen: set = set()
        self._load_existing_annotations()

    def _load_existing_annotations(self) -> None:
        ann_path = f"{self.output_uri}/annotations.json"
        try:
            with fsspec.open(ann_path, "r") as f:
                self.annotations = json.load(f)
            self.seen = {a["accession"] for a in self.annotations}
            print(f"Loaded {len(self.annotations)} existing annotations")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: could not load existing annotations: {e}")

    def download_and_save(self, accession: str) -> bool:
        if accession in self.seen:
            return False
        self.seen.add(accession)
        record = fetch_record(accession)
        if record is None:
            return False
        try:
            seq_len = len(record.seq)
        except Exception:
            print(f"  skip {accession}: undefined sequence")
            return False
        if seq_len < self.min_len:
            print(f"  skip {accession}: short ({seq_len}bp)")
            return False
        domains = parse_features(record)
        domains = {k: merge_intervals(v) for k, v in domains.items()}
        if self.require_domains and not any(domains.values()):
            print(f"  skip {accession}: no domains")
            return False

        rec_id = record.id or accession
        fasta_path = f"{self.sequences_uri}/{rec_id}.fasta"

        try:
            buf = io.StringIO()
            SeqIO.write(record, buf, "fasta")
            buf.seek(0)
            write_text(self.fs, fasta_path, buf.getvalue())
        except Exception as e:
            print(f"  fail {rec_id}: write error {e}")
            return False

        self.annotations.append({
            "accession": rec_id,
            "description": record.description,
            "length": seq_len,
            "domains": domains,
            "file": f"sequences/{rec_id}.fasta",
        })
        present = [d for d, v in domains.items() if v]
        print(f"  ok {rec_id} ({seq_len}bp) -> {present}")
        return True

    def download_dataset(self, queries: List[str], max_per_query: int) -> int:
        all_ids: List[str] = []
        for q in queries:
            all_ids.extend(search_ids(q, max_per_query))
            time.sleep(0.4)
        all_ids = list(dict.fromkeys(all_ids))
        print(f"\nUnique candidates: {len(all_ids)}\n")
        success = 0
        for i, acc in enumerate(all_ids, 1):
            if self.download_and_save(acc):
                success += 1
            time.sleep(0.25)
            if i % 10 == 0:
                self.save_annotations()
                print(f"  [{i}/{len(all_ids)}] saved {success} (checkpoint)")
        return success

    def save_annotations(self) -> None:
        path = f"{self.output_uri}/annotations.json"
        write_text(self.fs, path, json.dumps(self.annotations, indent=2))
        print(f"\nWrote {len(self.annotations)} annotations -> {path}")
        counts = {c: 0 for c in CLASSES}
        for ann in self.annotations:
            for d, v in ann["domains"].items():
                if v:
                    counts[d] = counts.get(d, 0) + 1
        print("Sequences with each domain class:")
        for c in CLASSES:
            print(f"  {c}: {counts.get(c, 0)}")


DEFAULT_QUERIES = [
    '(retrotransposon OR transposon) AND (integrase OR reverse transcriptase OR RNase)',
    'LTR retrotransposon AND complete',
    'long terminal repeat AND retrotransposon',
    'copia[organism] AND reverse transcriptase',
    'gypsy[organism] AND integrase',
    'ty1[organism] AND protein',
    'Ty3[organism] AND domain',
    'endogenous retrovirus AND LTR',
]


def default_output_uri() -> str:
    bucket = os.environ.get("GCS_BUCKET")
    prefix = os.environ.get("GCS_PREFIX", "annotated_sequences")
    if bucket:
        return f"gs://{bucket}/{prefix}"
    return str(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "annotated_sequences")))


def parse_dfam_header(header: str) -> Tuple[str, Dict[str, List[Tuple[int, int]]]]:
    accession = header.split()[0] if header else "unknown"
    domains: Dict[str, List[Tuple[int, int]]] = {c: [] for c in CLASSES}

    header_upper = header.upper()
    if "LTR" in header_upper:
        domains["LTR"] = [(0, 1)]
    if "COPIA" in header_upper or "TY1" in header_upper:
        domains["GAG"] = [(0, 1)]
        domains["RT"] = [(0, 1)]
        domains["INT"] = [(0, 1)]
    if "GYPSY" in header_upper or "TY3" in header_upper:
        domains["GAG"] = [(0, 1)]
        domains["RT"] = [(0, 1)]
        domains["INT"] = [(0, 1)]
    if "RETROVIRUS" in header_upper:
        domains["GAG"] = [(0, 1)]
        domains["RT"] = [(0, 1)]
        domains["INT"] = [(0, 1)]

    return accession, domains


def download_dfam_sequences(output_uri: str, min_len: int = 5000) -> int:
    print("Downloading DFAM sequences...")
    dfam_url = "https://www.dfam.org/releases/Dfam_3.8/families/Dfam_3.8.fasta.gz"

    with tempfile.TemporaryDirectory() as tmpdir:
        gz_path = os.path.join(tmpdir, "dfam.fasta.gz")
        fasta_path = os.path.join(tmpdir, "dfam.fasta")

        print(f"Downloading {dfam_url}...")
        try:
            subprocess.run(["wget", "-q", dfam_url, "-O", gz_path], check=True, timeout=300)
        except Exception as e:
            print(f"wget failed: {e}, trying curl...")
            subprocess.run(["curl", "-s", "-L", dfam_url, "-o", gz_path], check=True, timeout=300)

        print(f"Extracting...")
        subprocess.run(["gunzip", "-f", gz_path], check=True)

        print(f"Parsing {fasta_path}...")
        fs, _ = fsspec.core.url_to_fs(output_uri)
        sequences_uri = f"{output_uri.rstrip('/')}/sequences"
        fs.makedirs(sequences_uri, exist_ok=True)

        annotations = []
        success = 0

        for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
            if i % 500 == 0:
                print(f"  [{i}] processed, saved {success}")

            seq_len = len(record.seq)
            if seq_len < min_len:
                continue

            rec_id = record.id
            header = record.description

            _, domains = parse_dfam_header(header)
            if not any(domains.values()):
                continue

            try:
                buf = io.StringIO()
                SeqIO.write(record, buf, "fasta")
                buf.seek(0)
                fasta_uri = f"{sequences_uri}/{rec_id}.fasta"
                with fsspec.open(fasta_uri, "w") as f:
                    f.write(buf.getvalue())
            except Exception as e:
                continue

            domains_for_json = {k: [(0, seq_len)] if v else [] for k, v in domains.items()}
            annotations.append({
                "accession": rec_id,
                "description": header,
                "length": seq_len,
                "domains": domains_for_json,
                "file": f"sequences/{rec_id}.fasta",
            })
            success += 1

        print(f"\nSaved {success} DFAM sequences")

        ann_path = f"{output_uri.rstrip('/')}/annotations.json"
        with fsspec.open(ann_path, "w") as f:
            f.write(json.dumps(annotations, indent=2))
        print(f"Wrote annotations -> {ann_path}")

        counts = {c: 0 for c in CLASSES}
        for ann in annotations:
            for d, v in ann["domains"].items():
                if v:
                    counts[d] = counts.get(d, 0) + 1
        print("Sequences with each domain class:")
        for c in CLASSES:
            print(f"  {c}: {counts.get(c, 0)}")

        return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["ncbi", "dfam"], default="ncbi", help="data source (ncbi or dfam)")
    parser.add_argument("--out", default=default_output_uri(), help="local path or gs://bucket/prefix")
    parser.add_argument("--max-per-query", type=int, default=30, help="max results per NCBI query")
    parser.add_argument("--min-len", type=int, default=5000)
    parser.add_argument("--query", action="append", default=None, help="override default NCBI queries; repeat for multiple")
    parser.add_argument("--no-require-domains", action="store_true", help="save sequences even when no recognized features found (for downstream HMM annotation)")
    args = parser.parse_args()

    print(f"Source: {args.source}")
    print(f"Output: {args.out}")

    if args.source == "dfam":
        n = download_dfam_sequences(args.out, min_len=args.min_len)
    else:
        queries = args.query if args.query else DEFAULT_QUERIES
        downloader = AnnotatedSequenceDownloader(
            args.out,
            min_len=args.min_len,
            require_domains=not args.no_require_domains,
        )
        n = downloader.download_dataset(queries, args.max_per_query)
        downloader.save_annotations()

    print(f"\nDone. {n} sequences saved.")


if __name__ == "__main__":
    main()