"""
Download nucleotide sequences of RT / INT / RNaseH domains from GenBank.

Strategy: search NCBI nuccore for retrotransposon / retrovirus / LTR records,
fetch full GenBank entries, then extract mat_peptide (and CDS) features whose
/product qualifier matches one of the three target domains. Coordinates come
straight from the feature, so slicing is exact.

This bypasses Pfam/UniProt -> ENA cross-references entirely.
"""

import argparse
import re
import time
from pathlib import Path
from typing import Optional

from Bio import Entrez, SeqIO
from tqdm import tqdm

Entrez.email = "663vova@gmail.com"

QUERIES = {
    "RT":     '(retrotransposon[Title] OR retrovirus[Title] OR LTR[Title] OR "retroelement"[Title]) '
              'AND ("reverse transcriptase"[All Fields]) AND 60:5000[SLEN]',
    "INT":    '(retrotransposon[Title] OR retrovirus[Title] OR LTR[Title] OR "retroelement"[Title]) '
              'AND (integrase[All Fields]) AND 60:5000[SLEN]',
    "RNaseH": '(retrotransposon[Title] OR retrovirus[Title] OR LTR[Title] OR "retroelement"[Title]) '
              'AND ("ribonuclease H"[All Fields] OR "RNase H"[All Fields]) AND 60:5000[SLEN]',
}

PRODUCT_PATTERNS = {
    "RT":     re.compile(r"reverse\s*transcriptase|\bRT\b|p66|p51", re.IGNORECASE),
    "INT":    re.compile(r"\bintegrase\b|\bIN\b(?!hibitor)", re.IGNORECASE),
    "RNaseH": re.compile(r"ribonuclease\s*H|RNase\s*H|\bRNH\b", re.IGNORECASE),
}

EXCLUDE_PATTERNS = {
    "RT":     re.compile(r"polyprotein|pol\s*protein|gag-pol", re.IGNORECASE),
    "INT":    re.compile(r"polyprotein|pol\s*protein|gag-pol", re.IGNORECASE),
    "RNaseH": re.compile(r"polyprotein", re.IGNORECASE),
}

MIN_LEN, MAX_LEN = 60, 1000
BATCH_SIZE = 50


def append_fasta(fh, header: str, seq: str) -> None:
    fh.write(f">{header}\n")
    for i in range(0, len(seq), 80):
        fh.write(seq[i:i + 80] + "\n")
    fh.flush()


def entrez_call(fn, retries: int = 4, backoff: float = 2.0, **kwargs):
    for attempt in range(retries):
        try:
            return fn(**kwargs)
        except Exception:
            time.sleep(backoff ** attempt)
    return None


def search_ids(query: str, retmax: int) -> tuple[Optional[str], Optional[str], int]:
    """Return (WebEnv, QueryKey, count) for a search using history server."""
    h = entrez_call(Entrez.esearch, db="nuccore", term=query,
                    retmax=retmax, usehistory="y")
    if h is None:
        return None, None, 0
    res = Entrez.read(h)
    h.close()
    return res["WebEnv"], res["QueryKey"], min(int(res["Count"]), retmax)


def fetch_batch(webenv: str, query_key: str, start: int, size: int):
    h = entrez_call(Entrez.efetch, db="nuccore", rettype="gb", retmode="text",
                    retstart=start, retmax=size,
                    webenv=webenv, query_key=query_key)
    if h is None:
        return []
    try:
        records = list(SeqIO.parse(h, "genbank"))
    except Exception:
        records = []
    h.close()
    return records


def classify_product(product: str, target: str) -> bool:
    if EXCLUDE_PATTERNS[target].search(product):
        return False
    return bool(PRODUCT_PATTERNS[target].search(product))


def extract_domains(record, target: str) -> list[tuple[str, str]]:
    """Yield (header, nt_seq) tuples for features matching the target domain."""
    out = []
    for feature in record.features:
        if feature.type not in ("mat_peptide", "CDS", "Region", "misc_feature"):
            continue
        products = feature.qualifiers.get("product", []) + \
                   feature.qualifiers.get("note", []) + \
                   feature.qualifiers.get("region_name", [])
        if not any(classify_product(p, target) for p in products):
            continue
        try:
            seq = str(feature.extract(record.seq)).upper().replace("U", "T")
        except Exception:
            continue
        if not (MIN_LEN <= len(seq) <= MAX_LEN):
            continue
        # skip ambiguous-base junk
        if seq.count("N") > len(seq) * 0.1:
            continue
        product = products[0] if products else "unknown"
        product_clean = re.sub(r"[^A-Za-z0-9]+", "_", product)[:40]
        header = f"{record.id}|{feature.location}|{product_clean}|{target}"
        out.append((header, seq))
    return out


def process_class(name: str, query: str, out_dir: Path, retmax: int) -> None:
    print(f"\n[{name}] querying NCBI nuccore (retmax={retmax})")
    webenv, query_key, count = search_ids(query, retmax)
    if not webenv or count == 0:
        print(f"[{name}] no hits")
        return
    print(f"[{name}] {count} candidate records")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.fasta"
    written = 0
    seen_seqs: set[str] = set()

    with open(out_path, "w") as fh:
        for start in tqdm(range(0, count, BATCH_SIZE), desc=name):
            records = fetch_batch(webenv, query_key, start, BATCH_SIZE)
            for record in records:
                for header, seq in extract_domains(record, name):
                    if seq in seen_seqs:
                        continue
                    seen_seqs.add(seq)
                    append_fasta(fh, header, seq)
                    written += 1
            time.sleep(0.34)  # NCBI rate limit (3 req/s without API key)

    print(f"[{name}] wrote {written} unique domain sequences -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/retro_domains"))
    ap.add_argument("--retmax", type=int, default=2000,
                    help="max GenBank records to scan per class")
    args = ap.parse_args()

    for name, query in QUERIES.items():
        process_class(name, query, args.out, args.retmax)


if __name__ == "__main__":
    main()
