import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from Bio import Entrez, SeqIO
import time

Entrez.email = "663vova@gmail.com"

class AnnotatedSequenceDownloader:

    def __init__(self, output_dir: str = "../data/annotated_sequences"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequences_dir = self.output_dir / "sequences"
        self.sequences_dir.mkdir(exist_ok=True)
        self.annotations = []

    def search_retrotransposons(self, query: str, max_results: int = 50) -> List[str]:
        print(f"Searching GenBank for: {query}")
        try:
            handle = Entrez.esearch(
                db="nucleotide",
                term=query,
                retmax=max_results,
                sort="length"
            )
            record = Entrez.read(handle)
            handle.close()

            ids = record["IdList"]
            print(f"Found {len(ids)} sequences")
            return ids
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def parse_features(self, seq_record) -> Dict[str, List[Tuple[int, int]]]:
        domains = {"INT": [], "RT": [], "RN": [], "GAG": []}

        if not hasattr(seq_record, "features"):
            return domains

        for feature in seq_record.features:
            if feature.type not in ["CDS", "misc_feature", "protein_bind"]:
                continue

            location = feature.location
            start = int(location.start)
            end = int(location.end)

            desc = ""
            if "product" in feature.qualifiers:
                desc = str(feature.qualifiers["product"][0]).upper()
            elif "note" in feature.qualifiers:
                desc = str(feature.qualifiers["note"][0]).upper()

            # Map to domain types
            if "INTEGRASE" in desc or "INT" in desc:
                domains["INT"].append((start, end))
            elif "REVERSE TRANSCRIPTASE" in desc or "RT" in desc:
                domains["RT"].append((start, end))
            elif "RNASE" in desc or "RN" in desc:
                domains["RN"].append((start, end))
            elif "GAG" in desc:
                domains["GAG"].append((start, end))

        return domains

    def download_and_save(self, accession: str, attempt: int = 0) -> bool:
        if attempt > 3:
            return False

        try:
            print(f"  Downloading {accession}...", end=" ")
            handle = Entrez.efetch(
                db="nucleotide",
                id=accession,
                rettype="gb",
                retmode="text"
            )
            record = SeqIO.read(handle, "genbank")
            handle.close()

            seq_len = len(record.seq)
            if seq_len < 5000:  
                print(f"skipped (too short: {seq_len}bp)")
                return False

            domains = self.parse_features(record)

            if not any(domains.values()):
                print(f"skipped (no domains)")
                return False

            fasta_file = self.sequences_dir / f"{accession}.fasta"
            SeqIO.write(record, str(fasta_file), "fasta")

            self.annotations.append({
                "accession": accession,
                "description": record.description,
                "length": seq_len,
                "domains": domains,
                "file": f"sequences/{accession}.fasta"
            })

            print(f"✓ ({seq_len}bp, domains: {[d for d in domains if domains[d]]})")
            return True

        except Exception as e:
            print(f"error: {e}")
            time.sleep(1)
            return self.download_and_save(accession, attempt + 1)

    def download_dataset(self):
        """Download comprehensive retrotransposon dataset."""

        # Different search queries to get diverse data
        queries = [
            '(retrotransposon OR transposon) AND (integrase OR reverse transcriptase OR RNase)',
            'copia[organism] AND reverse transcriptase',
            'gypsy[organism] AND integrase',
            'ty1[organism] AND protein',
            'Ty3[organism] AND domain',
        ]

        all_ids = []
        for query in queries:
            ids = self.search_retrotransposons(query, max_results=30)
            all_ids.extend(ids)
            time.sleep(0.5)  # Be nice to NCBI

        # Remove duplicates
        all_ids = list(set(all_ids))
        print(f"\nTotal unique sequences: {len(all_ids)}\n")

        # Download each
        success_count = 0
        for i, accession in enumerate(all_ids, 1):
            if self.download_and_save(accession):
                success_count += 1
            time.sleep(0.3)  # Rate limiting

            if i % 20 == 0:
                print(f"  [{i}/{len(all_ids)}] Downloaded {success_count} sequences")

        return success_count

    def save_annotations(self):
        """Save annotations to JSON file."""
        annotation_file = self.output_dir / "annotations.json"
        with open(annotation_file, "w") as f:
            json.dump(self.annotations, f, indent=2)

        print(f"\nSaved {len(self.annotations)} annotations to {annotation_file}")

        # Summary
        domain_counts = {"INT": 0, "RT": 0, "RN": 0, "GAG": 0}
        for ann in self.annotations:
            for domain, coords in ann["domains"].items():
                if coords:
                    domain_counts[domain] += 1

        print("Domain statistics:")
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count} sequences")


if __name__ == "__main__":
    downloader = AnnotatedSequenceDownloader()
    success_count = downloader.download_dataset()
    downloader.save_annotations()
    print(f"\nDownload complete! Got {success_count} sequences")
