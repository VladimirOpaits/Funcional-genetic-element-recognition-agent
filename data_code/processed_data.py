import os
import sys
import torch
from pathlib import Path
from typing import Dict, Tuple
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.nt_model import NucleotideTransformer


class GeneticDataProcessor:
    def __init__(
        self,
        nt: NucleotideTransformer = None,
        model_name: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        device: str = "cuda",
    ):
        self.nt = nt or NucleotideTransformer(model_name=model_name, device=device)

    def get_embeddings(self, sequence: str) -> torch.Tensor:
        return self.nt.embed(sequence)

    @staticmethod
    def _iter_fasta_files(root: Path):
        if not root.exists():
            return
        for path in sorted(root.rglob("*")):
            if path.is_file():
                yield path

    @staticmethod
    def _category_from_filename(path: Path) -> str:
        return path.stem.split("_")[0]

    def _embed_fasta(self, path: Path, store: Dict[str, torch.Tensor]) -> int:
        added = 0
        try:
            for record in SeqIO.parse(str(path), "fasta"):
                if len(record.seq) == 0:
                    continue
                key = f"{path.stem}:{record.id}"
                if key in store:
                    key = f"{path.stem}:{record.id}#{added}"
                store[key] = self.get_embeddings(str(record.seq))
                added += 1
        except Exception as e:
            print(f"Error reading {path.name}: {e}")
        return added

    def process_dataset(self, data_dir: str) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        data_path = Path(data_dir)
        positives_dir = data_path / "positives"
        negatives_dir = data_path / "negatives"

        categories: Dict[str, Dict[str, torch.Tensor]] = {}
        negatives: Dict[str, torch.Tensor] = {}

        print("Processing positives...")
        for fasta_file in self._iter_fasta_files(positives_dir):
            category = self._category_from_filename(fasta_file)
            store = categories.setdefault(category, {})
            n = self._embed_fasta(fasta_file, store)
            print(f"  {fasta_file.relative_to(positives_dir)} -> {category}: +{n} (total {len(store)})")

        print("Processing negatives...")
        for fasta_file in self._iter_fasta_files(negatives_dir):
            n = self._embed_fasta(fasta_file, negatives)
            print(f"  {fasta_file.relative_to(negatives_dir)}: +{n} (total {len(negatives)})")

        return categories, negatives


if __name__ == "__main__":
    processor = GeneticDataProcessor()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    try:
        categories, neg_emb = processor.process_dataset(data_dir)

        for category, emb in categories.items():
            print(f"{category}: {len(emb)}")
            torch.save(emb, f"{category}_embeddings.pt")

        print(f"Negatives: {len(neg_emb)}")
        torch.save(neg_emb, "negatives_embeddings.pt")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
