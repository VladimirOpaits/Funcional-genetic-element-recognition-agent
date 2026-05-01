import os
import torch
from pathlib import Path
from typing import Dict, Tuple
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM


class GeneticDataProcessor:
    def __init__(self, model_name: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(device)
        self.model.eval()

    def get_embeddings(self, sequence: str) -> torch.Tensor:
        inputs = self.tokenizer(sequence, return_tensors="pt", max_length=1024, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        return outputs.hidden_states[-1]

    @staticmethod
    def _iter_fasta_files(root: Path):
        """Recursively yield FASTA-like files. Accepts files with or without extensions."""
        if not root.exists():
            return
        for path in sorted(root.rglob("*")):
            if path.is_file():
                yield path

    @staticmethod
    def _category_from_filename(path: Path) -> str:
        """
        Map a file to its category, supporting both naming schemes:
          old:  RT_copia, INT_gypsy, RNaseH_athila      -> 'RT' / 'INT' / 'RNaseH'
          new:  retro_domains/RT.fasta                  -> 'RT'
        """
        return path.stem.split("_")[0]

    def _embed_fasta(self, path: Path, store: Dict[str, torch.Tensor]) -> int:
        """Parse one FASTA, embed each non-empty sequence, write into store. Returns count."""
        added = 0
        try:
            for record in SeqIO.parse(str(path), "fasta"):
                if len(record.seq) == 0:
                    continue
                # Prefix with file stem so identical record IDs from different
                # source files (or repeated mat_peptides in one NCBI record)
                # don't overwrite each other.
                key = f"{path.stem}:{record.id}"
                if key in store:
                    key = f"{path.stem}:{record.id}#{added}"
                store[key] = self.get_embeddings(str(record.seq)).cpu()
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
