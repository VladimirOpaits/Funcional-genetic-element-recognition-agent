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

    def process_dataset(self, data_dir: str) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        positives_dir = Path(os.path.join(data_dir, "positives"))
        negatives_dir = Path(os.path.join(data_dir, "negatives"))

        categories: Dict[str, Dict[str, torch.Tensor]] = {}
        negatives: Dict[str, torch.Tensor] = {}

        if positives_dir.exists():
            print("Processing positives...")
            for fasta_file in sorted(positives_dir.glob("*")):
                if not fasta_file.is_file():
                    continue
                category = fasta_file.stem.split("_")[0]
                if category not in categories:
                    categories[category] = {}
                try:
                    for record in SeqIO.parse(str(fasta_file), "fasta"):
                        if len(record.seq) > 0:
                            categories[category][record.id] = self.get_embeddings(str(record.seq)).cpu()
                except Exception as e:
                    print(f"Error reading {fasta_file.name}: {e}")

        if negatives_dir.exists():
            print("Processing negatives...")
            for fasta_file in negatives_dir.glob("*"):
                if not fasta_file.is_file():
                    continue
                try:
                    for record in SeqIO.parse(str(fasta_file), "fasta"):
                        if len(record.seq) > 0:
                            negatives[record.id] = self.get_embeddings(str(record.seq)).cpu()
                except Exception as e:
                    print(f"Error reading {fasta_file.name}: {e}")

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
