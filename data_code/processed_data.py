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

    def load_fasta_sequences(self, directory: str) -> Dict[str, str]:
        sequences = {}
        fasta_dir = Path(directory)

        for fasta_file in fasta_dir.glob("*"):
            if fasta_file.is_file():
                try:
                    for record in SeqIO.parse(str(fasta_file), "fasta"):
                        sequences[record.id] = str(record.seq)
                except Exception as e:
                    print(f"Error reading {fasta_file}: {e}")

        return sequences

    def get_embeddings(self, sequence: str) -> torch.Tensor:
        inputs = self.tokenizer(sequence, return_tensors="pt", max_length=1024, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        embeddings = outputs.hidden_states[-1]
        return embeddings

    def process_dataset(self, data_dir: str):
        positives_dir = os.path.join(data_dir, "positives")
        negatives_dir = os.path.join(data_dir, "negatives")

        int_embeddings = {}
        rt_embeddings = {}
        negatives_embeddings = {}

        if os.path.exists(positives_dir):
            print("Processing positives...")
            pos_sequences = self.load_fasta_sequences(positives_dir)
            for name, seq in pos_sequences.items():
                if len(seq) > 0:
                    emb = self.get_embeddings(seq).cpu()
                    if name.startswith("INT"):
                        int_embeddings[name] = emb
                    elif name.startswith("RT"):
                        rt_embeddings[name] = emb

        if os.path.exists(negatives_dir):
            print("Processing negatives...")
            neg_sequences = self.load_fasta_sequences(negatives_dir)
            for name, seq in neg_sequences.items():
                if len(seq) > 0:
                    negatives_embeddings[name] = self.get_embeddings(seq).cpu()

        return int_embeddings, rt_embeddings, negatives_embeddings


if __name__ == "__main__":
    processor = GeneticDataProcessor()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    try:
        int_emb, rt_emb, neg_emb = processor.process_dataset(data_dir)
        print(f"INT: {len(int_emb)}")
        print(f"RT: {len(rt_emb)}")
        print(f"Negatives: {len(neg_emb)}")

        torch.save(int_emb, "INT_embeddings.pt")
        torch.save(rt_emb, "RT_embeddings.pt")
        torch.save(neg_emb, "negatives_embeddings.pt")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
