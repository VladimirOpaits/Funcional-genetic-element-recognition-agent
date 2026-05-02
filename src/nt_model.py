from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


DEFAULT_MODEL = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
DEFAULT_MAX_LENGTH = 1024


class NucleotideTransformer:
    """Wrapper around HuggingFace Nucleotide Transformer.

    Use ``embed`` for plain hidden-state extraction, ``embed_with_offsets``
    when you need to know which nucleotides each token represents
    (needed for token-level annotation/labeling).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
        self.model.eval()
        self._special_ids = set(self.tokenizer.all_special_ids)

    def embed(self, sequence: str) -> torch.Tensor:
        inputs = self.tokenizer(
            sequence, return_tensors="pt", max_length=self.max_length, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].cpu()

    def embed_with_offsets(
        self, sequence: str
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Return (hidden_states, offsets).

        ``hidden_states`` has shape (1, num_tokens, hidden_dim).
        ``offsets[i]`` is the (start, end) nucleotide span of token ``i``
        within ``sequence``. Special tokens get zero-width spans (e.g. (0, 0)).
        """
        inputs = self.tokenizer(
            sequence, return_tensors="pt", max_length=self.max_length, truncation=True
        )
        input_ids = inputs["input_ids"][0].tolist()
        offsets = self._compute_offsets(input_ids)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].cpu(), offsets

    def _compute_offsets(self, input_ids: List[int]) -> List[Tuple[int, int]]:
        offsets: List[Tuple[int, int]] = []
        cursor = 0
        for tid in input_ids:
            if tid in self._special_ids:
                offsets.append((cursor, cursor))
                continue
            token = self.tokenizer.convert_ids_to_tokens(tid)
            n = len(token)
            offsets.append((cursor, cursor + n))
            cursor += n
        return offsets
