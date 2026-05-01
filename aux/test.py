import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")

sequence = "ATTCCGATTCCGGTACGCCGTAGCTAGCTAGCTAGCTAGCTAG"

inputs = tokenizer(sequence, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

embeddings = outputs.hidden_states[-1]

print(embeddings.shape)
print(embeddings.device) 