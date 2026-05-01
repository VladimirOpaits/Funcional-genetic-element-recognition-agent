import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"

print("Start")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForMaskedLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.float16
).to("cuda")

print("Ready")