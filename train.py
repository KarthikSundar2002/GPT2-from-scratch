import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from config import GPTConfig
from GPT2 import GPT

enc = tiktoken.get_encoding("gpt2")
with open("input.txt", "r") as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buffer = torch.tensor(tokens[:B*T + 1], dtype=torch.long)
x = buffer[:-1].view(B, T)
y = buffer[1:].view(B, T)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = GPT(GPTConfig())
m = model.to(device).train()

logits, loss = m(x.to(device), y.to(device))

print(loss.item())
