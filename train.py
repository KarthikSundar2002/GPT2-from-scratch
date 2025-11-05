import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from config import GPTConfig
from GPT2 import GPT
from dataloader import DataLoader

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

optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)

dataloader = DataLoader(B, T)
for i in range(50):
    x, y = next(dataloader)
    optimizer.zero_grad()
    logits, loss = m(x.to(device), y.to(device))
    loss.backward()
    optimizer.step()
    print(f"Step {i}: loss = {loss.item()}")
