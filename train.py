import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from config import GPTConfig
from GPT2 import GPT
from dataloader import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = GPT(GPTConfig())
m = model.to(device).train()

optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)

dataloader = DataLoader(B=4, T=32)
for i in range(50):
    x, y = next(dataloader)
    optimizer.zero_grad()
    logits, loss = m(x.to(device), y.to(device))
    loss.backward()
    optimizer.step()
    print(f"Step {i}: loss = {loss.item()}")
