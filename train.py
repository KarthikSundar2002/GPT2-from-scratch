import torch
import torch.nn as nn
from torch.nn import functional as F
import time

from config import GPTConfig
from GPT2 import GPT
from dataloader import DataLoader

NUM_EPOCHS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig())
start_time = time.time()
m = model.to(device).train()
m = torch.compile(m)
end_time = time.time()
print(f"Time to load model: {end_time - start_time} seconds")

optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)

dataloader = DataLoader(B=2, T=1024)
start_time = time.time()
for epoch in range(NUM_EPOCHS):
    for i in range(len(dataloader)):
        x, y = next(dataloader)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = m(x.to(device), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Step {i}: loss = {loss.item()}")
torch.cuda.synchronize()
end_time = time.time()
print(f"Time to train: {end_time - start_time} seconds")