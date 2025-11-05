import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math

from config import GPTConfig
from GPT2 import GPT
from dataloader import DataLoader

NUM_EPOCHS = 100
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

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps=100
def get_lr(step):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > NUM_EPOCHS * len(dataloader)*0.9:
        return min_lr
    decay_ratio = (step - warmup_steps) / ((NUM_EPOCHS * len(dataloader))*0.9 - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = m.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, betas=(0.9, 0.95), eps=1e-8)

dataloader = DataLoader(B=2, T=1024)
start_time = time.time()
for epoch in range(NUM_EPOCHS):
    for i in range(len(dataloader)):
        x, y = next(dataloader)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = m(x.to(device), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        lr = get_lr(i + epoch * len(dataloader))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        print(f"Epoch {epoch}, Step {i}: loss = {loss.item()}, norm = {norm}")
torch.cuda.synchronize()
end_time = time.time()
torch.save(m.state_dict(), "/scratch/ks02450/model.pth")
torch.save(optimizer.state_dict(), "/scratch/ks02450/optimizer.pth")
print(f"Time to train: {end_time - start_time} seconds")