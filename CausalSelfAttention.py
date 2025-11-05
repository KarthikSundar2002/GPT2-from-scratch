import torch
import torch.nn as nn
from torch.nn import functional as F

import einops
import math

from config import GPTConfig
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # Causal Mask

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q,k,v = einops.rearrange(qkv, "b t (a h d) -> a b h t d", a=3, h=self.n_head)
       

        att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(C // self.n_head))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = einops.rearrange(y, "b h t d -> b t (h d)")
        return self.c_proj(y)
