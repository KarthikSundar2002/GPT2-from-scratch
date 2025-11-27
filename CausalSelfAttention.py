import torch
import torch.nn as nn
from torch.nn import functional as F

import einops
import math

from config import GPTConfig
from yarn import rotary
from MLP import CastedLinear

from torch.nn.attention.flex_attention import flex_attention
flex_attention = torch.compile(flex_attention, dynamic=False)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = CastedLinear(config.n_embd, 3 * config.n_embd)

     
        self.c_proj = CastedLinear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # Causal Mask
        with torch.no_grad():
            self.c_proj.weight.data.zero_()
            self.c_proj.bias.data.zero_()

    def forward(self, x, cos, sin, block_mask):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q,k,v = einops.rearrange(qkv, "b t (a h d) -> a b t h d", a=3, h=self.n_head)
        q, k = norm(q), norm(k)
        q = rotary(q, cos, sin)
        k = rotary(k, cos, sin)
        q = einops.rearrange(q, "b t h d -> b h t d")
        k = einops.rearrange(k, "b t h d -> b h t d")
        v = einops.rearrange(v, "b t h d -> b h t d")
        y = flex_attention(q,k,v, block_mask=block_mask)
        y = einops.rearrange(y, "b h t d -> b t (h d)")
        return self.c_proj(y)
