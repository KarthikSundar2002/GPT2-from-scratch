
import torch
import torch.nn as nn
from torch.nn import functional as F

from CausalSelfAttention import CausalSelfAttention
from MLP import MLP
from config import GPTConfig
from yarn import Yarn

from CausalSelfAttention import norm

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x, cos, sin):
        x = x + self.attn(norm(x), cos, sin)
        x = x + self.mlp(norm(x))
        return x


    

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word embedding table,
           # wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding,
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.yarn = Yarn(config.head_dim, config.block_size, config.block_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        for name, module in self.transformer.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                if "c_proj" in name:
                   continue
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
       # pos_emb = self.transformer.wpe(pos)
        x = tok_emb
        cos, sin = self.yarn.cos, self.yarn.sin
        for block in self.transformer.h:
            x = block(x, cos, sin)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": [p for p in decay_params], "weight_decay": weight_decay},
            {"params": [p for p in nodecay_params], "weight_decay": 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=True)
        return optimizer
    
    def load_state_dict(self, state_dict, optimizer_state_dict):
        cur_state_dict = self.state_dict() 
        for key in cur_state_dict.keys():
                cur_state_dict[key].data.copy_(state_dict['_orig_mod.'+key].data)
        optimizer = self.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), eps=1e-8)
        optimizer.load_state_dict(optimizer_state_dict)
        return self, optimizer