import torch
import torch.nn as nn
from torch.nn import functional as F


class CastedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = CastedLinear(config.n_embd, 4 * config.n_embd)
        self.c_proj = CastedLinear(4 * config.n_embd, config.n_embd)
        with torch.no_grad():
            self.c_proj.weight.data.zero_()
            self.c_proj.bias.data.zero_()
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x