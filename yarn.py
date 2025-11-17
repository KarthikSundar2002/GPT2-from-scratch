import torch
import torch.nn as nn
from torch.nn import functional as F

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len, block_size):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4,dtype=torch.float32, device=self.device)
        angular_freq = torch.cat([angualr_freq, angualar_freq.new_zeros()])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(
            theta.cos().to(torch.bfloat16), persistent=False
        )

        self.sin = nn.Buffer(
            theta.sin().to(torch.bfloat16), persistent=False
        )

        self.angular_freq = angular_freq
        self.attn_scale = 0.1
    
    def apply(self, old_window, new_window, alpha=1, beta=32):
        rotations = self.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)

        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window/old_window) + 1


