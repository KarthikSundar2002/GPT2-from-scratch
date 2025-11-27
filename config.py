from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024 * 64
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    head_dim: int = n_embd // n_head