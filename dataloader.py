import tiktoken
import torch

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open("combined.txt", "r") as f:
            text = f.read()
        self.tokens = tiktoken.get_encoding("gpt2").encode(text)
        self.data = torch.tensor(self.tokens)

        self.current_idx = 0

    def __len__(self):
        return len(self.data) // self.B // self.T
    
    def __iter__(self):
        return self

    def __next__(self):
        B, T = self.B, self.T
        buf = self.data[self.current_idx:self.current_idx + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_idx += B*T
        
        if self.current_idx >= len(self.data) - B*T - 1:
            self.current_idx = 0
        
        return x, y