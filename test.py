import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken
from GPT2 import GPT
from config import GPTConfig

model = GPT(GPTConfig())
model, optimizer = model.load_state_dict(state_dict=torch.load("/scratch/ks02450/model.pth"), optimizer_state_dict=torch.load("/scratch/ks02450/optimizer.pth"))
model.to('cuda')
model.eval()

text = "If I must not, I need not be barren of accusations"

tokens = tiktoken.get_encoding("gpt2").encode(text)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(5, 1) 
x = tokens.to('cuda')

while x.size(1) < 64: # max_length=30
    with torch.no_grad():
        logits = model(x)[0] # (B, T, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
     
        ix = torch.multinomial(topk_probs, 1) 
      
        xcol = torch.gather(topk_indices, -1, ix) 
        x = torch.cat((x, xcol), dim=1)


enc = tiktoken.get_encoding('gpt2')
for i in range(5):
    tokens = x[i, :30].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)