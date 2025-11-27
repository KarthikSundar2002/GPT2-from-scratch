
import torch
import torch.nn as nn
from torch.nn import functional as F

from CausalSelfAttention import CausalSelfAttention
from MLP import MLP, CastedLinear
from config import GPTConfig
from yarn import Yarn

from torch.nn.attention.flex_attention import create_block_mask

create_block_mask = torch.compile(create_block_mask, dynamic=False)

from CausalSelfAttention import norm

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_via_newtonschulz5
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1): # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)




class Block(nn.Module):
    def __init__(self, block_idx ,config: GPTConfig):
        super().__init__()
        self.config = config
        self.block_idx = block_idx 
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    
    def forward(self, x, cos, sin, block_mask, skip_out=None,skip_weight=None,first_weight=None,first_block_out=None):
        if self.block_idx != 0 and first_block_out is not None:
            x = first_weight * x + (1 - first_weight) * first_block_out
        if skip_out is not None:
            x = skip_weight[0] * x + skip_weight[1] * skip_out
        x = x + self.attn(norm(x), cos, sin, block_mask)
        x = x + self.mlp(norm(x))
        return x


    

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word embedding table,
#            vte = nn.Embedding(config.vocab_size, config.n_embd*config.n_layer), # value embedding table,
           # wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding,
            h = nn.ModuleList([Block(block_idx, config) for block_idx in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.yarn = Yarn(config.head_dim, config.block_size, config.block_size)
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)

       # self.transformer.wte.weight = self.lm_head.weight
        self.scalars = [
            [nn.Parameter(torch.tensor([1.0, 0.0])).to(device) for _ in range((config.n_layer // 2) - 1)],
            [nn.Parameter(torch.ones(1) * 0.5).to(device) for _ in range(config.n_layer - 1)],
        ]
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


        docs = (idx == 50256).cumsum(0)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[q_idx] == docs[kv_idx]
          window_mask = q_idx - kv_idx < 1024
          return causal_mask & document_mask & window_mask

        S = len(idx)
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)
        

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
       # pos_emb = self.transformer.wpe(pos)
        x = tok_emb
        cos, sin = self.yarn.cos, self.yarn.sin
        num_blocks = len(self.transformer.h)
        n = num_blocks // 2
        first_block_out = None
        skip_outs = []
        for block_idx, block in enumerate(self.transformer.h):
            if block_idx == 0:
                x = block(x, cos, sin, block_mask)
                first_block_out = x
            else:
                assert first_block_out is not None
                if block_idx < n:
                    x = block(x, cos, sin, block_mask, first_weight=self.scalars[1][block_idx - 1], first_block_out=first_block_out)
                    skip_outs.append(x)
                else:
                    if block_idx == self.config.n_layer - 1:
                        x = block(x, cos, sin, block_mask, first_weight=self.scalars[1][block_idx - 1], first_block_out=first_block_out)
                    else:
                        x = block(x, cos, sin, block_mask, skip_out=skip_outs.pop(), skip_weight=self.scalars[0][block_idx - n - 1], first_weight=self.scalars[1][block_idx - n - 1], first_block_out=first_block_out)
   
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        logits = 30 * torch.sigmoid(logits/7.5)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps):
        adam_param_dict = {pn: p for pn, p in self.named_parameters() if pn in ["lm_head.weight", "transformer.wte.weight", "transformer.ln_f.weight"]}
        muon_param_dict = {pn: p for pn, p in self.named_parameters() if pn not in ["lm_head.weight", "transformer.wte.weight", "transformer.ln_f.weight"] and p.dim() == 2}
        print(len(adam_param_dict), len(muon_param_dict))
        decay_params = [p for p in adam_param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in adam_param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": [p for p in decay_params], "weight_decay": weight_decay},
            {"params": [p for p in nodecay_params], "weight_decay": 0.0}
        ]
        optimizer1 = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=True)
        muon_params = [{"params": [p for p in muon_param_dict.values()]}]
        optimizer2 = Muon(muon_params, lr=0.1*learning_rate, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5)
        optimizers = [optimizer1, optimizer2]
        return optimizers
    
    def load_state_dict(self, state_dict, optimizer_state_dict):
        cur_state_dict = self.state_dict() 
        for key in cur_state_dict.keys():
                cur_state_dict[key].data.copy_(state_dict['_orig_mod.'+key].data)
        optimizer = self.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), eps=1e-8)
        optimizer[0].load_state_dict(optimizer_state_dict[0])
        optimizer[1].load_state_dict(optimizer_state_dict[1])
        return self, optimizer