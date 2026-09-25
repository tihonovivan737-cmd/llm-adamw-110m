import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 768
    layers: int = 12
    heads: int = 12
    ffn_dim: int = 2048
    seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if min(self.vocab_size, self.dim, self.layers, self.heads, self.ffn_dim, self.seq_len) <= 0:
            raise ValueError('Model dimensions must be positive')
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError('Head dimension must be integral and even')

    @property
    def parameter_count(self):
        return self.vocab_size * self.dim + self.layers * (
            4 * self.dim**2 + 3 * self.dim * self.ffn_dim + 2 * self.dim
        ) + self.dim


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        z = x.float()
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)
        return (z * self.weight.float()).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.dim // cfg.heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        angles = torch.outer(torch.arange(cfg.seq_len).float(), inv)
        self.register_buffer('cos', angles.cos()[None, None], persistent=False)
        self.register_buffer('sin', angles.sin()[None, None], persistent=False)

    def rotary(self, x):
        a, b = x.chunk(2, dim=-1)
        c = self.cos[:, :, :x.shape[-2]].to(x.dtype)
        s = self.sin[:, :, :x.shape[-2]].to(x.dtype)
        return torch.cat((a * c - b * s, b * c + a * s), dim=-1)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.view(batch, length, self.heads, self.head_dim).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(self.rotary(q), self.rotary(k), v,
                                          dropout_p=0.0, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        h = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, targets=None):
        if tokens.shape[1] > self.cfg.seq_len:
            raise ValueError('Sequence exceeds configured context')
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        # Tied output weights: the embedding is registered only once.
        logits = F.linear(self.norm(x), self.embedding.weight)
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().reshape(-1, self.cfg.vocab_size),
                               targets.reshape(-1), ignore_index=-100, reduction='sum')


def make_optimizer(model, cfg, device):
    # Embedding/LM head and norm weights are excluded from weight decay.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 or name == 'embedding.weight' else decay).append(param)
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': cfg['weight_decay']},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=cfg['learning_rate'], betas=tuple(cfg['betas']), eps=cfg['eps'],
        fused=device.type == 'cuda')
