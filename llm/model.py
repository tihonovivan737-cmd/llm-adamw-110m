"""Qwen3.5 text-backbone model wrapper used by the project's training loop."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


@dataclass
class ModelConfig:
    """Small Qwen3.5-style decoder configuration.

    The model keeps Qwen3.5's 3:1 Gated DeltaNet/full-attention layout and
    gated MLP, while the vocabulary remains the project's Russian tokenizer.
    """

    vocab_size: int = 50257
    hidden_size: int = 768
    intermediate_size: int = 2688
    num_hidden_layers: int = 16
    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    head_dim: int = 128
    seq_len: int = 1024
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 12
    linear_num_value_heads: int = 12
    full_attention_interval: int = 4
    rope_theta: float = 10000000.0

    def __post_init__(self):
        if min(self.vocab_size, self.hidden_size, self.intermediate_size,
               self.num_hidden_layers, self.num_attention_heads,
               self.num_key_value_heads, self.head_dim, self.seq_len,
               self.max_position_embeddings, self.full_attention_interval) <= 0:
            raise ValueError('Model dimensions must be positive')
        if self.num_hidden_layers % self.full_attention_interval:
            raise ValueError('num_hidden_layers must be divisible by full_attention_interval')
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError('num_attention_heads must be divisible by num_key_value_heads')

    @property
    def layer_types(self):
        return [
            'full_attention' if (index + 1) % self.full_attention_interval == 0
            else 'linear_attention'
            for index in range(self.num_hidden_layers)
        ]

    def to_transformers_config(self):
        return Qwen3_5TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            initializer_range=self.initializer_range,
            use_cache=False,
            tie_word_embeddings=True,
            linear_conv_kernel_dim=self.linear_conv_kernel_dim,
            linear_key_head_dim=self.linear_key_head_dim,
            linear_value_head_dim=self.linear_value_head_dim,
            linear_num_key_heads=self.linear_num_key_heads,
            linear_num_value_heads=self.linear_num_value_heads,
            layer_types=self.layer_types,
            rope_parameters={
                'rope_type': 'default',
                'rope_theta': self.rope_theta,
                'partial_rotary_factor': 0.25,
                'mrope_interleaved': True,
                'mrope_section': [5, 5, 6],
            },
        )


class LanguageModel(nn.Module):
    """Qwen3.5 text-only causal LM initialized from scratch."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3_5ForCausalLM(cfg.to_transformers_config())
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, tokens, targets=None):
        if tokens.shape[1] > self.cfg.seq_len:
            raise ValueError('Sequence exceeds configured context')
        output = self.model(input_ids=tokens, use_cache=False, return_dict=True)
        if targets is None:
            return output.logits
        return F.cross_entropy(output.logits.float().reshape(-1, self.cfg.vocab_size),
                               targets.reshape(-1), ignore_index=-100, reduction='sum')


def make_optimizer(model, cfg, device):
    # Keep embeddings/output head and norm vectors out of weight decay.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        is_embedding_or_head = 'embed_tokens' in name or 'lm_head' in name
        (no_decay if param.ndim < 2 or is_embedding_or_head else decay).append(param)
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': cfg['weight_decay']},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=cfg['learning_rate'], betas=tuple(cfg['betas']), eps=cfg['eps'],
        fused=device.type == 'cuda')
