"""Compact decoder-only Transformer with causal attention and RoPE."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)))
        self.register_buffer("cos", self._rope_cache(config.block_size, self.head_dim)[0])
        self.register_buffer("sin", self._rope_cache(config.block_size, self.head_dim)[1])

    @staticmethod
    def _rope_cache(length: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(length).float()
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return rotated.flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        shape = (batch, length, self.n_head, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        cos, sin = self.cos[..., :length, :], self.sin[..., :length, :]
        q, k = self._rotate(q, cos, sin), self._rotate(k, cos, sin)
        # For boolean masks, scaled_dot_product_attention keeps positions marked True.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.causal_mask[:length, :length], dropout_p=self.dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.dropout(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(), nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if input_ids.size(1) > self.config.block_size:
            raise ValueError(f"Sequence length exceeds block_size={self.config.block_size}")
        x = self.drop(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size:]
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :] / max(temperature, 1e-5), dim=-1)
            input_ids = torch.cat((input_ids, torch.multinomial(probs, 1)), dim=1)
        return input_ids


def prepare_lora_model(model_name: str, **kwargs):
    """Optional HF/PEFT adapter setup; imports stay lazy for offline base training."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    return get_peft_model(model, config)
