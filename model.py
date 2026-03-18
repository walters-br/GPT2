"""
EECE 4520 - Milestone 3 Part 1
GPT-2-like decoder-only Transformer architecture.

Components:
  - Token + learned positional embeddings
  - N x TransformerBlock (pre-LayerNorm, causal self-attention + FFN)
  - Language model head with weight tying
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass  – single place to change all hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size:  int   = 8000   # must match trained tokenizer
    block_size:  int   = 128    # max sequence / context length
    d_model:     int   = 256    # embedding & hidden dimension
    n_layers:    int   = 4      # number of transformer blocks
    n_heads:     int   = 4      # attention heads (d_model % n_heads == 0)
    dropout:     float = 0.1    # dropout probability
    bias:        bool  = False  # bias in Linear layers (GPT-2 uses False)


# ─────────────────────────────────────────────────────────────────────────────
# Causal (masked) multi-head self-attention
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    Uses a registered causal mask buffer so it is moved to the correct
    device automatically with model.to(device).
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model  = cfg.d_model

        # Fused QKV projection
        self.qkv_proj  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)

        # Causal mask — lower-triangular, shape (1,1,T,T)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn  = (q @ k.transpose(-2, -1)) / scale                # (B, nh, T, T)
        attn  = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )
        attn  = F.softmax(attn, dim=-1)
        attn  = self.attn_drop(attn)

        # Weighted sum over values
        y = (attn @ v)                                            # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)         # (B, T, C)
        return self.resid_drop(self.out_proj(y))


# ─────────────────────────────────────────────────────────────────────────────
# Position-wise Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """4× expansion FFN with GELU activation (GPT-2 style)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block  (Pre-LayerNorm — GPT-2 style)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-LN transformer block:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ff   = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full GPT-2-like Language Model
# ─────────────────────────────────────────────────────────────────────────────

class GPT2Like(nn.Module):
    """
    Decoder-only Transformer language model (GPT-2 architecture).

    Input  : (B, T) integer token ids
    Output : (B, T, vocab_size) logits  +  optional scalar cross-entropy loss
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model),
            pos_emb = nn.Embedding(cfg.block_size, cfg.d_model),
            drop    = nn.Dropout(cfg.dropout),
            blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)]),
            ln_f    = nn.LayerNorm(cfg.d_model),
        ))

        # LM head — no bias; weights tied with token embedding
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.tok_emb.weight   # weight tying

        self._init_weights()
        print(f"GPT-2-like model | params: {self.num_parameters():,}")

    # ── Weight initialisation ────────────────────────────────────────────────
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(
        self,
        idx:     torch.Tensor,               # (B, T)
        targets: torch.Tensor | None = None, # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"

        device = idx.device
        pos    = torch.arange(T, device=device)                  # (T,)

        tok = self.transformer.tok_emb(idx)                      # (B, T, d_model)
        pos = self.transformer.pos_emb(pos)                      # (T,  d_model)
        x   = self.transformer.drop(tok + pos)

        for block in self.transformer.blocks:
            x = block(x)

        x      = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                 # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, V)
                targets.view(-1),                   # (B*T,)
            )
        return logits, loss

    # ── Helpers ──────────────────────────────────────────────────────────────
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)