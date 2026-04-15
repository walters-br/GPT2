"""
EECE 4520 - Milestone 4: Design Patterns
Strategy Pattern — Decoding Strategies

Defines an abstract DecodingStrategy interface and three concrete
implementations (Greedy, TopK, Nucleus/TopP).  The generate() function
in generate.py accepts any DecodingStrategy object, making it trivial
to swap or extend decoding behaviour without touching core generation logic.

Pattern roles:
  Strategy (abstract)  : DecodingStrategy
  Concrete strategies  : GreedyStrategy, TopKStrategy, NucleusStrategy
  Context              : generate() in generate.py — calls strategy.select_token()
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Strategy
# ─────────────────────────────────────────────────────────────────────────────

class DecodingStrategy(ABC):
    """
    Abstract base class for autoregressive token-selection strategies.

    Each concrete strategy receives the raw logit vector for the next
    token position and returns a (1, 1) integer tensor with the chosen
    token id.
    """

    @abstractmethod
    def select_token(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Float tensor of shape (1, vocab_size) — the unnormalised
                    scores for the next token *after* temperature scaling has
                    already been applied by the caller.

        Returns:
            Integer tensor of shape (1, 1) containing the selected token id.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._repr_params()})"

    def _repr_params(self) -> str:          # override in subclasses if desired
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Strategy 1 — Greedy
# ─────────────────────────────────────────────────────────────────────────────

class GreedyStrategy(DecodingStrategy):
    """Always picks the single highest-probability token (argmax)."""

    def select_token(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1, keepdim=True)          # (1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Strategy 2 — Top-k Sampling
# ─────────────────────────────────────────────────────────────────────────────

class TopKStrategy(DecodingStrategy):
    """
    Samples from the k highest-probability tokens.

    Logits below the k-th largest value are masked to -inf before the
    softmax, concentrating probability mass on the top candidates.
    """

    def __init__(self, k: int = 50):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k

    def select_token(self, logits: torch.Tensor) -> torch.Tensor:
        topk_vals, _ = torch.topk(logits, self.k, dim=-1)
        threshold    = topk_vals[:, -1].unsqueeze(-1)
        filtered     = logits.masked_fill(logits < threshold, float("-inf"))
        probs        = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1)      # (1, 1)

    def _repr_params(self) -> str:
        return f"k={self.k}"


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Strategy 3 — Nucleus (Top-p) Sampling
# ─────────────────────────────────────────────────────────────────────────────

class NucleusStrategy(DecodingStrategy):
    """
    Samples from the smallest token set whose cumulative probability >= p.

    Also known as top-p or nucleus sampling (Holtzman et al., 2020).
    Dynamically adjusts the candidate set size per step, unlike top-k.
    """

    def __init__(self, p: float = 0.9):
        if not (0.0 < p <= 1.0):
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = p

    def select_token(self, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs  = F.softmax(sorted_logits, dim=-1)
        cumprobs      = torch.cumsum(sorted_probs, dim=-1)

        # Mask tokens whose inclusion would push cumulative prob above p,
        # but always keep at least one token.
        remove_mask              = cumprobs - sorted_probs > self.p
        sorted_logits[remove_mask] = float("-inf")

        # Scatter filtered logits back to original vocab ordering
        filtered = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
        probs    = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1)      # (1, 1)

    def _repr_params(self) -> str:
        return f"p={self.p}"
