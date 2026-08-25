"""Feature extractor abstraction.

Every extractor consumes the *same* single forward pass over prompt+answer and returns
a dict of named float arrays. Running them together is the point: ICR needs per-token
residual deltas that would cost ~126 MB/sample to persist, so it must be reduced inside
the loop rather than recomputed later from dumped activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..registry import Registry


@dataclass
class ForwardTrace:
    """One model forward over the full prompt+answer sequence.

    attentions:    tuple of L tensors, each [H, T, T] — row-stochastic, lower-triangular
    hidden_states: tuple of L+1 tensors, each [T, d] — raw residual stream (not normed)
    prompt_len:    number of prompt tokens; answer tokens are [prompt_len:]
    """

    attentions: tuple[torch.Tensor, ...]
    hidden_states: tuple[torch.Tensor, ...]
    prompt_len: int

    @property
    def n_layers(self) -> int:
        return len(self.attentions)

    @property
    def n_heads(self) -> int:
        return self.attentions[0].shape[0]

    @property
    def seq_len(self) -> int:
        return self.attentions[0].shape[-1]


class FeatureExtractor:
    """Base class. `name` keys the output dict; `extract` must be side-effect free."""

    name: str

    def extract(self, trace: ForwardTrace) -> dict[str, np.ndarray]:
        raise NotImplementedError


FEATURES: Registry[FeatureExtractor] = Registry("feature")


def top_k_sorted(values: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k largest along the last dim, descending, zero-padded if the sequence is short.

    Padding only bites on pathologically short sequences (T < k); with a 256-token answer
    budget T is in the hundreds, but a silent shape error here would be invisible until
    the probe trained on garbage.
    """
    n = values.shape[-1]
    if n >= k:
        return torch.topk(values, k, dim=-1, largest=True, sorted=True).values
    out = torch.zeros(*values.shape[:-1], k, dtype=values.dtype, device=values.device)
    out[..., :n] = torch.sort(values, dim=-1, descending=True).values
    return out
