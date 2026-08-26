"""Attention-spectrum features: LapEigvals and its no-Laplacian control.

Both produce the identical shape [L, H, k], so the pair isolates exactly one variable —
whether the Laplacian transform carries signal that raw attention does not.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import FEATURES, FeatureExtractor, ForwardTrace, top_k_sorted


@FEATURES.register("lapeigvals")
class LapEigvals(FeatureExtractor):
    """Top-k eigenvalues of the attention-graph Laplacian (arXiv 2502.17598).

    L = D - A with D the *out-degree* matrix, normalised by the number of outgoing
    edges so values do not scale with sequence length:

        d_ii = (sum_u a_ui) / (T - i)

    A is lower-triangular, so L is too, and its eigenvalues are just its diagonal
    (d_ii - a_ii) — no eigendecomposition needed. That is what makes this cheap enough
    to run for all 40x40 layer-head pairs.
    """

    name = "lapeigvals"

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def extract(self, trace: ForwardTrace) -> dict[str, np.ndarray]:
        per_layer = []
        for attn in trace.attentions:  # [H, T, T]
            a = attn.to(torch.float32)
            # Outgoing edges for token i = the tokens that actually attend to it, counted
            # from the attention matrix rather than assumed to be all T-i later tokens.
            # For dense causal attention the two are identical, so this leaves Qwen-style
            # models unchanged; for Gemma 3, whose local layers use a 1024 sliding window,
            # assuming T-i would understate the degree of early tokens in long sequences
            # and silently distort every eigenvalue.
            divisor = (a > 0).sum(dim=1).clamp(min=1).to(torch.float32)  # [H, T]
            out_degree = a.sum(dim=1) / divisor  # column sums -> [H, T]
            self_attn = torch.diagonal(a, dim1=-2, dim2=-1)  # [H, T]
            eigenvalues = out_degree - self_attn  # diag(L), = eigenvalues of L
            per_layer.append(top_k_sorted(eigenvalues, self.k).cpu())  # [H, k]
        return {self.name: torch.stack(per_layer).numpy().astype(np.float32)}


@FEATURES.register("attn_baseline")
class AttentionBaseline(FeatureExtractor):
    """Raw-attention control: eigenvalues of A itself, over all token positions.

    A is lower-triangular, so its eigenvalues are its diagonal — the self-attention
    scores. No Laplacian, no PCA downstream. Same [L, H, k] shape as LapEigvals so the
    two differ in exactly one respect.
    """

    name = "attn_baseline"

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def extract(self, trace: ForwardTrace) -> dict[str, np.ndarray]:
        per_layer = []
        for attn in trace.attentions:
            self_attn = torch.diagonal(attn.to(torch.float32), dim1=-2, dim2=-1)  # [H, T]
            per_layer.append(top_k_sorted(self_attn, self.k).cpu())
        return {self.name: torch.stack(per_layer).numpy().astype(np.float32)}
