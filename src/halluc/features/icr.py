"""ICR Probe features (arXiv 2507.16488, ACL 2025).

ICR = Information Contribution to Residual stream. For layer l and token i:

    p^l_ij   = (dx^l_i)^T . x^l_j / ||x^l_j||        projection of the residual update
    Proj^l_i = softmax_j(p^l_ij)                     over causally visible j
    ICR^l_i  = JSD(Proj^l_i, Attn^l_i)               over the top-k attended tokens

A low score means the update points where attention points (MHSA-driven); a high score
means the FFN is steering the update instead. The probe reads the layer profile, not any
single layer.

Note on normalisation order: the paper softmaxes the full projection vector and then
restricts to the top-k attended tokens. Softmax-then-renormalise-over-a-subset is
identical to softmax-over-the-subset, so the ambiguity is immaterial.

Unverified detail: the layer-pooling operator is in the paper's appendix (see the open
items in DESIGN.md). Both the last-token profile and the answer-token mean are persisted
so the choice can be made downstream without re-running extraction.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import FEATURES, FeatureExtractor, ForwardTrace

_EPS = 1e-12


def _jensen_shannon(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Row-wise JSD in bits, so values land in [0, 1]. Inputs are distributions."""
    m = 0.5 * (p + q)
    log_m = torch.log2(m + _EPS)
    kl_p = (p * (torch.log2(p + _EPS) - log_m)).sum(-1)
    kl_q = (q * (torch.log2(q + _EPS) - log_m)).sum(-1)
    return (0.5 * kl_p + 0.5 * kl_q).clamp(min=0.0)


@FEATURES.register("icr")
class IcrScore(FeatureExtractor):
    name = "icr"

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def extract(self, trace: ForwardTrace) -> dict[str, np.ndarray]:
        T = trace.seq_len

        last_profile, mean_profile = [], []
        for layer in range(trace.n_layers):
            # A sharded model may hold layer l's attention and hidden states on
            # different devices; align everything onto the attention's device, which
            # holds by far the largest tensor.
            device = trace.attentions[layer].device
            x = trace.hidden_states[layer].to(device, torch.float32)  # [T, d]
            # Score only answer tokens: prompt tokens carry no generation decision.
            rows = torch.arange(max(trace.prompt_len - 1, 0), T, device=device)
            # Causal visibility: token at row r may attend to j <= r.
            valid = torch.arange(T, device=device).unsqueeze(0) <= rows.unsqueeze(1)  # [R, T]

            dx = (trace.hidden_states[layer + 1].to(device, torch.float32) - x)[rows]  # [R, d]
            x_dir = x / (x.norm(dim=-1, keepdim=True) + _EPS)

            logits = dx @ x_dir.T  # [R, T] = p^l_ij
            attn = trace.attentions[layer].to(torch.float32).mean(0)[rows]  # head-averaged

            # Select the top-k *attended* tokens; -1 keeps masked positions out of topk
            # since attention weights are non-negative.
            k = min(self.top_k, T)
            sel = torch.topk(attn.masked_fill(~valid, -1.0), k, dim=-1).indices  # [R, k]

            sel_valid = torch.gather(valid, 1, sel)
            p = torch.softmax(torch.gather(logits, 1, sel).masked_fill(~sel_valid, -torch.inf), -1)
            q = torch.gather(attn, 1, sel).masked_fill(~sel_valid, 0.0)
            q = q / (q.sum(-1, keepdim=True) + _EPS)

            icr = _jensen_shannon(p, q)  # [R]
            last_profile.append(icr[-1].cpu())
            mean_profile.append(icr.mean().cpu())

        return {
            "icr": torch.stack(last_profile).cpu().numpy().astype(np.float32),
            "icr_mean": torch.stack(mean_profile).cpu().numpy().astype(np.float32),
        }
