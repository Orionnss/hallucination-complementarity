"""Hidden-state features: SAPLMA's probe input and the singular-value baseline."""

from __future__ import annotations

import numpy as np
import torch

from .base import FEATURES, FeatureExtractor, ForwardTrace


def _last_token_stack(trace: ForwardTrace) -> torch.Tensor:
    """Last token's hidden state at every layer -> [L+1, d].

    Collected on CPU because a sharded model spreads layers over several devices, so the
    per-layer slices cannot be stacked in place. Each slice is only d floats.
    """
    return torch.stack([h[-1].to("cpu", torch.float32) for h in trace.hidden_states])


@FEATURES.register("saplma")
class Saplma(FeatureExtractor):
    """SAPLMA (Azaria & Mitchell 2023): last-token hidden state at one layer.

    All L+1 layers are persisted rather than one, because the probe layer is a
    hyperparameter selected on the inner CV fold — fixing it here would either leak
    test information or arbitrarily handicap the method.
    """

    name = "saplma"

    def extract(self, trace: ForwardTrace) -> dict[str, np.ndarray]:
        # float16 halves the ~420 KB/sample cost; probe inputs are standardised anyway.
        return {self.name: _last_token_stack(trace).cpu().numpy().astype(np.float16)}


@FEATURES.register("svd_baseline")
class SvdBaseline(FeatureExtractor):
    """Singular values of the last token's across-layer hidden-state matrix.

    Stacking [L+1, d] and taking its spectrum summarises how the token's representation
    rotates through depth, in L+1 numbers that are independent of d.
    """

    name = "svd_baseline"

    def extract(self, trace: ForwardTrace) -> dict[str, np.ndarray]:
        stack = _last_token_stack(trace)  # [L+1, d]
        singular_values = torch.linalg.svdvals(stack)  # [min(L+1, d)] = [L+1]
        return {self.name: singular_values.cpu().numpy().astype(np.float32)}
