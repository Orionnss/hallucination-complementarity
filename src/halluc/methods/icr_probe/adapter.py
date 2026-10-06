"""Stage-1 adapter: the shared trace in, the official ICR Probe features out.

Official usage (README.md, "1. Compute ICR Scores"): `ICRScore(hidden_states, attentions,
skew_threshold=0, entropy_threshold=1e5, core_positions=..., icr_device=...)` on the
`generate()` outputs, then `compute_icr(top_k=20, top_p=0.1, pooling='mean', ...)`, which
returns one ICR score per (layer, response token). The probe reads one value per layer:
the mean over the response tokens (scripts/empirical_study.ipynb, `read_acd_scores`:
"[item, layer, token] -> [item, layer]" with `np.mean(item, axis=-1)`).

The native view of the shared trace is the `generate()` output ICRScore expects, cut to
the kept answer by the harness. `core_positions` locate the user prompt and the response
in the token sequence; the upstream repository does not show how its authors set them,
so they come from the harness's own prompt (ForwardTrace.user_span) and the response
starts at the first generated token (METHOD_CARD.md, "core_positions").
"""

from __future__ import annotations

import warnings

import numpy as np

from ...features.base import FEATURES, FeatureExtractor, ForwardTrace
from .upstream_spec import COMPUTE_ICR_KWARGS, ICR_SCORE_KWARGS, SPEC, load

ICR_BLOCK = "icr_official"
NTOK_BLOCK = "icr_official_ntok"


@FEATURES.register("icr_official")
class OfficialIcrFeatures(FeatureExtractor):
    name = "icr_official"
    needs = frozenset({"attentions", "hidden_states"})
    store = "methods"

    def __init__(self) -> None:
        self.up = load()

    def params(self) -> dict:
        return {"icr_score": ICR_SCORE_KWARGS, "compute_icr": COMPUTE_ICR_KWARGS,
                "token_pooling": "mean", "upstream": SPEC.url, "commit": SPEC.commit}

    def scores(self, trace: ForwardTrace) -> list[list[float]]:
        """ICR score per (layer, response token), from the official code."""
        if trace.step_attentions is None or trace.step_hidden_states is None:
            raise ValueError("icr_official needs the native generate() view with "
                             "attentions and hidden states (ADDING_A_METHOD.md §5.0)")
        if trace.user_span is None:
            raise ValueError("icr_official needs trace.user_span for core_positions")
        start, end = trace.user_span
        device = trace.step_attentions[0][0].device
        icr = self.up["icr_score"].ICRScore(
            hidden_states=trace.step_hidden_states,
            attentions=trace.step_attentions,
            core_positions={"user_prompt_start": start, "user_prompt_end": end,
                            "response_start": trace.prompt_len},
            icr_device=device,
            **ICR_SCORE_KWARGS,
        )
        scores, _top_p_mean = icr.compute_icr(**COMPUTE_ICR_KWARGS)
        return scores

    def extract(self, trace: ForwardTrace) -> dict[str, np.ndarray]:
        if trace.step_attentions is not None and len(trace.step_attentions) == 1:
            # No response token was read (empty answer): ICR is undefined, and upstream
            # cannot run (it stacks an empty list). Recorded as NaN.
            n_layers = len(trace.step_attentions[0])
            return {ICR_BLOCK: np.full(n_layers, np.nan, dtype=np.float32),
                    NTOK_BLOCK: np.array([0], dtype=np.int32)}
        scores = np.asarray(self.scores(trace), dtype=np.float64)  # [L, n_response_tokens]
        with warnings.catch_warnings():
            # An empty answer has no response token: the mean is NaN, kept as NaN.
            warnings.simplefilter("ignore", RuntimeWarning)
            per_layer = np.mean(scores, axis=-1)
        return {ICR_BLOCK: per_layer.astype(np.float32),
                NTOK_BLOCK: np.array([scores.shape[-1]], dtype=np.int32)}
