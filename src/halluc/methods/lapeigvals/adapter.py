"""Stage-1 adapter: the shared trace in, the official LapEigvals features out.

The official pipeline (hallucinations/llm/feature_storage.py, batch_size=1) does this per
item, on the output of `model.generate(output_attentions=True)` with eager attention:

  1. `_map_attentions_to_cpu`, then `stack_attention_matrix` - per-step attention rows ->
     one [1, H, T', T'] per layer, where T' = len(generated_tokens) - 1: the final
     generated token is never fed back, so it has no attention row
  2. `remove_padding_from_intermediate_states` - strips pad tokens at both ends
  3. `attention_diagonal`, `laplacian_diagonal_from_attn(vertical_edges=False)`, on CPU
     in the attention dtype (bf16), then cast to float32 when loaded
  4. `get_laplacian_eigvals_per_head_topk(layer_idx=None, top_k=k)` - all layers, top-k
     per head, flattened

The shared trace's native view *is* that `generate()` output, cut to the kept answer by
the harness (ForwardTrace.from_generate), with `input_ids` as `generated_tokens`. So every
step above is the upstream function, called on its own input format. Nothing is rebuilt
or converted here, and there is no second forward pass (ADDING_A_METHOD.md, rule R2).
"""

from __future__ import annotations

import numpy as np
import torch

from ...features.base import FEATURES, FeatureExtractor, ForwardTrace
from .upstream_spec import SPEC, TOP_K_EIGVALS, load

#: Every k the official sweep can offer. Stored once; smaller k are prefixes, because the
#: upstream top-k is `sort(descending)[..., :k]`. Values past an item's sequence length
#: are NaN: upstream cannot form them either, and drops any k above the shortest item.
K_MAX = max(TOP_K_EIGVALS)

LAP_BLOCK = "lapeigvals_official"
ATTN_BLOCK = "attneigvals_official"
LEN_BLOCK = "lapeigvals_official_T"


@FEATURES.register("lapeigvals_official")
class OfficialSpectralFeatures(FeatureExtractor):
    """LapEigvals and AttnEigvals blocks, computed by the official functions.

    AttnEigvals is the paper's own no-Laplacian control (same attention, same top-k, same
    probe), so storing it from the same pass gives the one-axis ablation of the
    Laplacian (PROTOCOL.md, M4). It replaced our own `attn_baseline`, which ran without
    PCA and so confounded the two.
    """

    name = "lapeigvals_official"
    needs = frozenset({"attentions"})
    store = "methods"

    def __init__(self, device: str = "cpu") -> None:
        # Upstream maps the attentions to CPU before stacking and computes there.
        self.device = device
        self.up = load()

    def params(self) -> dict:
        return {"device": self.device, "k_max": K_MAX, "upstream": SPEC.url,
                "commit": SPEC.commit}

    def diagonals(self, trace: ForwardTrace) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream steps 1-3. Returns (attn_diag, laplacian_diag), each [L, H, T''] float32."""
        if trace.step_attentions is None or trace.input_ids is None:
            raise ValueError("lapeigvals_official needs the native generate() view of the "
                             "trace, with attentions (ADDING_A_METHOD.md §5.0)")
        steps = tuple(tuple(a.to(self.device) for a in step) for step in trace.step_attentions)
        weights = self.up["attention_weights"]
        stacked = weights.stack_attention_matrix(steps)
        (example,) = self.up["processing"].remove_padding_from_intermediate_states(
            per_layer_batched_data=stacked,
            data_type="attn",
            generated_tokens=trace.input_ids.unsqueeze(0).to(self.device),
            pad_token_id=trace.pad_token_id,
        )
        attn_diag = weights.attention_diagonal(example)
        lap_diag = weights.laplacian_diagonal_from_attn(example, vertical_edges=False)
        # train_attn_vs_laplacian.load_and_prepare_data casts after loading.
        return attn_diag.float(), lap_diag.float()

    def topk(self, diag: torch.Tensor, laplacian: bool) -> np.ndarray:
        """Upstream step 4 at k = min(K_MAX, T''), reshaped to [L, H, K_MAX], NaN-padded."""
        feats = self.up["attn_feats"]
        n_layers, n_heads, n_tokens = diag.shape
        k = min(K_MAX, n_tokens)
        fn = (feats.get_laplacian_eigvals_per_head_topk if laplacian
              else feats.get_attn_eigvals_per_head_topk)
        flat = fn([diag], layer_idx=None, top_k=k)  # [1, L*H*k]
        out = np.full((n_layers, n_heads, K_MAX), np.nan, dtype=np.float32)
        out[:, :, :k] = flat.reshape(n_layers, n_heads, k).cpu().numpy()
        return out

    def extract(self, trace: ForwardTrace) -> dict[str, np.ndarray]:
        attn_diag, lap_diag = self.diagonals(trace)
        return {
            LAP_BLOCK: self.topk(lap_diag, laplacian=True),
            ATTN_BLOCK: self.topk(attn_diag, laplacian=False),
            LEN_BLOCK: np.array([lap_diag.shape[-1]], dtype=np.int32),
        }
