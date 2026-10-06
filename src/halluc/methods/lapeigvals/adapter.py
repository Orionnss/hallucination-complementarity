"""Extractor adapter: our forward trace in, the official LapEigvals features out.

The official pipeline (hallucinations/llm/feature_storage.py, batch_size=1) does this per
item:

  1. `model.generate(..., output_attentions=True)` with eager attention, in bf16
  2. `stack_attention_matrix`  - per-step attention rows -> one [1, H, T', T'] per layer,
                                 where T' = len(generated_tokens) - 1: the final generated
                                 token is never fed back, so it has no attention row
  3. `remove_padding_from_intermediate_states` - strips pad tokens at both ends
  4. `attention_diagonal`, `laplacian_diagonal_from_attn(vertical_edges=False)`, computed
     on CPU in the attention dtype (bf16), then cast to float32 when loaded
  5. `get_laplacian_eigvals_per_head_topk(layer_idx=None, top_k=k)` - all layers, top-k
     per head, flattened

This adapter replaces step 1 only. The harness owns generation (rule R2), so the input is
the stage-1 answer re-forwarded in one eager pass. Step 2 then reduces to choosing T': for
a causal model the stacked per-step rows and the rows of a single full forward are the
same matrix up to numerical noise (test F1 checks the stacking; the KV-cache noise is a
recorded deviation). Steps 3-5 are the upstream functions, called unchanged.

Choosing T' needs to know how generation ended, because the stage-1 sequence and the
upstream `generated_tokens` differ in their last token:

  stopped, terminator stripped by stage 1  upstream had answer + EOS, so T' = T and the
                                           EOS is appended to the token ids
  stopped, terminator kept (Gemma's 106)   upstream had the same ids, so T' = T - 1
  hit max_new_tokens                       upstream had the same ids, so T' = T - 1
"""

from __future__ import annotations

import numpy as np
import torch

from ...features.base import ForwardTrace
from .upstream_spec import TOP_K_EIGVALS, load

#: Every k the official sweep can offer. Stored once; smaller k are prefixes, because the
#: upstream top-k is `sort(descending)[..., :k]`. Values past an item's sequence length
#: are NaN: upstream cannot form them either, and drops any k above the shortest item.
K_MAX = max(TOP_K_EIGVALS)

LAP_BLOCK = "lapeigvals_official"
ATTN_BLOCK = "attneigvals_official"
LEN_BLOCK = "lapeigvals_official_T"


def official_input(
    attentions: tuple[torch.Tensor, ...],
    input_ids: torch.Tensor,
    stopped: bool,
    ends_with_terminator: bool,
    eos_token_id: int,
    device: str = "cpu",
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Rebuild upstream's (stacked attentions, generated_tokens) for one item.

    attentions: L tensors [H, T, T] from one eager forward over input_ids [T].
    Returns per-layer [1, H, T', T'] on `device` in the attention dtype, and [1, T'+1] ids.
    """
    T = int(input_ids.shape[0])
    if any(a.shape[-1] != T for a in attentions):
        raise ValueError("attention width does not match input_ids")
    ids = input_ids.to("cpu").long()
    if stopped and not ends_with_terminator:
        generated = torch.cat([ids, torch.tensor([eos_token_id])])
        t_prime = T
    else:
        generated = ids
        t_prime = T - 1
    stacked = [a[:, :t_prime, :t_prime].to(device).unsqueeze(0) for a in attentions]
    return stacked, generated.unsqueeze(0).to(device)


class OfficialSpectralFeatures:
    """LapEigvals and AttnEigvals blocks, computed by the official functions.

    AttnEigvals is the paper's own no-Laplacian control (same attention, same top-k, same
    probe), so storing it from the same pass gives the one-axis ablation of the
    Laplacian (PROTOCOL.md, M4). It replaced our own `attn_baseline`, which ran without
    PCA and so confounded the two.
    """

    def __init__(self, pad_token_id: int, eos_token_id: int, device: str = "cpu") -> None:
        self.up = load()
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.device = device

    def diagonals(
        self, trace: ForwardTrace, stopped: bool, ends_with_terminator: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream steps 2-4. Returns (attn_diag, laplacian_diag), each [L, H, T''] float32."""
        if trace.input_ids is None:
            raise ValueError("trace has no input_ids; produce it with HFGenerator._trace")
        stacked, generated = official_input(
            trace.attentions, trace.input_ids, stopped, ends_with_terminator,
            self.eos_token_id, self.device,
        )
        processing = self.up["processing"]
        weights = self.up["attention_weights"]
        (example,) = processing.remove_padding_from_intermediate_states(
            per_layer_batched_data=stacked,
            data_type="attn",
            generated_tokens=generated,
            pad_token_id=self.pad_token_id,
        )
        attn_diag = weights.attention_diagonal(example)
        lap_diag = weights.laplacian_diagonal_from_attn(example, vertical_edges=False)
        # train_attn_vs_laplacian.load_and_prepare_data casts after loading.
        return attn_diag.float(), lap_diag.float()

    def topk(self, diag: torch.Tensor, laplacian: bool) -> np.ndarray:
        """Upstream step 5 at k = min(K_MAX, T''), reshaped to [L, H, K_MAX], NaN-padded."""
        feats = self.up["attn_feats"]
        n_layers, n_heads, n_tokens = diag.shape
        k = min(K_MAX, n_tokens)
        fn = (feats.get_laplacian_eigvals_per_head_topk if laplacian
              else feats.get_attn_eigvals_per_head_topk)
        flat = fn([diag], layer_idx=None, top_k=k)  # [1, L*H*k]
        out = np.full((n_layers, n_heads, K_MAX), np.nan, dtype=np.float32)
        out[:, :, :k] = flat.reshape(n_layers, n_heads, k).cpu().numpy()
        return out

    def extract(
        self, trace: ForwardTrace, stopped: bool, ends_with_terminator: bool
    ) -> dict[str, np.ndarray]:
        attn_diag, lap_diag = self.diagonals(trace, stopped, ends_with_terminator)
        return {
            LAP_BLOCK: self.topk(lap_diag, laplacian=True),
            ATTN_BLOCK: self.topk(attn_diag, laplacian=False),
            LEN_BLOCK: np.array([lap_diag.shape[-1]], dtype=np.int32),
        }
