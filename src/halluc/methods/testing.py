"""Synthetic `generate()` outputs for the fidelity tests of every adapter (CPU only).

A causal attention and hidden-state sequence is drawn once at full length, then sliced
into exactly the per-step structure `model.generate(return_dict_in_generate=True,
output_attentions=True, output_hidden_states=True)` returns at batch size 1, and cut by
the harness (ForwardTrace.from_generate). Adapters are tested on that, like stage 1 runs
them, so no test needs a model.
"""

from __future__ import annotations

import torch

from ..features.base import ForwardTrace

PAD, EOS = 0, 2


def causal_attention(n_layers: int, n_heads: int, T: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Row-stochastic lower-triangular attention, in bf16 like the generator's output."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(T, T).tril().bool()
    layers = []
    for _ in range(n_layers):
        logits = torch.randn(n_heads, T, T, generator=g) * 3.0
        logits = logits.masked_fill(~mask, float("-inf"))
        layers.append(torch.softmax(logits, dim=-1).to(torch.bfloat16))
    return tuple(layers)


def hidden(n_layers: int, T: int, d: int, seed: int) -> tuple[torch.Tensor, ...]:
    """L + 1 hidden-state layers [T, d] (embeddings first), bf16."""
    g = torch.Generator().manual_seed(seed + 1000)
    return tuple(torch.randn(T, d, generator=g).to(torch.bfloat16) for _ in range(n_layers + 1))


def generate_outputs(attn_full, hid_full, prompt_len: int, n_generated: int):
    """Per-step attentions, hidden states and logits of n generated tokens.

    Step 0 reads the P prompt tokens; step j reads one token, at position P + j - 1. The
    last generated token is never read.
    """
    read = prompt_len + n_generated - 1
    att = [tuple(a[:, :prompt_len, :prompt_len].unsqueeze(0) for a in attn_full)]
    hs = [tuple(h[:prompt_len].unsqueeze(0) for h in hid_full)]
    for row in range(prompt_len, read):
        att.append(tuple(a[:, row:row + 1, :row + 1].unsqueeze(0) for a in attn_full))
        hs.append(tuple(h[row:row + 1].unsqueeze(0) for h in hid_full))
    logits = tuple(torch.zeros(1, 11) + j for j in range(n_generated))
    return tuple(att), tuple(hs), logits


def ids_for(T: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 1000, (T,), generator=g)


def make_case(P: int, n: int, keep: int, L: int, H: int, seed: int, d: int = 8,
              user_span: tuple[int, int] | None = None):
    """(trace, full attention, full hidden states, ids) of a simulated generation."""
    total = P + n
    attn, hid = causal_attention(L, H, total, seed), hidden(L, total, d, seed)
    seq = ids_for(total, seed)
    att_steps, hs_steps, logits = generate_outputs(attn, hid, P, n)
    trace = ForwardTrace.from_generate(
        seq, att_steps, hs_steps, logits, P, keep, pad_token_id=PAD, eos_token_id=EOS,
        user_span=user_span if user_span is not None else (1, max(2, P - 2)))
    return trace, attn, hid, seq
