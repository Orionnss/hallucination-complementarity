"""The shared trace, and the extractor (adapter) abstraction.

Every extractor reads the *same* trace: the outputs of the one `generate()` call that
produced the scored answer (ADDING_A_METHOD.md, rule R2 and §5.0). Nothing re-runs the
model. Running all extractors on that one trace is also what makes the pipeline
affordable: ICR needs per-token residual deltas that would cost ~126 MB/sample to
persist, and the attentions alone reach ~13 GB per item at T = 2000, so every method is
reduced to its feature blocks inside the loop and the trace itself is never stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import torch

from ..registry import Registry

#: The outputs an extractor can ask `generate()` for.
NEEDS = frozenset({"attentions", "hidden_states", "logits"})


class _StackedAttentions(Sequence):
    """The stacked view of the per-step attentions, built one layer at a time.

    A full [H, T', T'] matrix for every layer at once would double the attention memory
    (~13 GB at T = 2000 on Qwen3-14B), so a layer is stacked when it is read. The last
    layer read is kept, because extractors often read the same layer twice in a row.
    """

    def __init__(self, steps: tuple, prompt_len: int, seq_len: int) -> None:
        self._steps = steps
        self._prompt_len = prompt_len
        self._seq_len = seq_len
        self._cached: tuple[int, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self._steps[0])

    def __getitem__(self, layer: int) -> torch.Tensor:
        if layer < 0:
            layer += len(self)
        if not 0 <= layer < len(self):
            raise IndexError(layer)
        if self._cached is not None and self._cached[0] == layer:
            return self._cached[1]
        stacked = stack_attention_rows([step[layer] for step in self._steps],
                                       self._prompt_len, self._seq_len)
        self._cached = (layer, stacked)
        return stacked

    def __iter__(self) -> Iterator[torch.Tensor]:
        for layer in range(len(self)):
            yield self[layer]


def stack_attention_rows(per_step: list[torch.Tensor], prompt_len: int,
                         seq_len: int) -> torch.Tensor:
    """Per-step attention of one layer -> the full lower-triangular [H, T', T'] matrix.

    Step 0 holds the prompt rows [1, H, P, P]; step j >= 1 holds the one row of the token
    read at that step, [1, H, 1, P + j]. Entries right of the causal frontier are zero,
    which is what the softmax gives them anyway. This is the same construction as the
    official LapEigvals `stack_attention_matrix`, written here so the harness does not
    depend on upstream code.
    """
    first = per_step[0][0]
    out = torch.zeros(first.shape[0], seq_len, seq_len, dtype=first.dtype, device=first.device)
    out[:, :prompt_len, :prompt_len] = first
    for j, step in enumerate(per_step[1:], start=1):
        row = prompt_len + j - 1
        out[:, row, : row + 1] = step[0, :, 0, :].to(out.device)
    return out


@dataclass
class ForwardTrace:
    """The shared trace of one item (ADDING_A_METHOD.md §5.0).

    Native view, exactly as `generate()` returns them, cut to the kept answer:
      step_attentions:    steps x L x [1, H, q, k]
      step_hidden_states: steps x (L+1) x [1, q, d]
      step_logits:        steps x [1, V]

    Stacked view, built from the native one on demand:
      attentions:    L tensors [H, T', T']   (one layer is materialised at a time)
      hidden_states: L+1 tensors [T', d]

    T' is the number of tokens the model *read*. The last generated token is never read
    back, so it has no attention row and no hidden state.

      prompt_len:   number of prompt tokens
      input_ids:    [T' + 1] ids: every token read, plus the final generated one. This is
                    the `generated_tokens` row upstream code expects next to the stacked
                    attention (it drops the last token itself).
      pad_token_id, eos_token_id: the generator's, for upstream padding logic.
      user_span:    [start, end) token positions of the item's own text in the prompt
                    (the user message in a chat template; the final question block in a
                    base model's few-shot prompt). The prompt belongs to the harness, so
                    the harness locates it; ICR's `core_positions` need it.

    `from_full` builds a trace from full matrices instead (tests, legacy scripts). Such a
    trace has no native view, so adapters of upstream code refuse it.
    """

    prompt_len: int
    input_ids: torch.Tensor | None = None
    step_attentions: tuple | None = None
    step_hidden_states: tuple | None = None
    step_logits: tuple | None = None
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    user_span: tuple[int, int] | None = None
    _full_attentions: tuple | None = field(default=None, repr=False)
    _full_hidden_states: tuple | None = field(default=None, repr=False)
    _stacked: _StackedAttentions | None = field(default=None, repr=False)
    _stacked_hidden: tuple | None = field(default=None, repr=False)

    @classmethod
    def from_generate(cls, sequence: torch.Tensor, attentions, hidden_states, logits,
                      prompt_len: int, keep: int, pad_token_id: int | None = None,
                      eos_token_id: int | None = None,
                      user_span: tuple[int, int] | None = None) -> "ForwardTrace":
        """Cut `generate()` outputs to the kept answer (ADDING_A_METHOD.md §5.0).

        sequence: [P + n] ids (prompt + every generated token). keep: answer tokens the
        harness keeps after it removes the terminator or a base model's run-on text.

        The trace covers prompt + kept answer, i.e. T' = P + keep tokens read, but never
        more than were actually read (P + n - 1): when generation hit max_new_tokens the
        last answer token was never fed back. The token after the cut stays in input_ids
        as the "final generated token" upstream expects to drop.
        """
        n = int(sequence.shape[0]) - prompt_len
        if n < 1:
            raise ValueError("generate() returned no new token")
        t_prime = min(prompt_len + keep, prompt_len + n - 1)
        n_steps = 1 + (t_prime - prompt_len)
        cut = lambda steps: None if steps is None else tuple(steps[:n_steps])  # noqa: E731
        return cls(
            prompt_len=prompt_len,
            input_ids=sequence[: t_prime + 1].detach().cpu(),
            step_attentions=cut(attentions),
            step_hidden_states=cut(hidden_states),
            step_logits=cut(logits),
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            user_span=user_span,
        )

    @classmethod
    def from_full(cls, attentions, hidden_states, prompt_len: int,
                  input_ids: torch.Tensor | None = None) -> "ForwardTrace":
        """A trace from full [H, T, T] / [T, d] matrices (no native view)."""
        return cls(prompt_len=prompt_len, input_ids=input_ids,
                   _full_attentions=tuple(attentions), _full_hidden_states=tuple(hidden_states))

    @property
    def has_native(self) -> bool:
        return self.step_attentions is not None or self.step_hidden_states is not None

    @property
    def seq_len(self) -> int:
        """T', the number of tokens the model read."""
        if self.input_ids is not None and self.has_native:
            return int(self.input_ids.shape[0]) - 1
        if self._full_attentions:
            return self._full_attentions[0].shape[-1]
        if self._full_hidden_states:
            return self._full_hidden_states[0].shape[0]
        raise ValueError("trace has neither ids nor matrices")

    @property
    def attentions(self) -> Sequence[torch.Tensor]:
        if self._full_attentions is not None:
            return self._full_attentions
        if self.step_attentions is None:
            raise ValueError("attentions were not requested from generate() (needs)")
        if self._stacked is None:
            self._stacked = _StackedAttentions(self.step_attentions, self.prompt_len, self.seq_len)
        return self._stacked

    @property
    def hidden_states(self) -> tuple[torch.Tensor, ...]:
        if self._full_hidden_states is not None:
            return self._full_hidden_states
        if self.step_hidden_states is None:
            raise ValueError("hidden states were not requested from generate() (needs)")
        if self._stacked_hidden is None:
            n_layers = len(self.step_hidden_states[0])
            self._stacked_hidden = tuple(
                torch.cat([step[layer][0] for step in self.step_hidden_states], dim=0)
                for layer in range(n_layers)
            )
        return self._stacked_hidden

    @property
    def n_layers(self) -> int:
        return len(self.attentions)

    @property
    def n_heads(self) -> int:
        if self.step_attentions is not None:
            return self.step_attentions[0][0].shape[1]
        return self.attentions[0].shape[0]


class FeatureExtractor:
    """A stage-1 extractor (adapter). Runs inside the stage-1 loop on the shared trace.

    name:   keys the output dict and, for `store = "methods"`, the store directory
    needs:  what it reads from generate(); stage 1 requests the union over all extractors
    store:  "stage1" -> blocks go in stage1_extract/<ds>/ (our reimplementations);
            "methods" -> runs/<run>/methods/<name>/<ds>/ with its own spec.json
    `extract` must be side-effect free, must not run the model, and must not keep a
    reference to the trace.
    """

    name: str
    needs: frozenset[str] = frozenset({"attentions", "hidden_states"})
    store: str = "stage1"

    def extract(self, trace: ForwardTrace) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def params(self) -> dict:
        """Settings that change the blocks; recorded in the store's spec.json."""
        return {k: v for k, v in vars(self).items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}


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
