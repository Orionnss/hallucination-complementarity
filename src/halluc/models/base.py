"""Generator abstraction: produce an answer and, from the same call, the shared trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..datasets.base import QAItem
from ..features.base import ForwardTrace
from ..registry import Registry


@dataclass
class Generation:
    """One generated answer plus the metadata that lands in the stage-1 JSON."""

    item_id: str
    answer: str
    prompt_tokens: int
    answer_tokens: int
    #: "stop" if the model emitted EOS, "length" if it hit max_new_tokens. A high
    #: "length" rate means answers are being truncated mid-sentence, which corrupts
    #: both the judge's view and the last-token features.
    finish_reason: str
    seconds: float
    meta: dict[str, Any] = field(default_factory=dict)


class Generator:
    name: str

    def generate(
        self, item: QAItem, needs: frozenset[str] | None = None
    ) -> tuple[Generation, ForwardTrace]:
        """Answer the item; the trace comes from the same call (ADDING_A_METHOD.md R2)."""
        raise NotImplementedError

    def unload(self) -> None:
        """Free GPU memory. Stage 1 calls this before stage 2 loads the judges."""


GENERATORS: Registry[Generator] = Registry("generator")
