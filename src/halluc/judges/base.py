"""Judge abstraction and the label vocabulary.

A judge sees the question, the reference answer(s), and the model's answer, and returns
one of three labels. INVALID exists so that refusals ("I don't know") are not scored as
hallucinations — they are dropped from training and evaluation entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..registry import Registry


class Label(StrEnum):
    HALLUCINATED = "HALLUCINATED"
    NOT_HALLUCINATED = "NOT_HALLUCINATED"
    INVALID = "INVALID"
    #: The judge emitted something unparseable. Kept distinct from INVALID so a broken
    #: judge shows up in the JSON as a parse-rate problem instead of silently inflating
    #: the refusal class.
    UNPARSEABLE = "UNPARSEABLE"


def parse_label(text: str) -> Label:
    """Map raw judge output onto a label.

    NOT_HALLUCINATED is tested before HALLUCINATED: it contains the other as a
    substring, so the order is load-bearing.
    """
    upper = text.strip().upper().replace("-", "_").replace(" ", "_")
    for candidate in (Label.NOT_HALLUCINATED, Label.INVALID, Label.HALLUCINATED):
        if candidate.value in upper:
            return candidate
    return Label.UNPARSEABLE


@dataclass
class Verdict:
    item_id: str
    judge: str
    label: Label
    raw: str
    seconds: float


class Judge:
    name: str

    def judge(self, question: str, gold_answers: list[str], answer: str) -> tuple[Label, str]:
        raise NotImplementedError

    def unload(self) -> None:
        """Free GPU memory so the next judge in the pool can load."""


JUDGES: Registry[Judge] = Registry("judge")
