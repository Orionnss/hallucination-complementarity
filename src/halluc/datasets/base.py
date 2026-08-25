"""QA dataset abstraction.

A dataset adapter's only job is to yield `QAItem`s with a stable `item_id`. Stability
matters: `item_id` is the key for checkpoints, feature shards, and labels, so a resumed
or re-run pipeline must produce the same ids for the same source rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..registry import Registry


@dataclass(frozen=True)
class QAItem:
    item_id: str
    question: str
    gold_answers: list[str]
    #: Passage the question depends on. `None` for closed-book datasets — see the
    #: context policy in DESIGN.md: only CoQA keeps its passage.
    context: str | None = None
    #: Cross-validation grouping key. Items sharing a group must never be split across
    #: train and test: CoQA turns share a story, SQuAD questions share an article, so
    #: random folds would leak the passage. Defaults to `item_id` (every item its own
    #: group) for genuinely independent datasets.
    group_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def group(self) -> str:
        return self.group_id if self.group_id is not None else self.item_id


class QADataset:
    """Base class for dataset adapters.

    Subclasses set `name` and implement `load()`.
    """

    name: str

    def load(self) -> list[QAItem]:
        raise NotImplementedError

    def sample_pool(self, pool_size: int, pool_seed: int = 0) -> list[QAItem]:
        """Deterministic pool of at most `pool_size` items.

        The pool is drawn once per dataset and shared across experiment seeds; the
        per-seed draws in stage 3 subsample *from* this pool. Because generation is
        greedy, that is exactly equivalent to re-generating per seed, at 1/n the cost.
        """
        import numpy as np

        items = self.load()
        if len(items) <= pool_size:
            return items
        rng = np.random.default_rng(pool_seed)
        idx = rng.choice(len(items), size=pool_size, replace=False)
        return [items[i] for i in sorted(idx)]


DATASETS: Registry[QADataset] = Registry("dataset")
