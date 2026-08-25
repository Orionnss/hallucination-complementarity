"""Vote aggregation and inter-judge agreement.

With three judges from three different model families, agreement is a meaningful
reliability statistic rather than an artifact of shared pretraining.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from sklearn.metrics import cohen_kappa_score

from .base import Label

#: Labels that survive into training and evaluation. INVALID and UNPARSEABLE items are
#: dropped from both, per the study design.
SCORED = (Label.HALLUCINATED, Label.NOT_HALLUCINATED)


def majority_vote(labels: list[Label]) -> tuple[Label, int]:
    """Majority label and its vote count.

    No strict majority means the judges genuinely disagree about what the answer even
    is, so the item is marked INVALID and dropped rather than assigned a coin-flip
    label that would add noise to every method's score equally.
    """
    label, votes = Counter(labels).most_common(1)[0]
    if votes * 2 > len(labels):  # strict majority
        return Label(label), votes
    return Label.INVALID, votes


def agreement_stats(
    verdicts_by_judge: dict[str, dict[str, Label]], item_ids: list[str]
) -> dict:
    """Pairwise raw agreement and Cohen's kappa for every judge pair."""
    judges = sorted(verdicts_by_judge)
    pairs = {}
    for a, b in combinations(judges, 2):
        labels_a = [verdicts_by_judge[a][i].value for i in item_ids]
        labels_b = [verdicts_by_judge[b][i].value for i in item_ids]
        matches = sum(x == y for x, y in zip(labels_a, labels_b))
        # Kappa is undefined when both raters use exactly one label; report null rather
        # than a NaN that would silently poison downstream aggregation.
        distinct = len(set(labels_a) | set(labels_b))
        kappa = (
            float(cohen_kappa_score(labels_a, labels_b))
            if distinct > 1 and len(item_ids) > 1
            else None
        )
        pairs[f"{a}|{b}"] = {
            "agreement_rate": round(matches / len(item_ids), 4) if item_ids else None,
            "cohen_kappa": round(kappa, 4) if kappa is not None else None,
            "n": len(item_ids),
        }

    per_judge = {
        judge: dict(Counter(v.value for v in verdicts.values()))
        for judge, verdicts in verdicts_by_judge.items()
    }
    unanimous = sum(
        len({verdicts_by_judge[j][i] for j in judges}) == 1 for i in item_ids
    )
    return {
        "judges": judges,
        "pairwise": pairs,
        "label_counts_per_judge": per_judge,
        "unanimous_rate": round(unanimous / len(item_ids), 4) if item_ids else None,
        "n_items": len(item_ids),
    }
