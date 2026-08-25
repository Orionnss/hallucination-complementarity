"""Scoring and predictor-vs-predictor agreement tests.

MCC is the headline agreement-with-reference score; AUROC is reported alongside because
it is threshold-free and therefore comparable with the numbers the original papers
publish. Cohen's kappa and McNemar operate on hard decisions and answer a different
question: not "who is more accurate" but "do these two detectors make the same mistakes".
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximising MCC on a validation split.

    Candidates are the midpoints between consecutive distinct scores, so every
    achievable split of the data is considered exactly once.
    """
    order = np.unique(scores)
    if order.size < 2:
        return 0.5, 0.0
    candidates = (order[:-1] + order[1:]) / 2.0
    best_mcc, best_t = -2.0, 0.5
    for t in candidates:
        mcc = matthews_corrcoef(y_true, (scores >= t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    return best_t, best_mcc


def score_predictions(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    # AUROC is undefined if the fold happens to be single-class.
    auroc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None
    return {
        "auroc": auroc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test on the discordant pairs of two predictors.

    b = A right / B wrong, c = A wrong / B right. Only discordant pairs carry
    information about whether the two differ. The exact binomial test is used when the
    discordant count is small, where the chi-square approximation is unreliable.
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "statistic": None, "p_value": 1.0,
                "test": "none", "note": "predictors made identical decisions"}
    if n_discordant < 25:
        p = float(binomtest(b, n_discordant, 0.5).pvalue)
        return {"b": b, "c": c, "n_discordant": n_discordant, "statistic": None,
                "p_value": p, "test": "exact_binomial"}
    # Continuity-corrected chi-square, 1 dof.
    statistic = (abs(b - c) - 1) ** 2 / n_discordant
    return {"b": b, "c": c, "n_discordant": n_discordant,
            "statistic": float(statistic),
            "p_value": float(chi2.sf(statistic, 1)), "test": "chi2_continuity"}


def pairwise_agreement(y_true: np.ndarray, preds: dict[str, np.ndarray]) -> dict:
    """Cohen's kappa and McNemar for every pair of predictors."""
    names = sorted(preds)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pa, pb = preds[a], preds[b]
            both = len(set(pa) | set(pb)) > 1
            out[f"{a}|{b}"] = {
                "cohen_kappa": float(cohen_kappa_score(pa, pb)) if both else None,
                "raw_agreement": float(np.mean(pa == pb)),
                "mcnemar": mcnemar(y_true, pa, pb),
            }
    return out


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni correction.

    With 7 predictors there are 21 pairwise McNemar tests, so uncorrected p-values would
    turn chance into significance.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, previous = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        adjusted = min(max((m - rank) * p, previous), 1.0)
        previous = adjusted
        out[key] = {"p_raw": p, "p_holm": adjusted, "significant": adjusted < alpha}
    return out
