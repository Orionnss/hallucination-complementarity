"""How many voters does a vote need? All 31 non-empty subsets of the five detectors.

The voting results are reported over all five detectors at once, which leaves two things
unseparated: whether the gain comes from having MANY voters or from having the RIGHT ones,
and whether adding a weak detector to a strong one helps or hurts. Sweeping every subset
answers both, because at each size k the spread across subsets is the effect of membership
and the trend across k is the effect of count.

For every subset S of size k:

  soft   mean of the |S| probability scores, thresholded once
  rank   mean of the |S| within-method normalised ranks, thresholded once
  hard   strict majority, votes > k/2. At k=2 that is unanimity, and at even k there is no
         tie-break, which is why hard voting is not comparable across adjacent sizes the
         way soft and rank are.

Thresholds are chosen by grouped 5-fold cross-fitting over the out-of-fold scores, exactly
as in the five-detector run, so a subset is never scored under a threshold fitted on the
item being scored. At k=1 soft and rank reduce to the single detector re-thresholded by
that rule, which is the correct baseline for the sweep: it isolates the effect of
combining from the effect of the threshold rule.

Reported per size: the mean over subsets, the best and worst subset, and the membership of
the best one. Detectors are under their own published probes.

Usage: uv run python scripts/voter_ablation.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import rankdata
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from halluc.io import write_json

METHODS = ["saplma", "lapeigvals", "icr", "attn_baseline", "svd_baseline"]
SHORT = {"saplma": "SAP", "lapeigvals": "LAP", "icr": "ICR",
         "attn_baseline": "ATT", "svd_baseline": "SVD"}
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b",
        "llama3.2-3b-base", "gemma3-12b-pt"]
SCOPES = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]


def cross_fit_predict(y, s, groups, seed) -> np.ndarray:
    """Threshold each fold using only the other folds, and return the predictions.

    The earlier form of this returned a single float: the mean of the five per-fold
    optima, applied to every item. That leaks. Each item sits in the training part of
    four of the five folds, so the averaged threshold has seen its label, and pooled MCC
    came out about +0.0035 too high across all six generators -- landing between the
    honest value and one fitted on the whole vector, which is exactly what averaging
    four-fifths-seen optima produces.

    Grouping is not optional: CoQA turns share a story and SQuAD questions share a
    paragraph, so an ungrouped split puts near-duplicates on both sides of it.
    """
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            s.reshape(-1, 1), y, groups):
        grid = np.quantile(s[tr], np.linspace(0.05, 0.95, 91))
        thr = max(grid, key=lambda t: matthews_corrcoef(y[tr], (s[tr] >= t).astype(int)))
        pred[te] = (s[te] >= thr).astype(int)
    return pred

def run_one(run: str):
    files = sorted(glob.glob(f"runs/{run}/stage3_train/pooled/seed*/predictions.npz"),
                   key=lambda p: int(Path(p).parent.name.replace("seed", "")))
    if not files:
        return None
    acc = defaultdict(lambda: defaultdict(list))
    for f in files:
        seed = int(Path(f).parent.name.replace("seed", ""))
        d = np.load(f, allow_pickle=True)
        y = d["y"].astype(int); ds = d["dataset"].astype(str)
        groups = d["groups"].astype(str)
        P = {m: d[f"preds__{m}"].astype(int) for m in METHODS}
        S = {m: d[f"scores__{m}"].astype(float) for m in METHODS}
        Rk = {m: rankdata(S[m]) / len(y) for m in METHODS}

        for k in range(1, len(METHODS) + 1):
            for sub in itertools.combinations(METHODS, k):
                tag = "+".join(SHORT[m] for m in sub)
                soft = np.mean([S[m] for m in sub], axis=0)
                rank = np.mean([Rk[m] for m in sub], axis=0)
                votes = np.sum([P[m] for m in sub], axis=0)
                cand = {
                    "soft": (soft, cross_fit_predict(y, soft, groups, seed)),
                    "rank": (rank, cross_fit_predict(y, rank, groups, seed)),
                    "hard": (votes / k, votes > k / 2.0),
                }
                for scope in SCOPES:
                    msk = np.ones(len(y), bool) if scope == "pooled" else (ds == scope)
                    if len(np.unique(y[msk])) < 2:
                        continue
                    for name, (sc, pr) in cand.items():
                        key = f"{scope}|{name}|{k}|{tag}"
                        acc[key]["mcc"].append(
                            float(matthews_corrcoef(y[msk], pr[msk].astype(int))))
                        acc[key]["auroc"].append(float(roc_auc_score(y[msk], sc[msk])))
    return {k: {m: [round(float(x), 6) for x in v] for m, v in dd.items()}
            for k, dd in acc.items()}


def report(out, scope="pooled"):
    for metric in ("mcc", "auroc"):
        print(f"\n{'=' * 118}\n  {metric.upper()}   scope={scope}   "
              f"mean over subsets of each size, with best and worst subset\n{'=' * 118}")
        for run, blob in out.items():
            print(f"\n  {run}")
            print(f"    {'rule':6s}{'k':>3s}{'n_sub':>7s}{'mean':>9s}{'worst':>9s}"
                  f"{'best':>9s}   best subset")
            for rule in ("soft", "rank", "hard"):
                for k in range(1, 6):
                    vals = [(float(np.mean(v[metric])), key.split("|")[3])
                            for key, v in blob.items()
                            if key.startswith(f"{scope}|{rule}|{k}|")]
                    if not vals:
                        continue
                    best = max(vals); worst = min(vals)
                    print(f"    {rule:6s}{k:>3d}{len(vals):>7d}"
                          f"{np.mean([v for v, _ in vals]):>9.4f}{worst[0]:>9.4f}"
                          f"{best[0]:>9.4f}   {best[1]}")
                print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--scope", default="pooled")
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        res = run_one(run)
        if res is None:
            print(f"  {run}: no stage-3 predictions, skipped"); continue
        out[run] = res
        print(f"  {run}: done", flush=True)

    write_json(Path("runs/voter_ablation.json"), out)
    with open("runs/voter_ablation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "dataset", "rule", "k", "subset", "metric",
                    "mean", "sd", "n_seeds"])
        for run, blob in out.items():
            for key, d in blob.items():
                scope, rule, k, tag = key.split("|")
                for metric, vals in d.items():
                    w.writerow([run, scope, rule, k, tag, metric,
                                round(float(np.mean(vals)), 6),
                                round(float(np.std(vals)), 6), len(vals)])
    report(out, args.scope)
    print("\nwrote runs/voter_ablation.json and .csv")


if __name__ == "__main__":
    main()
