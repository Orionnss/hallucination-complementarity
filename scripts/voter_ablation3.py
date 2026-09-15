"""Voter-count ablation over SAPLMA, LapEigvals and ICR, each in its published setup.

Three detectors, so seven non-empty subsets. Every detector is used exactly as its paper
specifies: its own published probe AND its own published decision threshold, both taken
from stage 3's stored out-of-fold arrays. At k=1 the reported figure is therefore the
detector's reported figure, not a re-thresholded variant of it.

That fixes a confound in the earlier five-detector sweep. There, k=1 re-thresholded each
single detector by the same cross-fitting rule the combiners use, which on Qwen3-14B moved
SAPLMA from 0.4827 to 0.5178 MCC -- a +0.035 gain from the threshold rule alone, before any
voting. The k=1 to k=2 step mixed that with the effect of adding a voter.

One asymmetry cannot be removed and is stated rather than hidden:

  k=1            uses each detector's published threshold
  vote_hard      needs no threshold: majority of the published decisions
  vote_soft      has no published threshold to inherit, so one is fitted by grouped 5-fold
  vote_rank      cross-fitting over the out-of-fold scores

So vote_hard is the only strictly like-for-like comparison against the k=1 rows: it
consumes the same published decisions and adds nothing. vote_soft and vote_rank are given
a fitted threshold the singles do not get, which favours them. Read vote_hard as the clean
test and the other two as an upper estimate.

Majority is defined only at odd k. At k=3 it is 2 of 3. At k=2 there is no majority, so
vote_hard is reported only at k=1 (trivially the detector itself) and k=3.

Usage: uv run python scripts/voter_ablation3.py
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

METHODS = ["saplma", "lapeigvals", "icr"]
SHORT = {"saplma": "SAPLMA", "lapeigvals": "LapEig", "icr": "ICR"}
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

        cand = {}
        for m in METHODS:
            cand[(1, SHORT[m], "published")] = (S[m], P[m])
        for k in (2, 3):
            for sub in itertools.combinations(METHODS, k):
                tag = "+".join(SHORT[m] for m in sub)
                soft = np.mean([S[m] for m in sub], axis=0)
                rank = np.mean([Rk[m] for m in sub], axis=0)
                cand[(k, tag, "soft")] = (
                    soft, cross_fit_predict(y, soft, groups, seed))
                cand[(k, tag, "rank")] = (
                    rank, cross_fit_predict(y, rank, groups, seed))
                if k == 3:      # majority is defined only at odd k
                    v = np.sum([P[m] for m in sub], axis=0)
                    cand[(k, tag, "hard")] = (v / k, (v >= 2).astype(int))

        for scope in SCOPES:
            msk = np.ones(len(y), bool) if scope == "pooled" else (ds == scope)
            if len(np.unique(y[msk])) < 2:
                continue
            for (k, tag, rule), (sc, pr) in cand.items():
                key = f"{scope}|{k}|{tag}|{rule}"
                acc[key]["mcc"].append(float(matthews_corrcoef(y[msk], pr[msk])))
                acc[key]["auroc"].append(float(roc_auc_score(y[msk], sc[msk])))
    return {k: {m: [round(float(x), 6) for x in v] for m, v in dd.items()}
            for k, dd in acc.items()}


def report(out, scope):
    for metric in ("mcc", "auroc"):
        print(f"\n{'=' * 96}\n  {metric.upper()}   scope={scope}   mean $\\pm$ sd over 5 seeds"
              f"\n{'=' * 96}")
        for run, blob in out.items():
            print(f"\n  {run}")
            print(f"    {'k':>2s}  {'subset':26s}{'rule':12s}{'value':>10s}{'sd':>8s}")
            rows = [(int(key.split("|")[1]), key.split("|")[2], key.split("|")[3], v)
                    for key, v in blob.items() if key.startswith(f"{scope}|")]
            order = {"published": 0, "hard": 1, "rank": 2, "soft": 3}
            for k, tag, rule, v in sorted(rows, key=lambda r: (r[0], order[r[2]], r[1])):
                print(f"    {k:>2d}  {tag:26s}{rule:12s}"
                      f"{np.mean(v[metric]):>10.4f}{np.std(v[metric]):>8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--scope", default="pooled")
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        res = run_one(run)
        if res is None:
            print(f"  {run}: skipped"); continue
        out[run] = res
        print(f"  {run}: done", flush=True)

    write_json(Path("runs/voter_ablation3.json"), out)
    with open("runs/voter_ablation3.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "dataset", "k", "subset", "rule", "metric",
                    "mean", "sd", "n_seeds"])
        for run, blob in out.items():
            for key, d in blob.items():
                scope, k, tag, rule = key.split("|")
                for metric, vals in d.items():
                    w.writerow([run, scope, k, tag, rule, metric,
                                round(float(np.mean(vals)), 6),
                                round(float(np.std(vals)), 6), len(vals)])
    report(out, args.scope)
    print("\nwrote runs/voter_ablation3.json and .csv")


if __name__ == "__main__":
    main()
