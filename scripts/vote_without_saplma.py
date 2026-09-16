"""Can the other four detectors, voting together, replace SAPLMA?

SAPLMA wins every head-to-head under an equal classifier search, but that is a
one-against-one comparison. This asks the practical question behind it: if you pooled
everything *except* SAPLMA — attention spectra, raw attention, internal routing,
hidden-state spectra — would four methods together match the one you left out?

It matters for two reasons. If four combined beat one, the case for SAPLMA's primacy is
about convenience rather than information. If they do not, then SAPLMA is not merely the
best single method but carries signal the rest do not collectively hold, which is a much
stronger claim and the one the paper would rest on.

Three voting rules, since with four voters the choice is not innocent: hard majority has a
2-2 tie region, so its threshold is tuned like any other rather than fixed at "3 of 4".
Soft voting averages probabilities, rank voting averages within-method ranks to remove the
differing score scales.

Predictions come from the stored out-of-fold file, so nothing is refitted and the votes are
exactly the predictions used elsewhere in the study. Cohen's kappa is reported alongside
MCC because the question is about agreement with the reference, and kappa is the statistic
the original specification named.

Usage: uv run python scripts/vote_without_saplma.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import rankdata
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from halluc.io import write_json

OTHERS = ["lapeigvals", "attn_baseline", "icr", "svd_baseline"]
ALL = ["saplma"] + OTHERS
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b"]


def cross_fit_predict(y, s, groups, seed):
    """Threshold each fold on the other four, and return the predictions.

    This previously averaged the five per-fold optima and applied that one value to every
    item, under the claim that no item's own fold set it. That is not what it did: each
    item sits in the training part of four of the five folds, so the averaged threshold
    had seen its label. Measured elsewhere in this repo the leak was worth about +0.0035
    pooled MCC. Grouping stays mandatory -- CoQA turns share a story and SQuAD questions
    share a paragraph.
    """
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            s.reshape(-1, 1), y, groups):
        grid = np.unique(np.quantile(s[tr], np.linspace(0.02, 0.98, 60)))
        thr = max(grid, key=lambda t: matthews_corrcoef(y[tr], (s[tr] >= t).astype(int)))
        pred[te] = (s[te] >= thr).astype(int)
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        acc = defaultdict(lambda: defaultdict(list))
        for si, f in enumerate(sorted(glob.glob(
                f"runs/{run}/stage5_posthoc/block_oof/*.npz"))):
            if si not in args.seeds:
                continue
            d = np.load(f, allow_pickle=True)
            y, ds_arr = d["y"], d["dataset"].astype(str)
            groups = d["groups"].astype(str)
            P = {m: d[f"preds__{m}"] for m in ALL}
            S = {m: d[f"scores__{m}"] for m in ALL}

            v4 = np.stack([P[m] for m in OTHERS]).sum(0) / 4.0
            v5 = np.stack([P[m] for m in ALL]).sum(0) / 5.0
            soft4 = np.mean([S[m] for m in OTHERS], axis=0)
            rank4 = np.mean([rankdata(S[m]) / len(y) for m in OTHERS], axis=0)

            cand = {
                "saplma alone": (P["saplma"], S["saplma"]),
                "no-saplma vote_hard": (None, v4),
                "no-saplma vote_soft": (None, soft4),
                "no-saplma vote_rank": (None, rank4),
                "all-5 vote_soft": (None, np.mean([S[m] for m in ALL], axis=0)),
                "all-5 vote_hard": (None, v5),
            }
            for m in OTHERS:
                cand[f"  {m} alone"] = (P[m], S[m])

            for name, (pred, sc) in cand.items():
                if pred is None:
                    pred = cross_fit_predict(y, sc, groups, si)
                for scope in ["pooled"] + sorted(set(ds_arr)):
                    msk = (np.ones(len(y), bool) if scope == "pooled"
                           else (ds_arr == scope))
                    if len(np.unique(y[msk])) < 2:
                        continue
                    a = acc[(name, scope)]
                    a["mcc"].append(float(matthews_corrcoef(y[msk], pred[msk])))
                    a["auroc"].append(float(roc_auc_score(y[msk], sc[msk])))
                    a["kappa"].append(float(cohen_kappa_score(y[msk], pred[msk])))
        out[run] = {f"{n}|{s}": {k: round(float(np.mean(v)), 4) for k, v in dd.items()}
                    for (n, s), dd in acc.items()}

    order = ["saplma alone", "no-saplma vote_soft", "no-saplma vote_rank",
             "no-saplma vote_hard", "all-5 vote_soft", "all-5 vote_hard"] + \
            [f"  {m} alone" for m in OTHERS]
    scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]

    for metric in ("mcc", "auroc", "kappa"):
        print(f"\n{'=' * 92}\n  {metric.upper()}\n{'=' * 92}")
        for run in args.runs:
            r = out[run]
            print(f"\n  {run}")
            print(f"    {'variant':24s}" + "".join(f"{s[:11]:>13s}" for s in scopes))
            for n in order:
                if f"{n}|pooled" not in r:
                    continue
                print(f"    {n:24s}" + "".join(
                    f"{r[f'{n}|{s}'][metric]:13.4f}" if f"{n}|{s}" in r
                    else f"{'-':>13s}" for s in scopes))
            b = r["saplma alone|pooled"][metric]
            best = max((r[f"{n}|pooled"][metric], n) for n in order[1:4]
                       if f"{n}|pooled" in r)
            print(f"    {'best no-saplma vote':24s}{best[0]:13.4f}  ({best[1].strip()}), "
                  f"vs saplma {best[0] - b:+.4f}")

    write_json(Path("runs/vote_without_saplma.json"), out)
    print("\nwrote runs/vote_without_saplma.json")


if __name__ == "__main__":
    main()
