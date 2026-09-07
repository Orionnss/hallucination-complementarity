"""Plain voting over the five detectors — no stacking, no refitting.

Every combination tried so far learned a classifier on top of the blocks. This asks the
simpler question: if you just let the five methods vote, what do you get? It is the
cheapest possible combiner, and the one with no parameters to overfit.

Two readings of "voting", because they answer different questions:

  hard   majority of the five binary predictions (>= 3 of 5). The literal reading. Its
         weakness is that each method's threshold was tuned separately to maximise its own
         MCC, so the five votes are not calibrated to a common operating point and a
         method that fires often carries more weight than its accuracy earns.
  soft   mean of the five probability scores, thresholded once. Fixes the calibration
         problem, at the cost of no longer being a vote in the strict sense.

Both are computed from the stored out-of-fold predictions, so no model is refitted and
nothing new is fitted to the test data. The soft threshold is the only free parameter; it
is chosen per seed on the training folds' out-of-fold scores, never on the fold being
scored — see `_thresh`.

Rank-mean voting is included as a third row: scores from different methods live on
different scales, and averaging raw probabilities lets an over-confident method dominate.
Averaging within-method ranks removes the scale without needing calibration.

Reference rows are the strongest single method and the learned union, so the reader can
see what the classifier was buying.

Usage: uv run python scripts/vote.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import rankdata
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

from halluc.io import write_json

METHODS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline"]
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b"]


def _thresh(y, s, groups, seed):
    """Pick one threshold by grouped 5-fold cross-fitting, so no item's own fold sets it.

    Averaging the per-fold optima rather than optimising on all of the data keeps this
    honest: the stored scores are already out-of-fold with respect to the detectors, but
    a threshold chosen on the full vector would still peek.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    picks = []
    for tr, _ in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            s.reshape(-1, 1), y, groups):
        grid = np.quantile(s[tr], np.linspace(0.05, 0.95, 91))
        picks.append(max(grid, key=lambda t: matthews_corrcoef(y[tr], (s[tr] >= t).astype(int))))
    return float(np.mean(picks))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    args = ap.parse_args()

    fm = json.loads(Path("runs/full_metrics.json").read_text())
    pc = json.loads(Path("runs/saplma_pcalr_metrics.json").read_text())
    out = {}

    for run in args.runs:
        acc = defaultdict(lambda: defaultdict(list))
        for f in sorted(glob.glob(f"runs/{run}/stage5_posthoc/block_oof/*.npz")):
            seed = int(Path(f).stem.replace("seed", ""))
            d = np.load(f, allow_pickle=True)
            y, ds_arr = d["y"], d["dataset"].astype(str)
            groups = d["groups"].astype(str)
            P = np.stack([d[f"preds__{m}"] for m in METHODS])          # [5, N] binary
            S = np.stack([d[f"scores__{m}"] for m in METHODS])         # [5, N] probability

            votes = P.sum(0)
            R = np.stack([rankdata(s) / len(s) for s in S])

            cand = {"vote_hard": ((votes >= 3).astype(int), votes / 5.0),
                    "vote_hard_4of5": ((votes >= 4).astype(int), votes / 5.0),
                    "vote_unanimous": ((votes == 5).astype(int), votes / 5.0)}
            for name, sc in (("vote_soft", S.mean(0)), ("vote_rank", R.mean(0))):
                cand[name] = ((sc >= _thresh(y, sc, groups, seed)).astype(int), sc)

            for name, (pred, sc) in cand.items():
                for scope in ["pooled", "pooled_no_coqa"] + sorted(set(ds_arr)):
                    m = (np.ones(len(y), bool) if scope == "pooled"
                         else (ds_arr != "coqa") if scope == "pooled_no_coqa"
                         else (ds_arr == scope))
                    if len(np.unique(y[m])) < 2:
                        continue
                    a = acc[(name, scope)]
                    a["mcc"].append(float(matthews_corrcoef(y[m], pred[m])))
                    a["auroc"].append(float(roc_auc_score(y[m], sc[m])))
                    a["accuracy"].append(float(accuracy_score(y[m], pred[m])))
                    a["balanced_accuracy"].append(
                        float(balanced_accuracy_score(y[m], pred[m])))
                    a["f1"].append(float(f1_score(y[m], pred[m], zero_division=0)))

        out[run] = {f"{n}|{s}": {k: round(float(np.mean(v)), 4) for k, v in dd.items()}
                    for (n, s), dd in acc.items()}

    scopes = ["pooled", "pooled_no_coqa", "triviaqa", "nq_open", "squad_v2", "coqa"]
    order = ["vote_hard", "vote_soft", "vote_rank", "vote_hard_4of5", "vote_unanimous"]
    for metric in ("mcc", "auroc", "balanced_accuracy"):
        print(f"\n{'=' * 96}\n  {metric.upper()}\n{'=' * 96}")
        for run in args.runs:
            print(f"\n  {run}")
            print(f"    {'variant':20s}" + "".join(f"{s[:13]:>13s}" for s in scopes))
            for n in order:
                print(f"    {n:20s}" + "".join(
                    f"{out[run][f'{n}|{s}'][metric]:13.4f}" if f"{n}|{s}" in out[run]
                    else f"{'-':>13s}" for s in scopes))
            # references, from the already-computed runs
            for lab, src, key in (("saplma (MLP)", fm, "saplma"),
                                  ("saplma (PCA+LR)", pc, None),
                                  ("union_equal", fm, "union_equal")):
                row = ""
                for s in scopes:
                    sc = "pooled" if s == "pooled_no_coqa" else s   # not stored no-coqa
                    if s == "pooled_no_coqa":
                        row += f"{'-':>13s}"; continue
                    v = src[run].get(sc, {})
                    v = v.get(key) if key else v
                    row += (f"{v[metric]['mean']:13.4f}"
                            if v and metric in v else f"{'-':>13s}")
                print(f"    {lab:20s}{row}")

    write_json(Path("runs/vote.json"), out)
    print("\nwrote runs/vote.json")


if __name__ == "__main__":
    main()
