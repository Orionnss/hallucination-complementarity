"""The accuracy-independence tradeoff without a reference method.

Every conditional result in this study is anchored on SAPLMA's errors. That was a defensible
choice while SAPLMA was the champion, but it makes the tradeoff look like a fact about one
method rather than about the pool. This removes the anchor: for every ordered pair (A, B),
how often does B rescue A's errors, against what independence would predict?

  rescue(A, B)  = accuracy of B on the items A gets wrong
  null(A, B)    = B's per-class accuracy over ALL items, reweighted to the class mix of
                  A's error slice — what an independent method of B's strength would reach
  ratio         = rescue / null.  1.0 means independent, below 1 means B fails where A does

Two claims are then testable without privileging anyone:

  (1) is ratio < 1 for every ordered pair, not just those anchored on SAPLMA?
  (2) does Spearman(B's overall accuracy, ratio) stay negative for every choice of A?

(1) says the redundancy is pool-wide. (2) says the tradeoff — better methods being more
redundant — holds no matter which method's errors you look at. Together they make the
mechanism a property of the family rather than of a reference, which is what lets the paper
drop its champion.

The diagonal is undefined by construction: A rescues 0% of its own errors, so those cells
are left empty rather than filled with a meaningless zero.

Usage: uv run python scripts/pairwise_independence.py --run main
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

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from halluc.io import write_json

RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b"]


def methods_in(path):
    d = np.load(path, allow_pickle=True)
    return [k.replace("preds__", "") for k in d.files if k.startswith("preds__")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        files = [f for i, f in enumerate(sorted(glob.glob(
            f"runs/{run}/stage5_posthoc/block_oof/*.npz"))) if i in args.seeds]
        if not files:
            continue
        M = methods_in(files[0])
        cell = defaultdict(lambda: defaultdict(list))
        overall = defaultdict(list)

        for f in files:
            d = np.load(f, allow_pickle=True)
            y = d["y"]
            P = {m: d[f"preds__{m}"] for m in M}
            for m in M:
                overall[m].append(float((P[m] == y).mean()))
            for a in M:
                wrong = P[a] != y
                if wrong.sum() < 20 or len(np.unique(y[wrong])) < 2:
                    continue
                mix = {c: (y[wrong] == c).mean() for c in (0, 1)}
                for b in M:
                    if a == b:
                        continue
                    p = P[b]
                    resc = float((p[wrong] == y[wrong]).mean())
                    per_c = {c: (p[y == c] == c).mean() for c in (0, 1)}
                    null = float(sum(per_c[c] * mix[c] for c in (0, 1)))
                    cell[(a, b)]["rescue"].append(resc)
                    cell[(a, b)]["null"].append(null)
                    cell[(a, b)]["ratio"].append(resc / null if null else np.nan)
                    cell[(a, b)]["kappa"].append(
                        float(cohen_kappa_score(y[wrong], p[wrong]))
                        if len(np.unique(p[wrong])) > 1 else 0.0)

        acc_m = {m: float(np.mean(v)) for m, v in overall.items()}
        order = sorted(M, key=lambda m: -acc_m[m])

        print(f"\n{'=' * 96}\n  {run}: ratio of rescue to independence, "
              f"rows = whose errors, cols = who rescues\n{'=' * 96}")
        print(f"  {'errors of \\\\ rescuer':22s}" + "".join(f"{b[:9]:>11s}" for b in order)
              + f"{'row mean':>11s}")
        ratios = []
        for a in order:
            vals = [np.mean(cell[(a, b)]["ratio"]) if (a, b) in cell else np.nan
                    for b in order]
            ratios.append(vals)
            rm = np.nanmean(vals)
            print(f"  {a:22s}" + "".join(
                f"{'—':>11s}" if np.isnan(v) else f"{v:11.2f}" for v in vals)
                + f"{rm:11.2f}")
        R = np.array(ratios)

        n_pairs = int(np.sum(~np.isnan(R)))
        n_below = int(np.nansum(R < 1.0))
        worst = np.nanmax(R)
        print(f"\n  (1) ratio < 1 in {n_below} of {n_pairs} ordered pairs; "
              f"largest ratio anywhere = {worst:.2f}")

        print(f"\n  (2) Spearman(rescuer's overall accuracy, ratio), per row")
        rhos = []
        for i, a in enumerate(order):
            xs = [acc_m[b] for j, b in enumerate(order) if not np.isnan(R[i, j])]
            ys = [R[i, j] for j in range(len(order)) if not np.isnan(R[i, j])]
            rho = spearmanr(xs, ys).statistic if len(xs) > 2 else np.nan
            rhos.append(rho)
            print(f"      errors of {a:20s} rho = {rho:+.3f}  (n={len(xs)})")
        neg = int(np.nansum(np.array(rhos) < 0))
        print(f"    negative in {neg} of {len(rhos)} rows"
              f"   mean rho = {np.nanmean(rhos):+.3f}")

        print(f"\n  overall accuracy, for reference")
        print("    " + "  ".join(f"{m}:{acc_m[m]:.3f}" for m in order))

        out[run] = {
            "methods_by_accuracy": order,
            "overall_accuracy": {m: round(acc_m[m], 4) for m in order},
            "ratio": {f"{a}|{b}": round(float(np.mean(cell[(a, b)]["ratio"])), 4)
                      for (a, b) in cell},
            "rescue": {f"{a}|{b}": round(float(np.mean(cell[(a, b)]["rescue"])), 4)
                       for (a, b) in cell},
            "null": {f"{a}|{b}": round(float(np.mean(cell[(a, b)]["null"])), 4)
                     for (a, b) in cell},
            "kappa_on_errors": {f"{a}|{b}": round(float(np.mean(cell[(a, b)]["kappa"])), 4)
                                for (a, b) in cell},
            "pairs_below_one": [n_below, n_pairs],
            "max_ratio": round(float(worst), 4),
            "spearman_per_row": {a: (None if np.isnan(r) else round(float(r), 4))
                                 for a, r in zip(order, rhos)},
        }

    write_json(Path("runs/pairwise_independence.json"), out)
    print("\nwrote runs/pairwise_independence.json")


if __name__ == "__main__":
    main()
