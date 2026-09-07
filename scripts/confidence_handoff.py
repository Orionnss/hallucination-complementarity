"""Is there a band of SAPLMA confidence where another method should take over?

Every router tried so far predicted which method to trust from *features*, and all of them
landed at chance. This uses the one signal already shown to carry information about
SAPLMA's errors: its own distance from its decision threshold, which separates its true
from false alarms at AUROC 0.71-0.77. If SAPLMA degrades faster than LapEigvals as
confidence falls, a hand-off band exists and is trivial to deploy - no probe, no extra
features, just a rule on a number you already have.

Two parts, and the second is the one that counts:

  descriptive   accuracy of each method within deciles of |score - threshold|. Shows
                whether a crossover exists at all.
  honest        a hand-off rule whose band is chosen on training folds and applied to
                held-out data, measured end-to-end. Reading the crossover off the full
                data and then reporting the gain would be fitting the router to the test
                set - the same error this study criticises elsewhere.

The null result to expect, if the mechanism holds: methods that agree with SAPLMA overall
also lose confidence on the same items, so the alternative degrades in step and there is no
crossover. A crossover would mean the redundancy is not uniform across the score range,
which would be genuinely new.

Usage: uv run python scripts/confidence_handoff.py
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

from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from halluc.io import write_json

METHODS = ["lapeigvals", "attn_baseline", "icr", "svd_baseline"]
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b"]
NBIN = 10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        bins = defaultdict(lambda: defaultdict(list))
        hyb = defaultdict(lambda: defaultdict(list))
        bands = defaultdict(list)

        for si, f in enumerate(sorted(glob.glob(
                f"runs/{run}/stage5_posthoc/block_oof/*.npz"))):
            if si not in args.seeds:
                continue
            d = np.load(f, allow_pickle=True)
            y, groups = d["y"], d["groups"].astype(str)
            s, sap = d["scores__saplma"], d["preds__saplma"]
            # threshold recovered from where the stored predictions switch
            thr = float(s[sap == 1].min()) if (sap == 1).any() else 0.5
            conf = np.abs(s - thr)
            P = {m: d[f"preds__{m}"] for m in METHODS}

            # --- descriptive: accuracy per confidence decile ---------------------
            edges = np.quantile(conf, np.linspace(0, 1, NBIN + 1))
            idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, NBIN - 1)
            for b in range(NBIN):
                m = idx == b
                if m.sum() < 20:
                    continue
                bins[b]["n"].append(int(m.sum()))
                bins[b]["saplma"].append(float((sap[m] == y[m]).mean()))
                for meth in METHODS:
                    bins[b][meth].append(float((P[meth][m] == y[m]).mean()))

            # --- honest: band chosen on train folds, applied to held-out ---------
            for meth in METHODS:
                pred = np.full(len(y), -1, dtype=int)
                for tr, te in StratifiedGroupKFold(
                        5, shuffle=True, random_state=si).split(
                            conf.reshape(-1, 1), y, groups):
                    grid = np.quantile(conf[tr], np.linspace(0.0, 0.6, 31))
                    best_t, best_m = 0.0, -2.0
                    for t in grid:
                        cand = np.where(conf[tr] < t, P[meth][tr], sap[tr])
                        mm = matthews_corrcoef(y[tr], cand)
                        if mm > best_m:
                            best_m, best_t = mm, float(t)
                    bands[meth].append(float((conf[te] < best_t).mean()))
                    pred[te] = np.where(conf[te] < best_t, P[meth][te], sap[te])
                hyb[meth]["mcc"].append(float(matthews_corrcoef(y, pred)))
                hyb[meth]["kappa"].append(float(cohen_kappa_score(y, pred)))
                hyb[meth]["acc"].append(float((pred == y).mean()))
            hyb["saplma alone"]["mcc"].append(float(matthews_corrcoef(y, sap)))
            hyb["saplma alone"]["kappa"].append(float(cohen_kappa_score(y, sap)))
            hyb["saplma alone"]["acc"].append(float((sap == y).mean()))
            hyb["saplma alone"]["auroc"].append(float(roc_auc_score(y, s)))

        print(f"\n{'=' * 94}\n  {run}: accuracy by SAPLMA confidence decile "
              f"(|score - threshold|)\n{'=' * 94}")
        print(f"  {'decile':10s}{'n':>7s}{'SAPLMA':>10s}" +
              "".join(f"{m[:12]:>13s}" for m in METHODS) + "   crossover")
        for b in range(NBIN):
            if "n" not in bins[b]:
                continue
            sa = np.mean(bins[b]["saplma"])
            row = "".join(f"{np.mean(bins[b][m]):13.1%}" for m in METHODS)
            better = [m for m in METHODS if np.mean(bins[b][m]) > sa]
            print(f"  {b + 1:<10d}{np.mean(bins[b]['n']):7.0f}{sa:10.1%}{row}"
                  f"   {','.join(better) if better else '-'}")

        print(f"\n  hand-off rule, band chosen on training folds only")
        base = np.mean(hyb["saplma alone"]["mcc"])
        print(f"    {'rule':28s}{'MCC':>9s}{'kappa':>9s}{'accuracy':>11s}"
              f"{'deferred':>10s}{'vs SAPLMA':>11s}")
        print(f"    {'saplma alone':28s}{base:9.4f}"
              f"{np.mean(hyb['saplma alone']['kappa']):9.4f}"
              f"{np.mean(hyb['saplma alone']['acc']):11.1%}{'0.0%':>10s}{'':>11s}")
        for meth in METHODS:
            v = hyb[meth]
            print(f"    {'defer to ' + meth:28s}{np.mean(v['mcc']):9.4f}"
                  f"{np.mean(v['kappa']):9.4f}{np.mean(v['acc']):11.1%}"
                  f"{np.mean(bands[meth]):10.1%}"
                  f"{np.mean(v['mcc']) - base:+11.4f}")

        out[run] = {
            "deciles": {str(b + 1): {k: round(float(np.mean(v)), 4)
                                     for k, v in bins[b].items()}
                        for b in range(NBIN) if "n" in bins[b]},
            "handoff": {k: {m: round(float(np.mean(v)), 4) for m, v in dd.items()}
                        for k, dd in hyb.items()},
            "deferred_fraction": {m: round(float(np.mean(v)), 4)
                                  for m, v in bands.items()},
        }

    write_json(Path("runs/confidence_handoff.json"), out)
    print("\nwrote runs/confidence_handoff.json")


if __name__ == "__main__":
    main()
