"""How much labelled data does the SAPLMA probe actually need?

Everything in this study trains on ~6,400 items per fold. That is a lot of judge verdicts to
buy — 3 judges x 15,610 answers per generator here — so the practical question for anyone
reproducing this is how far down that number can go before the detector degrades.

The training fold is subsampled to a grid of sizes; the test fold is always left whole, so
every point is scored on the same items and only the training budget varies. The reported
answer is the smallest size reaching 95% and 99% of the full-data metric, linearly
interpolated between grid points rather than rounded up to the next one.

Two details that would otherwise distort the answer:

  PCA width is capped at min(128, n-1). At n=25 a 128-component projection is not merely
  ill-advised, it is undefined, and silently keeping 128 would make small sizes look worse
  than they are for the wrong reason.

  Subsampling is by group, not by item. CoQA turns share a passage and SQuAD questions
  share an article, so drawing items independently would count near-duplicates as
  independent examples and understate how much genuinely new data is needed.

AUROC and MCC are reported separately because they answer different questions: ranking
quality recovers much faster than a calibrated decision threshold, and a study that quoted
only one would give a misleading budget.

Usage: uv run python scripts/learning_curve.py --run main
"""

from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}
C_GRID = (0.003, 0.03, 0.3, 3.0)
SIZES = (50, 100, 200, 400, 800, 1600, 3200, 6400)
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b"]


def load_saplma(cfg, ids, layer):
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        a = load_features(cfg.stage_dir("stage1_extract", ds), "saplma", sub)
        a = a[:, min(layer, a.shape[1] - 1), :].astype(np.float32)
        if out is None:
            out = np.zeros((len(ids), a.shape[1]), np.float32)
        out[[pos[i] for i in sub]] = a
        del a
    return out


def subsample_by_group(rows, groups, n_target, rng):
    """Draw whole groups until n_target items are reached.

    Item-wise sampling would treat CoQA turns from one passage as independent examples,
    which inflates the effective sample size exactly where the probe benefits least.
    """
    g = groups[rows]
    uniq = rng.permutation(np.unique(g))
    keep, taken = [], 0
    for gid in uniq:
        idx = rows[g == gid]
        keep.append(idx)
        taken += len(idx)
        if taken >= n_target:
            break
    return np.concatenate(keep)[:n_target]


def fit_eval(Xtr, ytr, Xte, groups_tr, strat_tr, seed):
    n = len(ytr)
    dim = max(2, min(128, n - 1))
    s = StandardScaler().fit(Xtr)
    A, B = s.transform(Xtr), s.transform(Xte)
    p = PCA(n_components=min(dim, A.shape[1]), svd_solver="randomized",
            random_state=seed).fit(A)
    A, B = p.transform(A), p.transform(B)

    n_inner = 4 if n >= 200 else 2
    try:
        inner = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed)
        i_tr, i_va = next(inner.split(np.zeros(n), strat_tr, groups_tr))
    except ValueError:                    # too few groups or a one-class inner split
        cut = max(2, int(0.75 * n))
        i_tr, i_va = np.arange(cut), np.arange(cut, n)
    if len(np.unique(ytr[i_va])) < 2:
        cut = max(2, int(0.75 * n))
        i_tr, i_va = np.arange(cut), np.arange(cut, n)

    best_c, best_a = C_GRID[0], -1.0
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000,
                               class_weight="balanced").fit(A[i_tr], ytr[i_tr])
        a = roc_auc_score(ytr[i_va], m.predict_proba(A[i_va])[:, 1])
        if a > best_a:
            best_a, best_c = float(a), C
    m = LogisticRegression(C=best_c, max_iter=3000,
                           class_weight="balanced").fit(A[i_tr], ytr[i_tr])
    thr, _ = best_threshold(ytr[i_va], m.predict_proba(A[i_va])[:, 1])
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(A, ytr)
    sc = m.predict_proba(B)[:, 1]
    return sc, (sc >= thr).astype(int)


def crossing(sizes, vals, full, frac):
    """Smallest training size reaching `frac` of the full-data value, interpolated."""
    target = frac * full
    for i, (n, v) in enumerate(zip(sizes, vals)):
        if v >= target:
            if i == 0:
                return float(n)
            n0, v0 = sizes[i - 1], vals[i - 1]
            if v == v0:
                return float(n)
            return float(n0 + (n - n0) * (target - v0) / (v - v0))
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    dest = Path("runs/learning_curve.json")
    import json
    out = json.loads(dest.read_text()) if dest.exists() else {}

    for run in args.runs:
        cfg = Config(); cfg.run_id = run
        acc = defaultdict(lambda: defaultdict(list))
        for seed in args.seeds:
            t0 = time.perf_counter()
            sd = load_seed(cfg, seed)
            ids, y = list(sd["item_ids"]), sd["y"]
            groups, ds_arr = sd["groups"].astype(str), sd["dataset"].astype(str)
            strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
            X = load_saplma(cfg, ids, LAYER.get(run, 24))
            rng = np.random.default_rng(seed)

            for n in list(SIZES) + ["full"]:
                sc = np.full(len(y), np.nan); pr = np.full(len(y), -1, dtype=int)
                for tr, te in _folds(sd, seed, cfg.n_folds):
                    sub = tr if n == "full" else subsample_by_group(tr, groups, n, rng)
                    if len(np.unique(y[sub])) < 2:
                        continue
                    a, b = fit_eval(X[sub], y[sub], X[te], groups[sub], strat[sub], seed)
                    sc[te], pr[te] = a, b
                ok = ~np.isnan(sc)
                for scope in ["pooled"] + sorted(set(ds_arr)):
                    m = ok & (np.ones(len(y), bool) if scope == "pooled"
                              else (ds_arr == scope))
                    if m.sum() < 50 or len(np.unique(y[m])) < 2:
                        continue
                    acc[(n, scope)]["auroc"].append(float(roc_auc_score(y[m], sc[m])))
                    acc[(n, scope)]["mcc"].append(float(matthews_corrcoef(y[m], pr[m])))
            print(f"  {run} seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min",
                  flush=True)
            del X

        scopes = ["pooled"] + sorted({s for (_, s) in acc} - {"pooled"})
        res = {}
        print(f"\n{'=' * 90}\n  {run}: training-set size vs performance "
              f"({len(args.seeds)} seeds)\n{'=' * 90}")
        for metric in ("auroc", "mcc"):
            print(f"\n  {metric.upper()}")
            print(f"    {'n_train':>8s}" + "".join(f"{s[:11]:>13s}" for s in scopes))
            for n in list(SIZES) + ["full"]:
                if (n, "pooled") not in acc:
                    continue
                print(f"    {str(n):>8s}" + "".join(
                    f"{np.mean(acc[(n, s)][metric]):13.4f}" if (n, s) in acc
                    else f"{'-':>13s}" for s in scopes))
            print(f"    {'-> n for':>8s}" + "".join(f"{s[:11]:>13s}" for s in scopes))
            for frac, lab in ((0.95, "95%"), (0.99, "99%")):
                row = ""
                for s in scopes:
                    full = np.mean(acc[("full", s)][metric])
                    vals = [np.mean(acc[(n, s)][metric]) for n in SIZES if (n, s) in acc]
                    szs = [n for n in SIZES if (n, s) in acc]
                    c = crossing(szs, vals, full, frac)
                    row += f"{'>6400' if np.isnan(c) else f'{c:.0f}':>13s}"
                    res[f"{metric}|{s}|{lab}"] = None if np.isnan(c) else round(c)
                print(f"    {lab:>8s}{row}")
        out[run] = {"sizes": list(SIZES), "seeds": args.seeds,
                    "curve": {f"{n}|{s}|{m}": round(float(np.mean(v[m])), 4)
                              for (n, s), v in acc.items() for m in v},
                    "n_for_fraction": res}
        write_json(dest, out)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
