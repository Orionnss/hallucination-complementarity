"""Add PCA components one at a time, best-first, and watch AUROC climb.

The importance analysis ranked components by |coef| x std and found the useful ones are
not the high-variance ones: on Qwen3-14B the order is 5, 6, 0, 9, 2, while PCA's own order
is 0, 1, 2, 3, 4. This measures what that costs — how much AUROC the first k components
buy under each ordering.

The ranking is recomputed inside every training fold and never sees the test rows.
Selecting components by an importance score fitted on all the data would choose directions
because they happen to work on the items being scored, which inflates exactly the number
this is meant to report. Two references bound the result:

  by variance   PCA's own ordering, 0..k-1. What you get without looking at the label.
  by importance train-fold ranking, best-first. What a supervised selection reaches.
  pca128        the full-width baseline used throughout the study.

Usage: uv run python scripts/pca_greedy_sweep.py --run main
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
DIM = 128
KS = (1, 2, 3, 4, 5, 6, 8, 10, 16)


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


def fit_eval(A, B, y, tr, i_tr, i_va, seed):
    best_c, best_a = C_GRID[0], -1.0
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(
            A[i_tr], y[tr][i_tr])
        a = roc_auc_score(y[tr][i_va], m.predict_proba(A[i_va])[:, 1])
        if a > best_a:
            best_a, best_c = float(a), C
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(
        A[i_tr], y[tr][i_tr])
    thr, _ = best_threshold(y[tr][i_va], m.predict_proba(A[i_va])[:, 1])
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(A, y[tr])
    s = m.predict_proba(B)[:, 1]
    return s, (s >= thr).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    variants = [f"imp{k}" for k in KS] + [f"var{k}" for k in KS] + ["pca128"]
    acc = defaultdict(lambda: defaultdict(list))
    picked = defaultdict(int)

    for seed in args.seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"]
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = load_saplma(cfg, ids, LAYER.get(args.run, 24))

        sc = {v: np.full(len(y), np.nan) for v in variants}
        pr = {v: np.full(len(y), -1, dtype=int) for v in variants}

        for tr, te in _folds(sd, seed, cfg.n_folds):
            s = StandardScaler().fit(X[tr])
            p = PCA(n_components=DIM, svd_solver="randomized", random_state=seed).fit(
                s.transform(X[tr]))
            A, B = p.transform(s.transform(X[tr])), p.transform(s.transform(X[te]))
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

            # Rank on the training fold only. A ranking fitted on all rows would pick
            # components because they work on the very items being scored.
            rk = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(
                A, y[tr])
            imp = np.abs(rk.coef_[0]) * A.std(axis=0)
            order = np.argsort(-imp)
            for j in order[:5]:
                picked[int(j)] += 1

            for k in KS:
                cols = order[:k]
                a, b = fit_eval(A[:, cols], B[:, cols], y, tr, i_tr, i_va, seed)
                sc[f"imp{k}"][te], pr[f"imp{k}"][te] = a, b
                a, b = fit_eval(A[:, :k], B[:, :k], y, tr, i_tr, i_va, seed)
                sc[f"var{k}"][te], pr[f"var{k}"][te] = a, b
            a, b = fit_eval(A, B, y, tr, i_tr, i_va, seed)
            sc["pca128"][te], pr["pca128"][te] = a, b

        for v in variants:
            for scope in ["pooled"] + sorted(set(ds_arr)):
                m = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                if len(np.unique(y[m])) < 2:
                    continue
                acc[(v, scope)]["auroc"].append(float(roc_auc_score(y[m], sc[v][m])))
                acc[(v, scope)]["mcc"].append(float(matthews_corrcoef(y[m], pr[v][m])))
        print(f"  seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del X

    full_a = np.mean(acc[("pca128", "pooled")]["auroc"])
    full_m = np.mean(acc[("pca128", "pooled")]["mcc"])
    print(f"\n=== {args.run}: components added one at a time ({len(args.seeds)} seeds) ===")
    print(f"  {'k':>3s}{'AUROC best-first':>19s}{'% of pca128':>13s}"
          f"{'AUROC by variance':>20s}{'gap':>9s}{'MCC best-first':>17s}")
    for k in KS:
        ai = np.mean(acc[(f"imp{k}", "pooled")]["auroc"])
        av = np.mean(acc[(f"var{k}", "pooled")]["auroc"])
        mi = np.mean(acc[(f"imp{k}", "pooled")]["mcc"])
        print(f"  {k:3d}{ai:19.4f}{ai / full_a:13.1%}{av:20.4f}"
              f"{ai - av:+9.4f}{mi:17.4f}")
    print(f"  {'128':>3s}{full_a:19.4f}{1.0:13.1%}{full_a:20.4f}{0.0:+9.4f}{full_m:17.4f}")

    tot = sum(picked.values())
    top = sorted(picked.items(), key=lambda x: -x[1])[:10]
    print(f"\n  components most often in a fold's top 5 "
          f"({len(args.seeds) * cfg.n_folds} folds):")
    print("    " + "  ".join(f"c{j}:{n}" for j, n in top))

    write_json(Path(f"runs/{args.run}/stage5_posthoc/pca_greedy_sweep.json"),
               {f"{v}|{s}": {m: round(float(np.mean(d[m])), 4) for m in d}
                for (v, s), d in acc.items()} |
               {"_top5_frequency": {str(j): n for j, n in sorted(picked.items())}})
    print(f"\nwrote runs/{args.run}/stage5_posthoc/pca_greedy_sweep.json")


if __name__ == "__main__":
    main()
