"""Trace the PLS probe back to individual hidden-state dimensions.

PLS reaches 95% of the full probe's AUROC in four components, so the signal is a handful of
directions. This asks what those directions are made of: which of the 5,120 residual-stream
dimensions carry them, and whether the same ones carry them every time.

The projection and the classifier collapse into one linear map. PLS transforms
Z = (Xs - mean) @ x_rotations_, and the probe scores logit = Z @ coef, so

    logit = (Xs - mean) @ (x_rotations_ @ coef)

and w = x_rotations_ @ coef is an effective weight per original dimension. Xs is
standardised, so |w_i| is the change in log-odds per standard deviation of dimension i and
the entries are directly comparable — which raw PCA loadings are not.

"Important in 80% of scenarios" is read as stability across folds: a dimension counts if it
lands in the top-k of at least 80% of the 15 folds (3 seeds x 5 outer folds). Ranking within
each training fold and then asking how often the same dimensions reappear separates real
structure from per-split noise. A dimension selected on all the data would look important by
construction.

Three things are reported: how concentrated the weight is, which dimensions are stably
important, and how much a probe restricted to only those dimensions actually scores — the
last being the check that the selection means anything.

Usage: uv run python scripts/pls_traceback.py --run main
"""

from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14,
         # base generators, same depths as their instruct twins
         "llama3.2-3b-base": 14, "gemma3-12b-pt": 29}
C_GRID = (0.003, 0.03, 0.3, 3.0)
TOPK = (16, 64, 256)
STABLE_AT = 0.8


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
    ap.add_argument("--n-components", type=int, default=4)
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    counts = {k: Counter() for k in TOPK}
    mass, n_for_80, W = [], [], []
    acc = defaultdict(lambda: defaultdict(list))
    n_folds_total = 0

    for seed in args.seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"]
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = load_saplma(cfg, ids, LAYER.get(args.run, 24))
        D = X.shape[1]

        for tr, te in _folds(sd, seed, cfg.n_folds):
            sc_ = StandardScaler().fit(X[tr])
            A0 = sc_.transform(X[tr])
            pls = PLSRegression(n_components=args.n_components, scale=False).fit(
                A0, y[tr].astype(np.float64))
            Z = pls.transform(A0)
            lr = LogisticRegression(C=0.3, max_iter=3000,
                                    class_weight="balanced").fit(Z, y[tr])
            w = pls.x_rotations_ @ lr.coef_[0]           # [D] effective per-SD weight
            W.append(w)
            a = np.abs(w)
            order = np.argsort(-a)
            for k in TOPK:
                counts[k].update(order[:k].tolist())
            csum = np.cumsum(a[order]) / a.sum()
            mass.append(float(a[order][:64].sum() / a.sum()))
            n_for_80.append(int(np.searchsorted(csum, 0.80) + 1))
            n_folds_total += 1
        del X
        print(f"  seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    print(f"\n=== {args.run}: PLS-{args.n_components} traced to hidden dimensions "
          f"({n_folds_total} folds, D={D}) ===")
    print(f"  weight concentration: top 64 of {D} dims hold "
          f"{np.mean(mass):.1%} of total |w|")
    print(f"  dims needed for 80% of |w| mass: {np.mean(n_for_80):.0f} "
          f"({np.mean(n_for_80) / D:.1%} of the state)")

    print(f"\n  stability: dims appearing in a fold's top-k in >= {STABLE_AT:.0%} of folds")
    stable = {}
    for k in TOPK:
        st = [d for d, n in counts[k].items() if n >= STABLE_AT * n_folds_total]
        stable[k] = sorted(st)
        print(f"    top-{k:<4d} {len(st):4d} stable dims "
              f"({len(st) / k:.0%} of the slots), "
              f"{len(counts[k])} distinct dims ever selected")
    print(f"\n  the 12 most persistent dimensions (top-64 membership):")
    print("    " + "  ".join(f"d{d}:{n}/{n_folds_total}"
                             for d, n in counts[64].most_common(12)))

    # sign agreement: does a stable dimension push the same way every fold?
    Wm = np.vstack(W)
    for k in TOPK:
        if not stable[k]:
            continue
        sgn = np.sign(Wm[:, stable[k]])
        agree = np.abs(sgn.mean(axis=0))
        print(f"  mean |sign agreement| over top-{k} stable dims: {agree.mean():.2f}"
              f"   (1.0 = always the same direction)")

    # does a probe restricted to the stable dims work? selection is fold-honest above,
    # so this is re-fitted per fold using only that fold's own stable set.
    for seed in args.seeds[:1]:
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"]
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = load_saplma(cfg, ids, LAYER.get(args.run, 24))
        for k in TOPK:
            s_all = np.full(len(y), np.nan); p_all = np.full(len(y), -1, dtype=int)
            for tr, te in _folds(sd, seed, cfg.n_folds):
                sc_ = StandardScaler().fit(X[tr])
                A0 = sc_.transform(X[tr])
                pls = PLSRegression(n_components=args.n_components, scale=False).fit(
                    A0, y[tr].astype(np.float64))
                lr = LogisticRegression(C=0.3, max_iter=3000,
                                        class_weight="balanced").fit(
                    pls.transform(A0), y[tr])
                cols = np.argsort(-np.abs(pls.x_rotations_ @ lr.coef_[0]))[:k]
                inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
                i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
                a, b = fit_eval(A0[:, cols], sc_.transform(X[te])[:, cols],
                                y, tr, i_tr, i_va, seed)
                s_all[te], p_all[te] = a, b
            acc[f"top{k}_dims"]["auroc"].append(float(roc_auc_score(y, s_all)))
            acc[f"top{k}_dims"]["mcc"].append(float(matthews_corrcoef(y, p_all)))
        del X

    print(f"\n  probe on the top-k raw dimensions only (seed {args.seeds[0]}, "
          f"selection refitted per fold)")
    for k in TOPK:
        v = acc[f"top{k}_dims"]
        print(f"    {k:4d} dims: AUROC {np.mean(v['auroc']):.4f}  MCC {np.mean(v['mcc']):.4f}")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/pls_traceback.json"),
               {"n_components": args.n_components, "n_folds": n_folds_total, "D": int(D),
                "dims_for_80pct_mass": float(np.mean(n_for_80)),
                "top64_mass_share": float(np.mean(mass)),
                "stable_dims": {str(k): stable[k] for k in TOPK},
                "top64_frequency": {str(d): n for d, n in counts[64].most_common(64)},
                "restricted_probe": {k: {m: round(float(np.mean(v[m])), 4) for m in v}
                                     for k, v in acc.items()}})
    print(f"\nwrote runs/{args.run}/stage5_posthoc/pls_traceback.json")


if __name__ == "__main__":
    main()
