"""Supervised reduction against PCA: does using the label to choose directions help?

PCA orders directions by variance, which is not what the probe needs. The component
importance analysis showed the point concretely: on Qwen3-14B the most useful direction is
component 5, holding 1.78% of the variance but reaching univariate AUROC 0.702, while
component 0 holds 10.65% and reaches 0.579. PCA had no way to know that, because it never
sees the label.

PLS orders by covariance with the label instead, so its first components should be the ones
PCA happens to place at rank 5. LDA goes further and collapses everything to the single
best separating direction — for two classes there is only one — which needs shrinkage here,
since the within-class covariance of 5,120 features from ~6,400 rows is badly conditioned.

The honest expectation is efficiency rather than accuracy. Logistic regression on 128 PCA
components can already find the discriminative direction *within that span*, so PLS should
mostly buy the same performance in far fewer dimensions. It wins on accuracy only if useful
signal lies outside the top-128 PC subspace — which the width sweep argues against, since
performance falls past 128 and the raw 5,120-dim control scores below every width from 32
up. This measures which it is.

Both reductions use the label, so both are fitted strictly inside the training fold. Fitting
before the split would leak labels into the representation itself — a worse failure than
any PCA leakage, and invisible in the resulting numbers.

Usage: uv run python scripts/supervised_reduction.py --run main
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

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
PLS_DIMS = (2, 4, 8, 16, 32, 64)
PCA_DIMS = (8, 16, 128)


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
    variants = ([f"pca{d}" for d in PCA_DIMS] + [f"pls{d}" for d in PLS_DIMS]
                + ["lda_shrunk"])
    acc = defaultdict(lambda: defaultdict(list))

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
            A0, B0 = s.transform(X[tr]), s.transform(X[te])
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

            # PCA reference: unsupervised, prefixes of one fit
            p = PCA(n_components=max(PCA_DIMS), svd_solver="randomized",
                    random_state=seed).fit(A0)
            Ap, Bp = p.transform(A0), p.transform(B0)
            for d in PCA_DIMS:
                a, b = fit_eval(Ap[:, :d], Bp[:, :d], y, tr, i_tr, i_va, seed)
                sc[f"pca{d}"][te], pr[f"pca{d}"][te] = a, b

            # PLS: fitted on the training fold only, since it consumes the label.
            # Components are nested, so one fit at the widest setting serves every d.
            pls = PLSRegression(n_components=max(PLS_DIMS), scale=False).fit(
                A0, y[tr].astype(np.float64))
            Al, Bl = pls.transform(A0), pls.transform(B0)
            for d in PLS_DIMS:
                a, b = fit_eval(Al[:, :d], Bl[:, :d], y, tr, i_tr, i_va, seed)
                sc[f"pls{d}"][te], pr[f"pls{d}"][te] = a, b

            # LDA with Ledoit-Wolf shrinkage: 5,120 features from ~6,400 rows leaves the
            # within-class covariance singular, so the plain solver is not usable here.
            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(
                A0, y[tr])
            a_s = lda.decision_function(A0)
            thr, _ = best_threshold(y[tr][i_va], a_s[i_va])
            sc["lda_shrunk"][te] = lda.decision_function(B0)
            pr["lda_shrunk"][te] = (sc["lda_shrunk"][te] >= thr).astype(int)

        for v in variants:
            for scope in ["pooled"] + sorted(set(ds_arr)):
                m = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                if len(np.unique(y[m])) < 2:
                    continue
                acc[(v, scope)]["mcc"].append(float(matthews_corrcoef(y[m], pr[v][m])))
                acc[(v, scope)]["auroc"].append(float(roc_auc_score(y[m], sc[v][m])))
        print(f"  seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del X

    scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
    for metric in ("mcc", "auroc"):
        print(f"\n### {metric.upper()}  ({len(args.seeds)} seeds)")
        print(f"  {'reduction':14s}{'dims':>6s}" + "".join(f"{s[:11]:>12s}" for s in scopes))
        for v in variants:
            d = (v.replace("pca", "").replace("pls", "") if v != "lda_shrunk" else "1")
            print(f"  {v:14s}{d:>6s}" + "".join(
                f"{np.mean(acc[(v, s)][metric]):12.4f}" if (v, s) in acc
                else f"{'-':>12s}" for s in scopes))
        b = np.mean(acc[("pca128", "pooled")][metric])
        best = max((np.mean(acc[(v, "pooled")][metric]), v) for v in variants)
        print(f"  -> best: {best[1]} = {best[0]:.4f}   vs pca128 {b:.4f} "
              f"({best[0] - b:+.4f})")

    # Per-seed values are kept alongside the means: PLS-8 and PCA-128 differ by under
    # .01 in most cells, and a mean cannot say whether that gap survives seed noise.
    write_json(Path(f"runs/{args.run}/stage5_posthoc/supervised_reduction.json"),
               {"mean": {f"{v}|{s}": {m: round(float(np.mean(d[m])), 4) for m in d}
                         for (v, s), d in acc.items()},
                "per_seed": {f"{v}|{s}": {m: [round(float(x), 5) for x in d[m]] for m in d}
                             for (v, s), d in acc.items()}})
    print(f"\nwrote runs/{args.run}/stage5_posthoc/supervised_reduction.json")


if __name__ == "__main__":
    main()
