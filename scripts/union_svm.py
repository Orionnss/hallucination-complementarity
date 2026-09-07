"""Does an SVM — linear, RBF, polynomial or sigmoid — beat logistic regression here?

Logistic regression has already outscored richer features, more PCA components and a
nine-configuration MLP sweep. An SVM is worth trying anyway because the RBF kernel has a
different inductive bias from a feedforward net: local similarity rather than learned
hierarchical features.

`decision_function` is used rather than `predict_proba`. AUROC only needs a ranking, and
`probability=True` would fit an internal 5-fold Platt calibration per model, roughly
5x the cost for no benefit to the comparison.

Selection is nested, exactly as in union_mlp.py: block PCA on the training fold, an
inner split to choose the configuration, refit on the full training fold, then the
untouched test fold.

Usage: uv run python scripts/union_svm.py [--run main] [--seeds 0]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

from halluc.config import Config
from halluc.io import write_json
from halluc.pipeline.stage5_posthoc import _fast_mcc, _folds, load_seed
from union_mlp import BLOCKS, block_pca, load_blocks  # same feature harness

#: (kernel, C, gamma, degree)
# poly and sigmoid were dropped after a first attempt spent ~30 minutes without
# finishing a single seed: both converge slowly here and neither is a plausible winner
# given the MLP sweep improved monotonically toward linearity. What remains is the
# comparison that matters — does an RBF kernel beat a linear decision boundary?
SVM_CONFIGS = [
    ("linear", 0.01, "scale", 3),
    ("linear", 0.1, "scale", 3),
    ("linear", 1.0, "scale", 3),
    ("rbf", 1.0, "scale", 3),
    ("rbf", 10.0, "scale", 3),
    ("rbf", 1.0, 1e-3, 3),
    ("rbf", 10.0, 1e-4, 3),
]


def make_svm(cfg, seed):
    kernel, C, gamma, degree = cfg
    # A finite max_iter matters here: libsvm's default (-1) lets a badly conditioned
    # configuration spin indefinitely, which is how the first attempt stalled.
    return SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, max_iter=3_000_000,
               class_weight="balanced", cache_size=1000, random_state=seed)


def label(cfg) -> str:
    kernel, C, gamma, degree = cfg
    g = f" gamma={gamma}" if kernel != "linear" else ""
    d = f" deg={degree}" if kernel == "poly" else ""
    return f"svm_{kernel} C={C:g}{g}{d}"


def run_seed(cfg, seed, blocks, n_jobs):
    sd = load_seed(cfg, seed)
    y, groups = sd["y"], sd["groups"]
    strat = np.array([f"{d}_{v}" for d, v in zip(sd["dataset"], y)])
    oof = defaultdict(lambda: np.full(len(y), np.nan))
    picks = []

    for fold, (tr, te) in enumerate(_folds(sd, seed, cfg.n_folds)):
        t_fold = time.perf_counter()
        Xtr, Xte = block_pca(blocks, tr, te, 128, seed)
        print(f"  seed {seed} fold {fold + 1}/{cfg.n_folds}: pca done "
              f"({time.perf_counter() - t_fold:.0f}s), fitting {len(SVM_CONFIGS)} configs",
              flush=True)

        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

        def inner_score(c):
            m = make_svm(c, seed).fit(Xtr[i_tr], y[tr][i_tr])
            return float(roc_auc_score(y[tr][i_va], m.decision_function(Xtr[i_va]))), c

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(inner_score)(c) for c in SVM_CONFIGS
        )
        best = max(results, key=lambda r: r[0])[1]
        picks.append(label(best))

        oof["svm_tuned"][te] = make_svm(best, seed).fit(Xtr, y[tr]).decision_function(Xte)
        lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr])
        oof["logreg"][te] = lr.predict_proba(Xte)[:, 1]

        fitted = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(lambda c: make_svm(c, seed).fit(Xtr, y[tr]).decision_function(Xte))(c)
            for c in SVM_CONFIGS
        )
        for c, p in zip(SVM_CONFIGS, fitted):
            oof[label(c)][te] = p
        print(f"  seed {seed} fold {fold + 1}/{cfg.n_folds} done in "
              f"{(time.perf_counter() - t_fold) / 60:.1f} min (picked {label(best)})", flush=True)

    return y, oof, picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--saplma-layer", type=int, default=24)
    ap.add_argument("--n-jobs", type=int, default=9)
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    acc = defaultdict(lambda: defaultdict(list))
    picks = []
    for seed in args.seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        print(f"seed {seed}: loading blocks ...", flush=True)
        blocks = load_blocks(cfg, sd, args.saplma_layer)
        y, oof, chosen = run_seed(cfg, seed, blocks, args.n_jobs)
        picks += chosen
        for name, s in oof.items():
            acc[name]["auroc"].append(float(roc_auc_score(y, s)))
            grid = np.quantile(s, np.linspace(0.02, 0.98, 60))
            acc[name]["mcc_best"].append(max(_fast_mcc(y, (s >= t).astype(int)) for t in grid))
        print(f"  done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del blocks

    ms = lambda v: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4))
    write_json(Path(f"runs/{args.run}/stage5_posthoc/union_svm.json"),
               {"run": args.run, "seeds": args.seeds, "configs_selected_per_fold": picks,
                "results": {n: {k: {"mean": ms(v)[0], "std": ms(v)[1]} for k, v in d.items()}
                            for n, d in acc.items()}})

    print(f"\n=== {args.run} ===")
    print(f"{'variant':32s}{'AUROC':>16s}{'MCC@best':>11s}")
    for n, d in sorted(acc.items(), key=lambda kv: -np.mean(kv[1]["auroc"])):
        a, s = ms(d["auroc"])
        print(f"  {n:30s}{a:9.4f} ±{s:.4f}{ms(d['mcc_best'])[0]:11.4f}")
    print(f"\nconfigs picked by nested selection: {sorted(set(picks))}")


if __name__ == "__main__":
    main()
