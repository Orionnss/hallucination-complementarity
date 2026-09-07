"""Approximate the RBF kernel explicitly, then fit an ordinary linear model.

Exact RBF-SVC beat logistic regression on the union features (AUROC 0.868 vs 0.856).
That changed two things at once — the kernel *and* the hinge loss — so this isolates the
kernel: Nystroem (or random Fourier features) maps the data into an explicit
approximate-RBF space, and a plain logistic regression is fitted there.

If this recovers the SVC's gain, the kernel is doing the work and we also get back
calibrated probabilities, which `SVC.decision_function` does not provide and which the
threshold tuning depends on. If it does not, the margin/hinge loss is the active
ingredient.

gamma is expressed as a multiple of sklearn's `gamma='scale'` (1 / (n_features * X.var()))
computed on the training fold, so the settings are comparable to what SVC received.

Folds are the same ones stage 3 used, so numbers line up with union_svm.py directly.

Usage: uv run python scripts/union_nystroem.py [--run main] [--seeds 0 1 2 3 4]
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
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from halluc.config import Config
from halluc.io import write_json
from halluc.pipeline.stage5_posthoc import _fast_mcc, _folds, load_seed
from union_mlp import block_pca, load_blocks

#: (mapper, n_components, gamma multiple of 'scale', C)
CONFIGS = [
    ("nystroem", 512, 1.0, 0.1),
    ("nystroem", 512, 1.0, 1.0),
    ("nystroem", 1024, 1.0, 0.1),
    ("nystroem", 1024, 1.0, 1.0),
    ("nystroem", 1024, 0.5, 1.0),
    ("nystroem", 1024, 2.0, 1.0),
    ("nystroem", 2048, 1.0, 1.0),
    ("rff", 1024, 1.0, 1.0),
    ("rff", 2048, 1.0, 1.0),
]


def gamma_scale(X: np.ndarray) -> float:
    """sklearn's gamma='scale', computed explicitly so it can be scaled."""
    var = X.var()
    return 1.0 / (X.shape[1] * var) if var > 0 else 1.0


def build(cfg, X_fit, seed):
    mapper, n_comp, g_mult, C = cfg
    gamma = g_mult * gamma_scale(X_fit)
    n_comp = min(n_comp, len(X_fit))
    if mapper == "nystroem":
        m = Nystroem(kernel="rbf", gamma=gamma, n_components=n_comp, random_state=seed)
    else:
        m = RBFSampler(gamma=gamma, n_components=n_comp, random_state=seed)
    clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
    return m, clf


def label(cfg) -> str:
    mapper, n_comp, g_mult, C = cfg
    return f"{mapper}_d{n_comp}_g{g_mult:g}x_C{C:g}"


def fit_predict(cfg, Xtr, ytr, Xte, seed):
    m, clf = build(cfg, Xtr, seed)
    Ztr = m.fit_transform(Xtr)
    clf.fit(Ztr, ytr)
    return clf.predict_proba(m.transform(Xte))[:, 1]


def run_seed(cfg, seed, blocks, n_jobs):
    sd = load_seed(cfg, seed)
    y, groups = sd["y"], sd["groups"]
    strat = np.array([f"{d}_{v}" for d, v in zip(sd["dataset"], y)])
    oof = defaultdict(lambda: np.full(len(y), np.nan))
    picks = []

    for fold, (tr, te) in enumerate(_folds(sd, seed, cfg.n_folds)):
        t0 = time.perf_counter()
        Xtr, Xte = block_pca(blocks, tr, te, 128, seed)

        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

        def inner_score(c):
            p = fit_predict(c, Xtr[i_tr], y[tr][i_tr], Xtr[i_va], seed)
            return float(roc_auc_score(y[tr][i_va], p)), c

        res = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(inner_score)(c) for c in CONFIGS
        )
        best = max(res, key=lambda r: r[0])[1]
        picks.append(label(best))
        oof["kernel_tuned"][te] = fit_predict(best, Xtr, y[tr], Xte, seed)

        lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr])
        oof["logreg_linear"][te] = lr.predict_proba(Xte)[:, 1]

        fitted = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(fit_predict)(c, Xtr, y[tr], Xte, seed) for c in CONFIGS
        )
        for c, p in zip(CONFIGS, fitted):
            oof[label(c)][te] = p
        print(f"  seed {seed} fold {fold + 1}/{cfg.n_folds} done in "
              f"{(time.perf_counter() - t0) / 60:.1f} min (picked {label(best)})", flush=True)

    return y, oof, picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
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
            # Calibration, which SVC's decision_function cannot give.
            from sklearn.metrics import brier_score_loss
            acc[name]["brier"].append(float(brier_score_loss(y, np.clip(s, 0, 1))))
        print(f"  seed {seed} done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del blocks

    ms = lambda v: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4))
    write_json(Path(f"runs/{args.run}/stage5_posthoc/union_nystroem.json"),
               {"run": args.run, "seeds": args.seeds, "configs_selected_per_fold": picks,
                "results": {n: {k: {"mean": ms(v)[0], "std": ms(v)[1]} for k, v in d.items()}
                            for n, d in acc.items()}})

    print(f"\n=== {args.run} ({len(args.seeds)} seeds) ===")
    print(f"{'variant':30s}{'AUROC':>16s}{'MCC@best':>11s}{'Brier':>9s}")
    for n, d in sorted(acc.items(), key=lambda kv: -np.mean(kv[1]["auroc"])):
        a, s = ms(d["auroc"])
        print(f"  {n:28s}{a:9.4f} ±{s:.4f}{ms(d['mcc_best'])[0]:11.4f}{ms(d['brier'])[0]:9.4f}")
    print(f"\nconfigs picked: {sorted(set(picks))}")


if __name__ == "__main__":
    main()
