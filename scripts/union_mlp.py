"""Does an MLP extract more from the union features than logistic regression?

An earlier single configuration ((256,128), default alpha) scored below logistic
regression, but one untuned network is not a fair test — MLPs on ~6k rows are dominated
by the regularisation setting. This sweeps architecture, L2 (alpha) and learning rate.

Selection is nested: block PCA is fit on the training fold, the reduced training rows
are split again, the configuration is chosen on that inner split, then refit on the full
training fold and applied to the untouched test fold. So the reported "mlp (tuned)"
number is comparable to logreg rather than a best-of-sweep.

Per-configuration out-of-fold scores are also reported, to show the shape of the
sensitivity rather than only the winner.

Usage: uv run python scripts/union_mlp.py [--run main] [--seeds 0 1]
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _fast_mcc, _folds, load_seed

BLOCKS = ["lapeigvals", "attn_baseline", "saplma", "svd_baseline", "icr"]

#: architecture, L2 penalty, initial learning rate
MLP_CONFIGS = [
    ((256, 128), 1e-4, 1e-3),   # the earlier default that underperformed
    ((256, 128), 1e-2, 1e-3),
    ((256, 128), 1.0, 1e-3),
    ((256, 128), 10.0, 1e-3),
    ((128,), 1e-2, 1e-3),
    ((128,), 1.0, 1e-3),
    ((64,), 1.0, 1e-3),
    ((512, 256), 1.0, 1e-3),
    ((256, 128), 1.0, 3e-4),
]


def make_mlp(cfg, seed):
    hidden, alpha, lr = cfg
    return MLPClassifier(
        hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=lr,
        max_iter=800, early_stopping=True, n_iter_no_change=25,
        validation_fraction=0.12, random_state=seed,
    )


def load_blocks(cfg, sd, saplma_layer):
    ids = list(sd["item_ids"])
    pos = {i: k for k, i in enumerate(ids)}
    out = {}
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        d = cfg.stage_dir("stage1_extract", ds)
        rows = [pos[i] for i in sub]
        for b in BLOCKS:
            arr = load_features(d, b, sub)
            if b == "saplma":
                arr = arr[:, min(saplma_layer, arr.shape[1] - 1), :]
            arr = arr.reshape(len(sub), -1).astype(np.float32)
            if b not in out:
                out[b] = np.zeros((len(ids), arr.shape[1]), np.float32)
            out[b][rows] = arr
    return out


def block_pca(blocks, tr, te, n_comp, seed):
    parts_tr, parts_te = [], []
    for b in BLOCKS:
        s = StandardScaler().fit(blocks[b][tr])
        Xtr, Xte = s.transform(blocks[b][tr]), s.transform(blocks[b][te])
        k = min(n_comp, Xtr.shape[1], len(tr) - 1)
        if k < Xtr.shape[1]:
            p = PCA(n_components=k, random_state=seed).fit(Xtr)
            Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        parts_tr.append(Xtr); parts_te.append(Xte)
    return np.hstack(parts_tr), np.hstack(parts_te)


def run_seed(cfg, seed, blocks, n_jobs):
    sd = load_seed(cfg, seed)
    y, groups = sd["y"], sd["groups"]
    strat = np.array([f"{d}_{v}" for d, v in zip(sd["dataset"], y)])
    oof = defaultdict(lambda: np.full(len(y), np.nan))
    chosen = []

    for tr, te in _folds(sd, seed, cfg.n_folds):
        Xtr, Xte = block_pca(blocks, tr, te, 128, seed)

        # Inner split of the training fold only — the test fold is never touched.
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

        def score(c):
            m = make_mlp(c, seed).fit(Xtr[i_tr], y[tr][i_tr])
            return float(roc_auc_score(y[tr][i_va], m.predict_proba(Xtr[i_va])[:, 1])), c

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(score)(c) for c in MLP_CONFIGS
        )
        best = max(results, key=lambda r: r[0])[1]
        chosen.append(str(best))

        m = make_mlp(best, seed).fit(Xtr, y[tr])
        oof["mlp_tuned"][te] = m.predict_proba(Xte)[:, 1]

        lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr])
        oof["logreg"][te] = lr.predict_proba(Xte)[:, 1]

        # Every configuration on its own, to show sensitivity rather than only the winner.
        fitted = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(lambda c: make_mlp(c, seed).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1])(c)
            for c in MLP_CONFIGS
        )
        for c, p in zip(MLP_CONFIGS, fitted):
            oof[f"mlp{c[0]}_a{c[1]:g}_lr{c[2]:g}"][te] = p

    return y, oof, chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
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
    out = {"run": args.run, "seeds": args.seeds,
           "configs_selected_per_fold": picks,
           "results": {n: {k: {"mean": ms(v)[0], "std": ms(v)[1]} for k, v in d.items()}
                       for n, d in acc.items()}}
    write_json(Path(f"runs/{args.run}/stage5_posthoc/union_mlp.json"), out)

    print(f"\n=== {args.run} ===")
    print(f"{'variant':30s}{'AUROC':>16s}{'MCC@best':>11s}")
    for n, d in sorted(acc.items(), key=lambda kv: -np.mean(kv[1]["auroc"])):
        a, s = ms(d["auroc"])
        print(f"  {n:28s}{a:9.4f} ±{s:.4f}{ms(d['mcc_best'])[0]:11.4f}")
    print(f"\nconfigs picked by nested selection: {sorted(set(picks))}")


if __name__ == "__main__":
    main()
