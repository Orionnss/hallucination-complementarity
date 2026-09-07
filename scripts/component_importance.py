"""Which principal components does the probe actually use?

Three measurements, because each answers a different question and the naive one is wrong:

  |coef| x std   the correct importance for a linear model: change in log-odds per
                 standard deviation of that component. PCA.transform does not rescale its
                 output, so component variance falls with index by orders of magnitude and
                 raw |coef| is not comparable across components. This is the headline.

  univariate AUROC  how well each component separates the classes on its own, ignoring
                    the model. A component can matter to the fit while being useless
                    alone (suppressor variables) or vice versa, so the two disagree in an
                    informative way.

  stability      cosine similarity between the same-index component across folds. PCA is
                 refitted per fold, so "component 7" is only a meaningful object if fold
                 1's seventh direction is fold 2's seventh direction. Without this the
                 other two numbers are averages over incomparable things — sign is
                 arbitrary in PCA, so absolute cosine is used.

The question behind the question is whether the hallucination signal lives in the
high-variance directions. If it does not, PCA is an unsupervised projection being asked to
do a supervised job, and a supervised reduction would reach the same accuracy in fewer
dimensions. The width sweep already hints at this: 8 components reach MCC 0.484 against
0.552 at 128, so the signal is not concentrated at the top.

Usage: uv run python scripts/component_importance.py --run main
"""

from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}
C_GRID = (0.003, 0.03, 0.3, 3.0)
DIM = 128


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    imp, uni, evr = defaultdict(list), defaultdict(list), defaultdict(list)
    stab = defaultdict(list)

    for seed in args.seeds:
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"]
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = load_saplma(cfg, ids, LAYER.get(args.run, 24))

        prev_comp = None
        for tr, te in _folds(sd, seed, cfg.n_folds):
            sc = StandardScaler().fit(X[tr])
            A = sc.transform(X[tr])
            p = PCA(n_components=DIM, svd_solver="randomized", random_state=seed).fit(A)
            Z = p.transform(A)

            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
            best_c, best_a = C_GRID[0], -1.0
            for C in C_GRID:
                m = LogisticRegression(C=C, max_iter=3000,
                                       class_weight="balanced").fit(Z[i_tr], y[tr][i_tr])
                a = roc_auc_score(y[tr][i_va], m.predict_proba(Z[i_va])[:, 1])
                if a > best_a:
                    best_a, best_c = float(a), C
            m = LogisticRegression(C=best_c, max_iter=3000,
                                   class_weight="balanced").fit(Z, y[tr])

            sd_j = Z.std(axis=0)
            for j in range(DIM):
                imp[j].append(float(abs(m.coef_[0, j]) * sd_j[j]))
                uni[j].append(float(roc_auc_score(y[tr], Z[:, j])))
                evr[j].append(float(p.explained_variance_ratio_[j]))
            if prev_comp is not None:
                for j in range(DIM):
                    stab[j].append(float(abs(np.dot(prev_comp[j], p.components_[j]))))
            prev_comp = p.components_
        del X
        print(f"  seed {seed} done", flush=True)

    I = np.array([np.mean(imp[j]) for j in range(DIM)])
    U = np.array([abs(np.mean(uni[j]) - 0.5) + 0.5 for j in range(DIM)])
    V = np.array([np.mean(evr[j]) for j in range(DIM)])
    S = np.array([np.mean(stab[j]) if stab[j] else np.nan for j in range(DIM)])
    order = np.argsort(-I)

    print(f"\n=== {args.run}: top 15 components by |coef| x std ===")
    print(f"  {'rank':>5s}{'comp':>6s}{'importance':>12s}{'uni AUROC':>11s}"
          f"{'var %':>8s}{'stability':>11s}")
    for r, j in enumerate(order[:15], 1):
        print(f"  {r:5d}{j:6d}{I[j]:12.4f}{U[j]:11.4f}{V[j]:8.2%}{S[j]:11.3f}")

    tot = I.sum()
    print(f"\n  share of total importance held by the top-k components")
    for k in (1, 4, 8, 16, 32, 64, 128):
        print(f"    top {k:3d}: {np.sort(I)[::-1][:k].sum() / tot:6.1%}"
              f"   (first {k} by variance: {I[:k].sum() / tot:6.1%})")

    rho_v = spearmanr(np.arange(DIM), I).statistic
    rho_u = spearmanr(U, I).statistic
    print(f"\n  Spearman(component index, importance) = {rho_v:+.3f}"
          f"   -- negative means high-variance components matter more")
    print(f"  Spearman(univariate AUROC, importance) = {rho_u:+.3f}")
    print(f"  mean cross-fold stability: top-16 {np.nanmean(S[:16]):.3f}, "
          f"all {np.nanmean(S):.3f}")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/component_importance.json"),
               {"importance": [round(float(x), 5) for x in I],
                "univariate_auroc": [round(float(np.mean(uni[j])), 4) for j in range(DIM)],
                "explained_variance_ratio": [round(float(x), 6) for x in V],
                "cross_fold_stability": [round(float(x), 4) for x in S],
                "spearman_index_vs_importance": round(float(rho_v), 4)})
    print(f"\nwrote runs/{args.run}/stage5_posthoc/component_importance.json")


if __name__ == "__main__":
    main()
