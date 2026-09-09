"""Full metric breakdown for SAPLMA's features under PCA+logreg (or PCA+RBF-SVM).

The headline comparison in this study — union_equal beating saplma — confounds two
changes at once: more feature blocks AND a different classifier. The union is scored with
standardise / PCA 128 per block / logistic regression; SAPLMA is scored with the MLP from
the original paper. Pooled numbers already showed the classifier alone accounts for most
of the gap. This produces the same PCA+logreg treatment of SAPLMA's features across every
metric and every dataset, so the tables can compare like with like throughout.

Protocol is stage 3's exactly, so the rows drop straight into full_metrics.json's tables:
nested CV with the same folds, C chosen on the inner split by AUROC, decision threshold
chosen on the inner split by MCC, final model refitted on the whole training fold.

Usage: uv run python scripts/saplma_pcalr_metrics.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14,
         # base generators: same depths as their instruct twins, so the
         # instruct/base comparison varies the model and not the probe layer
         "llama3.2-3b-base": 14, "gemma3-12b-pt": 29}
C_GRID = (0.003, 0.03, 0.3, 3.0)
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b",
        "llama3.2-3b-base", "gemma3-12b-pt"]


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


def run_one(run: str, seeds: list[int], rbf: bool = False) -> dict:
    cfg = Config(); cfg.run_id = run
    layer = LAYER.get(run, 24)
    acc = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"]
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = load_saplma(cfg, ids, layer)

        score = np.full(len(y), np.nan)
        pred = np.full(len(y), -1, dtype=int)
        for tr, te in _folds(sd, seed, cfg.n_folds):
            s = StandardScaler().fit(X[tr])
            a, b = s.transform(X[tr]), s.transform(X[te])
            k = min(128, a.shape[1], len(tr) - 1)
            if k < a.shape[1]:
                p = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(a)
                a, b = p.transform(a), p.transform(b)

            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
            # The RBF arm uses the SAME grid, folds and threshold rule as the linear
            # one, so any difference is the kernel and not the tuning budget.
            grid = (0.1, 1.0, 10.0) if rbf else C_GRID
            def fit(C, X, yy):
                return (SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced",
                            max_iter=2_000_000, random_state=seed) if rbf
                        else LogisticRegression(C=C, max_iter=3000,
                                                class_weight="balanced")).fit(X, yy)
            def sc_of(m, X):
                return m.decision_function(X) if rbf else m.predict_proba(X)[:, 1]

            best_c, best_a = grid[0], -1.0
            for C in grid:
                m = fit(C, a[i_tr], y[tr][i_tr])
                auc = roc_auc_score(y[tr][i_va], sc_of(m, a[i_va]))
                if auc > best_a:
                    best_a, best_c = float(auc), C
            m = fit(best_c, a[i_tr], y[tr][i_tr])
            thr, _ = best_threshold(y[tr][i_va], sc_of(m, a[i_va]))
            m = fit(best_c, a, y[tr])
            score[te] = sc_of(m, b)
            pred[te] = (score[te] >= thr).astype(int)

        for scope in ["pooled"] + sorted(set(ds_arr)):
            msk = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
            if len(np.unique(y[msk])) < 2:
                continue
            d = acc[scope]
            d["accuracy"].append(float(accuracy_score(y[msk], pred[msk])))
            d["balanced_accuracy"].append(float(balanced_accuracy_score(y[msk], pred[msk])))
            d["mcc"].append(float(matthews_corrcoef(y[msk], pred[msk])))
            d["f1"].append(float(f1_score(y[msk], pred[msk], zero_division=0)))
            d["auroc"].append(float(roc_auc_score(y[msk], score[msk])))
        print(f"  {run} seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del X

    return {sc: {m: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                 for m, v in d.items()} for sc, d in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--rbf", action="store_true",
                    help="RBF-SVM instead of logistic regression, same protocol")
    args = ap.parse_args()

    # Merge into any existing results: background tasks get reaped on this host, so the
    # four models are run as separate foreground invocations.
    dest = Path("runs/saplma_pcarbf_metrics.json" if args.rbf
                else "runs/saplma_pcalr_metrics.json")
    out = json.loads(dest.read_text()) if dest.exists() else {}
    for run in args.runs:
        out[run] = run_one(run, args.seeds, args.rbf)

    write_json(dest, out)
    args.runs = [r for r in RUNS if r in out]
    scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
    for metric in ("mcc", "auroc", "accuracy", "balanced_accuracy", "f1"):
        print(f"\n### {metric.upper()} — saplma under PCA+logreg")
        print(f"  {'run':14s}" + "".join(f"{s[:11]:>15s}" for s in scopes))
        for run in args.runs:
            print(f"  {run:14s}" + "".join(
                f"{out[run][s][metric]['mean']:9.4f}±{out[run][s][metric]['std']:.3f}"
                if s in out[run] else f"{'-':>15s}" for s in scopes))
    print("\nwrote runs/saplma_pcalr_metrics.json")


if __name__ == "__main__":
    main()
