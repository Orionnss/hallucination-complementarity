"""Every non-CHARM method under the same five readers: PCA-128/256, PLS-4/8, published MLP.

Three separate runs have each covered part of this grid and none has covered it all:

  fair_comparison      PCA {64,128} + logreg, and an MLP arm, for all six methods -- but
                       the MLP arm reads PCA-128, not the raw state the published probe
                       consumes, and PCA-256 is outside its space
  supervised_reduction PLS 2-64, but only ever on SAPLMA (`load_saplma` is hardcoded)
  saplma_pca_sweep     PCA 8-1024 including 256, again SAPLMA only

So "does PLS help the other methods too?" and "does the published-style MLP change the
ordering?" are both unanswered. This closes the grid: five readers x six methods, on
stage 3's folds, so the rows drop straight into the existing tables.

  pca128 / pca256   one PCA fitted per fold at 256; the 128 row is a prefix of it, so the
                    two differ only in how many components the probe sees
  pls4 / pls8       one PLS fitted per fold at 8, likewise prefixed. PLS consumes the
                    label, so it is fitted strictly inside the training fold
  mlp_published     StandardScaler -> MLPClassifier(256, 128, 64) on the RAW features,
                    max_iter 600, early stopping, sklearn's default L2 -- byte-identical
                    to detectors.py's SaplmaDetector, applied to every method's block

Low-dimensional blocks make two of these rows the same measurement: icr is 40 dims,
svd_baseline 41 and logprob 14, so PCA at 128 and at 256 both clamp to the feature count
and no projection happens. Those rows are reported as `=pca128` rather than repeated, and
the effective width is recorded per method.

C and the decision threshold are chosen on the inner split, as everywhere else -- AUROC
for C, MCC for the threshold. The MLP arm has nothing to tune: the published
configuration is the whole point of it.

Usage: uv run python scripts/reduction_sweep.py --run main --seeds 0 1 2
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

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fair_comparison import LAYER, load_block  # identical block loading, incl. logprob

METHODS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline", "logprob"]
C_GRID = (0.003, 0.03, 0.3, 3.0)
PCA_DIMS = (128, 256)
PLS_DIMS = (4, 8)
VARIANTS = ["pca128", "pca256", "pls4", "pls8", "mlp_published"]


def fit_logreg(A, B, y_tr, i_tr, i_va, seed):
    """Stage 3's rule: C by AUROC on the inner split, threshold by MCC, refit on all."""
    best_c, best_a = C_GRID[0], -1.0
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(
            A[i_tr], y_tr[i_tr])
        a = roc_auc_score(y_tr[i_va], m.predict_proba(A[i_va])[:, 1])
        if a > best_a:
            best_a, best_c = float(a), C
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(
        A[i_tr], y_tr[i_tr])
    thr, _ = best_threshold(y_tr[i_va], m.predict_proba(A[i_va])[:, 1])
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(A, y_tr)
    s = m.predict_proba(B)[:, 1]
    return s, (s >= thr).astype(int)


def fit_published_mlp(A, B, y_tr, i_tr, i_va, seed):
    """detectors.py's SaplmaDetector, unchanged, on whatever block it is given."""
    def mk():
        return MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=600,
                             early_stopping=True, n_iter_no_change=20, random_state=seed)
    m = mk().fit(A[i_tr], y_tr[i_tr])
    thr, _ = best_threshold(y_tr[i_va], m.predict_proba(A[i_va])[:, 1])
    m = mk().fit(A, y_tr)
    s = m.predict_proba(B)[:, 1]
    return s, (s >= thr).astype(int)


def run_method(cfg, run, method, seeds):
    acc = defaultdict(lambda: defaultdict(list))
    widths = {}
    for seed in seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"].astype(str)
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = load_block(cfg, ids, method, LAYER.get(run, 24))
        widths["n_features"] = int(X.shape[1])

        score = {v: np.full(len(y), np.nan) for v in VARIANTS}
        pred = {v: np.full(len(y), -1, dtype=int) for v in VARIANTS}

        for tr, te in _folds(sd, seed, cfg.n_folds):
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
            s = StandardScaler().fit(X[tr])
            A0, B0 = s.transform(X[tr]), s.transform(X[te])
            y_tr = y[tr]

            # one PCA per fold at the widest setting; narrower rows are prefixes of it
            kmax = min(max(PCA_DIMS), A0.shape[1], len(tr) - 1)
            pca = PCA(n_components=kmax, svd_solver="randomized",
                      random_state=seed).fit(A0)
            Ap, Bp = pca.transform(A0), pca.transform(B0)
            for d in PCA_DIMS:
                k = min(d, kmax)
                widths[f"pca{d}"] = k
                sc, pr = fit_logreg(Ap[:, :k], Bp[:, :k], y_tr, i_tr, i_va, seed)
                score[f"pca{d}"][te], pred[f"pca{d}"][te] = sc, pr

            # PLS consumes the label, so it is fitted inside the training fold only
            kpls = min(max(PLS_DIMS), A0.shape[1])
            pls = PLSRegression(n_components=kpls, scale=False).fit(A0, y_tr)
            Al, Bl = pls.transform(A0), pls.transform(B0)
            for d in PLS_DIMS:
                k = min(d, kpls)
                widths[f"pls{d}"] = k
                sc, pr = fit_logreg(Al[:, :k], Bl[:, :k], y_tr, i_tr, i_va, seed)
                score[f"pls{d}"][te], pred[f"pls{d}"][te] = sc, pr

            widths["mlp_published"] = int(A0.shape[1])
            sc, pr = fit_published_mlp(A0, B0, y_tr, i_tr, i_va, seed)
            score["mlp_published"][te], pred["mlp_published"][te] = sc, pr

        for v in VARIANTS:
            for scope in ["pooled"] + sorted(set(ds_arr)):
                m = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                if len(np.unique(y[m])) < 2:
                    continue
                acc[f"{v}|{scope}"]["mcc"].append(float(matthews_corrcoef(y[m], pred[v][m])))
                acc[f"{v}|{scope}"]["auroc"].append(float(roc_auc_score(y[m], score[v][m])))
        print(f"    {run}/{method} seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min",
              flush=True)
        del X
    return {k: {m: [round(float(x), 5) for x in v] for m, v in d.items()}
            for k, d in acc.items()}, widths


SCOPES = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]


def report(run, res):
    for metric in ("mcc", "auroc"):
        print(f"\n### {metric.upper()}  —  {run}")
        print(f"  {'method':16s}{'reader':16s}{'dims':>6s}"
              + "".join(f"{s[:10]:>12s}" for s in SCOPES))
        for method in METHODS:
            if method not in res:
                continue
            raw, w = res[method]["raw"], res[method]["widths"]
            base = None
            for v in VARIANTS:
                if f"{v}|pooled" not in raw:
                    continue
                # PCA-256 on a 40-dim block is PCA-128 on a 40-dim block; say so rather
                # than print the same measurement twice as if it were two results.
                dup = v == "pca256" and w.get("pca256") == w.get("pca128")
                cells = "".join(
                    f"{np.mean(raw[f'{v}|{s}'][metric]):12.4f}" if f"{v}|{s}" in raw
                    else f"{'-':>12s}" for s in SCOPES)
                tag = "=pca128" if dup else f"{w.get(v, '?')}"
                print(f"  {method if v == VARIANTS[0] else '':16s}{v:16s}{tag:>6s}{cells}")
                if v == "pca128":
                    base = np.mean(raw[f"{v}|pooled"][metric])
            best = max((np.mean(raw[f'{v}|pooled'][metric]), v) for v in VARIANTS
                       if f"{v}|pooled" in raw)
            print(f"  {'':16s}{'-> best':16s}{'':>6s}{best[1]:>12s}"
                  f"{best[0] - base:+12.4f} vs pca128")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    args = ap.parse_args()

    if args.merge is not None:
        out = {}
        for f in args.merge:
            out.update(json.loads(Path(f).read_text()))
        write_json(Path("runs/reduction_sweep.json"), out)
        for run, res in out.items():
            report(run, res)
        print(f"\nmerged {len(args.merge)} files -> runs/reduction_sweep.json")
        return

    cfg = Config(); cfg.run_id = args.run
    res = {}
    for method in args.methods:
        try:
            raw, widths = run_method(cfg, args.run, method, args.seeds)
        except (FileNotFoundError, KeyError) as exc:
            print(f"    {args.run}/{method}: unavailable ({type(exc).__name__}), skipped",
                  flush=True)
            continue
        res[method] = {"raw": raw, "widths": widths, "seeds": args.seeds}
    if not res:
        print(f"nothing computed for {args.run}")
        return
    dest = Path(args.out or f"runs/reduction_sweep_parts/{args.run}.json")
    write_json(dest, {args.run: res})
    report(args.run, {**res})
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
