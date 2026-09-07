"""How many principal components does SAPLMA's hidden state actually need?

PCA-128 was inherited from the union protocol, where it was a per-block budget chosen so
five blocks would fit a comparable width — never validated for SAPLMA on its own. Since
the study's central claim now rests on how strong a properly-configured single-method
baseline is, the width that baseline runs at should be measured rather than assumed.

Sweeps 8 -> 1024 components plus the no-PCA control (the raw standardised state, which is
what the published probe consumes). The control matters: if raw beats every truncation,
PCA is costing signal rather than denoising, and the "PCA+logreg" framing is wrong.

Protocol is stage 3's, identical to saplma_pcalr_metrics.py so the rows are comparable:
same folds, C tuned on the inner split by AUROC, threshold by MCC, refit on the full
training fold.

Implementation note: PCA components are ordered, so one fit at max(DIMS) contains every
smaller width as a prefix. Fitting per width would multiply cost for identical numbers.

Usage: uv run python scripts/saplma_pca_sweep.py --runs main
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}
C_GRID = (0.003, 0.03, 0.3, 3.0)
DIMS = (8, 16, 32, 64, 128, 256, 512, 1024)


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


def fit_eval(Xtr, Xte, ytr, i_tr, i_va, seed):
    best_c, best_a = C_GRID[0], -1.0
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(
            Xtr[i_tr], ytr[i_tr])
        auc = roc_auc_score(ytr[i_va], m.predict_proba(Xtr[i_va])[:, 1])
        if auc > best_a:
            best_a, best_c = float(auc), C
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(
        Xtr[i_tr], ytr[i_tr])
    thr, _ = best_threshold(ytr[i_va], m.predict_proba(Xtr[i_va])[:, 1])
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    s = m.predict_proba(Xte)[:, 1]
    return s, (s >= thr).astype(int), best_c


def run_one(run, seeds, with_raw):
    cfg = Config(); cfg.run_id = run
    acc = defaultdict(lambda: defaultdict(list))
    chosen = defaultdict(list)

    for seed in seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"]
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = load_saplma(cfg, ids, LAYER.get(run, 24))

        names = [f"pca{d}" for d in DIMS] + (["raw"] if with_raw else [])
        score = {v: np.full(len(y), np.nan) for v in names}
        pred = {v: np.full(len(y), -1, dtype=int) for v in names}

        for fold, (tr, te) in enumerate(_folds(sd, seed, cfg.n_folds)):
            s = StandardScaler().fit(X[tr])
            a, b = s.transform(X[tr]), s.transform(X[te])
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

            kmax = min(max(DIMS), a.shape[1], len(tr) - 1)
            p = PCA(n_components=kmax, svd_solver="randomized", random_state=seed).fit(a)
            Ptr, Pte = p.transform(a), p.transform(b)
            evr = np.cumsum(p.explained_variance_ratio_)

            for d in DIMS:
                dd = min(d, kmax)
                s_, pr_, C = fit_eval(Ptr[:, :dd], Pte[:, :dd], y[tr], i_tr, i_va, seed)
                score[f"pca{d}"][te], pred[f"pca{d}"][te] = s_, pr_
                chosen[f"pca{d}"].append(C)
                if fold == 0 and seed == seeds[0]:
                    acc[(f"pca{d}", "_evr")]["evr"].append(float(evr[dd - 1]))
            if with_raw:
                s_, pr_, C = fit_eval(a, b, y[tr], i_tr, i_va, seed)
                score["raw"][te], pred["raw"][te] = s_, pr_
                chosen["raw"].append(C)
            print(f"    fold {fold + 1}/{cfg.n_folds}", flush=True)

        for v in names:
            for scope in ["pooled"] + sorted(set(ds_arr)):
                msk = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                if len(np.unique(y[msk])) < 2:
                    continue
                d = acc[(v, scope)]
                d["mcc"].append(float(matthews_corrcoef(y[msk], pred[v][msk])))
                d["auroc"].append(float(roc_auc_score(y[msk], score[v][msk])))
                d["accuracy"].append(float(accuracy_score(y[msk], pred[v][msk])))
                d["balanced_accuracy"].append(
                    float(balanced_accuracy_score(y[msk], pred[v][msk])))
                d["f1"].append(float(f1_score(y[msk], pred[v][msk], zero_division=0)))
        print(f"  {run} seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del X

    return ({f"{v}|{s}": dict(d) for (v, s), d in acc.items()},
            {k: float(np.mean(v)) for k, v in chosen.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--raw", action="store_true",
                    help="also fit on the raw standardised state (no PCA) — slow but it is "
                         "the control that says whether PCA helps or only shrinks")
    args = ap.parse_args()

    dest = Path("runs/saplma_pca_sweep.json")
    out = json.loads(dest.read_text()) if dest.exists() else {}
    for run in args.runs:
        res, cs = run_one(run, args.seeds, args.raw)
        prev = out.get(run, {"raw_metrics": {}, "seeds": [], "mean_C": {}})
        for key, d in res.items():
            for m, v in d.items():
                prev["raw_metrics"].setdefault(key, {}).setdefault(m, []).extend(v)
        prev["seeds"] = sorted(set(prev["seeds"]) | set(args.seeds))
        prev["mean_C"].update(cs)
        out[run] = prev
    write_json(dest, out)

    scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
    names = [f"pca{d}" for d in DIMS] + (["raw"] if args.raw else [])
    for metric in ("mcc", "auroc"):
        print(f"\n{'=' * 88}\n  {metric.upper()}\n{'=' * 88}")
        for run in args.runs:
            r = out[run]["raw_metrics"]
            print(f"\n  {run}  [{len(out[run]['seeds'])} seeds]")
            print(f"    {'width':10s}" + "".join(f"{s[:11]:>13s}" for s in scopes)
                  + "   var explained")
            for v in names:
                if f"{v}|pooled" not in r:
                    continue
                e = r.get(f"{v}|_evr", {}).get("evr")
                print(f"    {v:10s}" + "".join(
                    f"{np.mean(r[f'{v}|{s}'][metric]):13.4f}" if f"{v}|{s}" in r
                    else f"{'-':>13s}" for s in scopes)
                    + (f"{np.mean(e):15.1%}" if e else f"{'-':>15s}"))
            best = max((np.mean(r[f"{v}|pooled"][metric]), v) for v in names
                       if f"{v}|pooled" in r)
            b128 = np.mean(r["pca128|pooled"][metric])
            print(f"    best: {best[1]} = {best[0]:.4f}   (pca128 = {b128:.4f}, "
                  f"delta {best[0] - b128:+.4f})")
    print("\nwrote runs/saplma_pca_sweep.json")


if __name__ == "__main__":
    main()
