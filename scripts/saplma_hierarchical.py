"""Hierarchical SAPLMA: k layers ending at l, one PCA per layer, concatenated.

SAPLMA reads a single layer. If truthfulness is encoded progressively rather than at one
depth, the layers feeding into l should carry signal that l alone has discarded. This
builds that: take layers l-k+1 ... l, fit a separate PCA per layer, concatenate the
projections, and fit one logistic regression over the result.

Two ways to grow k, and the difference between them is the whole experiment:

  per_layer   each layer keeps DIM components, so total width grows as k * DIM. More
              layers AND more capacity - a gain here is ambiguous.
  budget      each layer keeps DIM // k components, so total width is ~DIM regardless of
              k. Capacity is held constant, so a gain is attributable to the extra layers
              and not to the extra dimensions. This is the row that answers the question.

  joint       control: one PCA over the concatenated raw layers, same total width as
              `budget`. If per-layer PCA beats this, the per-layer structure matters; if
              they tie, "one PCA over everything" was sufficient and the hierarchy is
              decoration.

k=1 reproduces the existing single-layer PCA+logreg, so it is the built-in baseline and
any claim is a delta against it under an identical protocol.

Implementation note: PCA components are ordered, so a DIM-component fit contains every
smaller fit as a prefix. One PCA per layer per fold is therefore enough to serve every k
and both modes — refitting per k would multiply the cost for identical numbers.

Usage: uv run python scripts/saplma_hierarchical.py --runs main
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
KS = (1, 2, 3, 4, 6, 8)
DIM = 128


def load_layers(cfg, ids, layers):
    """[N, len(layers), H] for the requested absolute layer indices."""
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        a = load_features(cfg.stage_dir("stage1_extract", ds), "saplma", sub)
        a = a[:, layers, :].astype(np.float32)
        if out is None:
            out = np.zeros((len(ids), len(layers), a.shape[2]), np.float32)
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
    return s, (s >= thr).astype(int)


def run_one(run, seeds, joint):
    cfg = Config(); cfg.run_id = run
    l = LAYER.get(run, 24)
    kmax = max(KS)
    layers = list(range(l - kmax + 1, l + 1))          # oldest .. l, l is last
    acc = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"]
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = load_layers(cfg, ids, layers)

        variants = [f"k{k}_{m}" for k in KS for m in ("per_layer", "budget")]
        if joint:
            variants += [f"k{k}_joint" for k in KS]
        score = {v: np.full(len(y), np.nan) for v in variants}
        pred = {v: np.full(len(y), -1, dtype=int) for v in variants}

        for fold, (tr, te) in enumerate(_folds(sd, seed, cfg.n_folds)):
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

            # One DIM-component PCA per layer; every smaller k reuses a prefix of it.
            proj = []
            for j in range(len(layers)):
                s = StandardScaler().fit(X[tr, j])
                a, b = s.transform(X[tr, j]), s.transform(X[te, j])
                p = PCA(n_components=min(DIM, a.shape[1], len(tr) - 1),
                        svd_solver="randomized", random_state=seed).fit(a)
                proj.append((p.transform(a), p.transform(b)))

            for k in KS:
                sel = range(len(layers) - k, len(layers))   # the k layers ending at l
                for mode, d in (("per_layer", DIM), ("budget", max(4, DIM // k))):
                    A = np.hstack([proj[j][0][:, :d] for j in sel])
                    B = np.hstack([proj[j][1][:, :d] for j in sel])
                    s_, p_ = fit_eval(A, B, y[tr], i_tr, i_va, seed)
                    score[f"k{k}_{mode}"][te], pred[f"k{k}_{mode}"][te] = s_, p_

                if joint:
                    # One PCA over the concatenated raw layers, same total width as budget
                    cat_tr = X[tr][:, sel].reshape(len(tr), -1)
                    cat_te = X[te][:, sel].reshape(len(te), -1)
                    sc = StandardScaler().fit(cat_tr)
                    a, b = sc.transform(cat_tr), sc.transform(cat_te)
                    w = max(4, DIM // k) * k
                    p = PCA(n_components=min(w, a.shape[1], len(tr) - 1),
                            svd_solver="randomized", random_state=seed).fit(a)
                    s_, p_ = fit_eval(p.transform(a), p.transform(b), y[tr], i_tr, i_va, seed)
                    score[f"k{k}_joint"][te], pred[f"k{k}_joint"][te] = s_, p_
            print(f"    fold {fold + 1}/{cfg.n_folds}", flush=True)

        for v in variants:
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

    return {f"{v}|{s}": dict(d) for (v, s), d in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--joint", action="store_true",
                    help="also fit the single-PCA-over-concatenated-layers control")
    args = ap.parse_args()

    dest = Path("runs/saplma_hierarchical.json")
    out = json.loads(dest.read_text()) if dest.exists() else {}
    for run in args.runs:
        res = run_one(run, args.seeds, args.joint)
        prev = out.get(run, {"raw": {}, "seeds": []})
        for key, d in res.items():
            for m, v in d.items():
                prev["raw"].setdefault(key, {}).setdefault(m, []).extend(v)
        prev["seeds"] = sorted(set(prev["seeds"]) | set(args.seeds))
        out[run] = prev
    write_json(dest, out)

    scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
    modes = ["per_layer", "budget"] + (["joint"] if args.joint else [])
    for metric in ("mcc", "auroc"):
        print(f"\n{'=' * 84}\n  {metric.upper()}\n{'=' * 84}")
        for run in args.runs:
            r = out[run]["raw"]
            print(f"\n  {run}  [{len(out[run]['seeds'])} seeds]")
            print(f"    {'variant':18s}" + "".join(f"{s[:11]:>13s}" for s in scopes)
                  + "   width")
            for mode in modes:
                for k in KS:
                    key = f"k{k}_{mode}"
                    if f"{key}|pooled" not in r:
                        continue
                    w = (DIM * k if mode == "per_layer" else max(4, DIM // k) * k)
                    print(f"    {key:18s}" + "".join(
                        f"{np.mean(r[f'{key}|{s}'][metric]):13.4f}"
                        if f"{key}|{s}" in r else f"{'-':>13s}" for s in scopes)
                        + f"{w:8d}")
                print()
            base = np.mean(r["k1_budget|pooled"][metric])
            best = max(((np.mean(r[f'{k}|pooled'][metric]), k)
                        for k in [f"k{kk}_{m}" for kk in KS for m in modes]
                        if f"{k}|pooled" in r))
            print(f"    best pooled {metric}: {best[1]} = {best[0]:.4f} "
                  f"(k=1 baseline {base:.4f}, delta {best[0] - base:+.4f})")
    print("\nwrote runs/saplma_hierarchical.json")


if __name__ == "__main__":
    main()
