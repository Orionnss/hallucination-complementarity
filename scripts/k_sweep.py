"""Does LapEigvals improve with more eigenvalues per head?

Stage 1 kept k=10; the re-extraction stored k=256, which contains k=64 and k=10 as
prefixes (the features are the top-k sorted descending). All three are therefore computed
on identical sequences — the k=10 slice reproduces the stored features bit-for-bit — so
differences here are attributable to k alone.

Two configurations per k, both under the stage-3 protocol (standardise, PCA 128 per
block, logistic regression with C and threshold tuned on an inner split):

  lapeigvals alone  - does the method itself benefit?
  union of 5 blocks - does the combination benefit?

Reported pooled and per dataset. The per-dataset split matters more than usual here:
outside CoQA over 96% of answers are shorter than 256 tokens, so the extra eigenvalues
are zero-padding and k=256 cannot differ from k=64 for any reason but noise. CoQA
(median T = 538) is the only slice where k=256 is genuinely supported.

Usage: uv run python scripts/k_sweep.py --run main --saplma-layer 24
"""

from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

OTHER_BLOCKS = ["attn_baseline", "saplma", "svd_baseline", "icr"]
KS = (10, 64, 256)
C_GRID = (0.003, 0.03, 0.3, 3.0)


def load_named(cfg, ids, name, layer=None, sub_dir=None):
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        d = cfg.stage_dir("stage1_extract", ds)
        if sub_dir:
            d = d / sub_dir
        arr = load_features(d, name, sub)
        if layer is not None:
            arr = arr[:, min(layer, arr.shape[1] - 1), :]
        arr = arr.reshape(len(sub), -1).astype(np.float32)
        if out is None:
            out = np.zeros((len(ids), arr.shape[1]), np.float32)
        for k, i in enumerate(sub):
            out[pos[i]] = arr[k]
    return out


def reduce(block, tr, te, seed, dim=128):
    s = StandardScaler().fit(block[tr])
    a, b = s.transform(block[tr]), s.transform(block[te])
    k = min(dim, a.shape[1], len(tr) - 1)
    if k < a.shape[1]:
        p = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(a)
        a, b = p.transform(a), p.transform(b)
    return a, b


def fit_eval(Xtr, Xte, y, tr, i_tr, i_va, seed):
    best_c, best_a = C_GRID[0], -1.0
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(
            Xtr[i_tr], y[tr][i_tr])
        a = roc_auc_score(y[tr][i_va], m.predict_proba(Xtr[i_va])[:, 1])
        if a > best_a:
            best_a, best_c = float(a), C
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(
        Xtr[i_tr], y[tr][i_tr])
    thr, _ = best_threshold(y[tr][i_va], m.predict_proba(Xtr[i_va])[:, 1])
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr])
    p = m.predict_proba(Xte)[:, 1]
    return p, (p >= thr).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--saplma-layer", type=int, required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    variants = [f"{c}_k{k}" for k in KS for c in ("lapeigvals", "union")]
    acc = defaultdict(lambda: defaultdict(list))

    for seed in args.seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        ids, y, groups = list(sd["item_ids"]), sd["y"], sd["groups"]
        ds_arr = sd["dataset"]
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        print(f"seed {seed}: loading blocks ...", flush=True)

        lap256 = load_named(cfg, ids, f"lapeigvals_k{256}", sub_dir="k256")
        n_head_layer = lap256.shape[1] // 256
        others = {b: load_named(cfg, ids, b,
                                layer=args.saplma_layer if b == "saplma" else None)
                  for b in OTHER_BLOCKS}
        # top-k sorted descending: k=64 and k=10 are prefixes of the k=256 block.
        lap = {k: lap256.reshape(len(ids), n_head_layer, 256)[:, :, :k]
                    .reshape(len(ids), -1) for k in KS}
        del lap256

        oof = {v: np.full(len(y), np.nan) for v in variants}
        prd = {v: np.full(len(y), -1, dtype=int) for v in variants}
        for fold, (tr, te) in enumerate(_folds(sd, seed, cfg.n_folds)):
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
            red_other = {b: reduce(others[b], tr, te, seed) for b in OTHER_BLOCKS}
            for k in KS:
                lt, le = reduce(lap[k], tr, te, seed)
                p, h = fit_eval(lt, le, y, tr, i_tr, i_va, seed)
                oof[f"lapeigvals_k{k}"][te], prd[f"lapeigvals_k{k}"][te] = p, h
                ut = np.hstack([lt] + [red_other[b][0] for b in OTHER_BLOCKS])
                ue = np.hstack([le] + [red_other[b][1] for b in OTHER_BLOCKS])
                p, h = fit_eval(ut, ue, y, tr, i_tr, i_va, seed)
                oof[f"union_k{k}"][te], prd[f"union_k{k}"][te] = p, h
            print(f"  fold {fold + 1}/{cfg.n_folds}", flush=True)

        for v in variants:
            for scope in ["pooled"] + sorted(set(ds_arr)):
                m = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                if len(np.unique(y[m])) < 2:
                    continue
                acc[(v, scope)]["auroc"].append(float(roc_auc_score(y[m], oof[v][m])))
                acc[(v, scope)]["mcc"].append(float(matthews_corrcoef(y[m], prd[v][m])))
        print(f"  seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del lap, others

    scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
    print(f"\n=== {args.run}: LapEigvals k sweep ===")
    for metric in ("auroc", "mcc"):
        print(f"\n  {metric.upper()}")
        print(f"    {'variant':18s}" + "".join(f"{s[:9]:>11s}" for s in scopes))
        for v in variants:
            row = "".join(f"{np.mean(acc[(v,s)][metric]):11.4f}" if (v, s) in acc
                          else f"{'-':>11s}" for s in scopes)
            print(f"    {v:18s}{row}")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/k_sweep.json"),
               {"run": args.run, "ks": list(KS), "seeds": args.seeds,
                "results": {f"{v}|{s}": {m: round(float(np.mean(d[m])), 4) for m in d}
                            for (v, s), d in acc.items()}})
    print(f"\nwrote runs/{args.run}/stage5_posthoc/k_sweep.json")


if __name__ == "__main__":
    main()
