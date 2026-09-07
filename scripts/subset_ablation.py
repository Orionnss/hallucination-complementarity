"""Every subset of the five detectors, scored under the stage-3 protocol.

31 non-empty subsets of {saplma, lapeigvals, attn_baseline, icr, svd_baseline}, each
trained exactly the way `union_equal` is: standardise, PCA each block to 128 dims,
concatenate, logistic regression with C tuned on an inner split, threshold tuned on the
same inner split, evaluated on the untouched outer fold.

The whole sweep is affordable because per-block PCA does not depend on which *other*
blocks are present: all five reductions are fitted once per fold and the subsets simply
select among the reduced blocks. A fold therefore costs 5 PCA fits, not 31 x 5.

Answers: how performance grows with the number of methods combined, which specific
combinations do the work, and what each method contributes at the margin.

Usage: uv run python scripts/subset_ablation.py --run main [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import os
import sys

# Must precede the numpy import: BLAS reads these at load time. Without the cap, PCA in
# the parent process claims all cores, and several concurrent runs thrash the machine
# (measured: load 470 on 80 cores, fold times rising rather than falling).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _fast_mcc, _folds, load_seed

METHODS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline"]
C_GRID = (0.003, 0.03, 0.3, 3.0)
N_COMPONENTS = 128


def all_subsets():
    for k in range(1, len(METHODS) + 1):
        yield from combinations(METHODS, k)


def load_blocks(cfg, sd, saplma_layer):
    ids = list(sd["item_ids"])
    pos = {i: k for k, i in enumerate(ids)}
    out: dict[str, np.ndarray] = {}
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        d = cfg.stage_dir("stage1_extract", ds)
        rows = [pos[i] for i in sub]
        for b in METHODS:
            arr = load_features(d, b, sub)
            if b == "saplma":
                arr = arr[:, min(saplma_layer, arr.shape[1] - 1), :]
            arr = arr.reshape(len(sub), -1).astype(np.float32)
            if b not in out:
                out[b] = np.zeros((len(ids), arr.shape[1]), np.float32)
            out[b][rows] = arr
    return out


def reduce_blocks(blocks, tr, te, seed):
    """Fit each block's scaler+PCA on the training rows once; reuse for every subset."""
    red = {}
    for b in METHODS:
        s = StandardScaler().fit(blocks[b][tr])
        Xtr, Xte = s.transform(blocks[b][tr]), s.transform(blocks[b][te])
        k = min(N_COMPONENTS, Xtr.shape[1], len(tr) - 1)
        if k < Xtr.shape[1]:
            p = PCA(n_components=k, random_state=seed).fit(Xtr)
            Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        red[b] = (Xtr, Xte)
    return red


def score_subset(subset, red, y, tr, i_tr, i_va, seed):
    """Tune C and the threshold on the inner split, then evaluate on the outer fold."""
    Xtr = np.hstack([red[b][0] for b in subset])
    Xte = np.hstack([red[b][1] for b in subset])

    best_c, best_auc = C_GRID[0], -1.0
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(
            Xtr[i_tr], y[tr][i_tr]
        )
        p = m.predict_proba(Xtr[i_va])[:, 1]
        a = roc_auc_score(y[tr][i_va], p) if len(np.unique(y[tr][i_va])) > 1 else 0.5
        if a > best_auc:
            best_auc, best_c = float(a), C

    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(
        Xtr[i_tr], y[tr][i_tr]
    )
    thr, _ = best_threshold(y[tr][i_va], m.predict_proba(Xtr[i_va])[:, 1])

    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr])
    return subset, m.predict_proba(Xte)[:, 1], thr, best_c


def run(cfg, seed, blocks, n_jobs, exclude=()):
    sd = load_seed(cfg, seed)
    y, groups = sd["y"], sd["groups"]
    strat = np.array([f"{d}_{v}" for d, v in zip(sd["dataset"], y)])
    subsets = list(all_subsets())
    # Keep stage 3's exact fold partition and simply drop the excluded rows from each
    # train/test set. Re-deriving folds on the subset would change the grouping and make
    # these numbers incomparable with every other result in the study.
    keep = ~np.isin(sd["dataset"], list(exclude))
    oof = {s: np.full(len(y), np.nan) for s in subsets}
    preds = {s: np.full(len(y), -1, dtype=int) for s in subsets}

    for fold, (tr, te) in enumerate(_folds(sd, seed, cfg.n_folds)):
        t0 = time.perf_counter()
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
        if len(exclude):
            keep_tr = keep[tr]
            tr = tr[keep_tr]
            te = te[keep[te]]
            # Re-index the inner split into the filtered training fold.
            remap = np.cumsum(keep_tr) - 1
            i_tr = remap[i_tr[keep_tr[i_tr]]]
            i_va = remap[i_va[keep_tr[i_va]]]
            if len(i_tr) < 100 or len(i_va) < 50 or len(te) < 20:
                continue
        red = reduce_blocks(blocks, tr, te, seed)

        results = Parallel(n_jobs=n_jobs, prefer="processes", inner_max_num_threads=1)(
            delayed(score_subset)(s, red, y, tr, i_tr, i_va, seed) for s in subsets
        )
        for s, scores, thr, _ in results:
            oof[s][te] = scores
            preds[s][te] = (scores >= thr).astype(int)
        print(f"  seed {seed} fold {fold + 1}/{cfg.n_folds}: {len(subsets)} subsets in "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    return y, oof, preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--saplma-layer", type=int, required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--exclude-datasets", nargs="*", default=[],
                    help="drop these datasets from training and evaluation, e.g. coqa")
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    acc: dict[tuple, dict[str, list]] = {s: {"auroc": [], "mcc": []} for s in all_subsets()}
    tag = "_no" + "".join(args.exclude_datasets) if args.exclude_datasets else ""
    cache_dir = Path(f"runs/{args.run}/stage5_posthoc/subset_seeds{tag}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        cache = cache_dir / f"seed{seed}.json"
        if cache.exists():
            import json
            for e in json.load(open(cache)):
                key = tuple(e["methods"])
                acc[key]["auroc"].append(e["auroc"]); acc[key]["mcc"].append(e["mcc"])
            print(f"seed {seed}: loaded from cache", flush=True)
            continue
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        print(f"seed {seed}: loading blocks ...", flush=True)
        blocks = load_blocks(cfg, sd, args.saplma_layer)
        y, oof, preds = run(cfg, seed, blocks, args.n_jobs, tuple(args.exclude_datasets))
        per_seed = []
        first = next(iter(acc))
        scored = preds[first] >= 0
        for s in acc:
            a = float(roc_auc_score(y[scored], oof[s][scored]))
            m = _fast_mcc(y[scored], preds[s][scored])
            acc[s]["auroc"].append(a); acc[s]["mcc"].append(m)
            per_seed.append({"methods": list(s), "auroc": a, "mcc": m})
        write_json(cache, per_seed)
        print(f"  seed {seed} done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del blocks

    out = {
        "run": args.run, "seeds": args.seeds, "saplma_layer": args.saplma_layer,
        "excluded_datasets": args.exclude_datasets,
        "methods": METHODS,
        "subsets": [
            {
                "methods": list(s), "n": len(s),
                "auroc_mean": round(float(np.mean(v["auroc"])), 4),
                "auroc_std": round(float(np.std(v["auroc"])), 4),
                "mcc_mean": round(float(np.mean(v["mcc"])), 4),
                "mcc_std": round(float(np.std(v["mcc"])), 4),
            }
            for s, v in acc.items()
        ],
    }
    write_json(Path(f"runs/{args.run}/stage5_posthoc/subset_ablation{tag}.json"), out)

    print(f"\n=== {args.run}: best subset at each size ===")
    for k in range(1, len(METHODS) + 1):
        at_k = [e for e in out["subsets"] if e["n"] == k]
        b = max(at_k, key=lambda e: e["mcc_mean"])
        print(f"  n={k}  MCC={b['mcc_mean']:.4f} ±{b['mcc_std']:.4f}  AUROC={b['auroc_mean']:.4f}"
              f"  {'+'.join(b['methods'])}")


if __name__ == "__main__":
    main()
