"""How many answers do the other methods get right that the best SAPLMA config misses?

SAPLMA's block dominates every subset in the ablation, and combining under PCA+concat
does not beat it. That raises the practical question: if a perfect router existed, how
much is actually there to route *to*?

Every block is fitted with the same pipeline — standardise, PCA to 128, logistic
regression, C and threshold tuned on the inner split — so no method is advantaged by its
classifier. Then, restricted to the answers SAPLMA gets wrong:

  * how many does each other method get right (individually)
  * how many does *at least one* other method get right (the exploitable set)
  * what MCC results from repairing exactly those (the routing ceiling over SAPLMA)

The ceiling is not achievable — identifying which answers to route needs the label — but
it bounds what any better combiner could win over SAPLMA alone.

Usage: uv run python scripts/headroom_over_saplma.py
"""

from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
from collections import defaultdict
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

BLOCKS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline"]
OTHERS = [b for b in BLOCKS if b != "saplma"]
C_GRID = (0.003, 0.03, 0.3, 3.0)
RUNS = [("Qwen3-14B", "main", 24), ("gemma-3-12b", "gemma3-12b", 29),
        ("gemma-3-4b", "gemma3-4b", 17), ("Llama-3.2-3B", "llama3.2-3b", 14),
        # base (non-instruction-tuned) generators; same depths as their instruct twins
        ("Llama-3.2-3B-base", "llama3.2-3b-base", 14),
        ("gemma-3-12b-pt", "gemma3-12b-pt", 29)]


def load_blocks(cfg, sd, layer):
    ids = list(sd["item_ids"])
    pos = {i: k for k, i in enumerate(ids)}
    out: dict[str, np.ndarray] = {}
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        d = cfg.stage_dir("stage1_extract", ds)
        rows = [pos[i] for i in sub]
        for b in BLOCKS:
            arr = load_features(d, b, sub)
            if b == "saplma":
                arr = arr[:, min(layer, arr.shape[1] - 1), :]
            arr = arr.reshape(len(sub), -1).astype(np.float32)
            if b not in out:
                out[b] = np.zeros((len(ids), arr.shape[1]), np.float32)
            out[b][rows] = arr
    return out


def block_oof(block, y, folds, strat, groups, seed):
    """One block under the shared pipeline; returns (scores, hard predictions)."""
    preds = np.full(len(y), -1, dtype=int)
    scores = np.full(len(y), np.nan)
    for tr, te in folds:
        s = StandardScaler().fit(block[tr])
        Xtr, Xte = s.transform(block[tr]), s.transform(block[te])
        k = min(128, Xtr.shape[1], len(tr) - 1)
        if k < Xtr.shape[1]:
            p = PCA(n_components=k, random_state=seed).fit(Xtr)
            Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
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
        p_te = m.predict_proba(Xte)[:, 1]
        scores[te], preds[te] = p_te, (p_te >= thr).astype(int)
    return scores, preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--runs", nargs="*", default=None,
                    help="restrict to these run-ids; base-model runs were added after "
                         "this script was written, so RUNS is not exhaustive")
    ap.add_argument("--n-jobs", type=int, default=5)
    args = ap.parse_args()

    results = {}
    sel = [r for r in RUNS if args.runs is None or r[1] in args.runs]
    for lbl, run, layer in sel:
        cfg = Config(); cfg.run_id = run
        agg = defaultdict(list)
        for seed in args.seeds:
            sd = load_seed(cfg, seed)
            blocks = load_blocks(cfg, sd, layer)
            y, groups = sd["y"], sd["groups"]
            strat = np.array([f"{d}_{v}" for d, v in zip(sd["dataset"], y)])
            folds = _folds(sd, seed, cfg.n_folds)

            out = dict(zip(BLOCKS, Parallel(n_jobs=args.n_jobs, prefer="processes")(
                delayed(block_oof)(blocks[b], y, folds, strat, groups, seed) for b in BLOCKS)))
            scores = {b: out[b][0] for b in BLOCKS}
            preds = {b: out[b][1] for b in BLOCKS}
            # Persist. These per-block out-of-fold predictions under one shared pipeline
            # are the substrate for every item-level analysis here, and recomputing them
            # each time costs ~10 min per model for nothing.
            oof_dir = cfg.stage_dir("stage5_posthoc", "block_oof")
            oof_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                oof_dir / f"seed{seed}.npz",
                y=y, item_ids=np.array(sd["item_ids"]), dataset=sd["dataset"],
                groups=groups, saplma_layer=np.array([layer]),
                **{f"scores__{b}": scores[b] for b in BLOCKS},
                **{f"preds__{b}": preds[b] for b in BLOCKS},
            )
            correct = {b: preds[b] == y for b in BLOCKS}
            sap_wrong = ~correct["saplma"]

            agg["n"].append(len(y))
            agg["saplma_mcc"].append(_fast_mcc(y, preds["saplma"]))
            agg["saplma_wrong"].append(int(sap_wrong.sum()))
            any_other = np.zeros(len(y), bool)
            for b in OTHERS:
                agg[f"rescue_{b}"].append(int((correct[b] & sap_wrong).sum()))
                any_other |= correct[b]
            exploitable = any_other & sap_wrong
            agg["exploitable"].append(int(exploitable.sum()))
            # Ceiling: SAPLMA, corrected wherever some other method was right.
            ceil = np.where(exploitable, y, preds["saplma"])
            agg["ceiling_mcc"].append(_fast_mcc(y, ceil))
            # And the reverse direction, which a router also risks losing.
            agg["saplma_only"].append(int((correct["saplma"] & ~any_other).sum()))
            # How many of the exploitable items only one method reaches.
            n_reach = np.sum([correct[b] for b in OTHERS], axis=0)
            agg["exploitable_unique"].append(int(((n_reach == 1) & exploitable).sum()))
            del blocks

        results[lbl] = {k: round(float(np.mean(v)), 2) for k, v in agg.items()}
        r = results[lbl]
        print(f"\n=== {lbl} ===  (mean per seed, n={r['n']:.0f})")
        print(f"  best SAPLMA config (PCA+logreg): MCC {r['saplma_mcc']:.4f}, "
              f"wrong on {r['saplma_wrong']:.0f} answers")
        for b in OTHERS:
            v = r[f"rescue_{b}"]
            print(f"    {b:16s} right on {v:6.0f} of those  ({v / r['saplma_wrong']:5.1%} of SAPLMA's errors)")
        print(f"  ANY other method right          : {r['exploitable']:6.0f}  "
              f"({r['exploitable'] / r['saplma_wrong']:.1%} of SAPLMA's errors, "
              f"{r['exploitable'] / r['n']:.1%} of all answers)")
        print(f"    of which only one method reaches: {r['exploitable_unique']:.0f}")
        print(f"  routing ceiling over SAPLMA     : MCC {r['ceiling_mcc']:.4f} "
              f"(+{r['ceiling_mcc'] - r['saplma_mcc']:.4f})")
        print(f"  answers only SAPLMA gets right  : {r['saplma_only']:.0f} (a router risks these)")

    write_json(Path("runs/headroom_over_saplma.json"), results)
    print("\nwrote runs/headroom_over_saplma.json")


if __name__ == "__main__":
    main()
