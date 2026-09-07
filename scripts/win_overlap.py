"""Are the union's wins the same items PCA+logreg would have won anyway?

Stage 3 gave SAPLMA its published MLP probe and the union PCA+logistic-regression, so
the reported union gain confounds two things. The aggregate numbers already show the
classifier swap is worth more than the union gain; this asks the sharper question at the
item level:

  A = items union_equal gets right that the SAPLMA MLP gets wrong   (the union's "wins")
  B = items SAPLMA+PCA+logreg gets right that the SAPLMA MLP gets wrong

If A and B largely coincide, the union is repairing the same answers a better classifier
on one feature block repairs — i.e. the gain is the classifier, not the extra methods.
If A contains substantial items outside B, the other four methods are contributing
something genuinely their own.

Requires refitting SAPLMA+PCA+logreg on stage 3's exact folds to obtain its predictions.

Usage: uv run python scripts/win_overlap.py
"""

from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _fast_mcc, _folds, load_seed

RUNS = [("Qwen3-14B", "main", 24), ("gemma-3-12b", "gemma3-12b", 29),
        ("gemma-3-4b", "gemma3-4b", 17), ("Llama-3.2-3B", "llama3.2-3b", 14)]
C_GRID = (0.003, 0.03, 0.3, 3.0)


def saplma_block(cfg, sd, layer):
    ids = list(sd["item_ids"])
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        arr = load_features(cfg.stage_dir("stage1_extract", ds), "saplma", sub)
        arr = arr[:, min(layer, arr.shape[1] - 1), :].astype(np.float32)
        if out is None:
            out = np.zeros((len(ids), arr.shape[1]), np.float32)
        for k, i in enumerate(sub):
            out[pos[i]] = arr[k]
    return out


def saplma_pca_logreg_oof(cfg, seed, block):
    """SAPLMA block under the union's own pipeline, on stage 3's exact folds."""
    sd = load_seed(cfg, seed)
    y, groups = sd["y"], sd["groups"]
    strat = np.array([f"{d}_{v}" for d, v in zip(sd["dataset"], y)])
    preds = np.full(len(y), -1, dtype=int)

    for tr, te in _folds(sd, seed, cfg.n_folds):
        s = StandardScaler().fit(block[tr])
        Xtr, Xte = s.transform(block[tr]), s.transform(block[te])
        p = PCA(n_components=min(128, Xtr.shape[1], len(tr) - 1), random_state=seed).fit(Xtr)
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
        preds[te] = (m.predict_proba(Xte)[:, 1] >= thr).astype(int)
    return y, preds, sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    results = {}
    for lbl, run, layer in RUNS:
        cfg = Config(); cfg.run_id = run
        agg = {k: [] for k in ("A", "B", "AB", "A_only", "B_only", "jaccard",
                               "frac_A_in_B", "mlp_mcc", "union_mcc", "pcalr_mcc")}
        for seed in args.seeds:
            sd0 = load_seed(cfg, seed)
            block = saplma_block(cfg, sd0, layer)
            y, pcalr, sd = saplma_pca_logreg_oof(cfg, seed, block)
            mlp, union = sd["preds"]["saplma"], sd["preds"]["union_equal"]

            mlp_wrong = mlp != y
            A = (union == y) & mlp_wrong      # union fixes what the MLP got wrong
            B = (pcalr == y) & mlp_wrong      # PCA+logreg fixes what the MLP got wrong
            inter = A & B
            agg["A"].append(int(A.sum())); agg["B"].append(int(B.sum()))
            agg["AB"].append(int(inter.sum()))
            agg["A_only"].append(int((A & ~B).sum())); agg["B_only"].append(int((B & ~A).sum()))
            agg["jaccard"].append(float(inter.sum() / max((A | B).sum(), 1)))
            agg["frac_A_in_B"].append(float(inter.sum() / max(A.sum(), 1)))
            agg["mlp_mcc"].append(_fast_mcc(y, mlp))
            agg["union_mcc"].append(_fast_mcc(y, union))
            agg["pcalr_mcc"].append(_fast_mcc(y, pcalr))
            del block
        results[lbl] = {k: round(float(np.mean(v)), 4) for k, v in agg.items()}
        r = results[lbl]
        print(f"\n=== {lbl} ===")
        print(f"  MCC: saplma-MLP={r['mlp_mcc']:.4f}  union_equal={r['union_mcc']:.4f}  "
              f"saplma-PCA+LR={r['pcalr_mcc']:.4f}")
        print(f"  A = union fixes MLP's errors      : {r['A']:.0f}")
        print(f"  B = PCA+LR fixes MLP's errors     : {r['B']:.0f}")
        print(f"  A n B (same items)                : {r['AB']:.0f}")
        print(f"  A only (union's own contribution) : {r['A_only']:.0f}")
        print(f"  B only (PCA+LR reaches, union not): {r['B_only']:.0f}")
        print(f"  fraction of union's wins also won by PCA+LR: {r['frac_A_in_B']:.1%}")
        print(f"  Jaccard(A,B)                      : {r['jaccard']:.3f}")

    write_json(Path("runs/win_overlap.json"), results)
    print("\nwrote runs/win_overlap.json")


if __name__ == "__main__":
    main()
