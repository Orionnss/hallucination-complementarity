"""Can a better classifier squeeze more out of the union features?

Baseline is stage 3's `union_equal`: standardise -> PCA 128 per block -> logistic
regression (pooled AUROC 0.860, MCC 0.534). Variants tried here:

  logreg            the baseline, re-run inside this harness for a like-for-like number
  logreg+context    plus dataset identity, answer/prompt length and per-method margins,
                    because the referee test found routing signal lives in exactly those
  mlp               non-linear, on the same reduced features
  mlp+context
  hist_gb           gradient boosting, which suits mixed tabular inputs
  logreg_pca256     twice the components per block, in case 128 is the bottleneck

Block PCA is fit once per fold on the training rows and shared across variants, so the
comparison is exactly like-for-like and the expensive step is paid once.

Usage: uv run python scripts/union_classifiers.py [--seeds 0 1]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.io import load_features, read_json, write_json
from halluc.pipeline.stage5_posthoc import _fast_mcc, _folds, load_seed

BLOCKS = ["lapeigvals", "attn_baseline", "saplma", "svd_baseline", "icr"]
SAPLMA_LAYER = 24


def load_blocks(cfg, sd):
    """Union feature blocks for one seed's items, in the seed's own order."""
    ids = list(sd["item_ids"])
    pos = {i: k for k, i in enumerate(ids)}
    out = {}
    shapes = {"lapeigvals": 16000, "attn_baseline": 16000, "saplma": 5120,
              "svd_baseline": 41, "icr": 40}
    for b in BLOCKS:
        out[b] = np.zeros((len(ids), shapes[b]), np.float32)
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        d = cfg.stage_dir("stage1_extract", ds)
        rows = [pos[i] for i in sub]
        for b in BLOCKS:
            arr = load_features(d, b, sub)
            if b == "saplma":
                arr = arr[:, SAPLMA_LAYER, :]
            out[b][rows] = arr.reshape(len(sub), -1).astype(np.float32)
    return out


def context_matrix(cfg, sd):
    """Dataset identity, lengths and per-method confidence margins."""
    meta = {}
    for ds in cfg.datasets:
        for r in read_json(cfg.stage_dir("stage1_extract", ds) / "manifest.json")["items"]:
            meta[r["item_id"]] = r
    ids = list(sd["item_ids"])
    tok = np.array([meta[i]["answer_tokens"] for i in ids], np.float32)
    ptok = np.array([meta[i]["prompt_tokens"] for i in ids], np.float32)
    onehot = np.column_stack([(sd["dataset"] == d).astype(np.float32)
                              for d in sorted(set(sd["dataset"]))])
    return np.column_stack([tok, ptok, onehot])


def variants(seed):
    return {
        "logreg":        lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
        "logreg+context": lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
        "mlp":           lambda: MLPClassifier((256, 128), max_iter=600, early_stopping=True,
                                               n_iter_no_change=20, random_state=seed),
        "mlp+context":   lambda: MLPClassifier((256, 128), max_iter=600, early_stopping=True,
                                               n_iter_no_change=20, random_state=seed),
        "hist_gb":       lambda: HistGradientBoostingClassifier(max_iter=300, random_state=seed),
        "logreg_pca256": lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
    }


def run_seed(cfg, seed, blocks, ctx):
    sd = load_seed(cfg, seed)
    y = sd["y"]
    folds = _folds(sd, seed, cfg.n_folds)
    oof = {name: np.full(len(y), np.nan) for name in variants(seed)}

    for tr, te in folds:
        # Fit the reduction once per fold, on training rows only, and share it.
        reduced = {}
        for n_comp in (128, 256):
            parts_tr, parts_te = [], []
            for b in BLOCKS:
                s = StandardScaler().fit(blocks[b][tr])
                Xtr, Xte = s.transform(blocks[b][tr]), s.transform(blocks[b][te])
                k = min(n_comp, Xtr.shape[1], len(tr) - 1)
                if k < Xtr.shape[1]:
                    p = PCA(n_components=k, random_state=seed).fit(Xtr)
                    Xtr, Xte = p.transform(Xtr), p.transform(Xte)
                parts_tr.append(Xtr); parts_te.append(Xte)
            reduced[n_comp] = (np.hstack(parts_tr), np.hstack(parts_te))

        cs = StandardScaler().fit(ctx[tr])
        ctr, cte = cs.transform(ctx[tr]), cs.transform(ctx[te])

        for name, make in variants(seed).items():
            base = reduced[256] if name.endswith("pca256") else reduced[128]
            Xtr, Xte = base
            if "context" in name:
                Xtr, Xte = np.hstack([Xtr, ctr]), np.hstack([Xte, cte])
            model = make()
            model.fit(Xtr, y[tr])
            oof[name][te] = model.predict_proba(Xte)[:, 1]

    return y, oof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    args = ap.parse_args()
    cfg = Config.load(args.config)

    acc = defaultdict(lambda: defaultdict(list))
    for seed in args.seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        print(f"seed {seed}: loading union blocks ...", flush=True)
        blocks = load_blocks(cfg, sd)
        ctx = context_matrix(cfg, sd)
        y, oof = run_seed(cfg, seed, blocks, ctx)
        for name, s in oof.items():
            acc[name]["auroc"].append(float(roc_auc_score(y, s)))
            acc[name]["mcc50"].append(_fast_mcc(y, (s >= 0.5).astype(int)))
            # Best achievable MCC over thresholds, as an upper reference per variant.
            grid = np.quantile(s, np.linspace(0.02, 0.98, 60))
            acc[name]["mcc_best"].append(max(_fast_mcc(y, (s >= t).astype(int)) for t in grid))
        print(f"  seed {seed} done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del blocks

    ms = lambda v: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4))
    summary = {"seeds": args.seeds,
               "baseline_union_equal": {"auroc": 0.860, "mcc": 0.534, "source": "stage 3, 5 seeds"},
               "variants": {n: {k: {"mean": ms(v)[0], "std": ms(v)[1]} for k, v in d.items()}
                            for n, d in acc.items()}}
    write_json(cfg.stage_dir("stage5_posthoc") / "union_classifiers.json", summary)

    print(f"\n{'variant':18s}{'AUROC':>16s}{'MCC@0.5':>12s}{'MCC@best':>12s}")
    for n, d in sorted(acc.items(), key=lambda kv: -np.mean(kv[1]["auroc"])):
        a, asd = ms(d["auroc"])
        print(f"  {n:16s}{a:9.4f} ±{asd:.4f}{ms(d['mcc50'])[0]:12.4f}{ms(d['mcc_best'])[0]:12.4f}")
    print(f"\n  {'union_equal (stage 3)':16s}{0.860:9.4f}         {'':12s}{0.534:12.4f}")


if __name__ == "__main__":
    main()
