"""Where SAPLMA fails, does token confidence disagree with it independently?

The accuracy-independence tradeoff was measured across four internal-representation
methods, all of which read the same forward pass SAPLMA reads. Token confidence comes from
a different level of the model entirely — the output distribution, not the residual stream
— so it is the strongest available test of whether the tradeoff is a property of the
methods or of the task.

Two outcomes, both informative. If logprob is the *most* independent (it is also the
weakest, at MCC .363 against SAPLMA's .547), the tradeoff extends to a signal that shares
no machinery with the hidden-state probes, and the mechanism generalises. If it is as
redundant as the rest, then what all these methods share is not architecture but the
difficulty of the items themselves.

SAPLMA's errors and every other method's predictions come from the stored stage-3
out-of-fold file, so they are exactly the numbers used elsewhere. Logprob predictions are
fitted here on the same folds under the same search, since it was extracted after that
file was written.

Usage: uv run python scripts/kappa_logprob.py --run main
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

C_GRID = (0.003, 0.03, 0.3, 3.0)
MLP_GRID = [((256, 128, 64), 1e-4), ((128,), 1e-2)]
OTHERS = ["lapeigvals", "attn_baseline", "icr", "svd_baseline"]


def logprob_matrix(cfg, ids):
    """Load logprob features, imputing the ~0.1% that failed the tokenizer round trip."""
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        d = cfg.stage_dir("stage1_extract", ds) / "logprob"
        have = set()
        for line in (d / "checkpoint.jsonl").open():
            r = json.loads(line)
            if "shard" in r:
                have.add(r.get("item_id") or r.get("id"))
        keep = [i for i in sub if i in have]
        got = load_features(d, "logprob", keep).reshape(len(keep), -1).astype(np.float32)
        if out is None:
            out = np.zeros((len(ids), got.shape[1]), np.float32)
        med = np.median(got, axis=0)
        for i in sub:
            out[pos[i]] = med
        for k, i in enumerate(keep):
            out[pos[i]] = got[k]
    return out


def oof_logprob(cfg, sd, X, seed):
    """Out-of-fold predictions on the same folds, same search as fair_comparison."""
    y, groups, ds_arr = sd["y"], sd["groups"], sd["dataset"]
    strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
    score = np.full(len(y), np.nan)
    pred = np.full(len(y), -1, dtype=int)
    for tr, te in _folds(sd, seed, cfg.n_folds):
        s = StandardScaler().fit(X[tr])
        A, B = s.transform(X[tr]), s.transform(X[te])
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
        i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

        def make(cfg_):
            kind, p1, p2 = cfg_
            return (LogisticRegression(C=p1, max_iter=3000, class_weight="balanced")
                    if kind == "lr" else
                    MLPClassifier(hidden_layer_sizes=p1, alpha=p2, max_iter=600,
                                  early_stopping=True, n_iter_no_change=20,
                                  random_state=seed))

        grid = [("lr", C, None) for C in C_GRID] + [("mlp", h, a) for h, a in MLP_GRID]
        best, best_a = grid[0], -1.0
        for g in grid:
            m = make(g).fit(A[i_tr], y[tr][i_tr])
            auc = roc_auc_score(y[tr][i_va], m.predict_proba(A[i_va])[:, 1])
            if auc > best_a:
                best_a, best = float(auc), g
        m = make(best).fit(A[i_tr], y[tr][i_tr])
        thr, _ = best_threshold(y[tr][i_va], m.predict_proba(A[i_va])[:, 1])
        m = make(best).fit(A, y[tr])
        score[te] = m.predict_proba(B)[:, 1]
        pred[te] = (score[te] >= thr).astype(int)
    return score, pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    acc = defaultdict(lambda: defaultdict(list))

    for seed in args.seeds:
        sd = load_seed(cfg, seed)
        ids = list(sd["item_ids"])
        f = sorted(glob.glob(f"runs/{args.run}/stage5_posthoc/block_oof/*.npz"))[seed]
        d = np.load(f, allow_pickle=True)
        assert list(d["item_ids"].astype(str)) == ids, "block_oof item order differs"
        y = d["y"]
        wrong = d["preds__saplma"] != y

        X = logprob_matrix(cfg, ids)
        _, lp_pred = oof_logprob(cfg, sd, X, seed)

        preds = {m: d[f"preds__{m}"] for m in OTHERS}
        preds["logprob"] = lp_pred
        for m, p in preds.items():
            acc[m]["kappa_all"].append(float(cohen_kappa_score(y, p)))
            acc[m]["mcc_all"].append(float(matthews_corrcoef(y, p)))
            acc[m]["kappa_err"].append(float(cohen_kappa_score(y[wrong], p[wrong])))
            acc[m]["acc_err"].append(float((p[wrong] == y[wrong]).mean()))
            # Independence null: per-class accuracy over all data, reweighted to the
            # class mix of the SAPLMA-wrong slice.
            per_c = {c: (p[y == c] == c).mean() for c in (0, 1)}
            mix = {c: (y[wrong] == c).mean() for c in (0, 1)}
            acc[m]["null"].append(float(sum(per_c[c] * mix[c] for c in (0, 1))))
        print(f"  seed {seed} done ({wrong.sum()} SAPLMA errors)", flush=True)

    print(f"\n=== {args.run}: agreement with the reference on SAPLMA's errors ===")
    print(f"  {'method':16s}{'MCC all':>10s}{'kappa all':>11s}{'kappa|err':>11s}"
          f"{'acc|err':>9s}{'null':>8s}{'ratio':>8s}")
    rows = []
    for m, v in acc.items():
        ratio = np.mean(v["acc_err"]) / np.mean(v["null"])
        rows.append((ratio, m, v))
    for ratio, m, v in sorted(rows):
        print(f"  {m:16s}{np.mean(v['mcc_all']):10.4f}{np.mean(v['kappa_all']):11.4f}"
              f"{np.mean(v['kappa_err']):11.4f}{np.mean(v['acc_err']):9.1%}"
              f"{np.mean(v['null']):8.1%}{ratio:8.2f}")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/kappa_logprob.json"),
               {m: {k: round(float(np.mean(v)), 4) for k, v in d.items()}
                for m, d in acc.items()})
    print(f"\nwrote runs/{args.run}/stage5_posthoc/kappa_logprob.json")


if __name__ == "__main__":
    main()
