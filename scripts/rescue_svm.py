"""Can an RBF SVM predict which of SAPLMA's errors another method will rescue?

The population is the answers SAPLMA gets wrong — the "conflicting" ones, where a router
would have to decide whether to defer. The target is binary: does method M recover this
answer? Linear and boosted probes on the method's own features both stalled near AUROC
0.55; RBF is the one family that beat linear elsewhere in this study (on the union
features, +0.017 MCC), so it is the strongest remaining candidate.

Three feature spaces per method, because *where* the routing signal would live is itself
the open question:

  own    the rescuing method's features  - "does M know when it wins?"
  saplma SAPLMA's features               - "does SAPLMA's own state predict its failure
                                            mode?" the routing-relevant view, untested
                                            until now
  union  all five blocks, PCA 128 each   - everything available to a combiner

5-fold grouped CV (each fold is an 80/20 split), grouped by passage so CoQA turns and
SQuAD questions sharing a context cannot straddle train and test. C and gamma are chosen
on an inner split, so the reported AUROC is not best-of-sweep. Logistic regression is
fitted on the identical folds as the reference.

Usage: uv run python scripts/rescue_svm.py --run main
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from halluc.config import Config
from halluc.io import load_features, write_json

BLOCKS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline"]
METHODS = [b for b in BLOCKS if b != "saplma"]
SVM_GRID = [("rbf", 1.0, "scale"), ("rbf", 10.0, "scale"),
            ("rbf", 1.0, 1e-3), ("rbf", 0.1, "scale")]
LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}


def load_block(cfg, ids, name, layer):
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        arr = load_features(cfg.stage_dir("stage1_extract", ds), name, sub)
        if name == "saplma":
            arr = arr[:, min(layer, arr.shape[1] - 1), :]
        arr = arr.reshape(len(sub), -1).astype(np.float32)
        if out is None:
            out = np.zeros((len(ids), arr.shape[1]), np.float32)
        for k, i in enumerate(sub):
            out[pos[i]] = arr[k]
    return out


def reduce_fit(Xtr, Xte, dim, seed=0):
    s = StandardScaler().fit(Xtr)
    a, b = s.transform(Xtr), s.transform(Xte)
    if a.shape[1] > dim:
        p = PCA(n_components=min(dim, len(a) - 1), random_state=seed).fit(a)
        a, b = p.transform(a), p.transform(b)
    return a, b


def evaluate(X, lab, groups, seed, n_folds=5):
    """5-fold grouped CV; C/gamma picked on an inner split of each training fold."""
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_svm = np.full(len(lab), np.nan)
    oof_lr = np.full(len(lab), np.nan)
    for tr, te in outer.split(X, lab, groups):
        Xtr, Xte = reduce_fit(X[tr], X[te], 128, seed)
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        i_tr, i_va = next(inner.split(Xtr, lab[tr], groups[tr]))
        best, best_a = SVM_GRID[0], -1.0
        for cfg_ in SVM_GRID:
            k, C, g = cfg_
            m = SVC(kernel=k, C=C, gamma=g, class_weight="balanced",
                    max_iter=2_000_000, random_state=seed).fit(Xtr[i_tr], lab[tr][i_tr])
            a = roc_auc_score(lab[tr][i_va], m.decision_function(Xtr[i_va]))
            if a > best_a:
                best_a, best = float(a), cfg_
        k, C, g = best
        oof_svm[te] = SVC(kernel=k, C=C, gamma=g, class_weight="balanced",
                          max_iter=2_000_000, random_state=seed).fit(
                              Xtr, lab[tr]).decision_function(Xte)
        oof_lr[te] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            Xtr, lab[tr]).predict_proba(Xte)[:, 1]
    return (float(roc_auc_score(lab, oof_svm)), float(roc_auc_score(lab, oof_lr)), best)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    layer = LAYER.get(args.run, 24)
    acc = defaultdict(lambda: defaultdict(list))
    picks = defaultdict(list)

    for si, f in enumerate(sorted(glob.glob(f"runs/{args.run}/stage5_posthoc/block_oof/*.npz"))):
        if si not in args.seeds:
            continue
        d = np.load(f, allow_pickle=True)
        y, ids = d["y"], d["item_ids"].astype(str)
        groups = d["groups"].astype(str)
        sel = np.flatnonzero(d["preds__saplma"] != y)
        blocks = {b: load_block(cfg, list(ids), b, layer) for b in BLOCKS}
        union = np.hstack([blocks[b] for b in BLOCKS])

        jobs = []
        for m in METHODS:
            lab = (d[f"preds__{m}"] == y)[sel].astype(int)
            for space, X in (("own", blocks[m][sel]), ("saplma", blocks["saplma"][sel]),
                             ("union", union[sel])):
                jobs.append((m, space, X, lab))
        res = Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(evaluate)(X, lab, groups[sel], si) for _, _, X, lab in jobs)
        for (m, space, _, lab), (a_svm, a_lr, best) in zip(jobs, res):
            acc[(m, space)]["svm"].append(a_svm)
            acc[(m, space)]["logreg"].append(a_lr)
            acc[(m, space)]["base"].append(float(lab.mean()))
            picks[(m, space)].append(str(best))
        print(f"  seed {si} done ({len(sel)} conflicting answers)", flush=True)
        del blocks, union

    print(f"\n=== {args.run}: predicting which SAPLMA errors each method rescues ===")
    print(f"  population = SAPLMA's errors; 5-fold grouped CV (80/20 per fold), "
          f"{len(args.seeds)} seeds")
    print(f"\n  {'method':15s}{'features':9s}{'rescue rate':>12s}{'SVM-RBF AUROC':>16s}{'logreg AUROC':>15s}")
    out = {}
    for (m, space), v in acc.items():
        s_m, s_s = np.mean(v["svm"]), np.std(v["svm"])
        l_m = np.mean(v["logreg"])
        print(f"  {m:15s}{space:9s}{np.mean(v['base']):12.3f}"
              f"{s_m:11.4f} ±{s_s:.4f}{l_m:15.4f}")
        out[f"{m}|{space}"] = {"svm_auroc_mean": round(s_m, 4), "svm_auroc_std": round(s_s, 4),
                               "logreg_auroc_mean": round(l_m, 4),
                               "rescue_rate": round(float(np.mean(v["base"])), 4),
                               "configs": picks[(m, space)]}
    write_json(Path(f"runs/{args.run}/stage5_posthoc/rescue_svm.json"), out)
    print(f"\nwrote runs/{args.run}/stage5_posthoc/rescue_svm.json")


if __name__ == "__main__":
    main()
