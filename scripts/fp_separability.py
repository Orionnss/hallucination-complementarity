"""Can another method's features tell SAPLMA's false alarms from its true ones?

Excluding CoQA, 57-64% of SAPLMA's errors are false positives — it cries hallucination
on a clean answer. That is the dominant failure mode, and the practical question is
whether anything can flag it.

Population: answers SAPLMA predicts HALLUCINATED. Target: was it right (TP) or wrong
(FP)? Each method's feature space is tested in turn, plus SAPLMA's own as the reference
— if another representation separates SAPLMA's false alarms better than SAPLMA's own
does, that is a concrete, deployable use for it, and a different claim from the rescue
analyses (which asked about all errors at once).

The mirror population (predicted NOT, TN vs FN) is reported alongside.

Blocks are loaded once per model and indexed per seed; reloading per seed made an
earlier version thrash. Grouped 5-fold CV, PCA fitted inside each fold.

Usage: uv run python scripts/fp_separability.py --run main
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from halluc.config import Config
from halluc.io import load_features, write_json

BLOCKS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline"]
LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}


def load_all(cfg, all_ids, layer):
    """Load every block once, for the union of item ids across seeds."""
    pos = {i: k for k, i in enumerate(all_ids)}
    out = {}
    for b in BLOCKS:
        arr_full = None
        for ds in cfg.datasets:
            sub = [i for i in all_ids if i.startswith(ds + ":")]
            if not sub:
                continue
            a = load_features(cfg.stage_dir("stage1_extract", ds), b, sub)
            if b == "saplma":
                a = a[:, min(layer, a.shape[1] - 1), :]
            a = a.reshape(len(sub), -1).astype(np.float32)
            if arr_full is None:
                arr_full = np.zeros((len(all_ids), a.shape[1]), np.float32)
            arr_full[[pos[i] for i in sub]] = a
            del a
        out[b] = arr_full
    return out, pos


def cv_auroc(X, lab, groups, seed, dim=128):
    o_lr = np.full(len(lab), np.nan)
    o_svm = np.full(len(lab), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, lab, groups):
        s = StandardScaler().fit(X[tr])
        a, b = s.transform(X[tr]), s.transform(X[te])
        if a.shape[1] > dim:
            p = PCA(n_components=min(dim, len(tr) - 1), random_state=seed).fit(a)
            a, b = p.transform(a), p.transform(b)
        o_lr[te] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            a, lab[tr]).predict_proba(b)[:, 1]
        o_svm[te] = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                        max_iter=2_000_000, random_state=seed).fit(
                            a, lab[tr]).decision_function(b)
    return float(roc_auc_score(lab, o_lr)), float(roc_auc_score(lab, o_svm))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--exclude-datasets", nargs="*", default=["coqa"],
                    help="CoQA is excluded by default: its 13-17%% base rate makes SAPLMA's "
                         "errors overwhelmingly false negatives, which is a different question")
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    files = [f for i, f in enumerate(sorted(
        glob.glob(f"runs/{args.run}/stage5_posthoc/block_oof/*.npz"))) if i in args.seeds]
    all_ids = sorted({i for f in files for i in np.load(f, allow_pickle=True)["item_ids"].astype(str)})
    print(f"{args.run}: loading blocks once for {len(all_ids)} distinct answers ...", flush=True)
    blocks, pos = load_all(cfg, all_ids, LAYER.get(args.run, 24))

    acc = defaultdict(lambda: defaultdict(list))
    for f in files:
        d = np.load(f, allow_pickle=True)
        y, p = d["y"], d["preds__saplma"]
        ids = d["item_ids"].astype(str)
        groups, ds_arr = d["groups"].astype(str), d["dataset"].astype(str)
        keep = ~np.isin(ds_arr, args.exclude_datasets)
        rows = np.array([pos[i] for i in ids])

        for pop, mask in (("pred HALLUC (TP vs FP)", (p == 1) & keep),
                          ("pred NOT (TN vs FN)", (p == 0) & keep)):
            idx = np.flatnonzero(mask)
            lab = y[idx].astype(int)
            if len(np.unique(lab)) < 2:
                continue
            # Baseline: SAPLMA's own score, no features and no second model. An answer
            # SAPLMA gets wrong is usually one it was unsure about, so distance from the
            # threshold separates TP from FP on its own. Any feature probe has to beat
            # this to be carrying information rather than restating calibration.
            acc[(pop, "saplma_score_only")]["lr"].append(
                float(roc_auc_score(lab, d["scores__saplma"][idx])))
            acc[(pop, "saplma_score_only")]["svm"].append(
                float(roc_auc_score(lab, d["scores__saplma"][idx])))
            for b in BLOCKS:
                a_lr, a_svm = cv_auroc(blocks[b][rows[idx]], lab, groups[idx], 0)
                acc[(pop, b)]["lr"].append(a_lr)
                acc[(pop, b)]["svm"].append(a_svm)
            acc[(pop, "_n")]["n"].append(len(idx))
            acc[(pop, "_n")]["err"].append(float(1 - lab.mean() if "HALLUC" in pop else lab.mean()))
        print(f"  seed done", flush=True)

    print(f"\n=== {args.run}: separating SAPLMA's own errors, CoQA excluded ===")
    for pop in ("pred HALLUC (TP vs FP)", "pred NOT (TN vs FN)"):
        info = acc[(pop, "_n")]
        if not info["n"]:
            continue
        print(f"\n  {pop}   n={np.mean(info['n']):.0f}, error rate {np.mean(info['err']):.1%}")
        print(f"    {'feature space':16s}{'logreg':>9s}{'RBF-SVM':>10s}")
        rows_ = [(b, np.mean(acc[(pop, b)]["lr"]), np.mean(acc[(pop, b)]["svm"]))
                 for b in BLOCKS + ["saplma_score_only"]]
        for b, lr, sv in sorted(rows_, key=lambda r: -r[2]):
            star = ("  <- SAPLMA's own" if b == "saplma"
                    else "  <- baseline: no model at all" if b == "saplma_score_only" else "")
            print(f"    {b:16s}{lr:9.4f}{sv:10.4f}{star}")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/fp_separability.json"),
               {f"{pop}|{b}": {"logreg": round(float(np.mean(v['lr'])), 4),
                               "svm": round(float(np.mean(v['svm'])), 4)}
                for (pop, b), v in acc.items() if b != "_n"})


if __name__ == "__main__":
    main()
