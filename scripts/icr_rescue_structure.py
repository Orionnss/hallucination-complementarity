"""Are ICR's rescues of SAPLMA's errors structured, or scattered?

Restricted to the answers SAPLMA gets wrong, some are recovered by ICR and some are not.
If the recovered ones form a coherent region, a router could learn to send those answers
to ICR; if they are scattered, routing is hopeless regardless of the combiner. Three
complementary probes:

1. **Descriptive.** How do rescued and non-rescued SAPLMA errors differ in dataset, true
   label, answer length, judge unanimity and ICR's own confidence?
2. **Separability.** Can a classifier predict "ICR rescues this" from the features, under
   grouped CV? This is the referee test narrowed to one method, and it is the number that
   decides whether routing is learnable.
3. **Local coherence.** For each rescued item, what share of its nearest neighbours in ICR
   feature space are also rescued? Compared against the base rate, this measures
   clustering without assuming a decision boundary.

`unique` restricts to answers ICR alone recovers among the four non-SAPLMA methods.

Usage: uv run python scripts/icr_rescue_structure.py --run main
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.io import load_features, read_json, write_json

OTHERS = ["lapeigvals", "attn_baseline", "icr", "svd_baseline"]


def load_block(cfg, item_ids, name):
    pos = {i: k for k, i in enumerate(item_ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in item_ids if i.startswith(ds + ":")]
        if not sub:
            continue
        arr = load_features(cfg.stage_dir("stage1_extract", ds), name, sub)
        arr = arr.reshape(len(sub), -1).astype(np.float32)
        if out is None:
            out = np.zeros((len(item_ids), arr.shape[1]), np.float32)
        for k, i in enumerate(sub):
            out[pos[i]] = arr[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--method", default="icr", choices=OTHERS,
                    help="which method's rescues of SAPLMA errors to analyse")
    ap.add_argument("--pca-dim", type=int, default=128,
                    help="components for wide blocks in the separability/geometry probes")
    ap.add_argument("--unique", action="store_true",
                    help="restrict to answers ICR alone recovers")
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    meta = {}
    for ds in cfg.datasets:
        for r in read_json(cfg.stage_dir("stage1_extract", ds) / "manifest.json")["items"]:
            meta[r["item_id"]] = r
        for e in read_json(cfg.stage_dir("stage2_judge", ds) / "labels.json")["labels"]:
            if e["item_id"] in meta:
                meta[e["item_id"]]["unanimous"] = bool(e["unanimous"])

    desc, sep, coh = [], [], []
    for f in sorted(glob.glob(f"runs/{args.run}/stage5_posthoc/block_oof/*.npz")):
        d = np.load(f, allow_pickle=True)
        y, ids, ds_arr = d["y"], d["item_ids"].astype(str), d["dataset"].astype(str)
        groups = d["groups"].astype(str)
        correct = {b: d[f"preds__{b}"] == y for b in ["saplma"] + OTHERS}
        sap_wrong = ~correct["saplma"]

        target = correct[args.method] & sap_wrong
        if args.unique:
            others = np.zeros(len(y), bool)
            for b in OTHERS:
                if b != args.method:
                    others |= correct[b]
            target &= ~others

        sel = np.flatnonzero(sap_wrong)          # the population: SAPLMA's errors
        lab = target[sel].astype(int)            # 1 = ICR rescued it
        tok = np.array([meta[i]["answer_tokens"] for i in ids], float)
        unan = np.array([bool(meta[i].get("unanimous", False)) for i in ids])

        desc.append({
            "n_saplma_wrong": len(sel), "n_rescued": int(lab.sum()),
            "rescue_rate": float(lab.mean()),
            "pos_rate_rescued": float(y[sel][lab == 1].mean()),
            "pos_rate_not": float(y[sel][lab == 0].mean()),
            "len_rescued": float(tok[sel][lab == 1].mean()),
            "len_not": float(tok[sel][lab == 0].mean()),
            "unanimous_rescued": float(unan[sel][lab == 1].mean()),
            "unanimous_not": float(unan[sel][lab == 0].mean()),
            "method_score_rescued": float(d[f"scores__{args.method}"][sel][lab == 1].mean()),
            "method_score_not": float(d[f"scores__{args.method}"][sel][lab == 0].mean()),
            **{f"frac_{x}_rescued": float((ds_arr[sel][lab == 1] == x).mean())
               for x in sorted(set(ds_arr))},
            **{f"frac_{x}_not": float((ds_arr[sel][lab == 0] == x).mean())
               for x in sorted(set(ds_arr))},
        })

        feat = load_block(cfg, list(ids), args.method)[sel]
        X = StandardScaler().fit_transform(feat)
        WIDE = X.shape[1] > 2 * args.pca_dim

        # 2. Separability under grouped CV: is "ICR rescues this" learnable?
        gkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
        for name, mk in (("logreg", lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
                         ("hist_gb", lambda: HistGradientBoostingClassifier(max_iter=200, random_state=0))):
            oof = np.full(len(lab), np.nan)
            for tr, te in gkf.split(X, lab, groups[sel]):
                Xtr, Xte = X[tr], X[te]
                if WIDE:
                    from sklearn.decomposition import PCA
                    pc = PCA(n_components=min(args.pca_dim, len(tr) - 1), random_state=0).fit(Xtr)
                    Xtr, Xte = pc.transform(Xtr), pc.transform(Xte)
                oof[te] = mk().fit(Xtr, lab[tr]).predict_proba(Xte)[:, 1]
            sep.append({"clf": name, "auroc": float(roc_auc_score(lab, oof))})

        # 3. Local coherence: neighbours of a rescued item, are they rescued too?
        Xg = X
        if WIDE:
            from sklearn.decomposition import PCA
            Xg = PCA(n_components=args.pca_dim, random_state=0).fit_transform(X)
        nn = NearestNeighbors(n_neighbors=11).fit(Xg)
        idx = nn.kneighbors(Xg, return_distance=False)[:, 1:]
        coh.append({
            "knn_purity_rescued": float(lab[idx][lab == 1].mean()),
            "knn_purity_all": float(lab[idx].mean()),
            "base_rate": float(lab.mean()),
        })

    agg = lambda rows, k: float(np.mean([r[k] for r in rows]))
    tag = "unique " if args.unique else ""
    print(f"=== {args.run}: {args.method} {tag}rescues of SAPLMA errors ===")
    print(f"  SAPLMA wrong on {agg(desc,'n_saplma_wrong'):.0f}; {args.method} rescues "
          f"{agg(desc,'n_rescued'):.0f} ({agg(desc,'rescue_rate'):.1%})")
    print("\n  1. DESCRIPTIVE (rescued vs not, within SAPLMA's errors)")
    for a, b, lbl in [("pos_rate_rescued", "pos_rate_not", "true positive rate"),
                      ("len_rescued", "len_not", "answer tokens"),
                      ("unanimous_rescued", "unanimous_not", "judges unanimous"),
                      ("method_score_rescued", "method_score_not", f"{args.method} score")]:
        print(f"    {lbl:22s} rescued={agg(desc,a):7.3f}   not={agg(desc,b):7.3f}")
    for x in cfg.datasets:
        if f"frac_{x}_rescued" in desc[0]:
            print(f"    share from {x:12s} rescued={agg(desc,f'frac_{x}_rescued'):7.3f}   "
                  f"not={agg(desc,f'frac_{x}_not'):7.3f}")
    print("\n  2. SEPARABILITY (predict rescue from method features, grouped CV)")
    for name in ("logreg", "hist_gb"):
        v = [r["auroc"] for r in sep if r["clf"] == name]
        print(f"    {name:10s} AUROC = {np.mean(v):.4f}  (0.5 = no structure)")
    print("\n  3. LOCAL COHERENCE (10-NN in method feature space)")
    print(f"    base rate of rescue          : {agg(coh,'base_rate'):.3f}")
    print(f"    neighbours of rescued rescued: {agg(coh,'knn_purity_rescued'):.3f}")
    print(f"    lift over base rate          : {agg(coh,'knn_purity_rescued')/agg(coh,'base_rate'):.2f}x")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/{args.method}_rescue_structure"
                    f"{'_unique' if args.unique else ''}.json"),
               {"descriptive": desc, "separability": sep, "coherence": coh})


if __name__ == "__main__":
    main()
