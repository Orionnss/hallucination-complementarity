"""Referee test: is it predictable *which* of two methods is right on a given answer?

Stacking learns `inputs -> label`. This learns `inputs -> which method to trust`, on the
subset where exactly one of SAPLMA / LapEigvals is correct (~21.7% of answers). If that
target is unpredictable, no router can beat direct prediction and the unexploited part
of the oracle gap is irreducible. If it is predictable, a router is worth building.

Three input sets, increasing in what they see:
  scores   - the two probabilities, i.e. exactly what the stacker gets
  context  - scores plus confidence margins, dataset and answer length
  features - PCA'd raw features from both methods (fit on train folds only)

Two classifiers per set; the more appropriate one wins on merit rather than assumption.
Gradient boosting suits the low-dimensional tabular sets, regularised logistic
regression the wide PCA'd one, but both are run everywhere to avoid begging the question.

Usage: uv run python scripts/referee_test.py [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.io import load_features, read_json, write_json
from halluc.pipeline.stage5_posthoc import _fast_mcc, _folds, load_seed

A, B = "saplma", "lapeigvals"
SAPLMA_LAYER = 24  # the layer stage 3 selected


def build_inputs(cfg, sd, seed):
    """Assemble the three input sets for one seed."""
    ids = list(sd["item_ids"])
    sa, sb = sd["scores"][A], sd["scores"][B]

    meta = {}
    for ds in cfg.datasets:
        for r in read_json(cfg.stage_dir("stage1_extract", ds) / "manifest.json")["items"]:
            meta[r["item_id"]] = r
    tok = np.array([meta[i]["answer_tokens"] for i in ids], float)
    ptok = np.array([meta[i]["prompt_tokens"] for i in ids], float)
    ds_onehot = np.column_stack([(sd["dataset"] == d).astype(float) for d in sorted(set(sd["dataset"]))])

    scores = np.column_stack([sa, sb])
    context = np.column_stack([
        sa, sb, np.abs(sa - 0.5), np.abs(sb - 0.5), np.abs(sa - sb),
        tok, ptok, ds_onehot,
    ])

    # Raw features, per-block, in the seed's item order.
    by_ds = {}
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if sub:
            by_ds[ds] = sub
    sap = np.zeros((len(ids), 5120), np.float32)
    lap = np.zeros((len(ids), 16000), np.float32)
    pos = {i: k for k, i in enumerate(ids)}
    for ds, sub in by_ds.items():
        d = cfg.stage_dir("stage1_extract", ds)
        s = load_features(d, "saplma", sub)[:, SAPLMA_LAYER, :].astype(np.float32)
        l = load_features(d, "lapeigvals", sub).reshape(len(sub), -1)
        for k, i in enumerate(sub):
            sap[pos[i]] = s[k]
            lap[pos[i]] = l[k]
    return {"scores": scores, "context": context}, (sap, lap)


def referee_for_seed(cfg, seed, n_pca=64):
    sd = load_seed(cfg, seed)
    y = sd["y"]
    ca = sd["preds"][A] == y
    cb = sd["preds"][B] == y
    routable = ca ^ cb                      # exactly one is right
    target = ca[routable].astype(int)       # 1 = trust SAPLMA, 0 = trust LapEigvals

    sets, (sap, lap) = build_inputs(cfg, sd, seed)
    folds = _folds(sd, seed, cfg.n_folds)
    out = {"n_routable": int(routable.sum()),
           "share_routable": float(routable.mean()),
           "majority_class": float(max(target.mean(), 1 - target.mean()))}

    def evaluate(name, make_train_test):
        res = {}
        for clf_name in ("logreg", "hist_gb"):
            oof = np.full(routable.sum(), np.nan)
            idx_routable = np.flatnonzero(routable)
            remap = {g: k for k, g in enumerate(idx_routable)}
            for tr, te in folds:
                tr_r = [i for i in tr if routable[i]]
                te_r = [i for i in te if routable[i]]
                if len(tr_r) < 50 or len(te_r) < 10:
                    continue
                Xtr, Xte = make_train_test(tr_r, te_r)
                ytr = ca[tr_r].astype(int)
                if len(np.unique(ytr)) < 2:
                    continue
                clf = (LogisticRegression(max_iter=2000, class_weight="balanced")
                       if clf_name == "logreg"
                       else HistGradientBoostingClassifier(max_iter=200, random_state=seed))
                clf.fit(Xtr, ytr)
                p = clf.predict_proba(Xte)[:, 1]
                for k, i in enumerate(te_r):
                    oof[remap[i]] = p[k]
            ok = ~np.isnan(oof)
            res[clf_name] = {
                "auroc": float(roc_auc_score(target[ok], oof[ok])) if len(np.unique(target[ok])) > 1 else None,
                "accuracy": float(((oof[ok] >= 0.5).astype(int) == target[ok]).mean()),
                "n": int(ok.sum()),
                "oof": oof,
            }
        return res

    # scores / context: standardise inside the fold.
    for name in ("scores", "context"):
        X = sets[name]
        def mk(tr_r, te_r, X=X):
            s = StandardScaler().fit(X[tr_r])
            return s.transform(X[tr_r]), s.transform(X[te_r])
        out[name] = evaluate(name, mk)

    # features: PCA each block on the training fold only.
    def mk_feat(tr_r, te_r):
        parts_tr, parts_te = [], []
        for block in (sap, lap):
            s = StandardScaler().fit(block[tr_r])
            p = PCA(n_components=min(n_pca, len(tr_r) - 1), random_state=seed)
            p.fit(s.transform(block[tr_r]))
            parts_tr.append(p.transform(s.transform(block[tr_r])))
            parts_te.append(p.transform(s.transform(block[te_r])))
        return np.hstack(parts_tr), np.hstack(parts_te)
    out["features"] = evaluate("features", mk_feat)

    # What routing would actually buy: follow the referee's choice on routable items.
    best = max(
        ((s, c, out[s][c]["auroc"]) for s in ("scores", "context", "features")
         for c in ("logreg", "hist_gb") if out[s][c]["auroc"] is not None),
        key=lambda t: t[2],
    )
    oof = out[best[0]][best[1]]["oof"]
    ok = ~np.isnan(oof)
    # Default to SAPLMA (the stronger single method); the referee overrides on routable
    # items where it prefers LapEigvals.
    routed = sd["preds"][A].copy()
    idx = np.flatnonzero(routable)
    for k, i in enumerate(idx):
        if ok[k]:
            routed[i] = sd["preds"][A][i] if oof[k] >= 0.5 else sd["preds"][B][i]
    out["routing"] = {
        "best_input_set": best[0], "best_classifier": best[1], "referee_auroc": round(best[2], 4),
        "routed_mcc": round(_fast_mcc(y, routed), 4),
        "saplma_mcc": round(_fast_mcc(y, sd["preds"][A]), 4),
        "lapeigvals_mcc": round(_fast_mcc(y, sd["preds"][B]), 4),
        "pair_oracle_mcc": round(_fast_mcc(y, np.where(ca | cb, y, 1 - y)), 4),
    }
    for s in ("scores", "context", "features"):
        for c in ("logreg", "hist_gb"):
            out[s][c].pop("oof", None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    seeds = args.seeds if args.seeds else cfg.seeds

    per_seed = {}
    for s in seeds:
        print(f"seed {s} ...", flush=True)
        per_seed[s] = referee_for_seed(cfg, s)

    agg = {}
    for iset in ("scores", "context", "features"):
        for clf in ("logreg", "hist_gb"):
            a = [per_seed[s][iset][clf]["auroc"] for s in seeds]
            agg[f"{iset}/{clf}"] = {
                "auroc_mean": round(float(np.mean(a)), 4),
                "auroc_std": round(float(np.std(a)), 4),
            }
    summary = {
        "target": "which of saplma/lapeigvals is correct, on items where exactly one is",
        "n_routable_mean": int(np.mean([per_seed[s]["n_routable"] for s in seeds])),
        "share_routable": round(float(np.mean([per_seed[s]["share_routable"] for s in seeds])), 4),
        "majority_class_rate": round(float(np.mean([per_seed[s]["majority_class"] for s in seeds])), 4),
        "referee_auroc": agg,
        "routing": {
            k: round(float(np.mean([per_seed[s]["routing"][k] for s in seeds])), 4)
            for k in ("routed_mcc", "saplma_mcc", "lapeigvals_mcc", "pair_oracle_mcc", "referee_auroc")
        },
        "per_seed": per_seed,
    }
    out = Config.load(args.config).stage_dir("stage5_posthoc") / "referee_test.json"
    write_json(out, summary)

    print(f"\nroutable items: {summary['n_routable_mean']} ({summary['share_routable']:.1%}) "
          f"| majority-class baseline {summary['majority_class_rate']:.3f}")
    print(f"\n{'input set / classifier':28s}{'referee AUROC':>16s}")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]['auroc_mean']):
        print(f"  {k:26s}{v['auroc_mean']:9.4f} ±{v['auroc_std']:.4f}")
    r = summary["routing"]
    print(f"\nrouting outcome (best referee, AUROC {r['referee_auroc']:.4f}):")
    print(f"  saplma alone      {r['saplma_mcc']:.4f}")
    print(f"  lapeigvals alone  {r['lapeigvals_mcc']:.4f}")
    print(f"  routed            {r['routed_mcc']:.4f}")
    print(f"  pair oracle       {r['pair_oracle_mcc']:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
