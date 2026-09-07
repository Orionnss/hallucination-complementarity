"""Two-stage detector: SAPLMA, then a corrector that predicts when SAPLMA is wrong.

Separating SAPLMA's true from false alarms in its own feature space reached AUROC
0.72-0.76 — far above anything the rescue-routing probes managed (0.55-0.65). This tests
whether that signal converts into a better detector, rather than only being measurable.

Protocol, per outer fold:

  1. fit SAPLMA on inner-train; tune its threshold on inner-val
  2. train the corrector on **inner-val**, where SAPLMA's errors are observable without
     touching test. Target: did SAPLMA get this wrong?
  3. choose the flip threshold by cross-fitting inside inner-val, so it is not picked on
     the same predictions the corrector was fitted to
  4. apply both to the untouched outer test fold; flip SAPLMA where the corrector is
     confident it erred

SAPLMA stays fitted on inner-train (not refitted on the full training fold), so the
errors the corrector learned from are the errors the deployed model actually makes. The
baseline is that same inner-train SAPLMA, making the comparison like-for-like.

A `crossfit` variant trains the corrector on the whole training fold via out-of-fold
SAPLMA predictions — 4x the corrector training data, at the cost of a mild mismatch
between OOF and final SAPLMA. Both are reported.

Usage: uv run python scripts/two_stage.py --run main
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
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}
C_GRID = (0.003, 0.03, 0.3, 3.0)


def load_saplma(cfg, ids, layer):
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        a = load_features(cfg.stage_dir("stage1_extract", ds), "saplma", sub)
        a = a[:, min(layer, a.shape[1] - 1), :].astype(np.float32)
        if out is None:
            out = np.zeros((len(ids), a.shape[1]), np.float32)
        out[[pos[i] for i in sub]] = a
    return out


def reduce(Xfit, others, seed, dim=128):
    s = StandardScaler().fit(Xfit)
    p = PCA(n_components=min(dim, Xfit.shape[1], len(Xfit) - 1), random_state=seed).fit(
        s.transform(Xfit))
    return [p.transform(s.transform(o)) for o in others]


def tuned_logreg(Xtr, ytr, Xva, yva, seed):
    best_c, best_a = C_GRID[0], -1.0
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        a = roc_auc_score(yva, m.predict_proba(Xva)[:, 1]) if len(np.unique(yva)) > 1 else 0.5
        if a > best_a:
            best_a, best_c = float(a), C
    return LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    variants = ["saplma_base", "corrected_val", "corrected_crossfit", "blended_val"]
    acc = defaultdict(lambda: defaultdict(list))
    flips = defaultdict(list)

    for seed in args.seeds:
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"].astype(str), sd["dataset"].astype(str)
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = load_saplma(cfg, ids, LAYER.get(args.run, 24))
        oof = {v: np.full(len(y), -1, dtype=int) for v in variants}
        sc = {v: np.full(len(y), np.nan) for v in variants}

        for tr, te in _folds(sd, seed, cfg.n_folds):
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
            a_tr, a_va = tr[i_tr], tr[i_va]

            Rtr, Rva, Rte = reduce(X[a_tr], [X[a_tr], X[a_va], X[te]], seed)
            base = tuned_logreg(Rtr, y[a_tr], Rva, y[a_va], seed)
            p_va, p_te = base.predict_proba(Rva)[:, 1], base.predict_proba(Rte)[:, 1]
            thr, _ = best_threshold(y[a_va], p_va)
            pred_va, pred_te = (p_va >= thr).astype(int), (p_te >= thr).astype(int)
            sc["saplma_base"][te], oof["saplma_base"][te] = p_te, pred_te

            # --- corrector trained on the validation split only -------------------
            wrong_va = (pred_va != y[a_va]).astype(int)
            if len(np.unique(wrong_va)) > 1:
                # Cross-fit inside inner-val so the flip threshold is not chosen on the
                # same predictions the corrector was fitted to.
                cf = np.full(len(a_va), np.nan)
                for c_tr, c_te in StratifiedGroupKFold(
                        3, shuffle=True, random_state=seed).split(
                            Rva, wrong_va, groups[a_va]):
                    cf[c_te] = LogisticRegression(
                        max_iter=3000, class_weight="balanced").fit(
                            Rva[c_tr], wrong_va[c_tr]).predict_proba(Rva[c_te])[:, 1]
                best_t, best_m = 0.5, -2.0
                for t in np.quantile(cf, np.linspace(0.5, 0.99, 40)):
                    cand = np.where(cf >= t, 1 - pred_va, pred_va)
                    m = matthews_corrcoef(y[a_va], cand)
                    if m > best_m:
                        best_m, best_t = m, float(t)
                corr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
                    Rva, wrong_va)
                w_te = corr.predict_proba(Rte)[:, 1]
                # Soft blend: rather than betting the prediction on a flip threshold,
                # move the probability toward its complement in proportion to how likely
                # the corrector thinks SAPLMA erred. Degrades gracefully.
                #
                # This corrector is deliberately NOT class-balanced. The blend reads w as
                # a literal P(SAPLMA is wrong); balancing calibrates it to a 50/50 prior
                # against a true error rate near 20%, pushing w over 0.5 for much of the
                # data and inverting the ranking instead of nudging it.
                cf_u = np.full(len(a_va), np.nan)
                for c_tr, c_te in StratifiedGroupKFold(
                        3, shuffle=True, random_state=seed).split(
                            Rva, wrong_va, groups[a_va]):
                    cf_u[c_te] = LogisticRegression(max_iter=3000).fit(
                        Rva[c_tr], wrong_va[c_tr]).predict_proba(Rva[c_te])[:, 1]
                corr_u = LogisticRegression(max_iter=3000).fit(Rva, wrong_va)
                w_te = corr_u.predict_proba(Rte)[:, 1]
                w_va_cf = np.nan_to_num(cf_u, nan=float(np.nanmean(cf_u)))
                b_va = p_va * (1 - w_va_cf) + (1 - p_va) * w_va_cf
                b_thr, _ = best_threshold(y[a_va], b_va)
                b_te = p_te * (1 - w_te) + (1 - p_te) * w_te
                w_te = corr.predict_proba(Rte)[:, 1]  # restore balanced w for the hard flip
                sc["blended_val"][te] = b_te
                oof["blended_val"][te] = (b_te >= b_thr).astype(int)
                oof["corrected_val"][te] = np.where(w_te >= best_t, 1 - pred_te, pred_te)
                sc["corrected_val"][te] = np.where(w_te >= best_t, 1 - p_te, p_te)
                flips[seed].append(float((w_te >= best_t).mean()))
            else:
                oof["corrected_val"][te], sc["corrected_val"][te] = pred_te, p_te
                oof["blended_val"][te], sc["blended_val"][te] = pred_te, p_te

            # --- corrector trained on the whole training fold via OOF SAPLMA ------
            oof_pred = np.full(len(tr), -1, dtype=int)
            for c_tr, c_te in StratifiedGroupKFold(
                    4, shuffle=True, random_state=seed).split(
                        np.zeros(len(tr)), strat[tr], groups[tr]):
                Ra, Rb = reduce(X[tr[c_tr]], [X[tr[c_tr]], X[tr[c_te]]], seed)
                mdl = LogisticRegression(C=0.3, max_iter=3000,
                                         class_weight="balanced").fit(Ra, y[tr][c_tr])
                oof_pred[c_te] = (mdl.predict_proba(Rb)[:, 1] >= thr).astype(int)
            wrong_tr = (oof_pred != y[tr]).astype(int)
            if len(np.unique(wrong_tr)) > 1:
                Rtr_full, Rte2 = reduce(X[tr], [X[tr], X[te]], seed)
                corr2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
                    Rtr_full, wrong_tr)
                w2_va = corr2.predict_proba(Rva)[:, 1]
                best_t2, best_m2 = 0.5, -2.0
                for t in np.quantile(w2_va, np.linspace(0.5, 0.99, 40)):
                    cand = np.where(w2_va >= t, 1 - pred_va, pred_va)
                    m = matthews_corrcoef(y[a_va], cand)
                    if m > best_m2:
                        best_m2, best_t2 = m, float(t)
                w2_te = corr2.predict_proba(Rte2)[:, 1]
                oof["corrected_crossfit"][te] = np.where(w2_te >= best_t2, 1 - pred_te, pred_te)
                sc["corrected_crossfit"][te] = np.where(w2_te >= best_t2, 1 - p_te, p_te)
            else:
                oof["corrected_crossfit"][te], sc["corrected_crossfit"][te] = pred_te, p_te

        for v in variants:
            for scope in ["pooled", "pooled_no_coqa"] + sorted(set(ds_arr)):
                m = (np.ones(len(y), bool) if scope == "pooled"
                     else (ds_arr != "coqa") if scope == "pooled_no_coqa" else (ds_arr == scope))
                if len(np.unique(y[m])) < 2:
                    continue
                acc[(v, scope)]["mcc"].append(float(matthews_corrcoef(y[m], oof[v][m])))
        print(f"  seed {seed} done", flush=True)
        del X

    scopes = ["pooled", "pooled_no_coqa", "triviaqa", "nq_open", "squad_v2", "coqa"]
    print(f"\n=== {args.run}: two-stage correction (MCC, {len(args.seeds)} seeds) ===")
    print(f"  {'variant':20s}" + "".join(f"{s[:13]:>15s}" for s in scopes))
    for v in variants:
        print(f"  {v:20s}" + "".join(
            f"{np.mean(acc[(v,s)]['mcc']):15.4f}" if (v, s) in acc else f"{'-':>15s}"
            for s in scopes))
    base = {s: np.mean(acc[("saplma_base", s)]["mcc"]) for s in scopes if ("saplma_base", s) in acc}
    print(f"\n  {'delta vs base':20s}" + "".join(
        f"{'':>15s}" for _ in scopes[:0]))
    for v in variants[1:]:
        print(f"  {v:20s}" + "".join(
            f"{np.mean(acc[(v,s)]['mcc']) - base[s]:+15.4f}" if (v, s) in acc else f"{'-':>15s}"
            for s in scopes))
    if flips:
        print(f"\n  fraction of test predictions flipped: "
              f"{np.mean([x for v in flips.values() for x in v]):.1%}")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/two_stage.json"),
               {f"{v}|{s}": round(float(np.mean(d['mcc'])), 4) for (v, s), d in acc.items()})


if __name__ == "__main__":
    main()
