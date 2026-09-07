"""Cascade: SAPLMA decides on everything, a specialist re-examines only the positives.

Unlike the bidirectional corrector in two_stage.py, stage 2 here is a one-way filter. It
sees only the answers stage 1 flagged, and can demote them to NOT — it never rescues a
false negative. That matches the error profile: outside CoQA, 57-64% of SAPLMA's errors
are false alarms, so the flagged set is where the recoverable mass is.

The specialist is trained on the **validation** split (its predicted-positive subset
only), so it learns a boundary local to the flagged population rather than a global one,
and never sees test. Its own threshold is chosen by cross-fitting inside that subset, and
scored by end-to-end MCC over the whole validation fold — the cascade is tuned as one
system, not as two independent classifiers.

Three feature sets for the specialist, all with SAPLMA's score appended since it is free
at deployment time:

  score    SAPLMA's score alone. THE BASELINE THAT MATTERS: with a monotone stage 2, the
           cascade is algebraically identical to raising SAPLMA's single threshold, which
           was already tuned for max MCC. It should not gain. Anything the feature
           variants win over this row is what specialisation actually buys.
  saplma   SAPLMA's hidden states, PCA 128
  union    all five blocks, PCA 128 each - the complementarity question, asked where the
           errors actually are rather than over the whole dataset

Usage: uv run python scripts/cascade.py --run main
"""

from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

BLOCKS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline"]
LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}
C_GRID = (0.003, 0.03, 0.3, 3.0)
SPECIALISTS = ["score", "saplma", "union"]


def load_block(cfg, ids, name, layer):
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        a = load_features(cfg.stage_dir("stage1_extract", ds), name, sub)
        if name == "saplma":
            a = a[:, min(layer, a.shape[1] - 1), :]
        a = a.reshape(len(sub), -1).astype(np.float32)
        if out is None:
            out = np.zeros((len(ids), a.shape[1]), np.float32)
        out[[pos[i] for i in sub]] = a
        del a
    return out


def reduce(block, fit_rows, apply_rows, seed, dim=128):
    s = StandardScaler().fit(block[fit_rows])
    a = s.transform(block[fit_rows])
    outs = []
    k = min(dim, a.shape[1], len(fit_rows) - 1)
    if k < a.shape[1]:
        p = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(a)
        for r in apply_rows:
            outs.append(p.transform(s.transform(block[r])))
    else:
        for r in apply_rows:
            outs.append(s.transform(block[r]))
    return outs


def tuned(Xtr, ytr, seed):
    """C by 3-fold CV on the specialist's own training subset (it is small)."""
    if len(np.unique(ytr)) < 2 or len(ytr) < 30:
        return None
    best_c, best_a = 0.3, -1.0
    for C in C_GRID:
        sc = np.full(len(ytr), np.nan)
        try:
            for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(
                    Xtr, ytr, np.arange(len(ytr))):
                sc[b] = LogisticRegression(C=C, max_iter=3000,
                                           class_weight="balanced").fit(
                    Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
            auc = roc_auc_score(ytr, sc)
        except ValueError:
            continue
        if auc > best_a:
            best_a, best_c = float(auc), C
    return LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    layer = LAYER.get(args.run, 24)
    variants = ["saplma_base"] + [f"cascade_{s}" for s in SPECIALISTS]
    acc = defaultdict(lambda: defaultdict(list))
    demoted = defaultdict(list)

    for seed in args.seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"].astype(str), sd["dataset"].astype(str)
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        print(f"seed {seed}: loading {len(BLOCKS)} blocks ...", flush=True)
        blocks = {b: load_block(cfg, ids, b, layer) for b in BLOCKS}

        pred = {v: np.zeros(len(y), dtype=int) for v in variants}

        for fold, (tr, te) in enumerate(_folds(sd, seed, cfg.n_folds)):
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
            a_tr, a_va = tr[i_tr], tr[i_va]

            # --- stage 1: SAPLMA over the whole set ------------------------------
            Rtr, Rva, Rte = reduce(blocks["saplma"], a_tr, [a_tr, a_va, te], seed)
            best_c, best_a = C_GRID[0], -1.0
            for C in C_GRID:
                m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(
                    Rtr, y[a_tr])
                auc = roc_auc_score(y[a_va], m.predict_proba(Rva)[:, 1])
                if auc > best_a:
                    best_a, best_c = float(auc), C
            s1 = LogisticRegression(C=best_c, max_iter=3000,
                                    class_weight="balanced").fit(Rtr, y[a_tr])
            p_va, p_te = s1.predict_proba(Rva)[:, 1], s1.predict_proba(Rte)[:, 1]
            thr, _ = best_threshold(y[a_va], p_va)
            pos_va, pos_te = p_va >= thr, p_te >= thr
            pred["saplma_base"][te] = pos_te.astype(int)

            # --- stage 2: specialist on the flagged subset of validation ---------
            v_idx, t_idx = a_va[pos_va], te[pos_te]
            y_v = y[v_idx]
            for spec in SPECIALISTS:
                out = np.zeros(len(te), dtype=int)
                out[pos_te] = 1  # default: keep stage 1's call
                if len(v_idx) >= 40 and len(np.unique(y_v)) > 1 and len(t_idx):
                    if spec == "score":
                        Xv = p_va[pos_va][:, None]
                        Xt = p_te[pos_te][:, None]
                    else:
                        use = BLOCKS if spec == "union" else ["saplma"]
                        # Fit the reducer on the flagged validation subset itself, so the
                        # projection describes the population the specialist works on.
                        parts = [reduce(blocks[b], v_idx, [v_idx, t_idx], seed) for b in use]
                        Xv = np.hstack([q[0] for q in parts] + [p_va[pos_va][:, None]])
                        Xt = np.hstack([q[1] for q in parts] + [p_te[pos_te][:, None]])
                    mdl = tuned(Xv, y_v, seed)
                    if mdl is not None:
                        # Cross-fit inside the flagged subset for an honest threshold, and
                        # score it by end-to-end MCC over the whole validation fold.
                        cf = np.full(len(v_idx), np.nan)
                        for a, b in StratifiedGroupKFold(
                                3, shuffle=True, random_state=seed).split(
                                    Xv, y_v, groups[v_idx]):
                            sub = tuned(Xv[a], y_v[a], seed)
                            if sub is not None:
                                cf[b] = sub.predict_proba(Xv[b])[:, 1]
                        cf = np.nan_to_num(cf, nan=1.0)
                        best_t, best_m = -1.0, -2.0
                        for t in np.unique(np.quantile(cf, np.linspace(0.0, 0.6, 40))):
                            cand = pos_va.astype(int).copy()
                            cand[np.flatnonzero(pos_va)] = (cf >= t).astype(int)
                            m = matthews_corrcoef(y[a_va], cand)
                            if m > best_m:
                                best_m, best_t = m, float(t)
                        keep = mdl.predict_proba(Xt)[:, 1] >= best_t
                        out[pos_te] = keep.astype(int)
                        demoted[spec].append(float(1 - keep.mean()))
                pred[f"cascade_{spec}"][te] = out
            print(f"  fold {fold + 1}/{cfg.n_folds}", flush=True)

        for v in variants:
            for scope in ["pooled", "pooled_no_coqa"] + sorted(set(ds_arr)):
                m = (np.ones(len(y), bool) if scope == "pooled"
                     else (ds_arr != "coqa") if scope == "pooled_no_coqa"
                     else (ds_arr == scope))
                if len(np.unique(y[m])) < 2:
                    continue
                acc[(v, scope)]["mcc"].append(float(matthews_corrcoef(y[m], pred[v][m])))
                acc[(v, scope)]["prec"].append(
                    float(precision_score(y[m], pred[v][m], zero_division=0)))
                acc[(v, scope)]["rec"].append(
                    float(recall_score(y[m], pred[v][m], zero_division=0)))
        print(f"  seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del blocks

    scopes = ["pooled", "pooled_no_coqa", "triviaqa", "nq_open", "squad_v2", "coqa"]
    print(f"\n=== {args.run}: cascade — stage 2 re-examines only stage 1's positives ===")
    for metric, lab in (("mcc", "MCC"), ("prec", "precision"), ("rec", "recall")):
        print(f"\n  {lab}")
        print(f"    {'variant':18s}" + "".join(f"{s[:13]:>15s}" for s in scopes))
        for v in variants:
            print(f"    {v:18s}" + "".join(
                f"{np.mean(acc[(v,s)][metric]):15.4f}" if (v, s) in acc else f"{'-':>15s}"
                for s in scopes))
    base = {s: np.mean(acc[("saplma_base", s)]["mcc"]) for s in scopes
            if ("saplma_base", s) in acc}
    print(f"\n  delta MCC vs stage 1 alone")
    for v in variants[1:]:
        print(f"    {v:18s}" + "".join(
            f"{np.mean(acc[(v,s)]['mcc']) - base[s]:+15.4f}" if (v, s) in acc
            else f"{'-':>15s}" for s in scopes))
    print("\n  share of stage-1 positives demoted by stage 2:")
    for s in SPECIALISTS:
        if demoted[s]:
            print(f"    {s:10s}{np.mean(demoted[s]):.1%}")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/cascade.json"),
               {f"{v}|{s}": {m: round(float(np.mean(d[m])), 4) for m in d}
                for (v, s), d in acc.items()})


if __name__ == "__main__":
    main()
