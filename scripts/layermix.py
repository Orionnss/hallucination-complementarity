"""LayerMix (arXiv 2608.28930): score every layer by CV, keep the top-K, average their probes.

Three stages, following the paper:

  1  score each layer by stratified CV AUROC of an L2 logistic regression on the full
     hidden state — this replaces oracle layer selection, using training data only
  2  keep the top-K layers (K=5 default); in practice they form a contiguous band
  3  fit one probe per selected layer and average their *predicted probabilities*

Three of the paper's design choices are deliberate and are kept:

  full-dimensional probes with strong regularisation (C=0.001) rather than PCA, so no
    information is lost to a projection
  averaging predictions rather than concatenating features, so width does not scale with K
  no learned aggregation weights, since the selected layers are high quality by
    construction

The comparison this has to survive is not "does it beat a fixed heuristic" but "does it
beat the layer we would have picked anyway". Three references are therefore reported:

  saplma_pcalr   this study's configuration: one inherited layer, PCA-128 + tuned logreg
  best_single    the single best layer chosen by the same CV scoring — what LayerMix
                 buys over its own stage 1, which is the honest measure of the averaging
  oracle_single  the best layer chosen on the *test* fold. Unattainable by construction;
                 it bounds what any layer-selection scheme could reach.

Stage 1 scoring runs inside each training fold and never sees the test rows, so the
selected band is not chosen on the data it is scored against.

Usage: uv run python scripts/layermix.py --runs main --seeds 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import time
from collections import Counter, defaultdict
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

TUNED = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14,
         "llama3.2-3b-base": 14, "gemma3-12b-pt": 29}
LM_C = 0.001          # the paper's regularisation for the full-dimensional probes
C_GRID = (0.003, 0.03, 0.3, 3.0)
DIM = 128


def load_all_layers(cfg, ids):
    """[N, L+1, d] in float32 — Gemma's activations exceed the float16 range."""
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        a = load_features(cfg.stage_dir("stage1_extract", ds), "saplma", sub)
        if out is None:
            out = np.zeros((len(ids), a.shape[1], a.shape[2]), np.float32)
        out[[pos[i] for i in sub]] = a.astype(np.float32)
        del a
    if not np.isfinite(out).all():
        raise ValueError("non-finite values in the stored SAPLMA block")
    return out


def lm_probe(Xtr, ytr, Xte):
    """The paper's probe: standardise, then L2 logistic regression at C=0.001."""
    s = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=LM_C, max_iter=3000, class_weight="balanced").fit(
        s.transform(Xtr), ytr)
    return m.predict_proba(s.transform(Xte))[:, 1]


def pcalr_probe(Xtr, ytr, Xte, i_tr, i_va, seed):
    """This study's configuration, as the reference: PCA-128 + logreg with C tuned.

    Returns (test scores, inner-validation scores). The second is what the threshold has
    to be read off: calibrating this variant on some *other* classifier's inner-val
    scores silently mixes two score distributions and corrupts only the MCC column,
    where it is invisible next to an intact AUROC.
    """
    s = StandardScaler().fit(Xtr)
    A, B = s.transform(Xtr), s.transform(Xte)
    p = PCA(n_components=min(DIM, A.shape[1], len(A) - 1), svd_solver="randomized",
            random_state=seed).fit(A)
    A, B = p.transform(A), p.transform(B)
    best_c, best_a, va = C_GRID[0], -1.0, None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(
            A[i_tr], ytr[i_tr])
        q = m.predict_proba(A[i_va])[:, 1]
        a = roc_auc_score(ytr[i_va], q)
        if a > best_a:
            best_a, best_c, va = float(a), C, q
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(A, ytr)
    return m.predict_proba(B)[:, 1], va


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--score-folds", type=int, default=3,
                    help="CV folds for stage-1 layer scoring (paper uses 5; 3 keeps the "
                         "sweep affordable over every layer of every model)")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    dest = Path("runs/layermix.json")
    out = json.loads(dest.read_text()) if dest.exists() else {}

    for run in args.runs:
        cfg = Config(); cfg.run_id = run
        acc = defaultdict(lambda: defaultdict(list))
        picked = Counter(); bands = []

        for seed in args.seeds:
            t0 = time.perf_counter()
            sd = load_seed(cfg, seed)
            ids, y = list(sd["item_ids"]), sd["y"]
            groups, ds_arr = sd["groups"].astype(str), sd["dataset"].astype(str)
            strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
            X = load_all_layers(cfg, ids)
            nL = X.shape[1]
            cand = list(range(0, nL, args.stride))
            tuned_L = min(TUNED.get(run, 24), nL - 1)

            names = ["layermix", "best_single", "oracle_single", "saplma_pcalr"]
            sc = {v: np.full(len(y), np.nan) for v in names}
            pr = {v: np.full(len(y), -1, dtype=int) for v in names}

            for tr, te in _folds(sd, seed, cfg.n_folds):
                inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
                i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

                # --- stage 1: score every layer by CV *inside the training fold* ---
                scores = {}
                skf = StratifiedGroupKFold(args.score_folds, shuffle=True,
                                           random_state=seed)
                sp = list(skf.split(np.zeros(len(tr)), strat[tr], groups[tr]))
                for L in cand:
                    Xl = X[tr, L, :]
                    aucs = []
                    for a_i, b_i in sp:
                        p = lm_probe(Xl[a_i], y[tr][a_i], Xl[b_i])
                        aucs.append(roc_auc_score(y[tr][b_i], p))
                    scores[L] = float(np.mean(aucs))

                # --- stage 2: top-K ---
                sel = sorted(scores, key=scores.get, reverse=True)[:args.K]
                picked.update(sel); bands.append(sorted(sel))
                best_L = sel[0]

                # --- stage 3: one probe per selected layer, average the probabilities ---
                ps = [lm_probe(X[tr, L, :], y[tr], X[te, L, :]) for L in sel]
                mix = np.mean(ps, axis=0)
                sc["layermix"][te] = mix
                sc["best_single"][te] = ps[sel.index(best_L)]

                # oracle: the layer that turns out best on the test fold itself. Only the
                # layer choice is oracular; the probe is still trained on the training
                # fold. It bounds what any layer-selection scheme could reach, and the
                # selection criterion is AUROC, so it is an oracle for ranking and not
                # for any thresholded metric.
                best_o, best_v, best_oL = None, -1.0, None
                for L in cand:
                    p = lm_probe(X[tr, L, :], y[tr], X[te, L, :])
                    v = roc_auc_score(y[te], p)
                    if v > best_v:
                        best_v, best_o, best_oL = v, p, L
                sc["oracle_single"][te] = best_o

                sc["saplma_pcalr"][te], pcalr_va = pcalr_probe(
                    X[tr, tuned_L, :], y[tr], X[te, tuned_L, :], i_tr, i_va, seed)

                # Thresholds from the inner split of the same training fold. Each variant
                # must be calibrated on the inner-val scores of *its own* estimator --
                # same layer(s), same classifier -- or the cut-point is read off a
                # different score distribution than the one it is applied to.
                for v in names:
                    if v == "layermix":
                        tr_p = np.mean([lm_probe(X[tr][i_tr, L, :], y[tr][i_tr],
                                                 X[tr][i_va, L, :]) for L in sel], axis=0)
                    elif v == "best_single":
                        tr_p = lm_probe(X[tr][i_tr, best_L, :], y[tr][i_tr],
                                        X[tr][i_va, best_L, :])
                    elif v == "saplma_pcalr":
                        tr_p = pcalr_va
                    else:                                    # oracle_single
                        tr_p = lm_probe(X[tr][i_tr, best_oL, :], y[tr][i_tr],
                                        X[tr][i_va, best_oL, :])
                    thr, _ = best_threshold(y[tr][i_va], tr_p)
                    pr[v][te] = (sc[v][te] >= thr).astype(int)

            for v in names:
                for scope in ["pooled"] + sorted(set(ds_arr)):
                    m = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                    if len(np.unique(y[m])) < 2:
                        continue
                    acc[(v, scope)]["auroc"].append(float(roc_auc_score(y[m], sc[v][m])))
                    acc[(v, scope)]["mcc"].append(float(matthews_corrcoef(y[m], pr[v][m])))
            print(f"  {run} seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min",
                  flush=True)
            del X

        scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
        print(f"\n=== {run}: LayerMix K={args.K} ({len(args.seeds)} seeds) ===")
        for metric in ("auroc", "mcc"):
            print(f"\n  {metric.upper()}")
            print(f"    {'variant':16s}" + "".join(f"{s[:11]:>13s}" for s in scopes))
            for v in ["saplma_pcalr", "best_single", "layermix", "oracle_single"]:
                print(f"    {v:16s}" + "".join(
                    f"{np.mean(acc[(v, s)][metric]):13.4f}" if (v, s) in acc
                    else f"{'-':>13s}" for s in scopes))
            b = np.mean(acc[("best_single", "pooled")][metric])
            lm = np.mean(acc[("layermix", "pooled")][metric])
            ref = np.mean(acc[("saplma_pcalr", "pooled")][metric])
            print(f"    -> layermix - best_single {lm - b:+.4f}   "
                  f"layermix - saplma_pcalr {lm - ref:+.4f}")
        print(f"\n  layers selected (frequency over {len(bands)} folds): "
              f"{[f'L{L}:{n}' for L, n in picked.most_common(8)]}")
        # Contiguity is measured in units of the sweep stride: at stride 4 consecutive
        # candidates are 4 apart, so a band of adjacent candidates is not a run of
        # consecutive integers.
        st = args.stride
        contig = sum(1 for b in bands if max(b) - min(b) == (len(b) - 1) * st)
        print(f"  contiguous bands (stride {st}): {contig}/{len(bands)}")

        # Per-seed values are retained, not just their mean: the deltas here are small
        # (+.001 to +.017) and a mean alone cannot support a paired test against the
        # baseline, which is the only thing that makes a 6/6 win count as evidence.
        out[run] = {"K": args.K, "seeds": args.seeds,
                    "results": {f"{v}|{s}": {m: round(float(np.mean(d[m])), 4) for m in d}
                                for (v, s), d in acc.items()},
                    "per_seed": {f"{v}|{s}": {m: [round(float(x), 5) for x in d[m]]
                                              for m in d}
                                 for (v, s), d in acc.items()},
                    "layer_frequency": {str(L): n for L, n in picked.most_common()},
                    "bands": bands}
        write_json(dest, out)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
