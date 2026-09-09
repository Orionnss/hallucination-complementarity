"""One layer at a time: where in the network does the hallucination signal live?

Everything in this study reads SAPLMA at a single depth chosen as a fraction of model
height, and the fraction was inherited rather than measured. Stage 1 persists all L+1
layers exactly so that choice can be revisited, and this revisits it: the same
PCA+logreg probe fitted independently on each layer, under the study's protocol.

Three things it settles:

  where the peak is, and whether the inherited depth sits on it
  how sharp the peak is — a flat profile means the depth barely matters and every
    per-layer result elsewhere is robust; a sharp one means the opposite
  whether the profile is the same shape across models and datasets, which is the
    difference between a fact about transformers and a fact about Qwen

Layer 0 is the embedding output, before any transformer block, and acts as a built-in
control: it carries token identity and length but no computation, so whatever it scores
is the floor attributable to surface features rather than to the model's internal state.

Cost note: this is CPU-only. All layers are already extracted, so no generator runs.
Results accumulate per (run, seed) across invocations, since one seed exceeds a single
foreground call.

Usage: uv run python scripts/layer_ablation.py --runs main --seeds 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import time
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

C_GRID = (0.003, 0.03, 0.3, 3.0)
DIM = 128
#: The depth each run uses elsewhere, so the sweep can be read against it.
TUNED = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14,
         "llama3.2-3b-base": 14, "gemma3-12b-pt": 29}


def load_all_layers(cfg, ids):
    """[N, L+1, d] — every layer, loaded once and sliced per depth.

    Kept in float32, not float16. Gemma 3's residual stream reaches ~6.7e4 where float16
    tops out at 65,504, so a float16 buffer silently turns its largest activations into
    inf and every downstream fit fails or, worse, succeeds on corrupted data. Stage 1's
    own extractor guards the same hazard (hidden.py, FP16_SAFE_MAX); this is the memory
    cost of not repeating that mistake — about 7 GB for the largest run.
    """
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


def fit_eval(A0, B0, y, tr, i_tr, i_va, seed):
    s = StandardScaler().fit(A0)
    A, B = s.transform(A0), s.transform(B0)
    k = min(DIM, A.shape[1], len(tr) - 1)
    p = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(A)
    A, B = p.transform(A), p.transform(B)
    best_c, best_a = C_GRID[0], -1.0
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(
            A[i_tr], y[tr][i_tr])
        a = roc_auc_score(y[tr][i_va], m.predict_proba(A[i_va])[:, 1])
        if a > best_a:
            best_a, best_c = float(a), C
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(
        A[i_tr], y[tr][i_tr])
    thr, _ = best_threshold(y[tr][i_va], m.predict_proba(A[i_va])[:, 1])
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(A, y[tr])
    sc = m.predict_proba(B)[:, 1]
    return sc, (sc >= thr).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--stride", type=int, default=1, help="sweep every Nth layer")
    args = ap.parse_args()

    dest = Path("runs/layer_ablation.json")
    out = json.loads(dest.read_text()) if dest.exists() else {}

    for run in args.runs:
        cfg = Config(); cfg.run_id = run
        prev = out.get(run, {"raw": {}, "seeds": []})
        for seed in args.seeds:
            if seed in prev["seeds"]:
                print(f"  {run} seed {seed} already done, skipping"); continue
            t0 = time.perf_counter()
            sd = load_seed(cfg, seed)
            ids, y = list(sd["item_ids"]), sd["y"]
            groups, ds_arr = sd["groups"], sd["dataset"].astype(str)
            strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
            X = load_all_layers(cfg, ids)
            n_layers = X.shape[1]
            layers = list(range(0, n_layers, args.stride))
            print(f"  {run} seed {seed}: {n_layers} layers, sweeping {len(layers)}",
                  flush=True)

            folds = list(_folds(sd, seed, cfg.n_folds))
            splits = []
            for tr, te in folds:
                inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
                splits.append(next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr])))

            for li, L in enumerate(layers):
                Xl = X[:, L, :]
                sc = np.full(len(y), np.nan); pr = np.full(len(y), -1, dtype=int)
                for (tr, te), (i_tr, i_va) in zip(folds, splits):
                    a, b = fit_eval(Xl[tr], Xl[te], y, tr, i_tr, i_va, seed)
                    sc[te], pr[te] = a, b
                for scope in ["pooled"] + sorted(set(ds_arr)):
                    m = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                    if len(np.unique(y[m])) < 2:
                        continue
                    for k, v in (("auroc", roc_auc_score(y[m], sc[m])),
                                 ("mcc", matthews_corrcoef(y[m], pr[m]))):
                        prev["raw"].setdefault(f"{L}|{scope}|{k}", []).append(float(v))
                if (li + 1) % 8 == 0:
                    r = (time.perf_counter() - t0) / (li + 1)
                    print(f"    layer {L}/{n_layers - 1}  {r:.1f}s/layer  "
                          f"eta {r * (len(layers) - li - 1) / 60:.1f} min", flush=True)
                del Xl
            prev["seeds"].append(seed)
            prev["n_layers"] = int(n_layers)
            out[run] = prev
            write_json(dest, out)
            print(f"  {run} seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min",
                  flush=True)
            del X

    # report
    for run in out:
        r = out[run]["raw"]; nl = out[run].get("n_layers", 0)
        Ls = sorted({int(k.split("|")[0]) for k in r})
        if not Ls:
            continue
        print(f"\n{'=' * 78}\n  {run}: AUROC by probe layer "
              f"({len(out[run]['seeds'])} seeds, {nl} layers)\n{'=' * 78}")
        au = {L: np.mean(r[f"{L}|pooled|auroc"]) for L in Ls if f"{L}|pooled|auroc" in r}
        mc = {L: np.mean(r[f"{L}|pooled|mcc"]) for L in Ls if f"{L}|pooled|mcc" in r}
        best = max(au, key=au.get)
        tuned = TUNED.get(run)
        for L in Ls:
            if L not in au:
                continue
            bar = "#" * int(round((au[L] - 0.5) * 80))
            tag = ""
            if L == best:
                tag += "  <- peak"
            if tuned is not None and L == tuned:
                tag += "  <- depth used elsewhere"
            print(f"    L{L:<3d} {au[L]:.4f}  mcc {mc.get(L, float('nan')):.4f}  "
                  f"{bar}{tag}")
        print(f"\n    peak L{best} = {au[best]:.4f}")
        if tuned in au:
            print(f"    inherited depth L{tuned} = {au[tuned]:.4f} "
                  f"({au[tuned] - au[best]:+.4f} vs peak)")
        top = sorted(au.values(), reverse=True)
        within = sum(1 for v in au.values() if v >= au[best] - 0.005)
        print(f"    layers within 0.005 AUROC of the peak: {within} of {len(au)}"
              f"   (embedding L0 = {au.get(0, float('nan')):.4f})")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
