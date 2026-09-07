"""Give SAPLMA's MLP the same tuning budget the linear probes got.

PCA+logreg beat the published SAPLMA MLP by +.024 to +.065 MCC, and that gap is what the
"combining barely helps" conclusion rests on. But the comparison was asymmetric: C and the
threshold were tuned on inner splits for logreg, while the MLP ran at its published
(256,128,64) with default regularisation. Some unknown share of the gap could be "one was
tuned and the other was not" rather than anything about linear vs non-linear.

This closes that gap. Same folds, same threshold rule, same fixed probe layer, and a grid
over the things the published probe fixes: architecture, L2, and whether the MLP sees the
raw 5120-dim state or the same PCA-128 projection the linear probes get. Config is chosen
on the inner split by AUROC, never on the test fold.

One asymmetry cannot be removed: sklearn's MLPClassifier supports neither class_weight nor
sample_weight, while the linear probes ran class_weight="balanced". Tuning the decision
threshold on the inner split absorbs most of that, since the base rates here are 30-48%
rather than extreme, but the MLP is still handicapped on the imbalanced slices (CoQA at
13-17%). Read the MLP's CoQA numbers with that in mind.

If the tuned MLP closes the gap, the "linear beats MLP" claim is a tuning artefact and the
baseline-strength finding needs restating. If it does not, the claim survives a fair test.

Usage: uv run python scripts/saplma_mlp_tuned.py --runs main
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}

#: (input, hidden_layer_sizes, alpha). The first row is the published probe, so the
#: search can only improve on it — it is always in the grid, never tuned away.
GRID = [
    ("raw", (256, 128, 64), 1e-4),   # published SAPLMA
    ("raw", (256, 128, 64), 1e-2),
    ("raw", (256, 128, 64), 1e-1),
    ("raw", (128,), 1e-2),
    ("pca", (256, 128, 64), 1e-4),
    ("pca", (256, 128, 64), 1e-2),
    ("pca", (128, 64), 1e-2),
    ("pca", (64,), 1e-1),
]


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
        del a
    return out


def run_one(run: str, seeds: list[int], grid) -> tuple[dict, list[str]]:
    cfg = Config(); cfg.run_id = run
    acc = defaultdict(lambda: defaultdict(list))
    picks = []

    for seed in seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"]
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = load_saplma(cfg, ids, LAYER.get(run, 24))

        score = np.full(len(y), np.nan)
        pred = np.full(len(y), -1, dtype=int)
        for fold, (tr, te) in enumerate(_folds(sd, seed, cfg.n_folds)):
            s = StandardScaler().fit(X[tr])
            raw_tr, raw_te = s.transform(X[tr]), s.transform(X[te])
            p = PCA(n_components=min(128, raw_tr.shape[1], len(tr) - 1),
                    svd_solver="randomized", random_state=seed).fit(raw_tr)
            rep = {"raw": (raw_tr, raw_te), "pca": (p.transform(raw_tr), p.transform(raw_te))}

            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

            best, best_a = grid[0], -1.0
            for cfg_ in grid:
                kind, hidden, alpha = cfg_
                A = rep[kind][0]
                m = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=600,
                                  early_stopping=True, n_iter_no_change=20,
                                  random_state=seed).fit(A[i_tr], y[tr][i_tr])
                auc = roc_auc_score(y[tr][i_va], m.predict_proba(A[i_va])[:, 1])
                if auc > best_a:
                    best_a, best = float(auc), cfg_
            picks.append(str(best))

            kind, hidden, alpha = best
            A, B = rep[kind]
            m = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=600,
                              early_stopping=True, n_iter_no_change=20,
                              random_state=seed).fit(A[i_tr], y[tr][i_tr])
            thr, _ = best_threshold(y[tr][i_va], m.predict_proba(A[i_va])[:, 1])
            m = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=600,
                              early_stopping=True, n_iter_no_change=20,
                              random_state=seed).fit(A, y[tr])
            score[te] = m.predict_proba(B)[:, 1]
            pred[te] = (score[te] >= thr).astype(int)
            print(f"    fold {fold + 1}/{cfg.n_folds}  picked {best}", flush=True)

        for scope in ["pooled"] + sorted(set(ds_arr)):
            msk = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
            if len(np.unique(y[msk])) < 2:
                continue
            d = acc[scope]
            d["accuracy"].append(float(accuracy_score(y[msk], pred[msk])))
            d["balanced_accuracy"].append(float(balanced_accuracy_score(y[msk], pred[msk])))
            d["mcc"].append(float(matthews_corrcoef(y[msk], pred[msk])))
            d["f1"].append(float(f1_score(y[msk], pred[msk], zero_division=0)))
            d["auroc"].append(float(roc_auc_score(y[msk], score[msk])))
        print(f"  {run} seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        del X

    # Raw per-seed lists, not aggregates: this run is too slow for one foreground call,
    # so seeds are accumulated across invocations and averaged only at report time.
    return ({sc: dict(d) for sc, d in acc.items()}, picks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    dest = Path("runs/saplma_mlp_tuned.json")
    out = json.loads(dest.read_text()) if dest.exists() else {}
    for run in args.runs:
        res, picks = run_one(run, args.seeds, GRID)
        prev = out.get(run, {"raw": {}, "seeds": [], "configs_selected_per_fold": []})
        for sc, d in res.items():
            for m, v in d.items():
                prev["raw"].setdefault(sc, {}).setdefault(m, []).extend(v)
        prev["seeds"] = sorted(set(prev["seeds"]) | set(args.seeds))
        prev["configs_selected_per_fold"] += picks
        out[run] = prev
    write_json(dest, out)

    def agg(run, sc, m):
        return float(np.mean(out[run]["raw"][sc][m]))

    fm = json.loads(Path("runs/full_metrics.json").read_text())
    lr = json.loads(Path("runs/saplma_pcalr_metrics.json").read_text())
    scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
    for metric in ("mcc", "auroc"):
        print(f"\n### {metric.upper()}")
        print(f"  {'variant':24s}" + "".join(f"{s[:11]:>13s}" for s in scopes))
        for run in args.runs:
            print(f"  --- {run}")
            print(f"  {'saplma MLP (published)':24s}" + "".join(
                f"{fm[run][s]['saplma'][metric]['mean']:13.4f}" for s in scopes))
            print(f"  {'saplma MLP (tuned)':24s}" + "".join(
                f"{agg(run, s, metric):13.4f}" for s in scopes)
                + f"   [{len(out[run]['seeds'])} seeds]")
            print(f"  {'saplma PCA+logreg':24s}" + "".join(
                f"{lr[run][s][metric]['mean']:13.4f}" for s in scopes))
            print(f"  {'  logreg - tuned MLP':24s}" + "".join(
                f"{lr[run][s][metric]['mean'] - agg(run, s, metric):+13.4f}"
                for s in scopes))
        from collections import Counter
        if metric == "mcc":
            for run in args.runs:
                print(f"\n  configs chosen ({run}): "
                      f"{dict(Counter(out[run]['configs_selected_per_fold']))}")
    print("\nwrote runs/saplma_mlp_tuned.json")


if __name__ == "__main__":
    main()
