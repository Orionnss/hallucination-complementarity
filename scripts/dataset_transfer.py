"""Cross-dataset transfer for SAPLMA features under PCA + logistic regression.

Everything else in this study trains one probe on the four pooled datasets and reports
per-dataset slices of it (DESIGN.md "Evaluation"), which answers "does one shared
detector work everywhere?" and never "does a TriviaQA-fitted detector work on CoQA?".
This runs the two protocols that second question needs:

  single         train on ONE dataset, evaluate on all four -> a 4x4 source-by-target
                 matrix. The diagonal is the in-domain reference.
  lodo           train on the THREE others, evaluate on the held-out dataset.
  lodo_matched   the same three sources subsampled to the single-source training size,
                 balanced across them. Without it, lodo beating single confounds "more
                 diverse training data" with "three times as much of it".
  pooled         train on all four including the target. The in-distribution ceiling,
                 and the number the rest of the study reports.

Every arm is fitted on the SAME outer folds and evaluated on the SAME held-out rows, so
a row of the matrix is a like-for-like comparison. In particular the diagonal is fitted
on one dataset's training-fold rows only -- not on the whole dataset -- so it carries the
same training-set size as the off-diagonal cells and the drop across a row is the cost of
changing domain rather than the cost of changing n.

Model selection never sees the target: C (by AUROC) and the decision threshold (by MCC)
are both chosen on a validation split drawn from the SOURCE datasets only. Base rates
differ by a factor of three across datasets (coqa 0.16, nq_open 0.47 on main), so a
source-fitted threshold is expected to transfer badly on its own. To separate that from a
genuine loss of ranking ability, each cell reports three things:

  auroc      threshold-free -- did the probe keep its ability to rank at all
  mcc        at the source-fitted threshold -- what you could actually deploy blind
  mcc_oracle at a threshold refitted on the target's own out-of-fold scores -- an oracle,
             so (mcc_oracle - mcc) is the part of the damage that recalibration alone
             would repair, and mcc_oracle itself is what a handful of target labels buys

Folds are stage 3's exact outer folds, so these numbers sit alongside
saplma_pcalr_metrics.json and full_metrics.json without re-deriving anything.

Usage: uv run python scripts/dataset_transfer.py --run main --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

# Same probe layer per generator as saplma_pcalr_metrics.py, so the transfer numbers are
# comparable with the in-distribution ones already reported.
LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14,
         "llama3.2-3b-base": 14, "gemma3-12b-pt": 29}
C_GRID = (0.003, 0.03, 0.3, 3.0)
DIM = 128
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b",
        "llama3.2-3b-base", "gemma3-12b-pt"]


def load_saplma_union(cfg, all_ids: list[str], layer: int) -> dict[str, np.ndarray]:
    """One layer's SAPLMA features for the union of item ids over every seed.

    The shards hold every layer for every item and total ~19 GB per generator, so they
    are scanned once here and indexed per seed rather than re-read five times.
    """
    out: dict[str, np.ndarray] = {}
    for ds in cfg.datasets:
        sub = [i for i in all_ids if i.startswith(ds + ":")]
        if not sub:
            continue
        a = load_features(cfg.stage_dir("stage1_extract", ds), "saplma", sub)
        a = a[:, min(layer, a.shape[1] - 1), :].astype(np.float32)
        for k, i in enumerate(sub):
            out[i] = a[k]
        del a
    return out


class Probe:
    """Scaler + PCA + logistic regression, C and threshold from an inner split.

    Identical to the probe used for the pooled numbers, so the only thing this script
    varies is which rows the probe is allowed to see.
    """

    def fit(self, X, y, i_tr, i_va, seed):
        self.s = StandardScaler().fit(X[i_tr])
        A = self.s.transform(X[i_tr])
        k = min(DIM, A.shape[1], len(i_tr) - 1)
        self.p = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(A)
        A = self.p.transform(A)
        V = self.p.transform(self.s.transform(X[i_va]))

        best_c, best_a = C_GRID[0], -1.0
        for C in C_GRID:
            m = LogisticRegression(C=C, max_iter=3000,
                                   class_weight="balanced").fit(A, y[i_tr])
            a = (roc_auc_score(y[i_va], m.predict_proba(V)[:, 1])
                 if len(np.unique(y[i_va])) > 1 else 0.5)
            if a > best_a:
                best_a, best_c = float(a), C
        self.C = best_c

        m = LogisticRegression(C=best_c, max_iter=3000,
                               class_weight="balanced").fit(A, y[i_tr])
        self.thr, _ = best_threshold(y[i_va], m.predict_proba(V)[:, 1])
        # Refit on train+val once the threshold is set, as elsewhere in this study.
        allX = np.vstack([X[i_tr], X[i_va]])
        ally = np.concatenate([y[i_tr], y[i_va]])
        self.m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(
            self.p.transform(self.s.transform(allX)), ally)
        return self

    def score(self, X):
        return self.m.predict_proba(self.p.transform(self.s.transform(X)))[:, 1]


def balanced_subsample(idx, ds_arr, n_total, rng):
    """Draw n_total rows from `idx`, spread as evenly as the sources allow.

    Sources are drawn to an equal quota rather than proportionally, so lodo_matched
    differs from lodo only in size and not also in source mix.
    """
    names = sorted(set(ds_arr[idx]))
    quota, out = n_total // len(names), []
    for d in names:
        pool = idx[ds_arr[idx] == d]
        out.append(rng.choice(pool, size=min(quota, len(pool)), replace=False))
    return np.sort(np.concatenate(out))


def cell_metrics(y, score, pred) -> dict:
    """Metrics for one (arm, target) cell, given target-only out-of-fold predictions."""
    if len(np.unique(y)) < 2:
        return {}
    thr_t, mcc_o = best_threshold(y, score)
    return {"auroc": float(roc_auc_score(y, score)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "mcc_oracle": float(mcc_o),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "pos_rate": float(y.mean())}


def run_one(run: str, seeds: list[int]) -> dict:
    cfg = Config(); cfg.run_id = run
    layer = LAYER.get(run, 24)

    seed_data = {}
    for seed in seeds:
        sd = load_seed(cfg, seed)
        if sd is None:
            print(f"  {run} seed {seed}: no stage-3 predictions, skipped", flush=True)
            continue
        seed_data[seed] = sd
    if not seed_data:
        return {}

    t0 = time.perf_counter()
    union = sorted({i for sd in seed_data.values() for i in sd["item_ids"].tolist()})
    feats = load_saplma_union(cfg, union, layer)
    print(f"  {run}: {len(union)} items, dim {len(next(iter(feats.values())))}, "
          f"features read in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    acc = defaultdict(lambda: defaultdict(list))
    n_train = defaultdict(list)

    for seed, sd in seed_data.items():
        t0 = time.perf_counter()
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"].astype(str)
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        X = np.stack([feats[i] for i in ids])
        dsets = sorted(set(ds_arr))
        rng = np.random.default_rng(seed)

        arms = ([f"single:{d}" for d in dsets] + [f"lodo:{d}" for d in dsets]
                + [f"lodo_matched:{d}" for d in dsets] + ["pooled"])
        score = {a: np.full(len(y), np.nan) for a in arms}
        pred = {a: np.full(len(y), -1, dtype=int) for a in arms}

        for tr, te in _folds(sd, seed, cfg.n_folds):
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
            a_tr, a_va = tr[i_tr], tr[i_va]

            # one probe per single source; scored on the WHOLE test fold, then sliced by
            # target at aggregation time, so every cell of a row comes from one fit
            fit_sizes = {}
            for src in dsets:
                m_tr = a_tr[ds_arr[a_tr] == src]
                m_va = a_va[ds_arr[a_va] == src]
                if len(m_tr) < 50 or len(np.unique(y[m_tr])) < 2:
                    continue
                fit_sizes[src] = len(m_tr)
                pb = Probe().fit(X, y, m_tr, m_va, seed)
                s = pb.score(X[te])
                score[f"single:{src}"][te] = s
                pred[f"single:{src}"][te] = (s >= pb.thr).astype(int)

            # the size every matched arm is cut down to: the smallest single source, so
            # the quota is always satisfiable
            n_match = min(fit_sizes.values()) if fit_sizes else 0

            for tgt in dsets:
                m = te[ds_arr[te] == tgt]
                if len(m) == 0 or len(np.unique(y[m])) < 2:
                    continue
                o_tr = a_tr[ds_arr[a_tr] != tgt]
                o_va = a_va[ds_arr[a_va] != tgt]

                pb = Probe().fit(X, y, o_tr, o_va, seed)
                s = pb.score(X[m])
                score[f"lodo:{tgt}"][m] = s
                pred[f"lodo:{tgt}"][m] = (s >= pb.thr).astype(int)

                if n_match:
                    s_tr = balanced_subsample(o_tr, ds_arr, n_match, rng)
                    s_va = balanced_subsample(o_va, ds_arr, max(len(o_va) // 3, 1), rng)
                    if len(np.unique(y[s_tr])) > 1 and len(np.unique(y[s_va])) > 1:
                        pb = Probe().fit(X, y, s_tr, s_va, seed)
                        s = pb.score(X[m])
                        score[f"lodo_matched:{tgt}"][m] = s
                        pred[f"lodo_matched:{tgt}"][m] = (s >= pb.thr).astype(int)

            pb = Probe().fit(X, y, a_tr, a_va, seed)
            s = pb.score(X[te])
            score["pooled"][te] = s
            pred["pooled"][te] = (s >= pb.thr).astype(int)

            n_train["single"].append(float(n_match))
            n_train["lodo"].append(float(len(a_tr) - (ds_arr[a_tr] == dsets[0]).sum()))
            n_train["pooled"].append(float(len(a_tr)))

        for tgt in dsets:
            m = ds_arr == tgt
            for src in dsets:
                a = f"single:{src}"
                if np.isnan(score[a][m]).any():
                    continue
                key = f"single|{src}->{tgt}"
                for k, v in cell_metrics(y[m], score[a][m], pred[a][m]).items():
                    acc[key][k].append(v)
            for arm in (f"lodo:{tgt}", f"lodo_matched:{tgt}", "pooled"):
                if np.isnan(score[arm][m]).any():
                    continue
                key = f"{arm.split(':')[0]}|->{tgt}"
                for k, v in cell_metrics(y[m], score[arm][m], pred[arm][m]).items():
                    acc[key][k].append(v)

        print(f"  {run} seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min",
              flush=True)
        del X

    return {"cells": {c: {k: {"mean": round(float(np.mean(v)), 4),
                              "std": round(float(np.std(v)), 4), "n": len(v)}
                          for k, v in d.items()} for c, d in acc.items()},
            "train_rows": {k: round(float(np.mean(v)), 1) for k, v in n_train.items()},
            "seeds": sorted(seed_data)}


DS = ["triviaqa", "nq_open", "squad_v2", "coqa"]


def report(run: str, res: dict) -> None:
    cells = res["cells"]
    def g(key, metric):
        return cells.get(key, {}).get(metric, {}).get("mean")
    def fmt(v):
        return f"{v:9.4f}" if v is not None else f"{'-':>9s}"

    print(f"\n=== {run}: cross-dataset transfer, SAPLMA + PCA{DIM} + logreg "
          f"({len(res['seeds'])} seeds) ===")
    print(f"  train rows per fold: single {res['train_rows'].get('single', 0):.0f}, "
          f"lodo {res['train_rows'].get('lodo', 0):.0f}, "
          f"pooled {res['train_rows'].get('pooled', 0):.0f}")

    for metric in ("auroc", "mcc", "mcc_oracle"):
        print(f"\n  {metric.upper()}   rows = training source, columns = evaluation target")
        print(f"    {'train on':16s}" + "".join(f"{d[:9]:>10s}" for d in DS) + "   off-diag mean")
        for src in DS:
            vals = [g(f"single|{src}->{t}", metric) for t in DS]
            off = [v for t, v in zip(DS, vals) if t != src and v is not None]
            print(f"    {src:16s}" + "".join(fmt(v) for v in vals)
                  + (f"{np.mean(off):14.4f}" if off else f"{'-':>14s}"))
        for arm, label in (("lodo", "lodo (3 others)"),
                           ("lodo_matched", "lodo size-matched"),
                           ("pooled", "pooled (incl. tgt)")):
            vals = [g(f"{arm}|->{t}", metric) for t in DS]
            ok = [v for v in vals if v is not None]
            print(f"    {label:16s}" + "".join(fmt(v) for v in vals)
                  + (f"{np.mean(ok):14.4f}" if ok else f"{'-':>14s}"))

    print("\n  transfer cost per target (mean over the 3 foreign sources)")
    print(f"    {'target':16s}{'in-domain':>11s}{'foreign':>10s}{'drop':>9s}"
          f"{'lodo':>9s}{'matched':>9s}{'pooled':>9s}")
    for t in DS:
        ind = g(f"single|{t}->{t}", "auroc")
        for_ = [g(f"single|{s}->{t}", "auroc") for s in DS if s != t]
        for_ = [v for v in for_ if v is not None]
        row = [ind, np.mean(for_) if for_ else None,
               (np.mean(for_) - ind) if (for_ and ind is not None) else None,
               g(f"lodo|->{t}", "auroc"), g(f"lodo_matched|->{t}", "auroc"),
               g(f"pooled|->{t}", "auroc")]
        print(f"    {t:16s}" + "".join(
            f"{v:>10.4f}" if v is not None else f"{'-':>10s}" for v in row))

    print("\n  how much of the MCC damage is threshold miscalibration "
          "(mcc_oracle - mcc, mean over foreign sources)")
    for t in DS:
        gaps = [g(f"single|{s}->{t}", "mcc_oracle") - g(f"single|{s}->{t}", "mcc")
                for s in DS if s != t and g(f"single|{s}->{t}", "mcc") is not None]
        lg = (g(f"lodo|->{t}", "mcc_oracle") - g(f"lodo|->{t}", "mcc")
              if g(f"lodo|->{t}", "mcc") is not None else None)
        print(f"    {t:16s} single {np.mean(gaps):+.4f}" if gaps else f"    {t:16s} -",
              end="")
        print(f"   lodo {lg:+.4f}" if lg is not None else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main", help="one generator per invocation")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--merge", action="store_true",
                    help="collect the per-generator files into runs/dataset_transfer.json")
    args = ap.parse_args()

    if args.merge:
        out = {}
        for f in sorted(Path("runs/dataset_transfer").glob("*.json")):
            out[f.stem] = json.loads(f.read_text())
        for run in RUNS:
            if run in out:
                report(run, out[run])
        write_json(Path("runs/dataset_transfer.json"), out)
        print(f"\nmerged {len(out)} generators -> runs/dataset_transfer.json")
        return

    res = run_one(args.run, args.seeds)
    if not res:
        print(f"nothing computed for {args.run}")
        return
    # One file per generator rather than one shared dict: the generators run as parallel
    # processes, and a read-modify-write of a single JSON would drop whichever result
    # landed between another process's read and its write. `--merge` collects them.
    write_json(Path(f"runs/dataset_transfer/{args.run}.json"), res)
    report(args.run, res)
    print(f"\nwrote runs/dataset_transfer/{args.run}.json")


if __name__ == "__main__":
    main()
