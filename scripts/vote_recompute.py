"""Voting combiners over the five published probes, per dataset, per model, 5 seeds.

A full recomputation of the "Nothing else combines better either" section, widened from
pooled MCC on four generators to every dataset x generator x metric, with the individual
methods recomputed from the same arrays rather than quoted from another run's JSON.

Two prediction sources exist on disk and they are NOT interchangeable:

  --source published   stage3_train/pooled/seed*/predictions.npz -- each method under its
                       OWN published probe. SAPLMA is the (256,128,64) MLP, ICR its native
                       four-layer MLP, and so on. This is what "the published detectors
                       vote" means, and it is the default.
  --source pcalr       stage5_posthoc/block_oof/seed*.npz -- every block read by the
                       union's PCA-128 + logreg instead. Same features, one shared reader.

The two disagree by a lot, and not in one direction: on Qwen3-14B pooled MCC the swap
takes SAPLMA from 0.4827 to 0.5482 and ICR from 0.3978 to 0.3022. A vote over five
detectors is therefore a different experiment under each source, because the swap changes
which members are strong and how correlated their errors are.

Either way nothing is refitted here and no model sees its own test fold; both sources are
out-of-fold under stage 3's nested CV. That is what makes plain voting the clean test: a
learned stacker can launder extra capacity as a gain, a vote cannot.

  vote_hard        majority of the five binary predictions, >= 3 of 5
  vote_hard_4of5   >= 4 of 5
  vote_unanimous   5 of 5
  vote_soft        mean of the five probability scores, thresholded once
  vote_rank        mean of within-method ranks, thresholded once

Only vote_soft and vote_rank have a free parameter, and it is the one place cross
validation still has work to do: each fold's threshold is fitted on the other four folds
and applied to that fold alone, so no item is scored under a threshold that has seen its
label. The hard variants need no threshold -- the five constituent thresholds were already
set inside stage 3's inner splits, which also makes them the only combiners strictly
comparable with the individual detectors, since those do not get a refitted threshold
either.

`best_single` is chosen per seed on POOLED MCC and then reported on every dataset with
that choice held fixed. Picking the best method separately in each dataset column would
be selection on the test slice and would flatter the single-method reference.

Both MCC and AUROC are reported for every cell. They answer different questions -- where
the threshold sits, versus ranking ability -- and the voting variants are exactly the case
where they come apart, since hard voting quantises the score to six levels.

Per-seed values are written out, not just the means, so the paired tests these rows need
can be run without recomputing anything.

Usage: uv run python scripts/vote_recompute.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import rankdata
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from halluc.io import write_json

METHODS = ["saplma", "lapeigvals", "icr", "attn_baseline", "svd_baseline"]
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b",
        "llama3.2-3b-base", "gemma3-12b-pt"]
SCOPES = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
VOTES = ["vote_hard", "vote_hard_4of5", "vote_unanimous", "vote_soft", "vote_rank"]
METRICS = ["mcc", "auroc", "accuracy", "balanced_accuracy", "f1"]


def cross_fit_predict(y, s, groups, seed) -> np.ndarray:
    """Threshold each fold using only the other folds, and return the predictions.

    The earlier form of this returned a single float: the mean of the five per-fold
    optima, applied to every item. That leaks. Each item sits in the training part of
    four of the five folds, so the averaged threshold has seen its label, and pooled MCC
    came out about +0.0035 too high across all six generators -- landing between the
    honest value and one fitted on the whole vector, which is exactly what averaging
    four-fifths-seen optima produces.

    Grouping is not optional: CoQA turns share a story and SQuAD questions share a
    paragraph, so an ungrouped split puts near-duplicates on both sides of it.
    """
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            s.reshape(-1, 1), y, groups):
        grid = np.quantile(s[tr], np.linspace(0.05, 0.95, 91))
        thr = max(grid, key=lambda t: matthews_corrcoef(y[tr], (s[tr] >= t).astype(int)))
        pred[te] = (s[te] >= thr).astype(int)
    return pred

def score_all(y, pred, sc, mask) -> dict:
    if len(np.unique(y[mask])) < 2:
        return {}
    return {"mcc": float(matthews_corrcoef(y[mask], pred[mask])),
            "auroc": float(roc_auc_score(y[mask], sc[mask])),
            "accuracy": float(accuracy_score(y[mask], pred[mask])),
            "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred[mask])),
            "f1": float(f1_score(y[mask], pred[mask], zero_division=0))}


def source_files(run: str, source: str) -> list[str]:
    if source == "published":
        return sorted(glob.glob(f"runs/{run}/stage3_train/pooled/seed*/predictions.npz"),
                      key=lambda p: int(Path(p).parent.name.replace("seed", "")))
    return sorted(glob.glob(f"runs/{run}/stage5_posthoc/block_oof/seed*.npz"),
                  key=lambda p: int(Path(p).stem.replace("seed", "")))


def seed_of(path: str, source: str) -> int:
    return int((Path(path).parent.name if source == "published"
                else Path(path).stem).replace("seed", ""))


def run_one(run: str, source: str) -> tuple[dict, list[int], dict]:
    files = source_files(run, source)
    if not files:
        return {}, [], {}
    acc = defaultdict(lambda: defaultdict(list))
    seeds, picked = [], []

    for f in files:
        seed = seed_of(f, source)
        d = np.load(f, allow_pickle=True)
        y, ds_arr = d["y"].astype(int), d["dataset"].astype(str)
        groups = d["groups"].astype(str)
        if any(f"preds__{m}" not in d.files for m in METHODS):
            continue
        seeds.append(seed)

        P = np.stack([d[f"preds__{m}"].astype(int) for m in METHODS])
        S = np.stack([d[f"scores__{m}"].astype(float) for m in METHODS])
        R = np.stack([rankdata(s) / len(s) for s in S])
        votes = P.sum(0)

        cand: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for i, m in enumerate(METHODS):
            cand[m] = (P[i], S[i])
        # best_single is fixed on pooled MCC, then read across every dataset column, so
        # the reference is not re-selected inside the slice it is compared on.
        pooled_mcc = [matthews_corrcoef(y, P[i]) for i in range(len(METHODS))]
        bi = int(np.argmax(pooled_mcc))
        picked.append(METHODS[bi])
        cand["best_single"] = (P[bi], S[bi])

        cand["vote_hard"] = ((votes >= 3).astype(int), votes / 5.0)
        cand["vote_hard_4of5"] = ((votes >= 4).astype(int), votes / 5.0)
        cand["vote_unanimous"] = ((votes == 5).astype(int), votes / 5.0)
        for name, s in (("vote_soft", S.mean(0)), ("vote_rank", R.mean(0))):
            cand[name] = (cross_fit_predict(y, s, groups, seed), s)

        for name, (pred, sc) in cand.items():
            for scope in SCOPES:
                m = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                for k, v in score_all(y, pred, sc, m).items():
                    acc[f"{name}|{scope}"][k].append(v)
        print(f"    {run} seed {seed}: best_single={METHODS[bi]} "
              f"(pooled MCC {pooled_mcc[bi]:.4f})", flush=True)

    per_seed = {k: {m: [round(float(x), 6) for x in v] for m, v in dd.items()}
                for k, dd in acc.items()}
    return per_seed, sorted(seeds), {"best_single_per_seed": picked}


ROWS = METHODS + ["best_single"] + VOTES


def big_table(out, runs):
    """MCC and AUROC together, every generator x dataset x variant."""
    for metric in ("mcc", "auroc"):
        print(f"\n{'=' * 118}\n  {metric.upper()}   mean +- sd over seeds\n{'=' * 118}")
        for run in runs:
            if run not in out:
                continue
            res, n = out[run]["per_seed"], len(out[run]["seeds"])
            print(f"\n  {run}   ({n} seeds)")
            print(f"    {'variant':18s}" + "".join(f"{s[:14]:>17s}" for s in SCOPES))
            for r in ROWS:
                cells = ""
                for s in SCOPES:
                    v = res.get(f"{r}|{s}", {}).get(metric)
                    cells += (f"{np.mean(v):11.4f}±{np.std(v):.3f}" if v
                              else f"{'-':>17s}")
                # The three hard variants differ only in where the vote count is cut,
                # and all three rank by the same votes/5 score, so their AUROC is equal by
                # construction. Marked rather than silently repeated three times.
                mark = ("  =hard" if metric == "auroc"
                        and r in ("vote_hard_4of5", "vote_unanimous") else "")
                print(f"    {r + mark:18s}{cells}")
            # the comparison the section exists to make
            for v in VOTES:
                d_ = []
                for s in SCOPES:
                    a = res.get(f"{v}|{s}", {}).get(metric)
                    b = res.get(f"best_single|{s}", {}).get(metric)
                    d_.append(np.mean(a) - np.mean(b) if a and b else np.nan)
                print(f"    {'  ' + v + ' - best':18s}"
                      + "".join(f"{x:17.4f}" if np.isfinite(x) else f"{'-':>17s}"
                                for x in d_))


def write_csv(out, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "variant", "dataset", "metric", "mean", "sd",
                    "n_seeds"] + [f"seed_{i}" for i in range(5)])
        for run, blob in out.items():
            for key, d in blob["per_seed"].items():
                variant, scope = key.split("|")
                for metric, vals in d.items():
                    w.writerow([run, variant, scope, metric,
                                round(float(np.mean(vals)), 6),
                                round(float(np.std(vals)), 6), len(vals)]
                               + [round(v, 6) for v in vals] + [""] * (5 - len(vals)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--source", choices=("published", "pcalr"), default="published",
                    help="published = each method's own probe (stage 3); "
                         "pcalr = every block under the union's PCA-128 + logreg")
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        per_seed, seeds, meta = run_one(run, args.source)
        if not per_seed:
            print(f"    {run}: no block_oof predictions, skipped", flush=True)
            continue
        out[run] = {"per_seed": per_seed, "seeds": seeds, "methods": METHODS,
                    "source": args.source, **meta}

    tag = "" if args.source == "published" else "_pcalr"
    write_json(Path(f"runs/vote_recompute{tag}.json"), out)
    write_csv(out, Path(f"runs/vote_recompute{tag}.csv"))
    big_table(out, args.runs)
    print(f"\nsource: {args.source}")
    print(f"wrote runs/vote_recompute{tag}.json  (per-seed values, all {len(METRICS)} metrics)")
    print(f"wrote runs/vote_recompute{tag}.csv   (long format, one row per "
          "generator x variant x dataset x metric)")
    for run in out:
        print(f"  {run:18s} seeds {out[run]['seeds']}  "
              f"best_single per seed: {out[run]['best_single_per_seed']}")


if __name__ == "__main__":
    main()
