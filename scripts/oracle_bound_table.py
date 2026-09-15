"""Oracle bound, best single detector and best ensemble, per generator and per dataset.

Three quantities on the same rows and the same folds, so the headroom argument can be read
off directly:

  oracle        right whenever ANY of the five individual detectors is right, wrong only
                when all five fail. It is a bound, not a method: constructing it requires
                knowing which detector to trust per item, which is the routing signal the
                independence results say does not exist.
  best single   the strongest of the five individual detectors, each under its own
                published probe
  best ensemble the strongest combiner available: the three voting rules plus the two
                learned unions from stage 3

The oracle has no score, only a decision, so its AUROC is undefined and is reported as
`--` rather than filled in. What replaces it is COVERAGE: the fraction of items at least
one detector gets right, which is the oracle's accuracy and the quantity the bound is
actually made of.

Both metrics are reported for the two attainable rows. `unexploited` is oracle MCC minus
the better of the two, i.e. the headroom that exists in principle and is not reached.

Predictions come from stage 3's stored out-of-fold arrays, so nothing is refitted and no
model sees its own test fold.

Usage: uv run python scripts/oracle_bound_table.py
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
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from halluc.io import write_json

SINGLES = ["saplma", "lapeigvals", "icr", "attn_baseline", "svd_baseline"]
UNIONS = ["union_equal", "union_raw"]
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b",
        "llama3.2-3b-base", "gemma3-12b-pt"]
SCOPES = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]


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

def run_one(run: str):
    files = sorted(glob.glob(f"runs/{run}/stage3_train/pooled/seed*/predictions.npz"),
                   key=lambda p: int(Path(p).parent.name.replace("seed", "")))
    if not files:
        return None
    acc = defaultdict(lambda: defaultdict(list))
    for f in files:
        seed = int(Path(f).parent.name.replace("seed", ""))
        d = np.load(f, allow_pickle=True)
        y = d["y"].astype(int)
        ds = d["dataset"].astype(str)
        groups = d["groups"].astype(str)
        P = {m: d[f"preds__{m}"].astype(int) for m in SINGLES}
        S = {m: d[f"scores__{m}"].astype(float) for m in SINGLES}

        cand = {m: (P[m], S[m]) for m in SINGLES}
        for u in UNIONS:
            if f"preds__{u}" in d.files:
                cand[u] = (d[f"preds__{u}"].astype(int), d[f"scores__{u}"].astype(float))

        A = np.stack([P[m] for m in SINGLES])
        Sm = np.stack([S[m] for m in SINGLES])
        votes = A.sum(0)
        cand["vote_hard"] = ((votes >= 3).astype(int), votes / 5.0)
        R = np.stack([rankdata(s) / len(s) for s in Sm])
        for name, sc in (("vote_soft", Sm.mean(0)), ("vote_rank", R.mean(0))):
            cand[name] = (cross_fit_predict(y, sc, groups, seed), sc)

        # Oracle: correct wherever at least one single detector is correct. It has a
        # decision but no score, so no AUROC is defined for it.
        hit = np.any(A == y[None, :], axis=0)
        oracle = np.where(hit, y, 1 - y)

        for scope in SCOPES:
            m = np.ones(len(y), bool) if scope == "pooled" else (ds == scope)
            if len(np.unique(y[m])) < 2:
                continue
            acc[scope]["oracle_mcc"].append(float(matthews_corrcoef(y[m], oracle[m])))
            acc[scope]["coverage"].append(float(hit[m].mean()))
            acc[scope]["positive_rate"].append(float(y[m].mean()))
            for name, (pred, sc) in cand.items():
                acc[scope][f"{name}|mcc"].append(float(matthews_corrcoef(y[m], pred[m])))
                acc[scope][f"{name}|auroc"].append(float(roc_auc_score(y[m], sc[m])))
    ens = [c for c in ["vote_hard", "vote_soft", "vote_rank"] + UNIONS
           if f"{c}|mcc" in acc["pooled"]]
    return ({s: {k: [round(float(x), 6) for x in v] for k, v in dd.items()}
             for s, dd in acc.items()}, ens)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    args = ap.parse_args()

    out, rows = {}, []
    for run in args.runs:
        got = run_one(run)
        if got is None:
            print(f"  {run}: no stage-3 predictions, skipped")
            continue
        per_seed, ens = got
        out[run] = {"per_seed": per_seed, "ensembles": ens, "singles": SINGLES}
        for scope in SCOPES:
            if scope not in per_seed:
                continue
            d = per_seed[scope]
            mean = lambda k: float(np.mean(d[k]))
            bs = max(SINGLES, key=lambda m: mean(f"{m}|mcc"))
            be = max(ens, key=lambda m: mean(f"{m}|mcc"))
            best = max(mean(f"{bs}|mcc"), mean(f"{be}|mcc"))
            rows.append(dict(
                generator=run, dataset=scope,
                pos_rate=round(mean("positive_rate"), 4),
                oracle_mcc=round(mean("oracle_mcc"), 4),
                coverage=round(mean("coverage"), 4),
                best_single=bs, bs_mcc=round(mean(f"{bs}|mcc"), 4),
                bs_auroc=round(mean(f"{bs}|auroc"), 4),
                best_ensemble=be, be_mcc=round(mean(f"{be}|mcc"), 4),
                be_auroc=round(mean(f"{be}|auroc"), 4),
                unexploited=round(mean("oracle_mcc") - best, 4)))
        print(f"  {run}: {len(per_seed['pooled']['oracle_mcc'])} seeds", flush=True)

    write_json(Path("runs/oracle_bound_table.json"), out)
    with open("runs/oracle_bound_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"\n{'=' * 126}")
    print(f"  {'generator':18s}{'dataset':10s}{'pos':>6s}{'ORACLE':>9s}{'cover':>8s}"
          f"   {'best single':16s}{'MCC':>8s}{'AUROC':>8s}"
          f"   {'best ensemble':14s}{'MCC':>8s}{'AUROC':>8s}{'unexpl':>9s}")
    print(f"{'=' * 126}")
    last = None
    for r in rows:
        if last and r["generator"] != last:
            print()
        last = r["generator"]
        print(f"  {r['generator'] if r['dataset'] == 'pooled' else '':18s}"
              f"{r['dataset']:10s}{r['pos_rate']:>6.3f}{r['oracle_mcc']:>9.4f}"
              f"{r['coverage']:>8.3f}   {r['best_single']:16s}{r['bs_mcc']:>8.4f}"
              f"{r['bs_auroc']:>8.4f}   {r['best_ensemble']:14s}{r['be_mcc']:>8.4f}"
              f"{r['be_auroc']:>8.4f}{r['unexploited']:>9.4f}")
    print("\n  ORACLE has a decision but no score, so its AUROC is undefined; coverage is")
    print("  the fraction of items at least one single detector gets right.")
    print("\nwrote runs/oracle_bound_table.json and .csv")


if __name__ == "__main__":
    main()
