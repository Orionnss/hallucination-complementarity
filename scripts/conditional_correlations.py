"""Four correlation metrics of B against the label, restricted to the items A got wrong.

For every ordered pair (A, B) the slice is `A is wrong`, and on that slice B is compared
with the judge label by:

  kappa      Cohen's kappa between B's decision and the label
  agree      fraction of the slice B gets right -- the rescue rate
  pearson    point-biserial r between B's continuous score and the label
  spearman   rank version of the same

(A, B) and (B, A) are different questions, so the matrices are asymmetric and the diagonal
is undefined: a method rescues none of its own errors.

None of the four can be read against zero. Conditioning on one method's errors is collider
conditioning: any B correlated with A looks bad on that slice no matter how good B is, and
the slice's class balance is shifted, which moves every metric on its own. So each cell
also carries a null obtained by permuting B's (decision, score) pairs WITHIN each class.
That preserves B's per-class accuracy and its score distribution exactly while destroying
any dependence between B and A, which is precisely the counterfactual "an equally strong
but independent detector". The reported ratio is observed / null.

The null matters more than it might seem. An independent detector does not score kappa 0
on these slices -- it scores 0.33 to 0.51, because the slice is enriched for the class B
is better at. Read against 0, the observed negative kappas look mildly bad; read against
the null they are far worse.

Members are the five detectors under their published probes plus the three vote rules.
A vote contains every detector, so any vote-detector pair is partly definitional and is
marked in the output.

Usage: uv run python scripts/conditional_correlations.py --perms 30
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

from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold

from halluc.io import write_json

DETECTORS = ["saplma", "lapeigvals", "icr", "attn_baseline", "svd_baseline"]
VOTES = ["vote_hard", "vote_rank", "vote_soft"]
MEMBERS = DETECTORS + VOTES
SHORT = {"saplma": "SAPLMA", "lapeigvals": "LapEig", "icr": "ICR",
         "attn_baseline": "Attn", "svd_baseline": "SVD",
         "vote_hard": "V-hard", "vote_rank": "V-rank", "vote_soft": "V-soft"}
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b",
        "llama3.2-3b-base", "gemma3-12b-pt"]
SCOPES = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
METRICS = ["kappa", "agree", "pearson", "spearman"]


def cross_fit_predict(y, s, groups, seed) -> np.ndarray:
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            s.reshape(-1, 1), y, groups):
        grid = np.quantile(s[tr], np.linspace(0.05, 0.95, 91))
        thr = max(grid, key=lambda t: matthews_corrcoef(y[tr], (s[tr] >= t).astype(int)))
        pred[te] = (s[te] >= thr).astype(int)
    return pred


def four(ys, pr, sc) -> dict:
    """The four metrics on one slice. NaN where a vector is constant."""
    out = {}
    out["kappa"] = (float(cohen_kappa_score(ys, pr))
                    if len(np.unique(ys)) > 1 and len(np.unique(pr)) > 1 else float("nan"))
    out["agree"] = float((pr == ys).mean())
    ok = len(np.unique(ys)) > 1 and len(np.unique(sc)) > 1
    out["pearson"] = float(pearsonr(sc, ys)[0]) if ok else float("nan")
    out["spearman"] = float(spearmanr(sc, ys).statistic) if ok else float("nan")
    return out


def null_four(y, pr_full, sc_full, wrong, rng, perms) -> dict:
    """Permute B within each class, so B keeps its strength and loses its link to A."""
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    acc = defaultdict(list)
    for _ in range(perms):
        p = np.empty_like(pr_full); s = np.empty_like(sc_full)
        for idx in (idx0, idx1):
            perm = rng.permutation(idx)
            p[idx] = pr_full[perm]; s[idx] = sc_full[perm]
        for k, v in four(y[wrong], p[wrong], s[wrong]).items():
            acc[k].append(v)
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def run_one(run: str, perms: int, min_slice: int):
    files = sorted(glob.glob(f"runs/{run}/stage3_train/pooled/seed*/predictions.npz"),
                   key=lambda p: int(Path(p).parent.name.replace("seed", "")))
    if not files:
        return None
    acc = defaultdict(lambda: defaultdict(list))
    for f in files:
        seed = int(Path(f).parent.name.replace("seed", ""))
        d = np.load(f, allow_pickle=True)
        y = d["y"].astype(int); ds = d["dataset"].astype(str)
        groups = d["groups"].astype(str)
        P = {m: d[f"preds__{m}"].astype(int) for m in DETECTORS}
        S = {m: d[f"scores__{m}"].astype(float) for m in DETECTORS}
        A = np.stack([P[m] for m in DETECTORS]); Sm = np.stack([S[m] for m in DETECTORS])
        v = A.sum(0)
        P["vote_hard"], S["vote_hard"] = (v >= 3).astype(int), v / 5.0
        R = np.stack([rankdata(x) / len(y) for x in Sm])
        for nm, sc in (("vote_soft", Sm.mean(0)), ("vote_rank", R.mean(0))):
            S[nm] = sc; P[nm] = cross_fit_predict(y, sc, groups, seed)
        rng = np.random.default_rng(seed)

        for scope in SCOPES:
            msk = np.ones(len(y), bool) if scope == "pooled" else (ds == scope)
            ys = y[msk]
            for a in MEMBERS:
                wrong = P[a][msk] != ys
                if wrong.sum() < min_slice or len(np.unique(ys[wrong])) < 2:
                    continue
                for b in MEMBERS:
                    if a == b:
                        continue
                    k = f"{scope}|{a}|{b}"
                    obs = four(ys[wrong], P[b][msk][wrong], S[b][msk][wrong])
                    for m, val in obs.items():
                        acc[k][m].append(val)
                    acc[k]["n"].append(float(wrong.sum()))
                    if scope == "pooled" and perms:
                        nl = null_four(ys, P[b][msk], S[b][msk], wrong, rng, perms)
                        for m, val in nl.items():
                            acc[k][f"null_{m}"].append(val)
        print(f"    {run} seed {seed}", flush=True)
    return {k: {m: [round(float(x), 6) for x in v] for m, v in dd.items()}
            for k, dd in acc.items()}


def report(out, scope, perms):
    for metric in METRICS:
        print(f"\n{'=' * 96}\n  {metric.upper()} of B against the label, on A's errors"
              f"   scope={scope}\n  rows = A (whose errors), cols = B"
              f"   * = definitional pair\n{'=' * 96}")
        for run, blob in out.items():
            print(f"\n  {run}")
            print(f"    {'':9s}" + "".join(f"{SHORT[m]:>9s}" for m in MEMBERS))
            for a in MEMBERS:
                row = f"    {SHORT[a]:9s}"
                for b in MEMBERS:
                    if a == b:
                        row += f"{'—':>9s}"; continue
                    v = blob.get(f"{scope}|{a}|{b}", {}).get(metric)
                    if not v:
                        row += f"{'.':>9s}"; continue
                    d = (a in VOTES and b in DETECTORS) or (b in VOTES and a in DETECTORS)
                    row += f"{np.nanmean(v):8.3f}{'*' if d else ' '}"
                print(row)
    if not perms:
        return
    print(f"\n{'=' * 96}\n  OBSERVED / NULL ratio, pooled, detector pairs only"
          f"\n  null = B permuted within class: same strength, no link to A"
          f"\n  1.0 = independent, below 1 = B fails where A fails\n{'=' * 96}")
    for metric in METRICS:
        print(f"\n  {metric}")
        for run, blob in out.items():
            vals = []
            for a in DETECTORS:
                for b in DETECTORS:
                    if a == b:
                        continue
                    o = blob.get(f"pooled|{a}|{b}", {}).get(metric)
                    n = blob.get(f"pooled|{a}|{b}", {}).get(f"null_{metric}")
                    if o and n and abs(np.nanmean(n)) > 1e-9:
                        vals.append(np.nanmean(o) / np.nanmean(n))
            if vals:
                print(f"    {run:18s} mean {np.mean(vals):+.3f}   "
                      f"range {min(vals):+.3f} to {max(vals):+.3f}   "
                      f"above 1.0: {sum(1 for x in vals if x > 1)}/{len(vals)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--scope", default="pooled")
    ap.add_argument("--perms", type=int, default=30)
    ap.add_argument("--min-slice", type=int, default=50)
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        res = run_one(run, args.perms, args.min_slice)
        if res is None:
            print(f"  {run}: skipped"); continue
        out[run] = res

    write_json(Path("runs/conditional_correlations.json"), out)
    with open("runs/conditional_correlations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "dataset", "errors_of_A", "rescuer_B", "metric",
                    "mean", "sd", "n_seeds", "definitional"])
        for run, blob in out.items():
            for key, d in blob.items():
                scope, a, b = key.split("|")
                dfn = int((a in VOTES and b in DETECTORS) or (b in VOTES and a in DETECTORS))
                for metric, vals in d.items():
                    w.writerow([run, scope, a, b, metric,
                                round(float(np.nanmean(vals)), 6),
                                round(float(np.nanstd(vals)), 6), len(vals), dfn])
    report(out, args.scope, args.perms)
    print("\nwrote runs/conditional_correlations.json and .csv")


if __name__ == "__main__":
    main()
