"""Four agreement metrics between every pair of published detectors and vote variants.

Eight members: the five detectors under their own published probes, plus the three vote
rules built from them. All 28 unordered pairs, per generator, per dataset and pooled.

The four metrics do not measure the same object, and forcing all four onto one
representation would make them incomparable:

  kappa       Cohen's kappa on the BINARY decisions, chance-corrected
  agree       raw fraction of items where the two decisions match
  pearson     Pearson r on the continuous SCORES
  spearman    Spearman rho on the continuous SCORES, so monotone rescaling is ignored

Kappa and agreement therefore answer "do these two make the same calls", while Pearson and
Spearman answer "do these two order items the same way". A pair can agree on nearly every
decision and still rank differently inside each class, and the two halves of the table
separate that.

Two cells need reading with care:

  - A vote is constructed from the detectors, so its correlation with a member it contains
    is partly definitional rather than empirical. Those pairs are marked `*` in the printed
    matrices. vote_soft against saplma is not evidence about saplma.
  - vote_hard's score is the vote count over five, so it takes six values. Spearman against
    it carries heavy ties and its Pearson is attenuated by the coarse support. Its kappa
    and agreement are unaffected.

Source is stage 3's stored out-of-fold predictions -- each detector under its own published
probe and threshold. The block_oof arrays (PCA-128 + logreg) are not used.

Usage: uv run python scripts/pairwise_correlations.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
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
    """Threshold each fold on the other four; see scripts/vote_recompute.py."""
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            s.reshape(-1, 1), y, groups):
        grid = np.quantile(s[tr], np.linspace(0.05, 0.95, 91))
        thr = max(grid, key=lambda t: matthews_corrcoef(y[tr], (s[tr] >= t).astype(int)))
        pred[te] = (s[te] >= thr).astype(int)
    return pred


def safe(fn, a, b) -> float:
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    v = fn(a, b)
    return float(v if np.isscalar(v) else v[0])


def run_one(run: str):
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
        votes = A.sum(0)
        P["vote_hard"], S["vote_hard"] = (votes >= 3).astype(int), votes / 5.0
        R = np.stack([rankdata(s) / len(y) for s in Sm])
        for name, sc in (("vote_soft", Sm.mean(0)), ("vote_rank", R.mean(0))):
            S[name] = sc
            P[name] = cross_fit_predict(y, sc, groups, seed)

        for scope in SCOPES:
            msk = np.ones(len(y), bool) if scope == "pooled" else (ds == scope)
            for a, b in itertools.combinations(MEMBERS, 2):
                k = f"{scope}|{a}|{b}"
                acc[k]["kappa"].append(safe(cohen_kappa_score, P[a][msk], P[b][msk]))
                acc[k]["agree"].append(float((P[a][msk] == P[b][msk]).mean()))
                acc[k]["pearson"].append(safe(lambda x, z: pearsonr(x, z),
                                              S[a][msk], S[b][msk]))
                acc[k]["spearman"].append(safe(lambda x, z: spearmanr(x, z),
                                               S[a][msk], S[b][msk]))
        print(f"    {run} seed {seed}", flush=True)
    return {k: {m: [round(float(x), 6) for x in v] for m, v in dd.items()}
            for k, dd in acc.items()}


def contains(vote: str, det: str) -> bool:
    """Every vote is built from all five detectors, so any (vote, detector) pair is one."""
    return vote in VOTES and det in DETECTORS


def report(out, scope):
    for metric in METRICS:
        basis = "decisions" if metric in ("kappa", "agree") else "scores"
        print(f"\n{'=' * 92}\n  {metric.upper()}  ({basis})   scope={scope}"
              f"\n  * = the pair is partly definitional: the vote contains that detector"
              f"\n{'=' * 92}")
        for run, blob in out.items():
            print(f"\n  {run}")
            print(f"    {'':9s}" + "".join(f"{SHORT[m]:>9s}" for m in MEMBERS))
            for i, a in enumerate(MEMBERS):
                row = f"    {SHORT[a]:9s}"
                for j, b in enumerate(MEMBERS):
                    if i == j:
                        row += f"{'—':>9s}"; continue
                    x, z = (a, b) if i < j else (b, a)
                    v = blob.get(f"{scope}|{x}|{z}", {}).get(metric)
                    if not v:
                        row += f"{'.':>9s}"; continue
                    mark = "*" if (contains(a, b) or contains(b, a)) else " "
                    row += f"{np.mean(v):8.3f}{mark}"
                print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--scope", default="pooled")
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        res = run_one(run)
        if res is None:
            print(f"  {run}: no stage-3 predictions, skipped"); continue
        out[run] = res

    write_json(Path("runs/pairwise_correlations.json"), out)
    with open("runs/pairwise_correlations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["generator", "dataset", "member_a", "member_b", "metric",
                    "mean", "sd", "n_seeds", "definitional"])
        for run, blob in out.items():
            for key, d in blob.items():
                scope, a, b = key.split("|")
                for metric, vals in d.items():
                    w.writerow([run, scope, a, b, metric,
                                round(float(np.mean(vals)), 6),
                                round(float(np.std(vals)), 6), len(vals),
                                int(contains(a, b) or contains(b, a))])
    report(out, args.scope)
    print("\nwrote runs/pairwise_correlations.json and .csv")


if __name__ == "__main__":
    main()
