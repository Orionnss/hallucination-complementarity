"""On the answers SAPLMA gets wrong, does voting the other four rescue more than any one?

Individually, each of the other methods recovers SAPLMA's errors at 0.42-0.58x what
statistical independence would give. Voting is the obvious next question: if their errors
on that slice were even partly independent of one another, a majority should denoise and
recover more than any single voter. If it does not, the redundancy runs deeper than
pairwise - the four fail on the same items as each other, not just as SAPLMA.

Two things make this measurable rather than circular:

  SAPLMA is wrong on 100% of this slice by construction, so it scores MCC -1 there and
  any method above chance "beats" it. The meaningful references are therefore the
  individual methods, and the independence null - per-class accuracy over ALL data,
  reweighted to the class mix of the error slice.

  Vote thresholds are tuned on the FULL population, never on the slice. At deployment you
  do not know which items SAPLMA got wrong, so a threshold fitted to the slice would be an
  oracle. This is the honest version: fit the voter as you would ship it, then look at how
  it behaves where the primary detector failed.

Reported on both halves separately as well, since SAPLMA's false positives and false
negatives are different populations with different base rates.

Usage: uv run python scripts/vote_on_saplma_errors.py
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from halluc.io import write_json

OTHERS = ["lapeigvals", "attn_baseline", "icr", "svd_baseline", "charm", "logprob"]
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b"]


def tuned_threshold(y, s, groups, seed):
    picks = []
    for tr, _ in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            s.reshape(-1, 1), y, groups):
        grid = np.unique(np.quantile(s[tr], np.linspace(0.02, 0.98, 60)))
        picks.append(max(grid, key=lambda t: matthews_corrcoef(
            y[tr], (s[tr] >= t).astype(int))))
    return float(np.mean(picks))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        acc = defaultdict(lambda: defaultdict(list))
        for si, f in enumerate(sorted(glob.glob(
                f"runs/{run}/stage5_posthoc/block_oof/*.npz"))):
            if si not in args.seeds:
                continue
            d = np.load(f, allow_pickle=True)
            y, groups = d["y"], d["groups"].astype(str)
            P = {m: d[f"preds__{m}"] for m in OTHERS}
            S = {m: d[f"scores__{m}"] for m in OTHERS}
            sap = d["preds__saplma"]
            wrong = sap != y

            cand = {m: (P[m], S[m]) for m in OTHERS}
            for nm, sc in (("vote_hard", np.stack([P[m] for m in OTHERS]).sum(0) / len(OTHERS)),
                           ("vote_soft", np.mean([S[m] for m in OTHERS], axis=0)),
                           ("vote_rank", np.mean([rankdata(S[m]) / len(y)
                                                  for m in OTHERS], axis=0))):
                # threshold from the full population, not the error slice
                cand[nm] = ((sc >= tuned_threshold(y, sc, groups, si)).astype(int), sc)

            pops = {"SAPLMA wrong (all)": wrong,
                    "  its false positives": wrong & (sap == 1),
                    "  its false negatives": wrong & (sap == 0)}
            for pop, msk in pops.items():
                if msk.sum() < 20 or len(np.unique(y[msk])) < 2:
                    continue
                acc[("_n", pop)]["n"].append(int(msk.sum()))
                for name, (pred, sc) in cand.items():
                    a = acc[(name, pop)]
                    a["acc"].append(float((pred[msk] == y[msk]).mean()))
                    a["mcc"].append(float(matthews_corrcoef(y[msk], pred[msk])))
                    a["kappa"].append(float(cohen_kappa_score(y[msk], pred[msk])))
                    a["auroc"].append(float(roc_auc_score(y[msk], sc[msk])))
                    per_c = {c: (pred[y == c] == c).mean() for c in (0, 1)}
                    mix = {c: (y[msk] == c).mean() for c in (0, 1)}
                    a["null"].append(float(sum(per_c[c] * mix[c] for c in (0, 1))))
        out[run] = {f"{n}|{p}": {k: round(float(np.mean(v)), 4) for k, v in dd.items()}
                    for (n, p), dd in acc.items()}

    order = OTHERS + ["vote_hard", "vote_soft", "vote_rank"]
    for run in args.runs:
        r = out[run]
        print(f"\n{'=' * 96}\n  {run}\n{'=' * 96}")
        for pop in ["SAPLMA wrong (all)", "  its false positives",
                    "  its false negatives"]:
            if f"_n|{pop}" not in r:
                continue
            print(f"\n  {pop}   n={r[f'_n|{pop}']['n']:.0f}"
                  f"   (SAPLMA here: accuracy 0.0%, MCC -1.000 by construction)")
            print(f"    {'method':16s}{'rescued':>10s}{'null':>9s}{'ratio':>8s}"
                  f"{'MCC':>9s}{'kappa':>9s}{'AUROC':>9s}")
            for n in order:
                k = f"{n}|{pop}"
                if k not in r:
                    continue
                v = r[k]
                star = " <-" if n.startswith("vote") else ""
                print(f"    {n:16s}{v['acc']:10.1%}{v['null']:9.1%}"
                      f"{v['acc'] / v['null']:8.2f}{v['mcc']:9.3f}{v['kappa']:9.3f}"
                      f"{v['auroc']:9.4f}{star}")
            # Index slicing here breaks whenever OTHERS changes length; name the vote
            # rules explicitly so adding a method cannot silently count it as a vote.
            VOTES = ["vote_hard", "vote_soft", "vote_rank"]
            best_single = max(r[f"{m}|{pop}"]["acc"] for m in OTHERS)
            best_vote = max(r[f"{n}|{pop}"]["acc"] for n in VOTES)
            print(f"    -> best vote {best_vote:.1%} vs best single {best_single:.1%}"
                  f"  ({best_vote - best_single:+.1%})")

    write_json(Path("runs/vote_on_saplma_errors.json"), out)
    print("\nwrote runs/vote_on_saplma_errors.json")


if __name__ == "__main__":
    main()
