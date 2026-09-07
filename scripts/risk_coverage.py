"""Selective prediction: what a detector buys you if it is allowed to abstain.

Every metric in this study forces a decision on every answer. MCC, accuracy and F1 all
score a single operating point where the detector must label all 8,000 items. That is not
how a detector gets deployed. In deployment you hold a *score* per answer and use it to
decide what to act on: auto-approve the safest, route the riskiest to retrieval or a human,
and abstain in between. You never have to rule on everything.

Risk-coverage makes that measurable:

  coverage  fraction of answers kept (not abstained on)
  risk      hallucination rate among the answers kept

Sweeping the threshold traces a curve. At coverage 1.0 risk equals the base rate, by
definition. If the score carries information, risk falls as coverage drops. AURC is the
area under that curve — lower is better — and unlike AUROC it weights the high-precision
regime where the decision actually gets made, rather than treating every operating point
as equally interesting.

The mirror view is reported too: flag the k% riskiest for review, and measure precision
among the flagged. That is the same curve read from the other end, and it is the number to
quote when the use case is triage rather than auto-approval.

Why this can change conclusions rather than only presentation:

  Two detectors with identical AUROC can differ sharply at 20% coverage — one may be
  excellent at isolating the very worst answers while the other is merely good on average.
  Combination gains, in particular, tend to concentrate in the tails and vanish in
  threshold-optimal MCC, so a combiner that loses on MCC may still win where it matters.

  On slices where accuracy cannot beat the majority class (CoQA at a 15% base rate), the
  detector looks worthless by accuracy and may still rank well. Risk-coverage separates
  "cannot label" from "cannot rank".

Usage: uv run python scripts/risk_coverage.py --run main
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import rankdata

from halluc.io import write_json

METHODS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline"]
COVERAGES = (1.0, 0.9, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1)
FLAG_RATES = (0.05, 0.10, 0.20, 0.30)


def risk_at(y, s, cov):
    """Hallucination rate among the `cov` fraction judged least risky."""
    n = max(1, int(round(cov * len(y))))
    keep = np.argsort(s, kind="stable")[:n]      # lowest scores = safest
    return float(y[keep].mean())


def precision_at(y, s, rate):
    """Share of true hallucinations among the `rate` fraction flagged as riskiest."""
    n = max(1, int(round(rate * len(y))))
    flag = np.argsort(-s, kind="stable")[:n]
    return float(y[flag].mean())


def aurc(y, s, grid=None):
    grid = grid if grid is not None else np.linspace(0.05, 1.0, 40)
    return float(np.mean([risk_at(y, s, c) for c in grid]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    acc = defaultdict(lambda: defaultdict(list))
    for si, f in enumerate(sorted(glob.glob(
            f"runs/{args.run}/stage5_posthoc/block_oof/*.npz"))):
        if si not in args.seeds:
            continue
        d = np.load(f, allow_pickle=True)
        y, ds_arr = d["y"], d["dataset"].astype(str)
        S = {m: d[f"scores__{m}"] for m in METHODS}
        # Soft vote over per-method ranks: raw probabilities live on different scales,
        # so averaging ranks is the scale-free way to pool them.
        S["vote_soft"] = np.mean([rankdata(S[m]) / len(y) for m in METHODS], axis=0)
        S["saplma+lapeig"] = np.mean(
            [rankdata(S[m]) / len(y) for m in ("saplma", "lapeigvals")], axis=0)
        S["random"] = np.random.default_rng(si).random(len(y))

        for scope in ["pooled"] + sorted(set(ds_arr)):
            msk = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
            yy = y[msk]
            for name, s in S.items():
                ss = s[msk]
                k = (name, scope)
                acc[k]["aurc"].append(aurc(yy, ss))
                for c in COVERAGES:
                    acc[k][f"risk@{c}"].append(risk_at(yy, ss, c))
                for r in FLAG_RATES:
                    acc[k][f"prec@{r}"].append(precision_at(yy, ss, r))

    names = METHODS + ["vote_soft", "saplma+lapeig", "random"]
    for scope in ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]:
        if (names[0], scope) not in acc:
            continue
        base = np.mean(acc[(names[0], scope)]["risk@1.0"])
        print(f"\n{'=' * 92}\n  {args.run} · {scope} · base hallucination rate "
              f"{base:.1%}\n{'=' * 92}")
        print(f"  {'method':16s}{'AURC':>8s}" +
              "".join(f"{f'risk@{int(c * 100)}%':>10s}" for c in COVERAGES[1:]) +
              f"{'prec@10%':>10s}")
        for m in names:
            if (m, scope) not in acc:
                continue
            v = acc[(m, scope)]
            print(f"  {m:16s}{np.mean(v['aurc']):8.4f}" +
                  "".join(f"{np.mean(v[f'risk@{c}']):10.1%}" for c in COVERAGES[1:]) +
                  f"{np.mean(v['prec@0.1']):10.1%}")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/risk_coverage.json"),
               {f"{m}|{s}": {k: round(float(np.mean(v)), 4) for k, v in dd.items()}
                for (m, s), dd in acc.items()})
    print(f"\nwrote runs/{args.run}/stage5_posthoc/risk_coverage.json")


if __name__ == "__main__":
    main()
