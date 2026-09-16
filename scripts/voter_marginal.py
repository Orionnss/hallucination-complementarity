"""What each voter is worth: leave-one-out cost and mean marginal contribution.

Reads the 31-subset sweep in runs/voter_ablation.json and asks two questions the k-sweep
cannot answer on its own, because a mean over subsets of size k hides which member is
carrying it.

  remove   full five-detector vote minus one voter. The cost of dropping that voter from
           the deployed ensemble. One number per voter.
  add      the voter's mean marginal contribution: for every subset S not containing v,
           the change from vote(S) to vote(S + v), averaged over all 16 such subsets with
           Shapley weights. With five players this is the exact Shapley value, not an
           estimate, so the five values sum to the difference between the full vote and
           the empty one.

The two disagree whenever a voter is redundant with another. A voter can be worth little
to remove -- because a near-duplicate covers for it -- while still having a large average
marginal contribution over subsets where that duplicate is absent. That gap is the
quantity of interest here, so both are reported side by side rather than one summarised.

Marginal contributions are computed per rule, since the rules weight members differently:
hard voting is insensitive to a member's score scale, soft voting is not.

The k=1 term of the Shapley sum uses the subset sweep's own k=1 row, which re-thresholds a
single detector by the same cross-fitting rule the combiners use. That keeps the telescoping
sum internally consistent. It is NOT the detector's published figure, and the two differ by
about +0.035 MCC on Qwen3-14B, so these values measure marginal contribution inside the
ensemble and must not be quoted as single-detector performance.

Usage: uv run python scripts/voter_marginal.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from halluc.io import write_json

SHORT = ["SAP", "LAP", "ICR", "ATT", "SVD"]
FULL = {"SAP": "saplma", "LAP": "lapeigvals", "ICR": "icr",
        "ATT": "attn_baseline", "SVD": "svd_baseline"}
RULES = ["soft", "rank", "hard"]
SCOPES = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]


def value(blob, scope, rule, members, metric):
    """Performance of the vote over `members`, or None if that cell is absent."""
    if not members:
        return None
    tag = "+".join(m for m in SHORT if m in members)
    key = f"{scope}|{rule}|{len(members)}|{tag}"
    v = blob.get(key, {}).get(metric)
    return float(np.mean(v)) if v else None


def shapley(blob, scope, rule, metric):
    """Exact Shapley value per voter. n=5, so all 16 coalitions per player are enumerated."""
    n = len(SHORT)
    phi = {}
    for v in SHORT:
        others = [m for m in SHORT if m != v]
        total = 0.0
        ok = True
        for k in range(len(others) + 1):
            for sub in itertools.combinations(others, k):
                a = value(blob, scope, rule, set(sub) | {v}, metric)
                b = value(blob, scope, rule, set(sub), metric) if sub else 0.0
                if a is None or b is None:
                    ok = False; break
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                total += w * (a - b)
            if not ok:
                break
        phi[v] = total if ok else float("nan")
    return phi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="runs/voter_ablation.json")
    ap.add_argument("--scope", default="pooled")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"{src} not found -- run scripts/voter_ablation.py first")
    data = json.loads(src.read_text())

    out, rows = {}, []
    for run, blob in data.items():
        out[run] = {}
        for scope in SCOPES:
            for rule in RULES:
                full = value(blob, scope, rule, set(SHORT), "mcc")
                if full is None:
                    continue
                for metric in ("mcc", "auroc"):
                    f = value(blob, scope, rule, set(SHORT), metric)
                    phi = shapley(blob, scope, rule, metric)
                    for v in SHORT:
                        loo = value(blob, scope, rule, set(SHORT) - {v}, metric)
                        rec = dict(generator=run, dataset=scope, rule=rule, metric=metric,
                                   voter=FULL[v], full=round(f, 6),
                                   without=round(loo, 6) if loo is not None else None,
                                   remove_cost=round(f - loo, 6) if loo is not None else None,
                                   shapley=round(phi[v], 6))
                        rows.append(rec)
                        out[run][f"{scope}|{rule}|{metric}|{FULL[v]}"] = rec

    write_json(Path("runs/voter_marginal.json"), out)
    with open("runs/voter_marginal.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    for metric in ("mcc", "auroc"):
        print(f"\n{'=' * 104}\n  {metric.upper()}   scope={args.scope}"
              f"\n  remove = full vote minus this voter (higher = costlier to drop)"
              f"\n  shapley = mean marginal contribution over all 16 coalitions"
              f"\n{'=' * 104}")
        for run in data:
            sel = [r for r in rows if r["generator"] == run and r["dataset"] == args.scope
                   and r["metric"] == metric]
            if not sel:
                continue
            print(f"\n  {run}")
            for rule in RULES:
                rr = [r for r in sel if r["rule"] == rule]
                if not rr:
                    continue
                print(f"    {rule:6s} full={rr[0]['full']:.4f}"
                      f"   {'voter':16s}{'without':>10s}{'remove':>10s}{'shapley':>10s}")
                for r in sorted(rr, key=lambda r: -(r["shapley"] or 0)):
                    wo = f"{r['without']:.4f}" if r["without"] is not None else "-"
                    rc = f"{r['remove_cost']:+.4f}" if r["remove_cost"] is not None else "-"
                    print(f"    {'':22s}   {r['voter']:16s}{wo:>10s}{rc:>10s}"
                          f"{r['shapley']:>+10.4f}")
                s = sum(r["shapley"] for r in rr)
                print(f"    {'':22s}   {'sum of shapley':16s}{'':>10s}{'':>10s}{s:>+10.4f}")
    print("\nwrote runs/voter_marginal.json and .csv")


if __name__ == "__main__":
    main()
