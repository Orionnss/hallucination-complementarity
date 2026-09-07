"""Cohen's kappa against the judge reference, restricted to the answers SAPLMA gets wrong.

Overall agreement is dominated by the easy answers every method gets right. Conditioning
on SAPLMA's errors asks the question that matters for combination: on the answers the
strongest single detector fails, does ICR still agree with the reference better than
chance — and does it agree well enough to be worth deferring to?

Kappa is the right statistic for this slice because the conditioning shifts the class
balance sharply (SAPLMA's errors are mostly false positives outside CoQA), and kappa
already discounts the agreement that shift would produce by itself.

Three populations are reported so the conditional number can be read against something:

  all            every answer - the standard agreement figure
  SAPLMA wrong   the conditional question
  SAPLMA right   the complement, as a control. A method that only agrees where SAPLMA
                 already succeeds carries no combination value, however high its overall
                 kappa looks.

Every method is included, not just ICR: a kappa near zero on SAPLMA's errors means little
on its own, but means a great deal if ICR is the only one near zero, or if all of them are.

Usage: uv run python scripts/kappa_on_saplma_errors.py
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

from sklearn.metrics import cohen_kappa_score

from halluc.io import write_json

METHODS = ["icr", "lapeigvals", "attn_baseline", "svd_baseline", "charm", "saplma"]
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b"]
POPS = ["all", "SAPLMA wrong", "SAPLMA right"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        acc = defaultdict(lambda: defaultdict(list))
        files = [f for i, f in enumerate(sorted(
            glob.glob(f"runs/{run}/stage5_posthoc/block_oof/*.npz"))) if i in args.seeds]
        for f in files:
            d = np.load(f, allow_pickle=True)
            y, ds_arr = d["y"], d["dataset"].astype(str)
            wrong = d["preds__saplma"] != y

            for scope in ["pooled", "pooled_no_coqa"] + sorted(set(ds_arr)):
                base = (np.ones(len(y), bool) if scope == "pooled"
                        else (ds_arr != "coqa") if scope == "pooled_no_coqa"
                        else (ds_arr == scope))
                for pop in POPS:
                    m = base & (wrong if pop == "SAPLMA wrong"
                                else ~wrong if pop == "SAPLMA right" else True)
                    if m.sum() < 20 or len(np.unique(y[m])) < 2:
                        continue
                    acc[(scope, pop)]["n"].append(int(m.sum()))
                    acc[(scope, pop)]["pos_rate"].append(float(y[m].mean()))
                    for meth in METHODS:
                        p = d[f"preds__{meth}"][m]
                        acc[(scope, pop)][meth].append(
                            float(cohen_kappa_score(y[m], p)) if len(np.unique(p)) > 1
                            else 0.0)
                        acc[(scope, pop)][f"{meth}__acc"].append(float((p == y[m]).mean()))
        out[run] = {f"{s}|{p}": {k: round(float(np.mean(v)), 4) for k, v in dd.items()}
                    for (s, p), dd in acc.items()}

        print(f"\n{'=' * 78}\n{run}: Cohen's kappa vs the judge reference\n{'=' * 78}")
        for scope in ["pooled", "pooled_no_coqa", "triviaqa", "nq_open", "squad_v2", "coqa"]:
            rows = [(p, acc[(scope, p)]) for p in POPS if (scope, p) in acc]
            if not rows:
                continue
            print(f"\n  {scope}")
            print(f"    {'population':15s}{'n':>6s}{'halluc%':>9s}"
                  + "".join(f"{m[:11]:>12s}" for m in METHODS))
            for pop, dd in rows:
                print(f"    {pop:15s}{np.mean(dd['n']):6.0f}{np.mean(dd['pos_rate']):9.1%}"
                      + "".join(f"{np.mean(dd[m]):12.4f}" for m in METHODS))

    write_json(Path("runs/kappa_on_saplma_errors.json"), out)
    print("\nwrote runs/kappa_on_saplma_errors.json")


if __name__ == "__main__":
    main()
