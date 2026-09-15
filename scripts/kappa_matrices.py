"""Two kappa matrices over the five published detectors, per dataset and per generator.

  Table A  kappa(P_a, P_b) -- how much two detectors agree with EACH OTHER, ignoring the
           label. Symmetric. High kappa means the two make the same calls, which is what
           makes combining them pointless regardless of how accurate either one is.

  Table B  kappa(y, P_b) restricted to the items A got WRONG -- how much B agrees with the
           judge reference on A's error slice. Asymmetric: (A, B) and (B, A) are different
           questions. This is the "can B rescue A" measure.

Table B cannot be read on its own, and that is the whole reason the null column exists.
Restricting to one method's errors is collider conditioning: any B correlated with A looks
bad on that slice no matter how good it is. So each cell also carries

  rescue(A,B) = accuracy of B on the items A gets wrong
  null(A,B)   = B's per-class accuracy over ALL items, reweighted to the class mix of A's
                error slice -- what an INDEPENDENT method of B's exact strength would reach
  ratio       = rescue / null.  1.0 = independent, < 1 = B fails where A fails

The diagonal of Table B is undefined by construction -- A rescues none of its own errors --
and is left empty rather than filled with a meaningless zero.

Both tables use each detector under its OWN published probe by default (stage 3's stored
out-of-fold predictions). `--source pcalr` switches to the shared PCA-128 + logreg reader
in stage5_posthoc/block_oof. The two give different answers because the reader changes
which methods are strong and how correlated their errors are, so the source is recorded
in the output.

Per-dataset slices are small once conditioned -- a method's errors inside one dataset can
fall to a few hundred items -- so every Table B cell records the mean slice size and cells
below --min-slice are dropped rather than reported as noise.

Usage: uv run python scripts/kappa_matrices.py --source published
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

from sklearn.metrics import cohen_kappa_score

from halluc.io import write_json

METHODS = ["saplma", "lapeigvals", "icr", "attn_baseline", "svd_baseline"]
SHORT = {"saplma": "SAPLMA", "lapeigvals": "LapEig", "icr": "ICR",
         "attn_baseline": "Attn", "svd_baseline": "SVD"}
RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b",
        "llama3.2-3b-base", "gemma3-12b-pt"]
SCOPES = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]


def source_files(run: str, source: str) -> list[str]:
    if source == "published":
        return sorted(glob.glob(f"runs/{run}/stage3_train/pooled/seed*/predictions.npz"),
                      key=lambda p: int(Path(p).parent.name.replace("seed", "")))
    return sorted(glob.glob(f"runs/{run}/stage5_posthoc/block_oof/seed*.npz"),
                  key=lambda p: int(Path(p).stem.replace("seed", "")))


def kappa(a, b) -> float:
    """Cohen's kappa, guarding the degenerate case of a constant vector."""
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def run_one(run: str, source: str, min_slice: int):
    files = source_files(run, source)
    if not files:
        return None
    pair = defaultdict(lambda: defaultdict(list))     # Table A
    cond = defaultdict(lambda: defaultdict(list))     # Table B

    for f in files:
        d = np.load(f, allow_pickle=True)
        if any(f"preds__{m}" not in d.files for m in METHODS):
            return None
        y = d["y"].astype(int)
        ds = d["dataset"].astype(str)
        P = {m: d[f"preds__{m}"].astype(int) for m in METHODS}

        for scope in SCOPES:
            sel = np.ones(len(y), bool) if scope == "pooled" else (ds == scope)
            ys, Ps = y[sel], {m: P[m][sel] for m in METHODS}

            for i, a in enumerate(METHODS):
                for b in METHODS[i + 1:]:
                    pair[(scope, a, b)]["kappa"].append(kappa(Ps[a], Ps[b]))
                    pair[(scope, a, b)]["agree"].append(float((Ps[a] == Ps[b]).mean()))

            for a in METHODS:
                wrong = Ps[a] != ys
                if wrong.sum() < min_slice or len(np.unique(ys[wrong])) < 2:
                    continue
                mix = {c: float((ys[wrong] == c).mean()) for c in (0, 1)}
                for b in METHODS:
                    if a == b:
                        continue
                    p = Ps[b]
                    resc = float((p[wrong] == ys[wrong]).mean())
                    per_c = {c: float((p[ys == c] == c).mean()) for c in (0, 1)}
                    null = float(sum(per_c[c] * mix[c] for c in (0, 1)))
                    k = cond[(scope, a, b)]
                    k["kappa"].append(kappa(ys[wrong], p[wrong]))
                    k["rescue"].append(resc)
                    k["null"].append(null)
                    k["ratio"].append(resc / null if null else float("nan"))
                    k["n"].append(float(wrong.sum()))

    agg = lambda dd: {f"{a}|{b}|{s}" if False else k: {m: [round(float(x), 6) for x in v]
                      for m, v in d.items()} for k, d in dd.items()}
    return {
        "pairwise": {f"{s}|{a}|{b}": {m: [round(float(x), 6) for x in v]
                                      for m, v in d.items()}
                     for (s, a, b), d in pair.items()},
        "conditional": {f"{s}|{a}|{b}": {m: [round(float(x), 6) for x in v]
                                         for m, v in d.items()}
                        for (s, a, b), d in cond.items()},
        "n_seeds": len(files),
    }


def mean(d, key, metric):
    v = d.get(key, {}).get(metric)
    return float(np.mean(v)) if v else None


def report(out, scopes):
    for run, blob in out.items():
        print(f"\n{'=' * 92}\n  {run}   ({blob['n_seeds']} seeds)\n{'=' * 92}")
        for scope in scopes:
            print(f"\n  --- {scope} ---")
            print(f"\n  TABLE A  kappa between methods (symmetric, label ignored)")
            print(f"    {'':10s}" + "".join(f"{SHORT[m]:>9s}" for m in METHODS))
            for i, a in enumerate(METHODS):
                row = f"    {SHORT[a]:10s}"
                for j, b in enumerate(METHODS):
                    if i == j:
                        row += f"{'—':>9s}"
                    else:
                        x, yk = (a, b) if i < j else (b, a)
                        v = mean(blob["pairwise"], f"{scope}|{x}|{yk}", "kappa")
                        row += f"{v:9.3f}" if v is not None else f"{'.':>9s}"
                print(row)

            print(f"\n  TABLE B  rows = A (whose errors), cols = B (the rescuer)")
            for metric, lab in (("kappa", "kappa(y, B) on A's errors"),
                                ("ratio", "rescue / independence null")):
                print(f"    {lab}")
                print(f"    {'':10s}" + "".join(f"{SHORT[m]:>9s}" for m in METHODS))
                for a in METHODS:
                    row = f"    {SHORT[a]:10s}"
                    for b in METHODS:
                        if a == b:
                            row += f"{'—':>9s}"
                            continue
                        v = mean(blob["conditional"], f"{scope}|{a}|{b}", metric)
                        row += f"{v:9.3f}" if v is not None else f"{'.':>9s}"
                    # slice size is a property of A alone, so read it from any b != a
                    n = next((mean(blob["conditional"], f"{scope}|{a}|{b}", "n")
                              for b in METHODS if b != a
                              and f"{scope}|{a}|{b}" in blob["conditional"]), None)
                    print(row + (f"   n={n:.0f}" if n else ""))
                print()


def write_csv(out, path: Path, source: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["table", "generator", "source", "dataset", "method_a", "method_b",
                    "metric", "mean", "sd", "n_seeds"])
        for run, blob in out.items():
            for tbl in ("pairwise", "conditional"):
                for key, d in blob[tbl].items():
                    scope, a, b = key.split("|")
                    for metric, vals in d.items():
                        w.writerow([tbl, run, source, scope, a, b, metric,
                                    round(float(np.mean(vals)), 6),
                                    round(float(np.std(vals)), 6), len(vals)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--source", choices=("published", "pcalr"), default="published")
    ap.add_argument("--min-slice", type=int, default=50,
                    help="drop a conditional cell whose error slice is smaller than this")
    ap.add_argument("--scopes", nargs="*", default=SCOPES)
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        res = run_one(run, args.source, args.min_slice)
        if res is None:
            print(f"  {run}: no usable predictions for --source {args.source}, skipped")
            continue
        out[run] = res
        print(f"  {run}: {res['n_seeds']} seeds", flush=True)

    tag = "" if args.source == "published" else "_pcalr"
    write_json(Path(f"runs/kappa_matrices{tag}.json"), {"source": args.source, **out})
    write_csv(out, Path(f"runs/kappa_matrices{tag}.csv"), args.source)
    report(out, args.scopes)
    print(f"\nsource: {args.source}")
    print(f"wrote runs/kappa_matrices{tag}.json and .csv")


if __name__ == "__main__":
    main()
