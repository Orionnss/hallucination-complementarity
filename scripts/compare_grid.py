"""Compare grid cells written by run_grid.py, seed by seed.

A cell is given as tag/cell, where tag is the run_grid.py --tag and cell is
<block>__<reader>. For every cell: each metric of binary_metrics, pooled and per dataset,
mean +/- sd over seeds. For every pair given with --pair: the paired difference per seed
(mean +/- sd), a paired t-test over seeds, and McNemar on the pooled decisions of each
seed (Holm-corrected across seeds).

Before any number is reported, every cell must have the same draw fingerprint and the
same item ids in the same order for each seed. Otherwise the cells did not see the same
data, and a difference between them would mean nothing.

--reference cell=MCC,AUROC checks one cell's pooled means against stage-3 values (for
example the RECOMPUTE_PUBLISHED.md table), to show the grid reproduces stage 3.

Usage:
  uv run python scripts/compare_grid.py --run main --name lapeigvals \\
      --cells official/lapeigvals_official__lapeigvals_official \\
              ours/lapeigvals__lapeigvals_stage3 \\
      --pair official/lapeigvals_official__lapeigvals_official ours/lapeigvals__lapeigvals_stage3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluc.config import Config
from halluc.eval.metrics import holm_bonferroni, mcnemar
from halluc.io import provenance, read_json, write_json

METRICS = ["auroc", "auprc", "mcc", "balanced_accuracy", "f1", "accuracy",
           "tpr_at_fpr05", "tpr_at_fpr10"]
SLICES = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]


def load_cell(root: Path, ref: str, seeds: list[int]) -> dict:
    tag, cell = ref.split("/", 1)
    out = {"metrics": [], "preds": [], "item_ids": [], "y": [], "fingerprint": []}
    for seed in seeds:
        d = root / tag / f"seed{seed}"
        m = read_json(d / "metrics.json")
        if cell not in m["cells"]:
            raise SystemExit(f"{ref}: no cell {cell} in {d}; has {sorted(m['cells'])}")
        npz = np.load(d / "predictions.npz", allow_pickle=True)
        out["metrics"].append(m["cells"][cell])
        out["preds"].append(npz[f"preds__{cell}"])
        out["item_ids"].append(npz["item_ids"])
        out["y"].append(npz["y"])
        out["fingerprint"].append(m["draw_fingerprint"])
    return out


def mean_sd(values: list[float]) -> dict:
    a = np.array([v for v in values if v is not None], dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "per_seed": [None if v is None else float(v) for v in values]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True, help="output file stem")
    ap.add_argument("--cells", nargs="+", required=True, help="tag/cell")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--scope", default="pooled",
                    help="grid scope dir; run_grid.py --n-per-seed N writes pooled_smokeN")
    ap.add_argument("--reference", action="append", default=[],
                    help="tag/cell=MCC,AUROC (pooled stage-3 means)")
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    root = cfg.stage_dir("grid", args.scope)
    cells = {ref: load_cell(root, ref, args.seeds) for ref in args.cells}

    # Same data, or stop.
    first = next(iter(cells.values()))
    for ref, c in cells.items():
        for i, seed in enumerate(args.seeds):
            if c["fingerprint"][i] != first["fingerprint"][i] or not np.array_equal(
                    c["item_ids"][i], first["item_ids"][i]):
                raise SystemExit(f"{ref} seed {seed}: different draw from {args.cells[0]}")

    summary = {}
    for ref, c in cells.items():
        summary[ref] = {
            "budget": c["metrics"][0]["budget"],
            "harness": {s: {m: mean_sd([c["metrics"][i]["harness"][s][m]
                                        for i in range(len(args.seeds))])
                            for m in METRICS} for s in SLICES},
            "native": ({s: {m: mean_sd([c["metrics"][i]["native"][s][m]
                                       for i in range(len(args.seeds))])
                            for m in ("mcc", "balanced_accuracy", "f1", "accuracy")}
                        for s in SLICES} if c["metrics"][0]["native"] else None),
            "best_params": [[f["best_params"] for f in c["metrics"][i]["folds"]]
                            for i in range(len(args.seeds))],
        }

    pairs = {}
    for a, b in args.pair:
        ca, cb = cells[a], cells[b]
        per_slice = {}
        for s in SLICES:
            per_slice[s] = {}
            for m in METRICS:
                va = np.array([x["harness"][s][m] for x in ca["metrics"]], dtype=float)
                vb = np.array([x["harness"][s][m] for x in cb["metrics"]], dtype=float)
                diff = va - vb
                p = float(ttest_rel(va, vb).pvalue) if len(diff) > 1 and diff.std() > 0 else None
                per_slice[s][m] = {"diff_mean": float(diff.mean()),
                                   "diff_sd": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                                   "paired_t_p": p, "a_wins": int((diff > 0).sum()),
                                   "n_seeds": len(diff)}
        tests = {f"seed{seed}": mcnemar(ca["y"][i], ca["preds"][i], cb["preds"][i])
                 for i, seed in enumerate(args.seeds)}
        holm = holm_bonferroni({k: v["p_value"] for k, v in tests.items()})
        pairs[f"{a} - {b}"] = {"metrics": per_slice, "mcnemar_pooled": tests, "mcnemar_holm": holm}

    checks = {}
    for spec in args.reference:
        ref, values = spec.split("=")
        mcc_ref, auroc_ref = (float(v) for v in values.split(","))
        got = summary[ref]["harness"]["pooled"]
        checks[ref] = {"mcc": [got["mcc"]["mean"], mcc_ref],
                       "auroc": [got["auroc"]["mean"], auroc_ref],
                       "match_1e-4": abs(got["mcc"]["mean"] - mcc_ref) < 1e-4
                       and abs(got["auroc"]["mean"] - auroc_ref) < 1e-4}

    out = {"run": args.run, "seeds": args.seeds, "fingerprints": first["fingerprint"],
           "cells": summary, "pairs": pairs, "reference_checks": checks,
           "provenance": provenance()}
    write_json(root / f"compare_{args.name}.json", out)

    lines = [f"# {args.name} — run {args.run}, seeds {args.seeds}", "",
             "Harness decision (threshold tuned on the inner split by MCC). Mean ± sd over seeds.", ""]
    for s in SLICES:
        lines += [f"## {s}", "", "| cell | AUROC | AUPRC | MCC | Bal. acc | F1 |", "|---|---|---|---|---|---|"]
        for ref in cells:
            h = summary[ref]["harness"][s]
            lines.append(f"| `{ref}` | " + " | ".join(
                f"{h[m]['mean']:.4f} ± {h[m]['sd']:.4f}"
                for m in ("auroc", "auprc", "mcc", "balanced_accuracy", "f1")) + " |")
        lines.append("")
    for name, pr in pairs.items():
        lines += [f"## Paired difference: `{name}`", "",
                  "| slice | ΔAUROC | p | ΔMCC | p | A wins (MCC) |", "|---|---|---|---|---|---|"]
        for s in SLICES:
            au, mc = pr["metrics"][s]["auroc"], pr["metrics"][s]["mcc"]
            fmt = lambda p: "—" if p is None else f"{p:.3g}"  # noqa: E731
            lines.append(f"| {s} | {au['diff_mean']:+.4f} ± {au['diff_sd']:.4f} | {fmt(au['paired_t_p'])} "
                         f"| {mc['diff_mean']:+.4f} ± {mc['diff_sd']:.4f} | {fmt(mc['paired_t_p'])} "
                         f"| {mc['a_wins']}/{mc['n_seeds']} |")
        sig = sum(v["significant"] for v in pr["mcnemar_holm"].values())
        lines += ["", f"McNemar on pooled decisions: significant (Holm, α=0.05) in {sig}/"
                  f"{len(pr['mcnemar_holm'])} seeds.", ""]
    for ref, c in checks.items():
        lines.append(f"Reference check `{ref}`: MCC {c['mcc'][0]:.4f} vs {c['mcc'][1]:.4f}, "
                     f"AUROC {c['auroc'][0]:.4f} vs {c['auroc'][1]:.4f} -> "
                     f"{'match' if c['match_1e-4'] else 'NO MATCH'}")
    (root / f"compare_{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
