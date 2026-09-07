"""Oracle upper bound per model and per dataset.

For each answer, the oracle is right whenever *any* individual method is right, and
wrong only when every method is wrong. It is the ceiling a perfect selector over the
existing methods could reach — unreachable in practice (choosing the right method needs
the label), but it bounds what the current feature set can possibly support.

Union models are excluded from the pool: they are combinations, not individual methods,
so including them would fold a combiner into the bound it is meant to bound.

Reported pooled and per dataset, mean +/- sd over seeds.

Usage: uv run python scripts/oracle_bound.py [--runs main gemma3-4b llama3.2-3b]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluc.config import Config
from halluc.io import write_json
from halluc.pipeline.stage5_posthoc import _fast_mcc, load_seed

RUN_LABELS = {"main": "Qwen3-14B", "gemma3-4b": "gemma-3-4b", "llama3.2-3b": "Llama-3.2-3B"}


def oracle_for_run(run_id: str, seeds: list[int]) -> dict:
    cfg = Config()
    cfg.run_id = run_id
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        sd = load_seed(cfg, seed)
        if sd is None:
            continue
        y, ds = sd["y"], sd["dataset"]
        base = [m for m in sd["methods"] if not m.startswith("union")]
        correct = {m: sd["preds"][m] == y for m in base}

        any_correct = np.zeros(len(y), bool)
        for m in base:
            any_correct |= correct[m]
        # Right wherever some method is right; wrong only where all of them fail.
        oracle_pred = np.where(any_correct, y, 1 - y)

        for scope, mask in [("pooled", np.ones(len(y), bool))] + [
            (d, ds == d) for d in sorted(set(ds))
        ]:
            if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
                continue
            a = acc[scope]
            a["n"].append(int(mask.sum()))
            a["positive_rate"].append(float(y[mask].mean()))
            a["coverage"].append(float(any_correct[mask].mean()))
            a["oracle_mcc"].append(_fast_mcc(y[mask], oracle_pred[mask]))
            a["best_single_mcc"].append(max(_fast_mcc(y[mask], sd["preds"][m][mask]) for m in base))
            for m in sd["methods"]:
                a[f"mcc__{m}"].append(_fast_mcc(y[mask], sd["preds"][m][mask]))

    return {
        scope: {
            k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
            for k, v in d.items()
        }
        for scope, d in acc.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main", "gemma3-4b", "llama3.2-3b"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    results = {r: oracle_for_run(r, args.seeds) for r in args.runs}
    write_json(Path("runs/oracle_bound.json"), results)

    scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
    for run in args.runs:
        label = RUN_LABELS.get(run, run)
        print(f"\n=== {label} ({run}) ===")
        print(f"{'scope':10s}{'n':>6s}{'pos':>7s}{'best single':>13s}{'union_equal':>13s}"
              f"{'ORACLE':>9s}{'coverage':>10s}{'unexploited':>13s}")
        for s in scopes:
            if s not in results[run]:
                continue
            d = results[run][s]
            bs, orc = d["best_single_mcc"]["mean"], d["oracle_mcc"]["mean"]
            ue = d["mcc__union_equal"]["mean"]
            print(f"{s:10s}{d['n']['mean']:6.0f}{d['positive_rate']['mean']:7.3f}"
                  f"{bs:13.3f}{ue:13.3f}{orc:9.3f}{d['coverage']['mean']:10.1%}"
                  f"{orc - ue:13.3f}")
    print("\nwrote runs/oracle_bound.json")


if __name__ == "__main__":
    main()
