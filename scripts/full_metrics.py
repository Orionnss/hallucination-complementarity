"""Accuracy, balanced accuracy, MCC, AUROC and F1 for every method, model and dataset.

Accuracy alone is misleading at these base rates: hallucination rates range from 0.148
(CoQA under Qwen3-14B) to 0.657 (NQ-Open under gemma-3-4b), so a constant predictor
scores anywhere from 85% to 66% depending on the slice. Balanced accuracy — the mean of
per-class recall — is the accuracy figure that survives that, and is reported alongside.

A majority-class baseline is included per slice so every accuracy has its floor visible.

Usage: uv run python scripts/full_metrics.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from halluc.config import Config
from halluc.io import write_json
from halluc.pipeline.stage5_posthoc import load_seed

LABELS = {"main": "Qwen3-14B", "gemma3-4b": "gemma-3-4b", "llama3.2-3b": "Llama-3.2-3B"}
SCOPES = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]


def metrics_for_run(run_id: str, seeds: list[int]) -> dict:
    cfg = Config(); cfg.run_id = run_id
    acc: dict[str, dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for seed in seeds:
        sd = load_seed(cfg, seed)
        if sd is None:
            continue
        y, ds = sd["y"], sd["dataset"]
        base = [m for m in sd["methods"] if not m.startswith("union")]

        any_correct = np.zeros(len(y), bool)
        for m in base:
            any_correct |= sd["preds"][m] == y
        oracle = np.where(any_correct, y, 1 - y)

        for scope in SCOPES:
            mask = np.ones(len(y), bool) if scope == "pooled" else (ds == scope)
            if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
                continue
            yy = y[mask]
            acc[scope]["_slice"]["n"].append(int(mask.sum()))
            acc[scope]["_slice"]["positive_rate"].append(float(yy.mean()))
            # Floor for the accuracy column: always predict the majority class.
            acc[scope]["_slice"]["majority_accuracy"].append(float(max(yy.mean(), 1 - yy.mean())))

            entries = [(m, sd["preds"][m][mask], sd["scores"][m][mask]) for m in sd["methods"]]
            entries.append(("ORACLE", oracle[mask], None))
            for name, pred, score in entries:
                d = acc[scope][name]
                d["accuracy"].append(float(accuracy_score(yy, pred)))
                d["balanced_accuracy"].append(float(balanced_accuracy_score(yy, pred)))
                d["mcc"].append(float(matthews_corrcoef(yy, pred)))
                d["f1"].append(float(f1_score(yy, pred, zero_division=0)))
                if score is not None:
                    d["auroc"].append(float(roc_auc_score(yy, score)))

    return {
        scope: {
            name: {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                   for k, v in d.items()}
            for name, d in by_name.items()
        }
        for scope, by_name in acc.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main", "gemma3-4b", "llama3.2-3b"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    results = {r: metrics_for_run(r, args.seeds) for r in args.runs}
    write_json(Path("runs/full_metrics.json"), results)

    for run in args.runs:
        print(f"\n{'=' * 96}\n{LABELS.get(run, run)}\n{'=' * 96}")
        for scope in SCOPES:
            if scope not in results[run]:
                continue
            sl = results[run][scope]["_slice"]
            print(f"\n  [{scope}]  n={sl['n']['mean']:.0f}  positive_rate={sl['positive_rate']['mean']:.3f}"
                  f"  majority-class accuracy={sl['majority_accuracy']['mean']:.3f}")
            print(f"    {'method':16s}{'acc':>8s}{'bal.acc':>9s}{'MCC':>8s}{'F1':>8s}{'AUROC':>8s}")
            rows = {k: v for k, v in results[run][scope].items() if k != "_slice"}
            for name, v in sorted(rows.items(), key=lambda kv: -kv[1]["mcc"]["mean"]):
                au = f"{v['auroc']['mean']:8.3f}" if "auroc" in v else f"{'-':>8s}"
                print(f"    {name:16s}{v['accuracy']['mean']:8.3f}{v['balanced_accuracy']['mean']:9.3f}"
                      f"{v['mcc']['mean']:8.3f}{v['f1']['mean']:8.3f}{au}")
    print("\nwrote runs/full_metrics.json")


if __name__ == "__main__":
    main()
