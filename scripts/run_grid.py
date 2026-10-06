"""Run (block x reader) cells under stage 3's protocol (ADDING_A_METHOD.md §6).

Each cell goes through `stage3_train.run_fold` unchanged, on the same per-seed draw and
the same outer folds as stage 3: grouped, stratified on (dataset, label), inner split for
views, hyperparameters and the MCC threshold, refit on the full training fold. The draw
is fingerprinted (sha256 of the drawn item ids), so a grid seed can be checked against
the stage-3 seed it claims to match.

Output, runs/<run>/grid/pooled/<tag>/seed<k>/ (separate jobs use separate tags, so they
never overwrite each other's files):
  predictions.npz  scores__<cell>, preds__<cell> (harness threshold),
                   native_preds__<cell> (the method's own decision, when it has one),
                   y, dataset, groups, item_ids
  metrics.json     binary_metrics per dataset and pooled, for the harness and the native
                   decision; the budget of each cell; folds; the draw fingerprint

Cells are named block:reader. Example, the official method, the official features under
every shared reader, and our earlier features under the official reader:

  uv run python scripts/run_grid.py --run main --seeds 0 --tag official \\
      --cells lapeigvals_official:lapeigvals_official attneigvals_official:lapeigvals_official \\
              lapeigvals_official:pca_logreg lapeigvals:lapeigvals_official
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluc.config import Config
from halluc.eval.metrics import binary_metrics, score_predictions
from halluc.grid import BLOCKS, READERS, make_cell
from halluc.io import load_features, provenance, write_json
from halluc.pipeline.stage3_train import draw_and_combine, load_arrays, run_fold
import halluc.methods.lapeigvals.reader  # noqa: F401  (registers blocks and reader)

DATASETS = ["triviaqa", "nq_open", "squad_v2", "coqa"]


def load(cfg: Config, datasets: list[str], blocks: list[str]) -> dict[str, dict]:
    """stage3's load_arrays for ids, labels and groups; each block from its own store."""
    out = {}
    for ds in datasets:
        arrays = load_arrays(cfg, ds, [])
        for block in blocks:
            spec = BLOCKS.create(block)
            for name in spec.arrays:
                if name not in arrays["data"]:
                    arrays["data"][name] = load_features(spec.source(cfg, ds), name,
                                                         arrays["item_ids"])
        out[ds] = arrays
    return out


def fingerprint(item_ids) -> str:
    return hashlib.sha256("\n".join(item_ids).encode()).hexdigest()[:16]


def slice_metrics(y, datasets, scores, preds) -> dict:
    out = {"pooled": binary_metrics(y, scores, preds)}
    for ds in sorted(set(datasets)):
        mask = datasets == ds
        out[ds] = binary_metrics(y[mask], scores[mask], preds[mask])
    return out


def run_seed(cfg, seed, arrays, cells_spec, n_jobs, out_root) -> dict:
    combined = draw_and_combine(arrays, cfg.n_per_seed, seed)
    y, groups, datasets = combined["y"], combined["groups"], combined["dataset"]
    combined["strat"] = np.array([f"{d}_{lab}" for d, lab in zip(datasets, y)])
    outer = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=seed)
    folds = list(outer.split(np.zeros(len(y)), combined["strat"], groups))

    cells = {}
    for block, reader in cells_spec:
        cell = make_cell(block, reader, combined["data"])
        if not cell.grid:
            raise SystemExit(f"{cell.name}: empty grid for this draw (no k fits the shortest item?)")
        cells[cell.name] = cell

    scores = {n: np.full(len(y), np.nan) for n in cells}
    preds = {n: np.full(len(y), -1, dtype=int) for n in cells}
    records = {n: [] for n in cells}
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        for name, cell in cells.items():
            started = time.perf_counter()
            result = run_fold(cell, seed, combined, y, train_idx, test_idx, n_jobs)
            scores[name][test_idx] = result["scores"]
            preds[name][test_idx] = (result["scores"] >= result["threshold"]).astype(int)
            records[name].append({
                "fold": fold_id, "best_params": result["best_params"],
                "inner_auroc": round(result["inner_auroc"], 4),
                "seconds": round(time.perf_counter() - started, 1),
                **score_predictions(y[test_idx], result["scores"], result["threshold"]),
            })
            print(f"  seed {seed} fold {fold_id + 1}/{len(folds)} {name}: "
                  f"inner AUROC {result['inner_auroc']:.3f} {result['best_params']}", flush=True)

    native = {n: c.reader.native_decision(scores[n]) for n, c in cells.items()
              if c.reader.native_decision is not None}
    out_dir = out_root / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz", y=y, dataset=datasets, groups=groups,
        item_ids=np.array(combined["item_ids"]),
        **{f"scores__{n}": v for n, v in scores.items()},
        **{f"preds__{n}": v for n, v in preds.items()},
        **{f"native_preds__{n}": v for n, v in native.items()},
    )
    summary = {
        "seed": seed, "n_samples": int(len(y)), "n_per_seed": cfg.n_per_seed,
        "draw_fingerprint": fingerprint(combined["item_ids"]),
        "cells": {
            n: {
                "block": c.spec.name, "reader": c.reader.name, "reader_note": c.reader.note,
                "budget": {**c.budget(), "train_rows_per_fold": [len(t) for t, _ in folds]},
                "harness": slice_metrics(y, datasets, scores[n], preds[n]),
                "native": (slice_metrics(y, datasets, scores[n], native[n])
                           if n in native else None),
                "folds": records[n],
            }
            for n, c in cells.items()
        },
        "provenance": provenance(),
    }
    write_json(out_dir / "metrics.json", summary)
    for n in cells:
        m = summary["cells"][n]["harness"]["pooled"]
        print(f"  seed {seed} {n}: AUROC {m['auroc']:.4f} MCC {m['mcc']:.4f}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cells", nargs="+", required=True, help="block:reader")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--n-per-seed", type=int, default=None,
                    help="smaller draw for smoke tests; results then do not match stage 3")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--tag", required=True, help="output subdirectory for this set of cells")
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    if args.n_per_seed is not None:
        cfg.n_per_seed = args.n_per_seed
    cells_spec = [tuple(c.split(":")) for c in args.cells]
    for block, reader in cells_spec:
        if block not in BLOCKS or reader not in READERS:
            raise SystemExit(f"unknown cell {block}:{reader}; blocks={BLOCKS.names()} "
                             f"readers={READERS.names()}")
    arrays = load(cfg, args.datasets, sorted({b for b, _ in cells_spec}))
    scope = "pooled" if args.n_per_seed is None else f"pooled_smoke{args.n_per_seed}"
    out_root = cfg.stage_dir("grid", scope, args.tag)
    for seed in args.seeds:
        run_seed(cfg, seed, arrays, cells_spec, args.n_jobs, out_root)


if __name__ == "__main__":
    main()
