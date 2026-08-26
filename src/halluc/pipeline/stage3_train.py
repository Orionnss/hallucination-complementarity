"""Stage 3: nested, grouped cross-validation of every detector, repeated across seeds.

By default every dataset is **pooled into one training set** and the resulting detector
is **evaluated per dataset**, so the numbers answer "does one probe trained on
everything work everywhere?" rather than "does a probe fitted to TriviaQA work on
TriviaQA". `--scope per_dataset` restores independent per-dataset training.

Guards that matter for the result being real:

* **Grouped folds.** CoQA turns share a story and SQuAD questions share a paragraph, so
  random folds would put near-duplicates on both sides of the split. Group ids are
  dataset-prefixed, so they stay unique once pooled.
* **Folds stratified by (dataset, label).** Pooling four datasets with different
  hallucination rates would otherwise let a fold drift toward one dataset, and its
  per-dataset metrics would be computed on a handful of rows.
* **Nothing is fit on test.** Scalers, PCA, probe weights, hyperparameters and the
  decision threshold all come from the training folds only.
* **One global threshold.** The model is global, so its threshold is too — tuned on the
  inner validation split. Per-dataset thresholds are also recorded, to show how much a
  single operating point costs on each dataset.
* **INVALID dropped.** Refusals are excluded from training and evaluation alike.

The seed selects which samples are drawn from each dataset's pool, and also drives the
fold partition and probe initialisation.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ..config import Config
from ..detectors import UnionDetector, build_detectors
from ..eval.metrics import best_threshold, score_predictions
from ..io import load_features, provenance, read_json, write_json
from ..judges.base import Label


def load_arrays(cfg: Config, dataset_name: str, blocks: list[str]) -> dict:
    """Scored items only, with features, labels and CV groups aligned by item id."""
    labels_path = cfg.stage_dir("stage2_judge", dataset_name) / "labels.json"
    manifest_path = cfg.stage_dir("stage1_extract", dataset_name) / "manifest.json"
    labels = read_json(labels_path)["labels"]
    manifest = {r["item_id"]: r for r in read_json(manifest_path)["items"]}

    kept = [e for e in labels if e["scored"] and e["item_id"] in manifest]
    item_ids = [e["item_id"] for e in kept]
    y = np.array([int(e["label"] == Label.HALLUCINATED.value) for e in kept], dtype=int)
    groups = np.array([manifest[i]["group_id"] for i in item_ids])
    shard_dir = cfg.stage_dir("stage1_extract", dataset_name)
    data = {name: load_features(shard_dir, name, item_ids) for name in blocks}
    return {
        "item_ids": item_ids,
        "y": y,
        "groups": groups,
        "dataset": np.array([dataset_name] * len(item_ids)),
        "data": data,
    }


def draw_and_combine(arrays_by_dataset: dict[str, dict], n_per_dataset: int, seed: int) -> dict:
    """Draw `n_per_dataset` from each dataset's pool, then concatenate.

    Drawing per dataset rather than from the pooled set keeps the datasets balanced in
    the training set; a proportional draw would let the largest dataset dominate the
    shared probe and make the smaller datasets' evaluation a test of transfer only.
    """
    rng = np.random.default_rng(seed)
    picked = {}
    for name, arrays in arrays_by_dataset.items():
        available = len(arrays["y"])
        n_draw = min(n_per_dataset, available)
        picked[name] = np.sort(rng.choice(available, size=n_draw, replace=False))

    blocks = sorted(next(iter(arrays_by_dataset.values()))["data"])
    order = sorted(arrays_by_dataset)
    return {
        "y": np.concatenate([arrays_by_dataset[n]["y"][picked[n]] for n in order]),
        "groups": np.concatenate([arrays_by_dataset[n]["groups"][picked[n]] for n in order]),
        "dataset": np.concatenate([arrays_by_dataset[n]["dataset"][picked[n]] for n in order]),
        "item_ids": [i for n in order for i in np.array(arrays_by_dataset[n]["item_ids"])[picked[n]]],
        "data": {
            b: np.concatenate([arrays_by_dataset[n]["data"][b][picked[n]] for n in order])
            for b in blocks
        },
    }


def _subset(data: dict[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    return {name: array[index] for name, array in data.items()}


def _fit_score(detector, params, seed, X_fit, y_fit, X_eval) -> np.ndarray:
    model = detector.estimator(params, seed)
    model.fit(X_fit, y_fit)
    return model.predict_proba(X_eval)[:, 1]


def _with_slices(detector, params, data):
    if isinstance(detector, UnionDetector) and detector.equal:
        return {**params, "slices": detector.block_slices(data, params)}
    return params


def run_fold(detector, seed, combined, y, train_idx, test_idx, n_jobs) -> dict:
    """One outer fold for one detector: inner tuning, threshold, refit, test."""
    data = combined["data"]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    strat_train = combined["strat"][train_idx]
    rel_train, rel_val = next(
        inner.split(np.zeros(len(train_idx)), strat_train, combined["groups"][train_idx])
    )
    inner_train, inner_val = train_idx[rel_train], train_idx[rel_val]

    def evaluate(params):
        p = _with_slices(detector, params, _subset(data, inner_train))
        scores = _fit_score(
            detector, p, seed,
            detector.matrix(_subset(data, inner_train), p),
            y[inner_train],
            detector.matrix(_subset(data, inner_val), p),
        )
        auroc = roc_auc_score(y[inner_val], scores) if len(np.unique(y[inner_val])) > 1 else 0.5
        return float(auroc), params

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(evaluate)(params) for params in detector.grid
    )
    best_auroc, best_params = max(results, key=lambda r: r[0])

    # Threshold from the validation split, never from test.
    p = _with_slices(detector, best_params, _subset(data, inner_train))
    val_scores = _fit_score(
        detector, p, seed,
        detector.matrix(_subset(data, inner_train), p),
        y[inner_train],
        detector.matrix(_subset(data, inner_val), p),
    )
    threshold, val_mcc = best_threshold(y[inner_val], val_scores)
    # Per-dataset thresholds, to quantify what a single global operating point costs.
    per_dataset_threshold = {}
    val_datasets = combined["dataset"][inner_val]
    for name in np.unique(val_datasets):
        mask = val_datasets == name
        if mask.sum() > 1 and len(np.unique(y[inner_val][mask])) > 1:
            per_dataset_threshold[str(name)] = best_threshold(
                y[inner_val][mask], val_scores[mask]
            )[0]

    # Refit on the full training fold with the chosen hyperparameters.
    p = _with_slices(detector, best_params, _subset(data, train_idx))
    test_scores = _fit_score(
        detector, p, seed,
        detector.matrix(_subset(data, train_idx), p),
        y[train_idx],
        detector.matrix(_subset(data, test_idx), p),
    )
    return {
        "scores": test_scores,
        "threshold": threshold,
        "per_dataset_threshold": per_dataset_threshold,
        "best_params": {k: v for k, v in best_params.items() if k != "slices"},
        "inner_auroc": best_auroc,
        "inner_mcc": val_mcc,
    }


def run_seed(cfg, scope_name, seed, arrays_by_dataset, detectors, n_jobs) -> dict:
    combined = draw_and_combine(arrays_by_dataset, cfg.n_per_seed, seed)
    y, groups, datasets = combined["y"], combined["groups"], combined["dataset"]
    n = len(y)
    # Stratify on the (dataset, label) pair so every fold keeps the dataset mix and the
    # class balance; per-dataset metrics are otherwise computed on unstable slices.
    combined["strat"] = np.array([f"{d}_{lab}" for d, lab in zip(datasets, y)])

    outer = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=seed)
    folds = list(outer.split(np.zeros(n), combined["strat"], groups))

    oof_scores = {name: np.full(n, np.nan) for name in detectors}
    oof_preds = {name: np.full(n, -1, dtype=int) for name in detectors}
    fold_records: dict[str, list] = {name: [] for name in detectors}

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        for name, detector in detectors.items():
            started = time.perf_counter()
            result = run_fold(detector, seed, combined, y, train_idx, test_idx, n_jobs)
            oof_scores[name][test_idx] = result["scores"]
            oof_preds[name][test_idx] = (result["scores"] >= result["threshold"]).astype(int)
            fold_records[name].append(
                {
                    "fold": fold_id,
                    "threshold": result["threshold"],
                    "per_dataset_threshold": result["per_dataset_threshold"],
                    "best_params": result["best_params"],
                    "inner_auroc": round(result["inner_auroc"], 4),
                    "seconds": round(time.perf_counter() - started, 1),
                    **score_predictions(y[test_idx], result["scores"], result["threshold"]),
                }
            )
        print(f"  [{scope_name} seed={seed}] fold {fold_id + 1}/{len(folds)} done")

    out_dir = cfg.stage_dir("stage3_train", scope_name, f"seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        y=y,
        item_ids=np.array(combined["item_ids"]),
        groups=groups,
        dataset=datasets,
        **{f"scores__{k}": v for k, v in oof_scores.items()},
        **{f"preds__{k}": v for k, v in oof_preds.items()},
    )

    # Evaluate per dataset on the pooled out-of-fold predictions.
    per_method = {}
    for name in detectors:
        by_dataset = {}
        for ds in sorted(set(datasets)):
            mask = datasets == ds
            by_dataset[ds] = {
                "n": int(mask.sum()),
                "positive_rate": float(y[mask].mean()),
                "auroc": (
                    float(roc_auc_score(y[mask], oof_scores[name][mask]))
                    if len(np.unique(y[mask])) > 1 else None
                ),
                "mcc": float(matthews_corrcoef(y[mask], oof_preds[name][mask])),
            }
        per_method[name] = {
            "per_dataset": by_dataset,
            "pooled_auroc": (
                float(roc_auc_score(y, oof_scores[name])) if len(np.unique(y)) > 1 else None
            ),
            "pooled_mcc": float(matthews_corrcoef(y, oof_preds[name])),
            "folds": fold_records[name],
        }

    summary = {
        "scope": scope_name,
        "seed": seed,
        "n_samples": n,
        "datasets": {ds: int((datasets == ds).sum()) for ds in sorted(set(datasets))},
        "positive_rate": float(y.mean()),
        "n_groups": int(len(set(groups))),
        "per_method": per_method,
        "provenance": provenance(),
    }
    write_json(out_dir / "metrics.json", summary)
    ranked = sorted(per_method.items(), key=lambda kv: -kv[1]["pooled_mcc"])
    print(f"  [{scope_name} seed={seed}] pooled MCC: " + "  ".join(
        f"{n}={m['pooled_mcc']:.3f}" for n, m in ranked
    ))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--scope",
        choices=("pooled", "per_dataset"),
        default=None,
        help="pooled (default): one probe trained on all datasets, evaluated per dataset",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.datasets:
        cfg.datasets = args.datasets
    if args.seeds:
        cfg.seeds = args.seeds
    scope = args.scope or cfg.training_scope

    # Probe-layer grids depend on the generator's depth, so build a throwaway detector
    # set to learn which blocks to load, then rebuild once the real depth is known.
    needed = sorted({b for d in build_detectors().values() for b in d.blocks} | {"icr_mean"})
    arrays_by_dataset = {name: load_arrays(cfg, name, needed) for name in cfg.datasets}
    n_layers = next(iter(arrays_by_dataset.values()))["data"]["saplma"].shape[1] - 1
    detectors = build_detectors(n_layers=n_layers)
    print(f"generator depth: {n_layers} layers -> saplma probe grid "
          f"{[g['layer'] for g in detectors['saplma'].grid]}")
    for name, arrays in arrays_by_dataset.items():
        print(
            f"[{name}] scored={len(arrays['y'])} "
            f"positive_rate={arrays['y'].mean():.3f} groups={len(set(arrays['groups']))}"
        )

    if scope == "pooled":
        print(f"training scope: pooled over {len(arrays_by_dataset)} datasets, evaluated per dataset")
        for seed in cfg.seeds:
            run_seed(cfg, "pooled", seed, arrays_by_dataset, detectors, args.n_jobs)
    else:
        for name, arrays in arrays_by_dataset.items():
            for seed in cfg.seeds:
                run_seed(cfg, name, seed, {name: arrays}, detectors, args.n_jobs)


if __name__ == "__main__":
    main()
