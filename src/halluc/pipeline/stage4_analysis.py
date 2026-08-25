"""Stage 4: complementarity analysis.

Stage 3 answers "how good is each detector". This stage answers the actual research
question: **do they fail on the same items?**

* Cohen's kappa between two detectors' decisions measures how far their agreement
  exceeds chance. Two strong detectors with low kappa are complementary — that is the
  finding the study is looking for.
* McNemar tests whether their disagreements are asymmetric, i.e. whether one is
  genuinely better rather than differently-wrong.
* Both are computed per (dataset, seed) on out-of-fold predictions, then aggregated
  across seeds, so a result that only holds for one sample draw is visible as spread.

Holm-Bonferroni is applied within each (dataset, seed) family of pairwise tests: with 7
predictors there are 21 pairs, and uncorrected p-values would manufacture significance.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np

from ..config import Config
from ..eval.metrics import holm_bonferroni, pairwise_agreement, score_predictions
from ..io import provenance, read_json, write_json


def _load_seed(cfg: Config, scope: str, seed: int, dataset_name: str | None = None):
    """Out-of-fold predictions for one seed, optionally restricted to one dataset.

    Under the pooled scope every seed file holds all datasets, so a per-dataset view is
    a mask over the same arrays — the predictions still come from the single probe
    trained on everything.
    """
    path = cfg.stage_dir("stage3_train", scope, f"seed{seed}") / "predictions.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    methods = sorted(k.removeprefix("preds__") for k in data.files if k.startswith("preds__"))
    mask = slice(None)
    if dataset_name is not None and "dataset" in data.files:
        mask = data["dataset"].astype(str) == dataset_name
        if not np.any(mask):
            return None
    return {
        "y": data["y"][mask],
        "preds": {m: data[f"preds__{m}"][mask] for m in methods},
        "scores": {m: data[f"scores__{m}"][mask] for m in methods},
        "methods": methods,
    }


def analyse_dataset(cfg: Config, dataset_name: str, scope: str | None = None) -> dict:
    """Complementarity analysis for one evaluation slice.

    `dataset_name` may be a real dataset (a mask over the pooled predictions) or the
    literal "pooled", meaning every item the shared probe was evaluated on.
    """
    scope = scope or cfg.training_scope
    per_seed, agreement_by_pair = [], defaultdict(list)
    method_scores = defaultdict(lambda: defaultdict(list))

    if scope == "pooled":
        source, slice_name = "pooled", (None if dataset_name == "pooled" else dataset_name)
    else:
        source, slice_name = dataset_name, None

    for seed in cfg.seeds:
        loaded = _load_seed(cfg, source, seed, slice_name)
        if loaded is None:
            continue
        y, preds = loaded["y"], loaded["preds"]

        for method in loaded["methods"]:
            metrics = score_predictions(y, loaded["scores"][method], 0.5)
            # MCC/accuracy come from the thresholds stage 3 already applied; recomputing
            # at 0.5 here would discard that tuning, so take the hard predictions as-is.
            hard = preds[method]
            from sklearn.metrics import accuracy_score, matthews_corrcoef

            method_scores[method]["auroc"].append(metrics["auroc"])
            method_scores[method]["mcc"].append(float(matthews_corrcoef(y, hard)))
            method_scores[method]["accuracy"].append(float(accuracy_score(y, hard)))

        pairwise = pairwise_agreement(y, preds)
        corrected = holm_bonferroni(
            {pair: stats["mcnemar"]["p_value"] for pair, stats in pairwise.items()}
        )
        for pair, stats in pairwise.items():
            stats["mcnemar"]["p_holm"] = corrected[pair]["p_holm"]
            stats["mcnemar"]["significant_holm"] = corrected[pair]["significant"]
            agreement_by_pair[pair].append(stats)
        per_seed.append({"seed": seed, "n": int(len(y)), "pairwise": pairwise})

    if not per_seed:
        return {"dataset": dataset_name, "error": "no stage 3 predictions found"}

    pair_summary = {}
    for pair, entries in agreement_by_pair.items():
        kappas = [e["cohen_kappa"] for e in entries if e["cohen_kappa"] is not None]
        pair_summary[pair] = {
            "mean_cohen_kappa": _round(np.mean(kappas)) if kappas else None,
            "std_cohen_kappa": _round(np.std(kappas)) if kappas else None,
            "mean_raw_agreement": _round(np.mean([e["raw_agreement"] for e in entries])),
            "median_mcnemar_p_holm": _round(
                np.median([e["mcnemar"]["p_holm"] for e in entries])
            ),
            "n_seeds_significant": int(
                sum(e["mcnemar"]["significant_holm"] for e in entries)
            ),
            "n_seeds": len(entries),
            "mean_b_a_better": _round(np.mean([e["mcnemar"]["b"] for e in entries])),
            "mean_c_b_better": _round(np.mean([e["mcnemar"]["c"] for e in entries])),
        }

    methods = sorted(method_scores)
    summary = {
        "dataset": dataset_name,
        "n_seeds": len(per_seed),
        "per_method": {
            m: {
                f"{stat}_{agg}": _round(getattr(np, agg)(values))
                for stat, values in method_scores[m].items()
                for agg in ("mean", "std")
                if all(v is not None for v in values)
            }
            for m in methods
        },
        "pairwise": pair_summary,
        # Lowest-kappa pairs are the complementary ones: they disagree more than their
        # individual accuracies would predict.
        "most_complementary": sorted(
            (
                (pair, stats["mean_cohen_kappa"])
                for pair, stats in pair_summary.items()
                if stats["mean_cohen_kappa"] is not None
            ),
            key=lambda kv: kv[1],
        )[:5],
        "per_seed": per_seed,
        "provenance": provenance(),
    }
    return summary


def _round(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def _print_table(summary: dict) -> None:
    print(f"\n=== {summary['dataset']} ({summary['n_seeds']} seeds) ===")
    rows = sorted(
        summary["per_method"].items(), key=lambda kv: -(kv[1].get("mcc_mean") or -1)
    )
    print(f"{'method':16s} {'MCC':>16s} {'AUROC':>16s}")
    for name, stats in rows:
        mcc = f"{stats.get('mcc_mean', float('nan')):.3f} ± {stats.get('mcc_std', 0):.3f}"
        auroc = f"{stats.get('auroc_mean', float('nan')):.3f} ± {stats.get('auroc_std', 0):.3f}"
        print(f"{name:16s} {mcc:>16s} {auroc:>16s}")
    print("  most complementary (lowest kappa):")
    for pair, kappa in summary["most_complementary"]:
        print(f"    {pair:38s} kappa={kappa:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--scope", choices=("pooled", "per_dataset"), default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.datasets:
        cfg.datasets = args.datasets
    scope = args.scope or cfg.training_scope

    out_dir = cfg.stage_dir("stage4_analysis")
    summaries = {}
    # Under the pooled scope, also analyse every dataset together — complementarity may
    # look different in aggregate than within any single dataset.
    slices = [*cfg.datasets, "pooled"] if scope == "pooled" else list(cfg.datasets)
    for dataset_name in slices:
        summary = analyse_dataset(cfg, dataset_name, scope)
        summaries[dataset_name] = summary
        if "error" in summary:
            print(f"[{dataset_name}] {summary['error']}")
            continue
        write_json(out_dir / f"{dataset_name}.json", summary)
        _print_table(summary)

    write_json(
        out_dir / "summary.json",
        {
            "run_id": cfg.run_id,
            "datasets": {
                name: {k: v for k, v in s.items() if k != "per_seed"}
                for name, s in summaries.items()
                if "error" not in s
            },
            "provenance": provenance(),
        },
    )


if __name__ == "__main__":
    main()
