"""Stage 5: post-hoc analyses over stage 3's out-of-fold predictions.

Everything here reads data already on disk — no GPU, no re-extraction. Five analyses:

1. **Length confound.** Are the detectors partly reading answer *length* rather than
   hallucination? A length-only probe is fitted on the identical folds as the reference
   point; if it scores near a real method, that method's number needs re-reading.
2. **Oracle ceiling and stacking.** How much is left on the table by combining? The
   oracle is the ceiling if you could pick the right method per item; stacking is the
   achievable version.
3. **Bootstrap confidence intervals.** Seed spread describes 5 draws; bootstrapping
   groups gives an interval on the MCC *differences* the conclusions actually rest on.
   Resampling is by group, not item, because folds are grouped.
4. **Calibration.** Brier score and expected calibration error. Two detectors with equal
   AUROC can be very differently usable, and thresholds are tuned on these scores.
5. **Error stratification.** Where the errors concentrate: dataset, answer length,
   judge unanimity, gold-alias count.

Folds are reconstructed with the same seed and the same (dataset, label) stratification
stage 3 used, so the length-only probe is scored on identical splits.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..io import provenance, read_json, write_json


def _fast_mcc(y: np.ndarray, p: np.ndarray) -> float:
    """MCC from raw counts — the bootstrap calls this ~100k times."""
    tp = float(np.sum((y == 1) & (p == 1)))
    tn = float(np.sum((y == 0) & (p == 0)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom


def load_seed(cfg: Config, seed: int) -> dict | None:
    path = cfg.stage_dir("stage3_train", "pooled", f"seed{seed}") / "predictions.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    methods = sorted(k.removeprefix("preds__") for k in d.files if k.startswith("preds__"))
    return {
        "y": d["y"].astype(int),
        "groups": d["groups"].astype(str),
        "dataset": d["dataset"].astype(str),
        "item_ids": d["item_ids"].astype(str),
        "preds": {m: d[f"preds__{m}"].astype(int) for m in methods},
        "scores": {m: d[f"scores__{m}"].astype(float) for m in methods},
        "methods": methods,
    }


def item_metadata(cfg: Config) -> dict[str, dict]:
    """Per-item answer length, gold-alias count and judge unanimity."""
    meta: dict[str, dict] = {}
    for ds in cfg.datasets:
        for r in read_json(cfg.stage_dir("stage1_extract", ds) / "manifest.json")["items"]:
            meta[r["item_id"]] = {
                "answer_tokens": r["answer_tokens"],
                "prompt_tokens": r["prompt_tokens"],
                "n_gold": len(r["gold_answers"]),
                "dataset": ds,
            }
        for e in read_json(cfg.stage_dir("stage2_judge", ds) / "labels.json")["labels"]:
            if e["item_id"] in meta:
                meta[e["item_id"]]["unanimous"] = bool(e["unanimous"])
    return meta


def _folds(seed_data: dict, seed: int, n_folds: int):
    """Reconstruct stage 3's exact outer folds."""
    strat = np.array([f"{d}_{y}" for d, y in zip(seed_data["dataset"], seed_data["y"])])
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(strat)), strat, seed_data["groups"]))


# --------------------------------------------------------------------------- 1
def length_confound(cfg, seeds, meta) -> dict:
    per_method = defaultdict(lambda: defaultdict(list))
    length_only, label_corr = [], []
    quartile_mcc = defaultdict(lambda: defaultdict(list))

    for seed, sd in seeds:
        tokens = np.array([meta[i]["answer_tokens"] for i in sd["item_ids"]], dtype=float)
        y = sd["y"]
        label_corr.append(spearmanr(tokens, y).statistic)

        # A probe on answer length alone, fitted on the identical folds. This is the
        # reference every method has to beat to be doing more than counting tokens.
        oof = np.full(len(y), np.nan)
        for tr, te in _folds(sd, seed, cfg.n_folds):
            model = Pipeline([("s", StandardScaler()),
                              ("c", LogisticRegression(class_weight="balanced", max_iter=1000))])
            model.fit(tokens[tr].reshape(-1, 1), y[tr])
            oof[te] = model.predict_proba(tokens[te].reshape(-1, 1))[:, 1]
        length_only.append({
            "auroc": float(roc_auc_score(y, oof)),
            "mcc": float(_fast_mcc(y, (oof >= 0.5).astype(int))),
        })

        edges = np.quantile(tokens, [0.25, 0.5, 0.75])
        bins = np.digitize(tokens, edges)
        for m in sd["methods"]:
            per_method[m]["spearman_score_vs_len"].append(
                spearmanr(tokens, sd["scores"][m]).statistic
            )
            for b in range(4):
                mask = bins == b
                if mask.sum() > 20 and len(np.unique(y[mask])) > 1:
                    quartile_mcc[m][b].append(_fast_mcc(y[mask], sd["preds"][m][mask]))

    return {
        "spearman_label_vs_answer_tokens": _ms(label_corr),
        "length_only_probe": {
            "auroc": _ms([r["auroc"] for r in length_only]),
            "mcc": _ms([r["mcc"] for r in length_only]),
            "note": "logistic regression on answer_tokens alone, stage 3's folds",
        },
        "per_method": {
            m: {
                "spearman_score_vs_answer_tokens": _ms(v["spearman_score_vs_len"]),
                "mcc_by_length_quartile": {
                    f"q{b + 1}": _ms(quartile_mcc[m][b]) for b in sorted(quartile_mcc[m])
                },
            }
            for m, v in per_method.items()
        },
    }


# --------------------------------------------------------------------------- 2
def oracle_and_stacking(cfg, seeds) -> dict:
    single, oracle, stack, coverage, greedy = [], [], [], [], []
    for seed, sd in seeds:
        y, methods = sd["y"], sd["methods"]
        base = [m for m in methods if not m.startswith("union")]
        correct = {m: (sd["preds"][m] == y) for m in methods}

        any_correct = np.zeros(len(y), bool)
        for m in base:
            any_correct |= correct[m]
        coverage.append(float(any_correct.mean()))
        # Oracle: right wherever ANY base method is right.
        oracle.append(_fast_mcc(y, np.where(any_correct, y, 1 - y)))
        single.append(max(_fast_mcc(y, sd["preds"][m]) for m in base))

        # Stacked meta-probe over the base methods' out-of-fold probabilities.
        X = np.column_stack([sd["scores"][m] for m in base])
        oof = np.full(len(y), np.nan)
        for tr, te in _folds(sd, seed, cfg.n_folds):
            model = Pipeline([("s", StandardScaler()),
                              ("c", LogisticRegression(class_weight="balanced", max_iter=2000))])
            model.fit(X[tr], y[tr])
            oof[te] = model.predict_proba(X[te])[:, 1]
        stack.append({"auroc": float(roc_auc_score(y, oof)),
                      "mcc": float(_fast_mcc(y, (oof >= 0.5).astype(int)))})

        # How many methods does the stack actually need?
        chosen, curve, remaining = [], [], list(base)
        while remaining:
            best = None
            for cand in remaining:
                cols = [base.index(c) for c in chosen + [cand]]
                o = np.full(len(y), np.nan)
                for tr, te in _folds(sd, seed, cfg.n_folds):
                    mdl = Pipeline([("s", StandardScaler()),
                                    ("c", LogisticRegression(class_weight="balanced", max_iter=1000))])
                    mdl.fit(X[tr][:, cols], y[tr])
                    o[te] = mdl.predict_proba(X[te][:, cols])[:, 1]
                score = float(roc_auc_score(y, o))
                if best is None or score > best[0]:
                    best = (score, cand)
            chosen.append(best[1]); remaining.remove(best[1])
            curve.append({"added": best[1], "n": len(chosen), "auroc": round(best[0], 4)})
        greedy.append(curve)

    return {
        "best_single_mcc": _ms(single),
        "oracle_mcc": _ms(oracle),
        "oracle_coverage": _ms(coverage),
        "stacking": {"auroc": _ms([s["auroc"] for s in stack]),
                     "mcc": _ms([s["mcc"] for s in stack])},
        "greedy_forward_selection": greedy[0],
        "note": ("oracle is the ceiling if the right base method were chosen per item; "
                 "stacking uses out-of-fold base scores, so it is mildly optimistic"),
    }


# --------------------------------------------------------------------------- 3
def bootstrap_cis(cfg, seeds, n_boot: int, rng_seed: int = 0) -> dict:
    pairs = defaultdict(list)
    for seed, sd in seeds:
        y, methods = sd["y"], sd["methods"]
        uniq = np.unique(sd["groups"])
        index_of = {g: np.flatnonzero(sd["groups"] == g) for g in uniq}
        rng = np.random.default_rng(rng_seed + seed)
        # Resample GROUPS, not items: folds are grouped, so items within a group are
        # not independent and item-level resampling would understate the interval.
        draws = [np.concatenate([index_of[g] for g in rng.choice(uniq, len(uniq), replace=True)])
                 for _ in range(n_boot)]
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                diffs = [_fast_mcc(y[d], sd["preds"][a][d]) - _fast_mcc(y[d], sd["preds"][b][d])
                         for d in draws]
                pairs[f"{a}|{b}"].append({
                    "mean_diff": float(np.mean(diffs)),
                    "lo": float(np.percentile(diffs, 2.5)),
                    "hi": float(np.percentile(diffs, 97.5)),
                })
    out = {}
    for pair, per_seed in pairs.items():
        lo = float(np.mean([p["lo"] for p in per_seed]))
        hi = float(np.mean([p["hi"] for p in per_seed]))
        out[pair] = {
            "mean_mcc_diff": round(float(np.mean([p["mean_diff"] for p in per_seed])), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            # An interval excluding zero in every seed is the strong form of the claim.
            "excludes_zero_all_seeds": all(p["lo"] > 0 or p["hi"] < 0 for p in per_seed),
            "n_seeds": len(per_seed),
        }
    return {"n_bootstrap_per_seed": n_boot, "resampled_unit": "group", "pairs": out}


# --------------------------------------------------------------------------- 4
def calibration(cfg, seeds, n_bins: int = 10) -> dict:
    per_method = defaultdict(lambda: defaultdict(list))
    curves: dict[str, list] = {}
    for _, sd in seeds:
        y = sd["y"]
        for m in sd["methods"]:
            s = np.clip(sd["scores"][m], 0, 1)
            per_method[m]["brier"].append(float(brier_score_loss(y, s)))
            edges = np.linspace(0, 1, n_bins + 1)
            idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
            ece, bins = 0.0, []
            for b in range(n_bins):
                mask = idx == b
                if not mask.any():
                    continue
                conf, acc = float(s[mask].mean()), float(y[mask].mean())
                ece += mask.mean() * abs(acc - conf)
                bins.append({"bin": b, "confidence": round(conf, 4),
                             "observed": round(acc, 4), "n": int(mask.sum())})
            per_method[m]["ece"].append(float(ece))
            curves.setdefault(m, bins)
    return {
        "per_method": {m: {"brier": _ms(v["brier"]), "ece": _ms(v["ece"])}
                       for m, v in per_method.items()},
        "reliability_curve_seed0": curves,
        "note": "ECE over 10 equal-width bins; lower is better for both",
    }


# --------------------------------------------------------------------------- 7
def error_strata(cfg, seeds, meta) -> dict:
    strata: dict[str, dict] = {}
    for key, fn in [
        ("dataset", lambda i: meta[i]["dataset"]),
        ("judge_unanimous", lambda i: str(meta[i].get("unanimous"))),
        ("n_gold_answers", lambda i: "1" if meta[i]["n_gold"] <= 1
         else ("2-5" if meta[i]["n_gold"] <= 5 else "6+")),
    ]:
        acc = defaultdict(lambda: defaultdict(list))
        for _, sd in seeds:
            labels = np.array([fn(i) for i in sd["item_ids"]])
            for value in np.unique(labels):
                mask = labels == value
                if mask.sum() < 30 or len(np.unique(sd["y"][mask])) < 2:
                    continue
                for m in sd["methods"]:
                    acc[str(value)][m].append(_fast_mcc(sd["y"][mask], sd["preds"][m][mask]))
                acc[str(value)]["_n"].append(int(mask.sum()))
                acc[str(value)]["_pos_rate"].append(float(sd["y"][mask].mean()))
        strata[key] = {
            v: {"n": int(np.mean(d["_n"])), "positive_rate": round(float(np.mean(d["_pos_rate"])), 4),
                "mcc": {m: _ms(x) for m, x in d.items() if not m.startswith("_")}}
            for v, d in acc.items()
        }

    # Length quartiles are computed per seed, so they get their own pass.
    tok_strata = defaultdict(lambda: defaultdict(list))
    for _, sd in seeds:
        tokens = np.array([meta[i]["answer_tokens"] for i in sd["item_ids"]], dtype=float)
        bins = np.digitize(tokens, np.quantile(tokens, [0.25, 0.5, 0.75]))
        for b in range(4):
            mask = bins == b
            if mask.sum() < 30 or len(np.unique(sd["y"][mask])) < 2:
                continue
            for m in sd["methods"]:
                tok_strata[f"q{b + 1}"][m].append(_fast_mcc(sd["y"][mask], sd["preds"][m][mask]))
            tok_strata[f"q{b + 1}"]["_n"].append(int(mask.sum()))
            tok_strata[f"q{b + 1}"]["_pos_rate"].append(float(sd["y"][mask].mean()))
    strata["answer_length_quartile"] = {
        v: {"n": int(np.mean(d["_n"])), "positive_rate": round(float(np.mean(d["_pos_rate"])), 4),
            "mcc": {m: _ms(x) for m, x in d.items() if not m.startswith("_")}}
        for v, d in tok_strata.items()
    }
    return strata


def _ms(values) -> dict:
    clean = [v for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return {"mean": None, "std": None}
    return {"mean": round(float(np.mean(clean)), 4), "std": round(float(np.std(clean)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    seeds = [(s, d) for s in cfg.seeds if (d := load_seed(cfg, s)) is not None]
    if not seeds:
        raise SystemExit("no stage 3 predictions found")
    print(f"loaded {len(seeds)} seeds x {len(seeds[0][1]['y'])} items")
    meta = item_metadata(cfg)

    out = cfg.stage_dir("stage5_posthoc")
    results = {"n_seeds": len(seeds), "provenance": provenance()}

    print("1/5 length confound...");        results["length_confound"] = length_confound(cfg, seeds, meta)
    print("2/5 oracle + stacking...");      results["oracle_and_stacking"] = oracle_and_stacking(cfg, seeds)
    print("3/5 bootstrap CIs...");          results["bootstrap"] = bootstrap_cis(cfg, seeds, args.n_bootstrap)
    print("4/5 calibration...");            results["calibration"] = calibration(cfg, seeds)
    print("5/5 error stratification...");   results["error_strata"] = error_strata(cfg, seeds, meta)

    write_json(out / "posthoc.json", results)
    print(f"\nwrote {out / 'posthoc.json'}")


if __name__ == "__main__":
    main()
