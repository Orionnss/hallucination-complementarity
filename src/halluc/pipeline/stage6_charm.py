"""Stage 6: CHARM — build attention graphs and fit the GNN in one pass, nothing on disk.

Every other detector in this study reads feature blocks that stage 1 wrote to `.npz`
shards. CHARM cannot: its input is a whole attributed graph per sample, and a dense
edge-feature matrix runs to tens of megabytes per item at Qwen3-14B's L*H = 1600
channels. So this stage owns its own extraction loop — generator forward pass, graph
construction, RAM cache — and then trains from that cache. The graphs are never
persisted; only the out-of-fold predictions are.

**The evaluation protocol is deliberately stage 3's, unchanged.** Same per-seed draw
(`draw_and_combine`), same balanced per-dataset sampling, same grouped folds stratified
on (dataset, label), same inner split, same MCC-maximising threshold tuned on
validation. The predictions land in the same array layout as stage 3's, so stage 4 picks
CHARM up for kappa and McNemar against the other detectors with no special-casing.

Two places where CHARM necessarily differs from stage 3, both narrowing rather than
widening what it sees:

* **No refit on the full training fold.** A neural network needs a held-out split to
  early-stop on, so the reported model is trained on the inner-train portion and stopped
  on inner-val — the same split its hyperparameters and threshold come from. It
  therefore sees *less* data than the probes it is compared against, never more.
* **Answers are replayed, not regenerated.** Stage 1 recorded what the generator said
  under greedy decoding; this stage re-encodes that answer and runs a single forward
  pass. Items whose text does not re-encode to the same token count fall back to a real
  `generate()` call, so the trace always matches a sequence the model actually produced.

Extraction is the expensive half and the cache is RAM-only by design, so a crash loses
it. Run this stage to completion in one go.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from ..charm.checkpoint import FitStore, seed_fingerprint
from ..charm.data import GraphCache
from ..charm.graph import DEFAULT_TAU, GraphSpec, activation_layers, build_graph
from ..charm.train import GRIDS, TrainConfig, fit_predict
from ..config import Config
from ..datasets import DATASETS
from ..eval.metrics import best_threshold, score_predictions
from ..io import provenance, read_json, write_json
from ..models import GENERATORS
from .stage3_train import draw_and_combine, load_arrays

METHOD = "charm"


def required_items(cfg: Config, jobs: list[tuple]) -> dict[str, list[str]]:
    """Item ids each dataset must supply, as the union over the draws still to be run.

    Computed from `draw_and_combine` itself rather than assumed to be "everything", so
    extraction covers exactly the samples the pending folds will ask for and no more —
    a resumed run does not re-trace items only a finished seed needed.
    """
    wanted: dict[str, set[str]] = {}
    for _, arrays, seed in jobs:
        combined = draw_and_combine(arrays, cfg.n_per_seed, seed)
        for item_id, dataset in zip(combined["item_ids"], combined["dataset"]):
            wanted.setdefault(str(dataset), set()).add(str(item_id))
    return {name: sorted(ids) for name, ids in wanted.items()}


def extract_graphs(cfg, generator, needed, spec, cache, device) -> dict:
    """Run the generator over every needed item and cache its attention graph."""
    failures, regenerated, started = [], 0, time.perf_counter()

    for dataset_name, item_ids in needed.items():
        manifest = {
            r["item_id"]: r
            for r in read_json(
                cfg.stage_dir("stage1_extract", dataset_name) / "manifest.json"
            )["items"]
        }
        pool = {item.item_id: item for item in DATASETS.create(dataset_name).sample_pool(
            cfg.pool_size, cfg.pool_seed
        )}
        pending = [i for i in item_ids if i not in cache]
        print(f"[{dataset_name}] graphs needed={len(item_ids)} pending={len(pending)}")

        for item_id in tqdm(pending, desc=f"charm:{dataset_name}", unit="item"):
            item, record = pool.get(item_id), manifest.get(item_id)
            if item is None or record is None:
                failures.append({"item_id": item_id, "error": "missing from pool or manifest"})
                continue
            try:
                try:
                    trace = generator.retrace(item, record["answer"], record["answer_tokens"])
                except ValueError:
                    # Tokenizer round trip did not land back on stage 1's ids. Pay for a
                    # real generation rather than trace a sequence the model never emitted.
                    _, trace = generator.generate(item)
                    regenerated += 1
                cache.add(item_id, build_graph(trace, spec.tau, spec.act_layers))
                del trace
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                failures.append({"item_id": item_id, "error": "OOM", "detail": str(exc)[:200]})
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {"item_id": item_id, "error": type(exc).__name__, "detail": str(exc)[:200]}
                )

    stats = cache.stats()
    elapsed = (time.perf_counter() - started) / 60
    if stats["n_graphs"]:
        print(
            f"[charm] cached {stats['n_graphs']} graphs, {stats['resident_gib']} GiB resident "
            f"(mean {stats['mean_edges']:.0f} edges, edge-feature density "
            f"{stats['edge_feature_density']:.4f}) in {elapsed:.1f} min"
        )
    else:
        print(f"[charm] no graphs built in {elapsed:.1f} min — every item failed")
    if failures:
        print(f"[charm] {len(failures)} items failed extraction")
    return {
        "graph_stats": stats,
        "n_failed": len(failures),
        "failures": failures[:50],
        "n_regenerated": regenerated,
        "extract_seconds": round(time.perf_counter() - started, 1),
    }


def run_fold(
    graphs, y, combined, seed, train_idx, test_idx, spec, grid, train_cfg, store=None,
    fold_id=0,
) -> dict:
    """One outer fold: inner split, grid search, threshold, test scores.

    Mirrors `stage3_train.run_fold` step for step — same inner splitter, same selection
    criterion (inner AUROC), same threshold rule — so the only thing that differs
    between CHARM and the probes it is compared against is the detector itself.

    Each grid point is checkpointed as it finishes, so a crash costs at most the fit in
    flight. Which point wins and where the threshold lands are recomputed from the
    cached fits — both are cheap and deterministic.
    """
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    rel_train, rel_val = next(
        inner.split(
            np.zeros(len(train_idx)),
            combined["strat"][train_idx],
            combined["groups"][train_idx],
        )
    )
    inner_train, inner_val = train_idx[rel_train], train_idx[rel_val]

    best, n_resumed = None, 0
    for params in grid:
        cached = store.load(fold_id, params) if store is not None else None
        if cached is not None:
            val_scores, test_scores, record = (
                cached["val_scores"], cached["eval_scores"], cached["record"]
            )
            n_resumed += 1
        else:
            val_scores, test_scores, record, weights = fit_predict(
                graphs, y, inner_train, inner_val, test_idx, spec, params, train_cfg
            )
            if store is not None:
                store.save(fold_id, params, val_scores, test_scores, record, weights)
        auroc = record["val_auroc"]
        print(
            f"      {params.hidden}x{params.n_layers}L "
            f"val_auroc={auroc:.4f} epochs={record['epochs_run']}"
            + ("  (resumed)" if cached is not None else "")
        )
        if best is None or auroc > best["inner_auroc"]:
            best = {
                "inner_auroc": auroc,
                "params": params,
                "val_scores": val_scores,
                "test_scores": test_scores,
                "record": record,
            }

    threshold, val_mcc = best_threshold(y[inner_val], best["val_scores"])
    per_dataset_threshold = {}
    val_datasets = combined["dataset"][inner_val]
    for name in np.unique(val_datasets):
        mask = val_datasets == name
        if mask.sum() > 1 and len(np.unique(y[inner_val][mask])) > 1:
            per_dataset_threshold[str(name)] = best_threshold(
                y[inner_val][mask], best["val_scores"][mask]
            )[0]

    return {
        "scores": best["test_scores"],
        "threshold": threshold,
        "per_dataset_threshold": per_dataset_threshold,
        "best_params": best["params"].to_dict(),
        "inner_auroc": best["inner_auroc"],
        "inner_mcc": val_mcc,
        "epochs_run": best["record"]["epochs_run"],
        "n_parameters": best["record"]["n_parameters"],
        "n_resumed": n_resumed,
    }


def run_seed(
    cfg, scope_name, seed, arrays_by_dataset, cache, spec, grid, train_cfg,
    save_models=True, resume=True,
) -> dict:
    combined = draw_and_combine(arrays_by_dataset, cfg.n_per_seed, seed)
    y, groups, datasets = combined["y"], combined["groups"], combined["dataset"]
    item_ids = [str(i) for i in combined["item_ids"]]
    # Fingerprint of the draw itself, before any extraction failures are filtered out.
    # This is what `main` can recompute without a graph cache, so it is the key that
    # decides whether an existing metrics.json describes *this* experiment. The draw
    # depends on the dataset list — `draw_and_combine` consumes the RNG once per dataset
    # in sorted order — so a triviaqa-only run and a four-dataset run give a seed
    # different items, and must not satisfy each other's skip check.
    draw_fp = seed_fingerprint(item_ids, cfg.n_folds, scope_name, seed)

    # Any item that failed extraction is dropped from this seed's sample rather than
    # imputed. Stage 4 aligns on item ids, so a shorter CHARM vector stays comparable.
    keep = np.array([i in cache for i in item_ids])
    if not keep.all():
        print(f"  [{scope_name} seed={seed}] dropping {int((~keep).sum())} items with no graph")
        y, groups, datasets = y[keep], groups[keep], datasets[keep]
        item_ids = [i for i, k in zip(item_ids, keep) if k]
    combined.update({"y": y, "groups": groups, "dataset": datasets, "item_ids": item_ids})
    combined["strat"] = np.array([f"{d}_{lab}" for d, lab in zip(datasets, y)])

    graphs = cache.get(item_ids)
    n = len(y)
    outer = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=seed)
    folds = list(outer.split(np.zeros(n), combined["strat"], groups))

    out_dir = cfg.stage_dir("stage6_charm", scope_name, f"seed{seed}")
    store = None
    if resume:
        store = FitStore(
            out_dir / "checkpoints",
            seed_fingerprint(item_ids, cfg.n_folds, scope_name, seed),
            save_models=save_models,
        )
        if store.n_completed():
            print(f"  [{scope_name} seed={seed}] resuming: {store.n_completed()} fits on disk")

    oof_scores = np.full(n, np.nan)
    oof_preds = np.full(n, -1, dtype=int)
    fold_records = []

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        started = time.perf_counter()
        print(f"  [{scope_name} seed={seed}] fold {fold_id + 1}/{len(folds)}")
        result = run_fold(
            graphs, y, combined, seed, train_idx, test_idx, spec, grid,
            # Per-fold config differing only in the seed, so the network's
            # initialisation and batch order follow the same seed as the fold split.
            replace(train_cfg, seed=seed),
            store=store,
            fold_id=fold_id,
        )
        oof_scores[test_idx] = result["scores"]
        oof_preds[test_idx] = (result["scores"] >= result["threshold"]).astype(int)
        fold_records.append(
            {
                "fold": fold_id,
                "threshold": result["threshold"],
                "per_dataset_threshold": result["per_dataset_threshold"],
                "best_params": result["best_params"],
                "inner_auroc": round(result["inner_auroc"], 4),
                "epochs_run": result["epochs_run"],
                "n_parameters": result["n_parameters"],
                "seconds": round(time.perf_counter() - started, 1),
                **score_predictions(y[test_idx], result["scores"], result["threshold"]),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        y=y,
        item_ids=np.array(item_ids),
        groups=groups,
        dataset=datasets,
        **{f"scores__{METHOD}": oof_scores, f"preds__{METHOD}": oof_preds},
    )

    by_dataset = {}
    for ds in sorted(set(datasets)):
        mask = datasets == ds
        by_dataset[ds] = {
            "n": int(mask.sum()),
            "positive_rate": float(y[mask].mean()),
            "auroc": (
                float(roc_auc_score(y[mask], oof_scores[mask]))
                if len(np.unique(y[mask])) > 1 else None
            ),
            "mcc": float(matthews_corrcoef(y[mask], oof_preds[mask])),
        }

    summary = {
        "scope": scope_name,
        "seed": seed,
        "method": METHOD,
        "draw_fingerprint": draw_fp,
        "n_samples": n,
        "datasets": {ds: int((datasets == ds).sum()) for ds in sorted(set(datasets))},
        "positive_rate": float(y.mean()),
        "n_groups": int(len(set(groups))),
        "per_method": {
            METHOD: {
                "per_dataset": by_dataset,
                "pooled_auroc": (
                    float(roc_auc_score(y, oof_scores)) if len(np.unique(y)) > 1 else None
                ),
                "pooled_mcc": float(matthews_corrcoef(y, oof_preds)),
                "folds": fold_records,
            }
        },
        "provenance": provenance(),
    }
    write_json(out_dir / "metrics.json", summary)
    print(
        f"  [{scope_name} seed={seed}] charm pooled MCC="
        f"{summary['per_method'][METHOD]['pooled_mcc']:.3f} "
        f"AUROC={summary['per_method'][METHOD]['pooled_auroc']:.3f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None, help="must match the run stage 1 wrote")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--scope", choices=("pooled", "per_dataset"), default=None)
    parser.add_argument("--device", default=None, help="GPU for both extraction and training")
    parser.add_argument("--model", default=None, help="generator preset or HF id")
    # Defaults are None so a value set in the YAML config is not silently overridden by
    # an argparse default; the config's CharmConfig fills them in below.
    parser.add_argument(
        "--tau", type=float, default=None,
        help=f"attention sparsification threshold (paper default {DEFAULT_TAU})",
    )
    parser.add_argument(
        "--act-fractions", nargs="*", type=float, default=None,
        help="activation layers as fractions of depth; pass an empty list for the "
             "attention-only variant, CHARM (att)",
    )
    parser.add_argument("--grid", choices=tuple(GRIDS), default=None)
    parser.add_argument("--max-edges-per-batch", type=int, default=None)
    parser.add_argument(
        "--max-cache-gib", type=float, default=None,
        help="abort if the RAM graph cache would exceed this",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-run seeds that already have metrics.json (their cached per-fit "
             "checkpoints are still reused unless --no-resume)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="ignore per-fit checkpoints and re-train every grid point",
    )
    parser.add_argument(
        "--no-save-models", action="store_true",
        help="checkpoint scores but not weights (~7 MB per fit at hidden=128)",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.run_id:
        cfg.run_id = args.run_id
    if args.datasets:
        cfg.datasets = args.datasets
    if args.seeds:
        cfg.seeds = args.seeds
    if args.device:
        cfg.generator.device = args.device
    if args.model:
        from ..models.hf import resolve_model_id

        cfg.generator.model_id = resolve_model_id(args.model)
    if args.tau is not None:
        cfg.charm.tau = args.tau
    if args.act_fractions is not None:
        cfg.charm.act_fractions = args.act_fractions
    if args.grid is not None:
        cfg.charm.grid = args.grid
    if args.max_edges_per_batch is not None:
        cfg.charm.max_edges_per_batch = args.max_edges_per_batch
    if args.max_cache_gib is not None:
        cfg.charm.max_cache_gib = args.max_cache_gib
    scope = args.scope or cfg.training_scope
    device = cfg.generator.device

    # The generator that produced stage 1 is the one whose activations CHARM must read;
    # tracing a different model against those labels would be meaningless.
    recorded = read_json(
        cfg.stage_dir("stage1_extract", cfg.datasets[0]) / "manifest.json"
    ).get("generator")
    if recorded and recorded != cfg.generator.model_id:
        raise SystemExit(
            f"run '{cfg.run_id}' holds features from {recorded}, but this invocation uses "
            f"{cfg.generator.model_id}. Pass --model {recorded} or a different --run-id."
        )

    # No feature blocks: CHARM builds its own inputs, and this call is only for labels,
    # groups and the item ordering the per-seed draw depends on.
    arrays_by_dataset = {name: load_arrays(cfg, name, []) for name in cfg.datasets}
    for name, arrays in arrays_by_dataset.items():
        print(
            f"[{name}] scored={len(arrays['y'])} "
            f"positive_rate={arrays['y'].mean():.3f} groups={len(set(arrays['groups']))}"
        )

    # Jobs are (scope_name, arrays, seed). A seed whose metrics.json exists finished
    # cleanly, so it is skipped unless --force. This is settled *before* the generator is
    # touched: a run with nothing left to do must not spend minutes loading 28 GB of
    # weights only to exit.
    if scope == "pooled":
        jobs = [("pooled", arrays_by_dataset, seed) for seed in cfg.seeds]
    else:
        jobs = [
            (name, {name: arrays}, seed)
            for name, arrays in arrays_by_dataset.items()
            for seed in cfg.seeds
        ]
    done = {
        (scope_name, seed): cfg.stage_dir("stage6_charm", scope_name, f"seed{seed}")
        / "metrics.json"
        for scope_name, _, seed in jobs
    }
    def already_done(scope_name, arrays, seed) -> bool:
        path = done[(scope_name, seed)]
        if args.force or not path.exists():
            return False
        drawn = [str(i) for i in draw_and_combine(arrays, cfg.n_per_seed, seed)["item_ids"]]
        recorded = read_json(path).get("draw_fingerprint")
        expected = seed_fingerprint(drawn, cfg.n_folds, scope_name, seed)
        if recorded == expected:
            return True
        print(
            f"[charm] {scope_name}/seed{seed} on disk was computed over a different draw "
            f"({recorded} vs {expected}) — re-running it"
        )
        return False

    pending = [job for job in jobs if not already_done(*job)]
    if len(pending) < len(jobs):
        print(
            f"[charm] {len(jobs) - len(pending)}/{len(jobs)} seed-scopes already complete; "
            f"{len(pending)} to run"
        )
    if not pending:
        print("[charm] nothing to do — pass --force to re-run from scratch")
        return

    needed = required_items(cfg, pending)
    print(
        f"[charm] {sum(len(v) for v in needed.values())} distinct items "
        f"across {len(needed)} datasets"
    )

    generator = GENERATORS.create(
        cfg.generator.kind,
        model_id=cfg.generator.model_id,
        device=device,
        dtype=cfg.generator.dtype,
        max_new_tokens=cfg.generator.max_new_tokens,
        enable_thinking=cfg.generator.enable_thinking,
        max_seq_len=cfg.generator.max_seq_len,
        load_in_4bit=cfg.generator.load_in_4bit,
        reserve_gib=cfg.generator.reserve_gib,
    )
    dims = generator.dims
    act_layers = activation_layers(dims["n_layers"], tuple(cfg.charm.act_fractions))
    spec = GraphSpec(
        n_channels=dims["n_layers"] * dims["n_heads"],
        d_act=len(act_layers) * dims["hidden_size"],
        act_layers=act_layers,
        tau=cfg.charm.tau,
    )
    print(
        f"[charm] {cfg.generator.model_id}: L={dims['n_layers']} H={dims['n_heads']} "
        f"-> {spec.n_channels} attention channels; activations from layers {act_layers} "
        f"({spec.d_act} dims); tau={spec.tau}"
    )

    cache = GraphCache(spec)
    try:
        extraction = extract_graphs(cfg, generator, needed, spec, cache, device)
    finally:
        generator.unload()

    resident = cache.nbytes() / 1024**3
    if resident > cfg.charm.max_cache_gib:
        raise SystemExit(
            f"graph cache is {resident:.1f} GiB, over the {cfg.charm.max_cache_gib} GiB limit. "
            f"Raise --max-cache-gib, raise --tau to sparsify harder, or drop --act-fractions."
        )

    grid = GRIDS[cfg.charm.grid]()
    train_cfg = TrainConfig(device=device, max_edges_per_batch=cfg.charm.max_edges_per_batch)
    print(f"[charm] training on {device} with {len(grid)} grid points, scope={scope}")

    summaries = []
    for scope_name, arrays, seed in pending:
        summaries.append(
            run_seed(
                cfg, scope_name, seed, arrays, cache, spec, grid, train_cfg,
                save_models=not args.no_save_models, resume=not args.no_resume,
            )
        )
    # Seeds skipped above still belong in the summary, so their metrics are read back.
    for scope_name, _, seed in jobs:
        if any(s["scope"] == scope_name and s["seed"] == seed for s in summaries):
            continue
        summaries.append(read_json(done[(scope_name, seed)]))


    write_json(
        cfg.stage_dir("stage6_charm") / "summary.json",
        {
            "run_id": cfg.run_id,
            "scope": scope,
            "generator": cfg.generator.model_id,
            "spec": {
                "n_channels": spec.n_channels,
                "d_act": spec.d_act,
                "act_layers": list(spec.act_layers),
                "tau": spec.tau,
            },
            "grid": cfg.charm.grid,
            "config": cfg.to_dict(),
            "extraction": extraction,
            "seeds": {
                f"{s['scope']}/seed{s['seed']}": {
                    "scope": s["scope"],
                    "seed": s["seed"],
                    "pooled_mcc": s["per_method"][METHOD]["pooled_mcc"],
                    "pooled_auroc": s["per_method"][METHOD]["pooled_auroc"],
                }
                for s in summaries
            },
            "provenance": provenance(),
        },
    )


if __name__ == "__main__":
    main()
