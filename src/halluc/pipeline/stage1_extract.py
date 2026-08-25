"""Stage 1: generate answers and extract every method's features in one loop.

The generator runs exactly once per pool item. All five feature blocks are computed from
that single forward trace, because ICR's inputs (per-token residual deltas, ~126 MB per
sample) are far too large to persist and recompute later.

The judges are deliberately *not* loaded here — stage 2 runs after this process exits,
so the 28 GB generator and the quantised judge pool are never co-resident.

Resumable: completed item ids live in checkpoint.jsonl; a restart skips them.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..config import Config
from ..datasets import DATASETS
from ..features import FEATURES
from ..io import Checkpoint, ShardWriter, provenance, write_json
from ..models import GENERATORS


def run_dataset(cfg: Config, dataset_name: str, generator) -> dict:
    out_dir = cfg.stage_dir("stage1_extract", dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = DATASETS.create(dataset_name)
    pool = dataset.sample_pool(cfg.pool_size, cfg.pool_seed)
    by_id = {item.item_id: item for item in pool}

    extractors = [FEATURES.create(name) for name in cfg.features]

    manifest_path = out_dir / "manifest.json"
    checkpoint = Checkpoint(out_dir / "checkpoint.jsonl")
    # checkpoint.jsonl carries whole records, so it — not the manifest — is the source
    # of truth after an interrupted run.
    records: dict[str, dict] = {i: dict(r) for i, r in checkpoint.records().items()}
    pending = checkpoint.pending([item.item_id for item in pool])
    print(f"[{dataset_name}] pool={len(pool)} done={len(checkpoint)} pending={len(pending)}")

    writer = ShardWriter(out_dir, shard_size=cfg.shard_size)
    failures: list[dict] = []
    started = time.perf_counter()
    # Items whose features are still buffered in the writer. They are checkpointed only
    # once their shard is on disk, so the checkpoint never claims an item whose features
    # were lost with the buffer.
    staged: list[dict] = []

    def commit(shard_name: str | None) -> None:
        for record in staged:
            record["shard"] = shard_name
            checkpoint.mark(**record)
            records[record["item_id"]] = record
        staged.clear()

    with checkpoint:
        for item_id in tqdm(pending, desc=dataset_name, unit="item"):
            item = by_id[item_id]
            try:
                generation, trace = generator.generate(item)
                features: dict[str, np.ndarray] = {}
                for extractor in extractors:
                    features.update(extractor.extract(trace))
                del trace
            except torch.cuda.OutOfMemoryError as exc:
                # One pathological item must not kill a multi-hour run; record and move
                # on, so the failure is visible in the JSON rather than silent.
                torch.cuda.empty_cache()
                failures.append({"item_id": item_id, "error": "OOM", "detail": str(exc)[:200]})
                continue
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {"item_id": item_id, "error": type(exc).__name__, "detail": str(exc)[:200]}
                )
                continue

            staged.append(
                {
                    "item_id": item_id,
                    "question": item.question,
                    "gold_answers": item.gold_answers,
                    "group_id": item.group,
                    "has_context": item.context is not None,
                    "answer": generation.answer,
                    "prompt_tokens": generation.prompt_tokens,
                    "answer_tokens": generation.answer_tokens,
                    "finish_reason": generation.finish_reason,
                    "seconds": round(generation.seconds, 3),
                    "seq_len": generation.meta.get("seq_len"),
                    "feature_shapes": {k: list(v.shape) for k, v in features.items()},
                }
            )
            flushed = writer.add(item_id, features)
            if flushed:
                commit(flushed)
                write_json(manifest_path, _summary(cfg, dataset_name, pool, records, failures, started))

        commit(writer.flush())

    summary = _summary(cfg, dataset_name, pool, records, failures, started)
    write_json(manifest_path, summary)
    print(
        f"[{dataset_name}] extracted={summary['n_extracted']} failed={len(failures)} "
        f"truncated={summary['truncated_rate']:.1%} "
        f"in {summary['elapsed_seconds'] / 60:.1f} min"
    )
    return summary


def _summary(cfg, dataset_name, pool, records, failures, started) -> dict:
    ordered = [records[item.item_id] for item in pool if item.item_id in records]
    return {
        "dataset": dataset_name,
        "generator": cfg.generator.model_id,
        "pool_size": len(pool),
        "n_extracted": len(ordered),
        "n_failed": len(failures),
        "failures": failures,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        # A high truncation rate means answers are being cut mid-sentence, which would
        # corrupt both the judge's view and the last-token features.
        "truncated_rate": (
            round(sum(r["finish_reason"] == "length" for r in ordered) / max(len(ordered), 1), 4)
        ),
        "mean_answer_tokens": (
            round(float(np.mean([r["answer_tokens"] for r in ordered])), 2) if ordered else None
        ),
        "mean_seq_len": (
            round(float(np.mean([r["seq_len"] for r in ordered])), 2) if ordered else None
        ),
        "config": cfg.to_dict(),
        "provenance": provenance(),
        "items": ordered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config; defaults are the locked ones")
    parser.add_argument("--datasets", nargs="*", default=None, help="subset of configured datasets")
    parser.add_argument("--limit", type=int, default=None, help="cap pool size (smoke tests)")
    parser.add_argument("--device", default=None, help="override generator device, e.g. cuda:0")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.datasets:
        cfg.datasets = args.datasets
    if args.limit:
        cfg.pool_size = args.limit
    if args.device:
        cfg.generator.device = args.device

    generator = GENERATORS.create(
        cfg.generator.kind,
        model_id=cfg.generator.model_id,
        device=cfg.generator.device,
        dtype=cfg.generator.dtype,
        max_new_tokens=cfg.generator.max_new_tokens,
        enable_thinking=cfg.generator.enable_thinking,
        max_seq_len=cfg.generator.max_seq_len,
        load_in_4bit=cfg.generator.load_in_4bit,
        reserve_gib=cfg.generator.reserve_gib,
    )
    try:
        summaries = [run_dataset(cfg, name, generator) for name in cfg.datasets]
    finally:
        generator.unload()

    write_json(
        cfg.stage_dir("stage1_extract") / "summary.json",
        {
            "run_id": cfg.run_id,
            "datasets": {s["dataset"]: {k: v for k, v in s.items() if k != "items"} for s in summaries},
            "provenance": provenance(),
        },
    )


if __name__ == "__main__":
    main()
