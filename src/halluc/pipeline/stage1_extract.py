"""Stage 1: the one shared extraction pass (ADDING_A_METHOD.md, rule R2 and §5.0).

One `generate()` call per item gives the scored answer *and* the trace every method
reads: the per-step attentions, hidden states and logits of that same call, eager
attention, greedy, bf16. There is no second forward pass. Every enabled extractor runs on
the trace while it is in memory, and only their blocks are stored: the trace itself is
far too large to keep (~13 GB of attentions per item at T = 2000; ICR's per-token
residual deltas alone are ~126 MB).

Stores:
  stage1_extract/<ds>/checkpoint.jsonl   one generation record per item (answer, ids...)
  stage1_extract/<ds>/trace_spec.json    the trace fingerprint, written before generation
  stage1_extract/<ds>/manifest.json      the generation records + the fingerprint
  stage1_extract/<ds>/features_*.npz     blocks of extractors with store="stage1"
                                         (tracked by features_checkpoint.jsonl)
  methods/<name>/<ds>/                   blocks of extractors with store="methods",
                                         each with its own checkpoint and spec.json

Adding a method to an existing run: `--features <name>` runs this same pass for that
extractor only. It must reproduce every stored generated id and the trace fingerprint;
otherwise it stops, because the new method would read a different trace from the others
(PROTOCOL.md, M12).

The judges are deliberately *not* loaded here — stage 2 runs after this process exits,
so the generator and the quantised judge pool are never co-resident.

Resumable per store: an item is checkpointed in a store only once its shard is on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm

from .. import methods  # noqa: F401  (registers the adapters as stage-1 extractors)
from ..config import Config
from ..datasets import DATASETS
from ..datasets.base import QAItem
from ..features import FEATURES
from ..features.base import FeatureExtractor
from ..io import Checkpoint, ShardWriter, provenance, read_json, write_json
from ..models import GENERATORS


def trace_spec(cfg: Config, generator) -> dict:
    """The trace fingerprint (ADDING_A_METHOD.md §5.0).

    Everything that can change the bf16 arithmetic of the generate() call, or the tokens
    it reads. The prompt is fingerprinted by rendering it for two fixed probe items (with
    and without a passage), so a change to the template or to prompts.py is caught, but a
    change to a comment is not.
    """
    probes = [QAItem(item_id="probe:0", question="What is the capital of France?",
                     gold_answers=["Paris"]),
              QAItem(item_id="probe:1", question="Who is speaking?", gold_answers=["Ann"],
                     context="Ann said hello.")]
    rendered = "\x00".join(generator._prompt_text(p) for p in probes)
    device = cfg.generator.device
    gpu = (torch.cuda.get_device_name(int(device.split(":")[1]) if ":" in device else 0)
           if device.startswith("cuda") and torch.cuda.is_available() else device)
    return {
        "generator": cfg.generator.model_id,
        "dtype": cfg.generator.dtype,
        "load_in_4bit": cfg.generator.load_in_4bit,
        "attn_implementation": generator.model.config._attn_implementation,
        "decoding": {"do_sample": False, "max_new_tokens": cfg.generator.max_new_tokens,
                     "enable_thinking": cfg.generator.enable_thinking,
                     "completion_mode": generator.completion_mode},
        "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest()[:16],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }


@dataclass
class _Store:
    """Where one group of extractors writes its blocks, with its own checkpoint."""

    name: str
    directory: Path
    extractors: list[FeatureExtractor]
    writer: ShardWriter
    checkpoint: Checkpoint
    staged: list[dict] = field(default_factory=list)

    def commit(self, shard: str | None) -> None:
        for record in self.staged:
            self.checkpoint.mark(**record, shard=shard)
        self.staged.clear()


def _check_or_write_spec(path: Path, spec: dict) -> None:
    if path.exists():
        existing = read_json(path)
        if existing != spec:
            raise SystemExit(
                f"{path} was built under another trace fingerprint or extractor settings.\n"
                f"  stored: {existing}\n  now:    {spec}\n"
                "Refusing to mix blocks from different traces (PROTOCOL.md, M12)."
            )
    else:
        write_json(path, spec)


def _open_stores(cfg: Config, dataset_name: str, extractors: list[FeatureExtractor],
                 spec: dict) -> list[_Store]:
    stores = []
    stage1 = [e for e in extractors if e.store == "stage1"]
    if stage1:
        d = cfg.stage_dir("stage1_extract", dataset_name)
        _check_or_write_spec(d / "features_spec.json", {
            "trace_spec": spec, "extractors": {e.name: e.params() for e in stage1}})
        stores.append(_Store("stage1", d, stage1,
                             ShardWriter(d, shard_size=cfg.shard_size, prefix="features"),
                             Checkpoint(d / "features_checkpoint.jsonl")))
    for e in extractors:
        if e.store == "stage1":
            continue
        if e.store != "methods":
            raise ValueError(f"extractor {e.name}: unknown store {e.store!r}")
        d = cfg.stage_dir("methods", e.name, dataset_name)
        _check_or_write_spec(d / "spec.json", {"trace_spec": spec,
                                               "extractors": {e.name: e.params()}})
        stores.append(_Store(e.name, d, [e],
                             ShardWriter(d, shard_size=cfg.shard_size, prefix=e.name),
                             Checkpoint(d / "checkpoint.jsonl")))
    return stores


def run_dataset(cfg: Config, dataset_name: str, generator, extractors: list[FeatureExtractor],
                spec: dict) -> dict:
    out_dir = cfg.stage_dir("stage1_extract", dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    dataset = DATASETS.create(dataset_name)
    pool = dataset.sample_pool(cfg.pool_size, cfg.pool_seed)
    by_id = {item.item_id: item for item in pool}

    generations = Checkpoint(out_dir / "checkpoint.jsonl")
    # checkpoint.jsonl carries whole records, so it — not the manifest — is the source
    # of truth after an interrupted run.
    records: dict[str, dict] = {i: dict(r) for i, r in generations.records().items()}
    # The fingerprint is written before the first generation, so it exists whenever a
    # generation record does. Records without it come from the pre-R2 pipeline.
    spec_path = out_dir / "trace_spec.json"
    if spec_path.exists():
        stored_spec = read_json(spec_path)
        if stored_spec != spec:
            raise SystemExit(f"{spec_path}: trace fingerprint differs.\n"
                             f"  stored: {stored_spec}\n  now:    {spec}")
    elif records:
        raise SystemExit(
            f"{out_dir} was made before the shared extraction pass (no trace "
            "fingerprint). Its features cannot be mixed with new ones; use a new run-id."
        )
    else:
        write_json(spec_path, spec)

    stores = _open_stores(cfg, dataset_name, extractors, spec)
    # Stores written by earlier passes (other methods) stay listed in the manifest.
    earlier = read_json(manifest_path).get("stores", {}) if manifest_path.exists() else {}
    pending = [i.item_id for i in pool if any(i.item_id not in s.checkpoint for s in stores)]
    print(f"[{dataset_name}] pool={len(pool)} generated={len(records)} "
          + " ".join(f"{s.name}={len(s.checkpoint)}" for s in stores)
          + f" pending={len(pending)}", flush=True)

    failures: list[dict] = []
    regenerated = 0
    started = time.perf_counter()

    for item_id in tqdm(pending, desc=dataset_name, unit="item"):
        item = by_id[item_id]
        todo = [s for s in stores if item_id not in s.checkpoint]
        needs = frozenset().union(*(e.needs for s in todo for e in s.extractors))
        try:
            generation, trace = generator.generate(item, needs)
            blocks = {s.name: {} for s in todo}
            seconds = {}
            for s in todo:
                t0 = time.perf_counter()
                for extractor in s.extractors:
                    blocks[s.name].update(extractor.extract(trace))
                seconds[s.name] = round(time.perf_counter() - t0, 3)
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

        ids = generation.meta["generated_ids"]
        if item_id in records:
            # This item's trace was made before: the new one must be the same trace.
            if records[item_id].get("generated_ids") != ids:
                raise SystemExit(
                    f"{item_id}: generate() produced different ids from the stored run "
                    "(same fingerprint). The trace is not reproducible here; stopping "
                    "rather than mixing traces (ADDING_A_METHOD.md, test F3)."
                )
            regenerated += 1
        else:
            record = {
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
                "trace_len": generation.meta.get("trace_len"),
                "generated_ids": ids,
            }
            generations.mark(**record)
            records[item_id] = record

        for s in todo:
            s.staged.append({"item_id": item_id, "seconds": seconds[s.name],
                             "shapes": {k: list(v.shape) for k, v in blocks[s.name].items()}})
            flushed = s.writer.add(item_id, blocks[s.name])
            if flushed:
                s.commit(flushed)
                write_json(manifest_path, _summary(cfg, dataset_name, pool, records, stores,
                                                   failures, started, spec, earlier))

    for s in stores:
        s.commit(s.writer.flush())
        s.checkpoint.close()
    generations.close()

    summary = _summary(cfg, dataset_name, pool, records, stores, failures, started, spec,
                       earlier)
    summary["regenerated_and_verified"] = regenerated
    write_json(manifest_path, summary)
    print(
        f"[{dataset_name}] extracted={summary['n_extracted']} failed={len(failures)} "
        f"verified={regenerated} truncated={summary['truncated_rate']:.1%} "
        f"in {summary['elapsed_seconds'] / 60:.1f} min", flush=True,
    )
    return summary


def _guard_generator(cfg) -> None:
    """Refuse to mix generators inside one run directory.

    Feature shapes are model-specific ([L, H, k] and [L+1, d] both change), so writing a
    second generator into an existing run would produce shards that cannot be stacked —
    and the failure would surface much later, in stage 3, as a confusing shape error.
    """
    for dataset_name in cfg.datasets:
        manifest = cfg.stage_dir("stage1_extract", dataset_name) / "manifest.json"
        if not manifest.exists():
            continue
        existing = read_json(manifest).get("generator")
        if existing and existing != cfg.generator.model_id:
            raise SystemExit(
                f"run '{cfg.run_id}' already holds features from {existing}, but this run "
                f"uses {cfg.generator.model_id}. Pass --run-id to start a separate run "
                f"(e.g. --run-id {cfg.generator.model_id.split('/')[-1].lower()})."
            )


def _summary(cfg, dataset_name, pool, records, stores, failures, started, spec,
             earlier: dict | None = None) -> dict:
    ordered = [records[item.item_id] for item in pool if item.item_id in records]
    complete = [r for r in ordered if all(r["item_id"] in s.checkpoint for s in stores)]
    return {
        "dataset": dataset_name,
        "generator": cfg.generator.model_id,
        "trace_spec": spec,
        "stores": {**(earlier or {}),
                   **{s.name: {"directory": str(s.directory), "n_done": len(s.checkpoint),
                               "extractors": [e.name for e in s.extractors]} for s in stores}},
        "pool_size": len(pool),
        # Items generated *and* present in every store this pass wrote to.
        "n_extracted": len(complete),
        "n_generated": len(ordered),
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
    parser.add_argument(
        "--model", default=None,
        help="generator: a preset (qwen3-14b, qwen3-4b, llama3.2-3b, gemma3-4b, "
             "gemma3-12b) or any HF model id",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="output namespace. A different generator REQUIRES a different run-id: "
             "feature shapes are model-specific and would collide.",
    )
    parser.add_argument(
        "--features", nargs="+", default=None,
        help="extractors to run (default: the config's list). To add a method to an "
             "existing run, name only that method: the pass must then reproduce the "
             "stored ids and fingerprint.",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.datasets:
        cfg.datasets = args.datasets
    if args.limit:
        cfg.pool_size = args.limit
    if args.device:
        cfg.generator.device = args.device
    if args.model:
        from ..models.hf import resolve_model_id
        cfg.generator.model_id = resolve_model_id(args.model)
    if args.run_id:
        cfg.run_id = args.run_id
    if args.features:
        cfg.features = args.features
    unknown = [n for n in cfg.features if n not in FEATURES]
    if unknown:
        raise SystemExit(f"unknown extractors {unknown}; available: {FEATURES.names()}")
    _guard_generator(cfg)

    # Built once: adapters verify their upstream checkout here, before the model loads.
    extractors = [FEATURES.create(name) for name in cfg.features]
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
        spec = trace_spec(cfg, generator)
        summaries = [run_dataset(cfg, name, generator, extractors, spec)
                     for name in cfg.datasets]
    finally:
        generator.unload()

    write_json(
        cfg.stage_dir("stage1_extract") / "summary.json",
        {
            "run_id": cfg.run_id,
            "trace_spec": spec,
            "datasets": {s["dataset"]: {k: v for k, v in s.items() if k != "items"} for s in summaries},
            "provenance": provenance(),
        },
    )


if __name__ == "__main__":
    main()
