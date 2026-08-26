"""Stage 2: label each generated answer with a pool of voting judges.

Runs as its own process, after stage 1 has exited, so the bf16 generator and the
quantised judges are never resident at the same time. Within this stage the judges are
also sequenced: load one, label every item of every dataset, unload, load the next.

Each judge's verdicts are checkpointed separately, so an interrupted run resumes at the
judge and item it stopped on rather than re-labelling everything.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..io import Checkpoint, provenance, read_json, write_json
from ..judges import JUDGES
from ..judges.base import Label
from ..judges.pool import SCORED, agreement_stats, majority_vote


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def _load_items(cfg: Config, dataset_name: str) -> list[dict]:
    manifest = cfg.stage_dir("stage1_extract", dataset_name) / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"stage 1 has not run for {dataset_name}: {manifest} missing")
    return read_json(manifest)["items"]


def run_judge(cfg: Config, judge_cfg, items_by_dataset: dict[str, list[dict]]) -> None:
    """Label every item of every dataset with one judge, then free it."""
    slug = _slug(judge_cfg.model_id)
    todo = {
        name: Checkpoint(cfg.stage_dir("stage2_judge", name) / f"verdicts_{slug}.jsonl")
        for name in items_by_dataset
    }
    pending_total = sum(
        len(cp.pending([i["item_id"] for i in items_by_dataset[name]]))
        for name, cp in todo.items()
    )
    if pending_total == 0:
        print(f"[judge {judge_cfg.model_id}] already complete")
        for cp in todo.values():
            cp.close()
        return

    print(f"[judge {judge_cfg.model_id}] loading, {pending_total} items pending")
    judge = JUDGES.create(
        judge_cfg.kind,
        model_id=judge_cfg.model_id,
        device=judge_cfg.device,
        load_in_4bit=judge_cfg.load_in_4bit,
        max_new_tokens=judge_cfg.max_new_tokens,
        batch_size=judge_cfg.batch_size,
    )
    try:
        batch_size = getattr(judge, "batch_size", 1)
        for name, items in items_by_dataset.items():
            checkpoint = todo[name]
            pending = set(checkpoint.pending([i["item_id"] for i in items]))
            todo_items = [i for i in items if i["item_id"] in pending]
            with tqdm(total=len(todo_items), desc=f"{slug[:18]}/{name}", unit="item") as bar:
                for start in range(0, len(todo_items), batch_size):
                    chunk = todo_items[start : start + batch_size]
                    began = time.perf_counter()
                    results = judge.judge_batch(
                        [(i["question"], i["gold_answers"], i["answer"]) for i in chunk]
                    )
                    per_item = (time.perf_counter() - began) / len(chunk)
                    # Checkpoint only after the whole batch returns, so a kill mid-batch
                    # re-judges those items rather than recording partial results.
                    for item, (label, raw) in zip(chunk, results):
                        checkpoint.mark(
                            item["item_id"],
                            label=label.value,
                            raw=raw[:200],
                            seconds=round(per_item, 4),
                        )
                    bar.update(len(chunk))
    finally:
        judge.unload()
        for cp in todo.values():
            cp.close()


def aggregate(cfg: Config, dataset_name: str, items: list[dict]) -> dict:
    """Combine judge verdicts into final labels plus agreement statistics."""
    out_dir = cfg.stage_dir("stage2_judge", dataset_name)
    verdicts_by_judge: dict[str, dict[str, Label]] = {}
    raw_by_judge: dict[str, dict[str, str]] = {}
    for judge_cfg in cfg.judges:
        slug = _slug(judge_cfg.model_id)
        records = Checkpoint(out_dir / f"verdicts_{slug}.jsonl").records()
        verdicts_by_judge[judge_cfg.model_id] = {
            i: Label(r["label"]) for i, r in records.items()
        }
        raw_by_judge[judge_cfg.model_id] = {i: r.get("raw", "") for i, r in records.items()}

    # Only items every judge labelled can be voted on.
    complete = [
        item["item_id"]
        for item in items
        if all(item["item_id"] in v for v in verdicts_by_judge.values())
    ]

    labels = []
    for item_id in complete:
        votes = {j: verdicts_by_judge[j][item_id] for j in verdicts_by_judge}
        final, count = majority_vote(list(votes.values()))
        labels.append(
            {
                "item_id": item_id,
                "label": final.value,
                "votes": {j: v.value for j, v in votes.items()},
                "raw": {j: raw_by_judge[j].get(item_id, "") for j in verdicts_by_judge},
                "n_agreeing": count,
                "unanimous": len(set(votes.values())) == 1,
                "scored": final in SCORED,
            }
        )

    stats = agreement_stats(verdicts_by_judge, complete)
    n_scored = sum(entry["scored"] for entry in labels)
    n_hallucinated = sum(entry["label"] == Label.HALLUCINATED.value for entry in labels)
    summary = {
        "dataset": dataset_name,
        "n_items": len(items),
        "n_labelled": len(complete),
        "n_scored": n_scored,
        "n_dropped_invalid": len(complete) - n_scored,
        # The base rate the probes have to beat; a degenerate rate makes MCC unstable.
        "hallucination_rate_scored": (
            round(n_hallucinated / n_scored, 4) if n_scored else None
        ),
        "agreement": stats,
        "provenance": provenance(),
        "labels": labels,
    }
    write_json(out_dir / "labels.json", summary)
    write_json(out_dir / "agreement.json", stats)
    print(
        f"[{dataset_name}] labelled={len(complete)} scored={n_scored} "
        f"dropped={len(complete) - n_scored} "
        f"halluc_rate={summary['hallucination_rate_scored']} "
        f"unanimous={stats['unanimous_rate']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", default=None, help="override judge device")
    parser.add_argument(
        "--judges", nargs="*", default=None,
        help="run only judges whose model_id contains one of these substrings; "
             "lets the pool run concurrently, one judge per GPU",
    )
    parser.add_argument(
        "--skip-aggregate", action="store_true",
        help="label only. Concurrent single-judge processes must skip aggregation, "
             "which needs every judge finished; run --aggregate-only afterwards.",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true", help="recompute votes without running judges"
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.datasets:
        cfg.datasets = args.datasets
    if args.device:
        for judge_cfg in cfg.judges:
            judge_cfg.device = args.device

    items_by_dataset = {name: _load_items(cfg, name) for name in cfg.datasets}

    if not args.aggregate_only:
        selected = cfg.judges
        if args.judges:
            selected = [
                j for j in cfg.judges
                if any(pat.lower() in j.model_id.lower() for pat in args.judges)
            ]
            if not selected:
                raise SystemExit(f"no judge matched {args.judges}")
        for judge_cfg in selected:
            run_judge(cfg, judge_cfg, items_by_dataset)

    if args.skip_aggregate:
        return

    summaries = [aggregate(cfg, name, items) for name, items in items_by_dataset.items()]
    write_json(
        cfg.stage_dir("stage2_judge") / "summary.json",
        {
            "run_id": cfg.run_id,
            "judges": [j.model_id for j in cfg.judges],
            "datasets": {
                s["dataset"]: {k: v for k, v in s.items() if k != "labels"} for s in summaries
            },
            "provenance": provenance(),
        },
    )


if __name__ == "__main__":
    main()
