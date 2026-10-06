"""Extract the official LapEigvals / AttnEigvals blocks for one run (GPU).

Adapter stage for `halluc.methods.lapeigvals` (ADDING_A_METHOD.md §4.4). The answers are
stage 1's, never regenerated when avoidable (rule R2): each item is re-encoded and
re-forwarded once with eager attention (`HFGenerator.retrace` semantics). If the
tokenizer round trip does not land on stage 1's token count, the item falls back to a
real greedy `generate()`, so the trace is always of a sequence the model produced.

Gemma's instruct runs keep <end_of_turn> (106) as the final stored token (a known stage-1
defect, see scripts/reextract_lapeigvals.py), so it is appended back before the count
check. The adapter then treats it as the terminator the official pipeline would drop.

Output, per dataset, under runs/<run>/methods/lapeigvals_official/<dataset>/:
  spec.json          upstream commit, k_max, compute device; a store with another spec is
                     refused rather than mixed
  lapeig_*.npz       blocks lapeigvals_official [L,H,100], attneigvals_official [L,H,100],
                     lapeigvals_official_T [1]
  checkpoint.jsonl   resume; an item is marked only once its shard is on disk
  manifest.json      counts, fallbacks, failures, provenance

Usage: uv run python scripts/extract_lapeigvals_official.py --run main --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluc.config import Config
from halluc.datasets import DATASETS
from halluc.io import Checkpoint, ShardWriter, provenance, read_json, write_json
from halluc.methods.lapeigvals import K_MAX, SPEC, OfficialSpectralFeatures
from halluc.models.hf import HFGenerator

ANSWER_SUFFIX_ID = {"google/gemma-3-12b-it": 106, "google/gemma-3-4b-it": 106}


def terminator_ids(gen: HFGenerator) -> set[int]:
    eos = gen.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    return {i for i in eos | gen._stop_ids() if i is not None}


@torch.inference_mode()
def trace_for(gen: HFGenerator, item, record: dict, suffix_id: int | None):
    """(trace, stopped, how). Re-forward stage 1's answer; regenerate only on mismatch."""
    prompt = gen._prompt_text(item)
    p_ids = gen.tokenizer(prompt, return_tensors="pt").input_ids
    a_ids = gen.tokenizer(record["answer"], return_tensors="pt",
                          add_special_tokens=False).input_ids
    if suffix_id is not None:
        a_ids = torch.cat([a_ids, torch.tensor([[suffix_id]], dtype=a_ids.dtype)], dim=1)
    if a_ids.shape[1] == record["answer_tokens"]:
        seq = torch.cat([p_ids, a_ids], dim=1).to(gen.input_device)
        if seq.shape[1] > gen.max_seq_len:
            raise ValueError(f"sequence {seq.shape[1]} exceeds max_seq_len")
        return gen._trace(seq, p_ids.shape[1]), record["finish_reason"] == "stop", "retrace"
    generation, trace = gen.generate(item)
    return trace, generation.finish_reason == "stop", "regenerated"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--compute-device", default="cpu",
                    help="where the upstream functions run; upstream ran them on CPU")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="new items per dataset (smoke test)")
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    datasets = args.datasets or ["triviaqa", "nq_open", "squad_v2", "coqa"]
    model_id = read_json(cfg.stage_dir("stage1_extract", datasets[0]) / "manifest.json")["generator"]
    gen = HFGenerator(model_id=model_id, device=args.device, dtype="bfloat16")
    tok = gen.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    extractor = OfficialSpectralFeatures(pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                                         device=args.compute_device)
    terminators = terminator_ids(gen)
    suffix_id = ANSWER_SUFFIX_ID.get(model_id)
    spec = {"upstream": SPEC.url, "commit": SPEC.commit, "k_max": K_MAX,
            "generator": model_id, "compute_device": args.compute_device,
            "attention": "eager, one full forward over stage-1 prompt+answer, bf16"}
    print(f"[{args.run}] {model_id} on {args.device}; terminators={sorted(terminators)}",
          flush=True)

    for ds in datasets:
        out_dir = cfg.stage_dir("methods", "lapeigvals_official", ds)
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_path = out_dir / "spec.json"
        if spec_path.exists() and read_json(spec_path) != spec:
            raise SystemExit(f"{spec_path} was built under another spec; refusing to mix")
        write_json(spec_path, spec)

        records = {r["item_id"]: r for r in read_json(
            cfg.stage_dir("stage1_extract", ds) / "manifest.json")["items"]}
        pool = {i.item_id: i for i in DATASETS.create(ds).sample_pool(cfg.pool_size, cfg.pool_seed)}
        ckpt = Checkpoint(out_dir / "checkpoint.jsonl")
        pending = ckpt.pending([i for i in records if i in pool])
        if args.limit is not None:
            pending = pending[: args.limit]
        print(f"[{ds}] {len(records)} in stage 1, {len(pending)} pending", flush=True)

        writer = ShardWriter(out_dir, shard_size=200, prefix="lapeig")
        staged, failures, how_counts, started = [], [], {}, time.perf_counter()

        def commit(shard):
            for rec in staged:
                ckpt.mark(**rec, shard=shard)
            staged.clear()

        with ckpt:
            for iid in tqdm(pending, desc=f"{args.run}/{ds}", unit="item"):
                try:
                    trace, stopped, how = trace_for(gen, pool[iid], records[iid], suffix_id)
                    ends_term = int(trace.input_ids[-1]) in terminators
                    feats = extractor.extract(trace, stopped=stopped, ends_with_terminator=ends_term)
                    del trace
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    failures.append({"item_id": iid, "error": "OOM", "detail": str(exc)[:200]})
                    continue
                except Exception as exc:  # noqa: BLE001
                    failures.append({"item_id": iid, "error": type(exc).__name__,
                                     "detail": str(exc)[:200]})
                    continue
                how_counts[how] = how_counts.get(how, 0) + 1
                staged.append({"item_id": iid, "how": how, "stopped": stopped,
                               "ends_with_terminator": ends_term,
                               "n_tokens": int(feats["lapeigvals_official_T"][0])})
                flushed = writer.add(iid, feats)
                if flushed:
                    commit(flushed)
            flushed = writer.flush()
            if flushed:
                commit(flushed)

        write_json(out_dir / "manifest.json", {
            "run": args.run, "dataset": ds, "spec": spec, "n_done": len(ckpt),
            "this_call": how_counts, "failures": failures,
            "elapsed_min": round((time.perf_counter() - started) / 60, 1),
            "provenance": provenance(),
        })
        print(f"[{ds}] {how_counts}, {len(failures)} failed, "
              f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
