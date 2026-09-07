"""Re-extract LapEigvals at a larger k, without regenerating the answers.

`k` is fixed at extraction time — stage 1 kept only the top 10 eigenvalues per head — so
studying larger k needs the attention matrices again. Generation does not need repeating:
decoding was greedy, so re-forwarding prompt+answer is teacher-forced on the tokens the
model actually produced. That replaces ~256 decode steps with one forward pass.

Because the features are the top-k sorted descending, storing k=256 yields k=64 and k=10
as prefixes. One pass therefore serves every k, and all of them are computed on the same
sequences, so the k-comparison is internally consistent.

Reconstruction fidelity was verified against the recorded token counts: Qwen and Llama
round-trip exactly; Gemma's answers begin with a newline that decoding dropped, so it is
restored here (verified 799/800 exact).

Writes to <stage1_extract>/<dataset>/k256/ so existing k=10 shards are untouched.

Usage: uv run python scripts/reextract_lapeigvals.py --run main --device cuda:1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluc.config import Config
from halluc.datasets import DATASETS
from halluc.features.base import ForwardTrace, top_k_sorted
from halluc.io import Checkpoint, ShardWriter, provenance, read_json, write_json
from halluc.models.hf import _load_causal_lm, resolve_model_id
from halluc.prompts import generator_messages
from transformers import AutoTokenizer

#: Trailing terminator that stage 1 failed to strip, and which is therefore part of the
#: stored sequence. Gemma's tokenizer reports eos_token_id=1 (<eos>) while generation
#: actually ends on 106 (<end_of_turn>), which `_stop_ids()` did not collect — so every
#: Gemma sequence carries it as the final token. Reproducing the stored sequence exactly
#: requires appending it back after re-tokenising the decoded answer.
#: See DESIGN.md: this is a known defect in the Gemma runs, documented not fixed.
ANSWER_SUFFIX_ID = {"google/gemma-3-12b-it": 106, "google/gemma-3-4b-it": 106}


def lapeigvals_k(trace: ForwardTrace, k: int) -> np.ndarray:
    """Same definition as the pipeline's LapEigvals, at arbitrary k."""
    per_layer = []
    for attn in trace.attentions:
        a = attn.to(torch.float32)
        divisor = (a > 0).sum(dim=1).clamp(min=1).to(torch.float32)
        eig = a.sum(dim=1) / divisor - torch.diagonal(a, dim1=-2, dim2=-1)
        per_layer.append(top_k_sorted(eig, k).cpu())
    return torch.stack(per_layer).numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True, help="preset or HF id")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    datasets = args.datasets or cfg.datasets
    model_id = resolve_model_id(args.model)
    suffix_id = ANSWER_SUFFIX_ID.get(model_id)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = _load_causal_lm(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(args.device).eval()
    print(f"[{args.run}] {model_id} on {args.device}, k={args.k}", flush=True)

    for ds in datasets:
        src = cfg.stage_dir("stage1_extract", ds)
        out_dir = src / f"k{args.k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        items = {r["item_id"]: r for r in read_json(src / "manifest.json")["items"]}
        pool = {i.item_id: i for i in DATASETS.create(ds).sample_pool(cfg.pool_size, cfg.pool_seed)}

        ckpt = Checkpoint(out_dir / "checkpoint.jsonl")
        pending = ckpt.pending([i for i in items if i in pool])
        print(f"[{ds}] {len(items)} extracted, {len(pending)} pending", flush=True)
        if not pending:
            ckpt.close(); continue

        writer = ShardWriter(out_dir, shard_size=200, prefix=f"lapeig_k{args.k}")
        staged, mismatch, started = [], 0, time.perf_counter()

        def commit(shard):
            for rec in staged:
                rec["shard"] = shard
                ckpt.mark(**rec)
            staged.clear()

        with ckpt, torch.inference_mode():
            for iid in tqdm(pending, desc=f"{args.run}/{ds}", unit="item"):
                rec, item = items[iid], pool[iid]
                prompt = tok.apply_chat_template(
                    generator_messages(item), tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                p_ids = tok(prompt, return_tensors="pt").input_ids
                a_ids = tok(rec["answer"], add_special_tokens=False,
                            return_tensors="pt").input_ids
                if suffix_id is not None:
                    a_ids = torch.cat(
                        [a_ids, torch.tensor([[suffix_id]], dtype=a_ids.dtype)], dim=1)
                if a_ids.shape[1] != rec["answer_tokens"]:
                    mismatch += 1
                seq = torch.cat([p_ids, a_ids], dim=1).to(args.device)

                model.set_attn_implementation("eager")
                try:
                    o = model(seq, output_attentions=True, use_cache=False)
                finally:
                    model.set_attn_implementation("sdpa")
                trace = ForwardTrace(tuple(a[0] for a in o.attentions), (), p_ids.shape[1])
                feats = {f"lapeigvals_k{args.k}": lapeigvals_k(trace, args.k)}
                del o, trace

                staged.append({"item_id": iid, "seq_len": int(seq.shape[1]),
                               "orig_seq_len": rec["seq_len"]})
                flushed = writer.add(iid, feats)
                if flushed:
                    commit(flushed)
            commit(writer.flush())

        write_json(out_dir / "manifest.json", {
            "run": args.run, "dataset": ds, "model": model_id, "k": args.k,
            "n_items": len(pending), "token_count_mismatches": mismatch,
            "elapsed_min": round((time.perf_counter() - started) / 60, 1),
            "provenance": provenance(),
        })
        print(f"[{ds}] done in {(time.perf_counter() - started)/60:.1f} min, "
              f"token-count mismatches: {mismatch}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
