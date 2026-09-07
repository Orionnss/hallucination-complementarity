"""Token-level confidence features: the cheap baseline every white-box probe must beat.

The study compares hidden-state and attention probes against each other, but not against
the signal a practitioner would reach for first — the model's own token probabilities.
That baseline needs no probe, no training set, and no access beyond the logits head. If
SAPLMA's last-token state is largely a re-encoding of sequence confidence, the white-box
probing literature is measuring something it already had for free, and the paper's central
comparison changes meaning. This makes the question answerable.

Greedy decoding makes the forward pass exact rather than approximate: the stored answer is
the argmax continuation, so teacher-forcing over (prompt, stored answer) reproduces the
per-token distributions seen at generation time. `retrace`'s token-count check guards the
one failure mode — a tokenizer round trip that lands on a different id sequence — and
mismatched items are recorded and skipped rather than silently traced.

Attentions are not requested here (sdpa, no `output_attentions`), so this is far cheaper
than stage 1's eager trace: only the logits are needed.

Fourteen features per answer, all length-aware in different ways, because "confidence" has
several plausible operationalisations and picking one in advance would beg the question:
means and sums of log p, order statistics over the token sequence, predictive entropy,
margin between top-1 and top-2, and the fraction of tokens below fixed surprisal cuts.

Checkpointed per item, so the run can be chunked into foreground calls.

Usage: uv run python scripts/extract_logprob.py --run main --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluc.config import Config
from halluc.datasets import DATASETS
from halluc.io import Checkpoint, ShardWriter
from halluc.models.hf import HFGenerator
from halluc.prompts import generator_messages

FEATURES = [
    "mean_logp", "sum_logp", "min_logp", "max_logp", "median_logp", "std_logp",
    "logp_q10", "logp_q25", "mean_entropy", "max_entropy", "mean_margin",
    "frac_below_2", "frac_below_5", "n_tokens",
]


@torch.inference_mode()
def features_for(gen, item, answer: str, expected: int | None):
    prompt = gen._prompt_text(item)
    p_ids = gen.tokenizer(prompt, return_tensors="pt").input_ids.to(gen.input_device)
    a_ids = gen.tokenizer(answer, return_tensors="pt",
                          add_special_tokens=False).input_ids.to(gen.input_device)
    if expected is not None and a_ids.shape[1] != expected:
        return None, f"re-encoded {a_ids.shape[1]} tokens, stage 1 recorded {expected}"
    if a_ids.shape[1] == 0:
        return None, "empty answer"

    seq = torch.cat([p_ids, a_ids], dim=1)
    if seq.shape[1] > gen.max_seq_len:
        return None, f"sequence {seq.shape[1]} exceeds max_seq_len"

    out = gen.model(seq, use_cache=False)
    # Position i predicts token i+1, so answer token j is predicted from logits at
    # prompt_len + j - 1.
    start = p_ids.shape[1] - 1
    logits = out.logits[0, start:start + a_ids.shape[1], :].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    tok = a_ids[0]
    lp = logprobs.gather(1, tok.unsqueeze(1)).squeeze(1)          # [T_ans]
    ent = -(logprobs.exp() * logprobs).sum(dim=-1)
    top2 = logprobs.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]

    lp_np = lp.cpu().numpy().astype(np.float64)
    v = np.array([
        lp_np.mean(), lp_np.sum(), lp_np.min(), lp_np.max(),
        float(np.median(lp_np)), lp_np.std(),
        float(np.quantile(lp_np, 0.10)), float(np.quantile(lp_np, 0.25)),
        float(ent.mean()), float(ent.max()), float(margin.mean()),
        float((lp_np < -2.0).mean()), float((lp_np < -5.0).mean()),
        float(len(lp_np)),
    ], dtype=np.float32)
    return v, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many new items (chunking for foreground calls)")
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    gen = HFGenerator(model_id=cfg.generator.model_id, device=args.device,
                      dtype="bfloat16", load_in_4bit=False)
    gen.model.set_attn_implementation("sdpa")

    done_total = 0
    for ds_name in (args.datasets or cfg.datasets):
        d = cfg.stage_dir("stage1_extract", ds_name)
        stage1 = {}
        for line in (d / "checkpoint.jsonl").open():
            r = json.loads(line)
            stage1[r.get("item_id") or r["id"]] = r

        dataset = DATASETS.create(ds_name)
        by_id = {i.item_id: i for i in dataset.sample_pool(cfg.pool_size)}

        out_dir = d / "logprob"
        out_dir.mkdir(parents=True, exist_ok=True)
        ck = Checkpoint(out_dir / "checkpoint.jsonl")
        writer = ShardWriter(out_dir, prefix="logprob")
        pending = ck.pending([i for i in stage1 if i in by_id])
        print(f"{ds_name}: {len(pending)} pending of {len(stage1)}", flush=True)

        # Stage ids and commit only once their shard is on disk: a checkpoint entry
        # written before the flush would mark an item done whose features are still
        # in memory, and a resumed run would skip it forever.
        staged: list[str] = []
        skipped = 0
        t0 = time.perf_counter()
        with ck:
            for n, iid in enumerate(pending):
                if args.limit is not None and done_total >= args.limit:
                    break
                rec = stage1[iid]
                try:
                    v, err = features_for(gen, by_id[iid], rec["answer"],
                                          rec.get("answer_tokens"))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    v, err = None, "OOM"
                if err is not None:
                    skipped += 1
                    continue
                staged.append(iid)
                flushed = writer.add(iid, {"logprob": v})
                if flushed:
                    for s_id in staged:
                        ck.mark(s_id, shard=flushed)
                    staged.clear()
                done_total += 1
                if (n + 1) % 500 == 0:
                    r = done_total / max(time.perf_counter() - t0, 1e-9)
                    print(f"  {ds_name} {n + 1}/{len(pending)}  {r:.1f} it/s", flush=True)
            flushed = writer.flush()
            if flushed:
                for s_id in staged:
                    ck.mark(s_id, shard=flushed)
        print(f"  {ds_name} done, {done_total} new this call, {skipped} skipped",
              flush=True)
        if args.limit is not None and done_total >= args.limit:
            print("hit --limit, stopping (rerun to continue)", flush=True)
            break

    print(f"\nfeature order: {FEATURES}")


if __name__ == "__main__":
    main()
