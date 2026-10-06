"""F1 and F3 on real `generate()` outputs (GPU), for the shared trace (ADDING_A_METHOD.md §7).

Per item, three generate() calls through HFGenerator, exactly as stage 1 makes them:

  F1   the official pipeline (map to CPU -> stack_attention_matrix -> remove_padding ->
       diagonals -> top-k), run on the trace's native view and its ids, against the
       adapter on the same trace. Same input, so the result must be bitwise equal.
  F3   a second call with the same needs: same generated ids, identical blocks.
  F3b  a third call that asks for hidden states only (no attention weights): same
       generated ids. Requesting different outputs must not change the computation,
       because a later stage-1 run for another method may request different outputs.

Usage: uv run python src/halluc/methods/lapeigvals/tests/f1_real.py --model llama3.2-3b --n 5
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from halluc.config import Config
from halluc.datasets import DATASETS
from halluc.methods.lapeigvals import OfficialSpectralFeatures, load
from halluc.methods.lapeigvals.adapter import LAP_BLOCK
from halluc.models.hf import HFGenerator, resolve_model_id


def official_lap(up, trace) -> torch.Tensor:
    steps = tuple(tuple(a.cpu() for a in step) for step in trace.step_attentions)
    stacked = up["attention_weights"].stack_attention_matrix(steps)
    (example,) = up["processing"].remove_padding_from_intermediate_states(
        per_layer_batched_data=stacked, data_type="attn",
        generated_tokens=trace.input_ids.unsqueeze(0), pad_token_id=trace.pad_token_id)
    return up["attention_weights"].laplacian_diagonal_from_attn(
        example, vertical_edges=False).float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="preset or HF id")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=5, help="items per dataset")
    ap.add_argument("--datasets", nargs="*", default=["triviaqa", "nq_open", "squad_v2", "coqa"])
    args = ap.parse_args()

    cfg = Config()
    gen = HFGenerator(model_id=resolve_model_id(args.model), device=args.device, dtype="bfloat16")
    ext = OfficialSpectralFeatures()
    up = load()
    needs = frozenset({"attentions"})

    rows = []
    for ds in args.datasets:
        for item in DATASETS.create(ds).sample_pool(cfg.pool_size, cfg.pool_seed)[: args.n]:
            g1, t1 = gen.generate(item, needs)
            feats1 = ext.extract(t1)
            lap_up = official_lap(up, t1)
            k = min(10, lap_up.shape[-1])
            up_top = lap_up.sort(dim=-1, descending=True).values[..., :k].numpy()
            f1 = bool(np.array_equal(up_top, feats1[LAP_BLOCK][..., :k]))
            del t1

            g2, t2 = gen.generate(item, needs)
            feats2 = ext.extract(t2)
            del t2
            f3 = (g1.meta["generated_ids"] == g2.meta["generated_ids"]
                  and all(np.array_equal(feats1[b], feats2[b], equal_nan=True) for b in feats1))

            g3, t3 = gen.generate(item, frozenset({"hidden_states"}))
            del t3
            f3b = g1.meta["generated_ids"] == g3.meta["generated_ids"]

            rows.append({"dataset": ds, "item_id": item.item_id,
                         "prompt": g1.prompt_tokens, "answer": g1.answer_tokens,
                         "trace_len": g1.meta["trace_len"], "finish": g1.finish_reason,
                         "F1_bitwise": f1, "F3_same_ids_and_blocks": f3,
                         "F3b_same_ids_other_needs": f3b})
            print(json.dumps(rows[-1]), flush=True)
            torch.cuda.empty_cache()

    summary = {k: f"{sum(r[k] for r in rows)}/{len(rows)}"
               for k in ("F1_bitwise", "F3_same_ids_and_blocks", "F3b_same_ids_other_needs")}
    print(json.dumps({"model": args.model, "n": len(rows), **summary,
                      "max_trace_len": max(r["trace_len"] for r in rows),
                      "peak_gib": round(torch.cuda.max_memory_allocated(args.device) / 2**30, 1)}))


if __name__ == "__main__":
    main()
