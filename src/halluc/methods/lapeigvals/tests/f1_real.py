"""F1-real: deviation D1, measured on real items (GPU).

Upstream reads attention from `generate()`: one row per decoding step, computed against a
KV cache. The adapter reads one full forward over the same tokens. The two are the same
matrix in exact arithmetic; in bf16 they are not. This measures by how much, on the
features themselves, and whether it changes the top-k order.

For each item: greedy `generate(output_attentions=True)` with eager attention and the
harness prompt, run through the official pipeline exactly as feature_storage.py does it;
then the generated ids re-forwarded and run through the adapter. Both feed the same
official functions, so any difference is the attention input alone.

Usage: uv run python src/halluc/methods/lapeigvals/tests/f1_real.py --run llama3.2-3b --n 20
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from halluc.config import Config
from halluc.datasets import DATASETS
from halluc.io import read_json
from halluc.methods.lapeigvals import OfficialSpectralFeatures, load
from halluc.methods.lapeigvals.adapter import LAP_BLOCK
from halluc.models.hf import HFGenerator


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=20, help="items per dataset")
    ap.add_argument("--datasets", nargs="*", default=["triviaqa", "nq_open", "squad_v2", "coqa"])
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    model_id = read_json(cfg.stage_dir("stage1_extract", args.datasets[0]) / "manifest.json")["generator"]
    gen = HFGenerator(model_id=model_id, device=args.device, dtype="bfloat16")
    tok = gen.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ext = OfficialSpectralFeatures(pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
    up = load()

    rows = []
    for ds in args.datasets:
        items = DATASETS.create(ds).sample_pool(cfg.pool_size, cfg.pool_seed)[: args.n]
        for item in items:
            p_ids = tok(gen._prompt_text(item), return_tensors="pt").input_ids.to(args.device)
            gen.model.set_attn_implementation("eager")
            try:
                out = gen.model.generate(
                    p_ids, max_new_tokens=gen.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None, pad_token_id=pad_id,
                    output_attentions=True, return_dict_in_generate=True,
                )
            finally:
                gen.model.set_attn_implementation("sdpa")
            generated = out.sequences.cpu()
            steps = tuple(tuple(a.cpu() for a in step) for step in out.attentions)
            stacked = up["attention_weights"].stack_attention_matrix(steps)
            (example,) = up["processing"].remove_padding_from_intermediate_states(
                per_layer_batched_data=stacked, data_type="attn",
                generated_tokens=generated, pad_token_id=pad_id)
            official = up["attention_weights"].laplacian_diagonal_from_attn(
                example, vertical_edges=False).float()
            del out, steps, stacked, example

            # Adapter path: one full forward over all generated ids. Upstream never feeds
            # the last one back, so this is the T' = T - 1 case of official_input.
            trace = gen._trace(generated.to(args.device), p_ids.shape[1])
            _, adapted = ext.diagonals(trace, stopped=True, ends_with_terminator=True)
            del trace

            k = min(10, official.shape[-1])
            top_off = official.sort(dim=-1, descending=True).values[..., :k]
            top_ada = adapted.sort(dim=-1, descending=True).values[..., :k]
            rows.append({
                "dataset": ds, "item_id": item.item_id, "T": int(official.shape[-1]),
                "same_T": official.shape == adapted.shape,
                "max_abs_diag": float((official - adapted).abs().max()) if official.shape == adapted.shape else None,
                "max_abs_top10": float((top_off - top_ada).abs().max()) if official.shape == adapted.shape else None,
                "max_abs_value": float(official.abs().max()),
            })
            print(json.dumps(rows[-1]), flush=True)

    diffs = np.array([r["max_abs_top10"] for r in rows if r["max_abs_top10"] is not None])
    print(json.dumps({
        "run": args.run, "n": len(rows), "same_T": sum(r["same_T"] for r in rows),
        "top10_max_abs_diff": {"median": float(np.median(diffs)), "max": float(diffs.max())},
        "exact": int((diffs == 0).sum()),
    }))


if __name__ == "__main__":
    main()
