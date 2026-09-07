"""SAPLMA features from k sampled generations, not just the greedy one.

Consistency detectors (SelfCheckGPT, semantic entropy) compare sampled generations as
*text*. This reads the same disagreement in representation space instead: sample k answers,
take each one's last-token hidden state, and hand the classifier all k rather than a scalar
agreement score. If the model is unsure, the k states should scatter; if it knows, they
should coincide — and the classifier can use the shape of that scatter, not just its mean.

This is the one axis the mechanism predicts should work. Every failed combination so far
varied the readout, the combination rule, or the training data while holding the forward
pass fixed. Sampling produces genuinely different forward passes, so it is a change of
information source rather than of perspective on the same information.

Design decisions that keep the comparison honest:

  The greedy answer remains the labelled unit. Its judge verdict and its stored features are
  unchanged, so every item set and every label in the study stays identical and the sampled
  features slot in as an extra block. Labelling a sampled answer instead would silently make
  this arm incomparable with everything else.

  The k samples are exchangeable — slot 1 is arbitrary — so a raw concatenation would make
  the probe learn per-slot weights that mean nothing. They are sorted by mean sequence
  logprob, giving a canonical order: the resulting vector is the empirical quantile function
  of the sample set rather than an arbitrary permutation of it.

  Only the probe layer is persisted, not all L+1. Storing every layer for every sample would
  cost k times stage 1's footprint; the layer is fixed to the depth used throughout this
  study, which also keeps the sampled and greedy features directly comparable.

Generation is batched with num_return_sequences; hidden states come from one padded forward
pass over the whole batch afterwards, not from the generate call, because per-step hidden
states are far larger than the one position actually needed.

Usage: uv run python scripts/extract_sampled_saplma.py --run main --device cuda:0
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

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}


@torch.inference_mode()
def sample_batch(gen, items, k, temperature, top_p, max_new, layer):
    """Generate k samples per item, then read each one's last-token state and logprob."""
    tok = gen.tokenizer
    prompts = [gen._prompt_text(it) for it in items]
    old_side = tok.padding_side
    tok.padding_side = "left"          # left pad so every prompt ends at the same index
    enc = tok(prompts, return_tensors="pt", padding=True).to(gen.input_device)
    tok.padding_side = old_side

    out = gen.model.generate(
        **enc, do_sample=True, temperature=temperature, top_p=top_p,
        num_return_sequences=k, max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    plen = enc["input_ids"].shape[1]
    stop = gen._stop_ids() | set(getattr(gen.model.generation_config, "eos_token_id", []) or [])

    # One padded forward over all B*k sequences instead of B*k separate passes: the
    # per-sample loop was the throughput bottleneck. A hook on the target decoder layer
    # captures just that layer, because output_hidden_states=True would materialise all
    # L+1 of them (~6 GB at this batch size) to use one.
    lens, kept = [], []
    for row in out:
        ans = row[plen:]
        keep = ans.shape[0]
        while keep > 0 and ans[keep - 1].item() in stop:
            keep -= 1
        lens.append(keep)
        kept.append(ans[:keep])
    texts = [tok.decode(a, skip_special_tokens=True) for a in kept]

    width = plen + max(1, max(lens))
    seqs = out[:, :width].clone()
    mask = torch.zeros_like(seqs)
    mask[:, :plen] = enc["attention_mask"].repeat_interleave(k, dim=0)[:, :plen]
    for i, L in enumerate(lens):
        mask[i, plen:plen + L] = 1

    grabbed = {}
    layers = gen.model.model.layers if hasattr(gen.model, "model") else gen.model.layers
    h_idx = max(0, min(layer, len(layers)) - 1)

    def hook(_m, _i, o):
        grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    handle = layers[h_idx].register_forward_hook(hook)
    try:
        o = gen.model(seqs, attention_mask=mask, use_cache=False)
    finally:
        handle.remove()
    H = grabbed["h"]

    states, logps = [], []
    for i, L in enumerate(lens):
        if L == 0:
            states.append(None); logps.append(-1e9); continue
        states.append(H[i, plen + L - 1, :].float().cpu().numpy())
        # Per-row slice, not a whole-batch log_softmax: casting every logit to float32
        # at once is ~6 GB at this batch size, while one row's answer span is ~35 MB.
        lg = torch.log_softmax(o.logits[i, plen - 1:plen + L - 1, :].float(), dim=-1)
        logps.append(float(lg.gather(1, kept[i].unsqueeze(1)).mean()))
        del lg
    del o, H, grabbed

    dim = next((s.shape[0] for s in states if s is not None), 1)
    per_item = []
    for i in range(len(items)):
        sl = slice(i * k, (i + 1) * k)
        S = [np.zeros(dim, np.float32) if s is None else s for s in states[sl]]
        L = logps[sl]
        order = np.argsort(L)[::-1]                    # canonical order: most likely first
        per_item.append((np.stack([S[j] for j in order]),
                         np.array([L[j] for j in order], np.float32),
                         [texts[sl][j] for j in order]))
    return per_item


def collapse_rate(all_texts):
    """Share of items whose k samples are all the same string.

    At low temperature this is the number that decides whether the block carries anything:
    identical samples give identical hidden states, so those rows are constant and the
    probe can only learn from the items where sampling actually diverged.
    """
    return float(np.mean([len(set(t)) == 1 for t in all_texts])) if all_texts else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    layer = LAYER.get(args.run, 24)
    gen = HFGenerator(model_id=cfg.generator.model_id, device=args.device,
                      dtype="bfloat16", load_in_4bit=False)
    gen.model.set_attn_implementation("sdpa")
    if gen.tokenizer.pad_token_id is None:
        gen.tokenizer.pad_token = gen.tokenizer.eos_token

    tag = f"sampled_k{args.k}_t{args.temperature}"
    done_total = 0
    for ds_name in (args.datasets or cfg.datasets):
        d = cfg.stage_dir("stage1_extract", ds_name)
        stage1 = {json.loads(l).get("item_id") or json.loads(l)["id"]: 1
                  for l in (d / "checkpoint.jsonl").open()}
        by_id = {i.item_id: i for i in DATASETS.create(ds_name).sample_pool(cfg.pool_size)}

        out_dir = d / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        ck = Checkpoint(out_dir / "checkpoint.jsonl")
        writer = ShardWriter(out_dir, prefix="sampled")
        pending = ck.pending([i for i in stage1 if i in by_id])
        print(f"{ds_name}: {len(pending)} pending", flush=True)

        staged, t0, all_texts = [], time.perf_counter(), []
        with ck:
            for b in range(0, len(pending), args.batch):
                if args.limit is not None and done_total >= args.limit:
                    break
                ids = pending[b:b + args.batch]
                try:
                    res = sample_batch(gen, [by_id[i] for i in ids], args.k,
                                       args.temperature, args.top_p, args.max_new, layer)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  OOM on batch at {b}, skipping", flush=True)
                    continue
                for iid, (S, L, T) in zip(ids, res):
                    all_texts.append(T)
                    staged.append(iid)
                    flushed = writer.add(iid, {"sampled_saplma": S.astype(np.float16),
                                               "sample_logp": L})
                    if flushed:
                        for s_id in staged:
                            ck.mark(s_id, shard=flushed)
                        staged.clear()
                    done_total += 1
                if b and (b // args.batch) % 20 == 0:
                    r = done_total / max(time.perf_counter() - t0, 1e-9)
                    eta = (len(pending) - b) / max(r, 1e-9) / 3600
                    print(f"  {ds_name} {b}/{len(pending)}  {r:.1f} it/s  eta {eta:.1f}h"
                          f"  all-{args.k}-identical {collapse_rate(all_texts):.1%}",
                          flush=True)
            flushed = writer.flush()
            if flushed:
                for s_id in staged:
                    ck.mark(s_id, shard=flushed)
        print(f"  {ds_name}: {done_total} done this call, "
              f"all-{args.k}-identical {collapse_rate(all_texts):.1%}", flush=True)
        if args.limit is not None and done_total >= args.limit:
            break


if __name__ == "__main__":
    main()
