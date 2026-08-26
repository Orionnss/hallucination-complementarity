"""Score a candidate judge against the established gemma+qwen consensus.

Picking a third judge by reputation has failed twice: Mistral-Nemo over-flagged
(45.8% HALLUCINATED) and Llama-3.1-8B under-flagged (3.7%), against gemma 23.2% and
qwen 29.8%. Batched judging makes a few hundred items cheap, so candidates are measured
before committing to a full 15,610-item pass.

A good third judge should (a) have a hallucination rate in the same range as the two
reference judges, and (b) agree with their consensus at a kappa comparable to the
0.65-0.94 the two of them reach with each other.

Usage: uv run python scripts/pilot_judge.py <model_id> [--n 500] [--device cuda:1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import cohen_kappa_score

from halluc.io import read_json
from halluc.judges import JUDGES
from halluc.judges.base import Label
from halluc.judges.pool import majority_vote

DATASETS = ["triviaqa", "nq_open", "squad_v2", "coqa"]
REFERENCE = {
    "gemma": "google_gemma_3_12b_it",
    "qwen": "qwen_qwen2_5_14b_instruct",
}


def load_reference(dataset: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for name, slug in REFERENCE.items():
        path = Path(f"runs/main/stage2_judge/{dataset}/verdicts_{slug}.jsonl")
        out[name] = {
            r["id"]: r["label"] for r in (json.loads(l) for l in open(path))
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_id")
    ap.add_argument("--n", type=int, default=500, help="items sampled across all datasets")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    per_dataset = max(args.n // len(DATASETS), 1)
    work, refs = [], {}
    for ds in DATASETS:
        items = read_json(f"runs/main/stage1_extract/{ds}/manifest.json")["items"]
        refs[ds] = load_reference(ds)
        # Stride rather than head-slice, so the sample spans the whole pool.
        stride = max(len(items) // per_dataset, 1)
        for item in items[::stride][:per_dataset]:
            work.append((ds, item))
    print(f"piloting {args.model_id} on {len(work)} items ({per_dataset}/dataset)")

    judge = JUDGES.create(
        "hf_judge", model_id=args.model_id, device=args.device,
        load_in_4bit=True, max_new_tokens=8, batch_size=args.batch_size,
    )
    labels: list[str] = []
    for start in range(0, len(work), args.batch_size):
        chunk = work[start : start + args.batch_size]
        results = judge.judge_batch(
            [(i["question"], i["gold_answers"], i["answer"]) for _, i in chunk]
        )
        labels += [lab.value for lab, _ in results]

    cand, g, q, cons = [], [], [], []
    for (ds, item), lab in zip(work, labels):
        iid = item["item_id"]
        if iid not in refs[ds]["gemma"] or iid not in refs[ds]["qwen"]:
            continue
        gl, ql = refs[ds]["gemma"][iid], refs[ds]["qwen"][iid]
        cand.append(lab); g.append(gl); q.append(ql)
        # Consensus is defined only where the two reference judges agree.
        cons.append(gl if gl == ql else None)

    rate = lambda xs: sum(x == "HALLUCINATED" for x in xs) / len(xs)
    print(f"\n{'judge':10s}{'HALL rate':>11s}")
    for name, xs in [("gemma", g), ("qwen", q), ("CANDIDATE", cand)]:
        print(f"{name:10s}{rate(xs):11.1%}")

    print(f"\n{'vs':22s}{'kappa':>8s}{'agree':>8s}{'n':>7s}")
    for name, xs in [("gemma", g), ("qwen", q)]:
        k = cohen_kappa_score(cand, xs)
        a = sum(x == y for x, y in zip(cand, xs)) / len(xs)
        print(f"candidate vs {name:9s}{k:8.3f}{a:8.1%}{len(xs):7d}")

    both = [(c, r) for c, r in zip(cand, cons) if r is not None]
    k = cohen_kappa_score([c for c, _ in both], [r for _, r in both])
    a = sum(c == r for c, r in both) / len(both)
    print(f"candidate vs {'CONSENSUS':9s}{k:8.3f}{a:8.1%}{len(both):7d}")
    print(f"\nreference: gemma-qwen kappa on these items = "
          f"{cohen_kappa_score(g, q):.3f} ({sum(x==y for x,y in zip(g,q))/len(g):.1%} agree)")


if __name__ == "__main__":
    main()
