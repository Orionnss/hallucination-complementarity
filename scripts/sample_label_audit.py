"""Draw a blind sample of SAPLMA's errors for manual adjudication.

Every negative result in this study lives in one population: the answers SAPLMA gets
wrong. If a large share of those are judge mistakes rather than detector mistakes, the
"unrecoverable" residual is partly noise and the ceiling analyses are measuring the wrong
thing. This samples that population so a human (or a fourth model) can check.

Two files are written, deliberately separated:

  blind_{run}.jsonl  question, context where the dataset keeps one, gold answers, and the
                     generated answer. No label, no vote count, no SAPLMA prediction.
  key.json           the labels, per-judge votes, unanimity flag and SAPLMA's prediction

The adjudicator reads only the blind file and commits verdicts before key.json is opened.
Seeing the pool's verdict first would turn the exercise into a measure of anchoring.

Sampling is stratified by dataset and by error direction (false positive vs false
negative), because the mix differs sharply per dataset — CoQA's 13-17% base rate makes its
errors overwhelmingly false negatives while the others are mostly false alarms — and an
unstratified draw would over-represent whatever is most common.

CoQA's story is not stored in stage 1 (only a has_context flag), so it is rebuilt from the
source dataset by group_id. Its questions are anaphoric and unjudgeable without it.

Usage: uv run python scripts/sample_label_audit.py --n 100
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluc.config import Config
from halluc.io import write_json

RUNS = ["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b"]
OUT = Path("runs/label_audit")


def load_text(cfg, datasets):
    """question / gold / generated answer, keyed by item id."""
    txt = {}
    for ds in datasets:
        f = cfg.stage_dir("stage1_extract", ds) / "checkpoint.jsonl"
        for line in f.open():
            d = json.loads(line)
            iid = d.get("item_id") or d["id"]
            txt[iid] = {"question": d["question"], "gold_answers": d["gold_answers"],
                        "answer": d["answer"], "group_id": d.get("group_id", "")}
    return txt


def coqa_contexts(needed_groups):
    """Rebuild CoQA stories + dialogue history for the sampled turns only."""
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/coqa", split="validation")
    out = {}
    for conv_idx, row in enumerate(ds):
        gid = f"coqa:{conv_idx}"
        if gid not in needed_groups:
            continue
        for turn in range(len(row["questions"])):
            history = "\n".join(
                f"Q: {q}\nA: {a}" for q, a in
                zip(row["questions"][:turn], row["answers"]["input_text"][:turn]))
            out[f"coqa:{conv_idx}:{turn}"] = (
                row["story"] + (f"\n\n{history}" if history else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--n", type=int, default=100, help="items sampled per model")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    key = {}

    for run in args.runs:
        cfg = Config(); cfg.run_id = run
        f = sorted(glob.glob(f"runs/{run}/stage5_posthoc/block_oof/*.npz"))[0]
        d = np.load(f, allow_pickle=True)
        y, p = d["y"], d["preds__saplma"]
        ids, ds_arr = d["item_ids"].astype(str), d["dataset"].astype(str)
        wrong = np.flatnonzero(p != y)

        # stratify: equal share per dataset, FP/FN inside it in their natural proportion
        per_ds = args.n // len(set(ds_arr))
        picked = []
        for ds in sorted(set(ds_arr)):
            pool = wrong[ds_arr[wrong] == ds]
            fp = pool[y[pool] == 0]
            fn = pool[y[pool] == 1]
            n_fp = min(len(fp), int(round(per_ds * len(fp) / max(len(pool), 1))))
            n_fn = min(len(fn), per_ds - n_fp)
            picked += list(rng.choice(fp, n_fp, replace=False))
            picked += list(rng.choice(fn, n_fn, replace=False))
        picked = np.array(sorted(picked))

        txt = load_text(cfg, sorted(set(ds_arr)))
        labels = {}
        for ds in sorted(set(ds_arr)):
            j = json.loads((cfg.stage_dir("stage2_judge", ds) / "labels.json").read_text())
            for e in j["labels"]:
                labels[e["item_id"]] = e

        ctx = {}
        coqa_ids = [ids[i] for i in picked if ds_arr[i] == "coqa"]
        if coqa_ids:
            ctx = coqa_contexts({txt[i]["group_id"] for i in coqa_ids})

        blind = OUT / f"blind_{run}.jsonl"
        with blind.open("w") as fh:
            for i in picked:
                iid = ids[i]
                t = txt[iid]
                rec = {"item_id": iid, "dataset": ds_arr[i], "question": t["question"],
                       "gold_answers": t["gold_answers"], "generated_answer": t["answer"]}
                if iid in ctx:
                    rec["context"] = ctx[iid]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                e = labels.get(iid, {})
                # Key on run+item: the same question carries a different label
                # per model, because the judges rate that model's own answer.
                key[f"{run}|{iid}"] = {"run": run, "dataset": ds_arr[i],
                            "reference_label": "HALLUCINATED" if y[i] else "NOT_HALLUCINATED",
                            "saplma_pred": "HALLUCINATED" if p[i] else "NOT_HALLUCINATED",
                            "error_type": "false_positive" if p[i] == 1 else "false_negative",
                            "votes": e.get("votes", {}), "n_agreeing": e.get("n_agreeing"),
                            "unanimous": e.get("unanimous")}
        counts = defaultdict(int)
        for i in picked:
            counts[(ds_arr[i], "FP" if p[i] == 1 else "FN")] += 1
        print(f"{run}: {len(picked)} items -> {blind}")
        print("   " + "  ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(counts.items())))

    write_json(OUT / "key.json", key)
    print(f"\nwrote {OUT}/key.json  ({len(key)} items)  -- do not open before judging")


if __name__ == "__main__":
    main()
