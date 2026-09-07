"""Join blind verdicts to the judge labels and estimate the label-error rate.

Reads verdicts_{run}.json (committed before key.json was opened) against key.json, on the
sample of answers SAPLMA gets wrong. A disagreement is a candidate labelling error: the
detector was penalised for a verdict a careful reader would not have given.

What the number does and does not support. It bounds label noise on SAPLMA's error set;
it does not measure it. The adjudicator here is another language model, of the same
family as the judge pool, so its verdicts are not independent of theirs in the way a
second human annotator's would be — agreement is inflated and disagreement is the more
informative direction. The estimate applies only to SAPLMA's errors, which is the
population every ceiling and complementarity result in this study lives in; the label
error rate over all answers will be lower, since errors concentrate on hard items.

Cross-tabs reported, because the aggregate rate is the least useful number here:

  by judge unanimity   if noise drives the residual, disagreement should concentrate in
                       the split-vote items. If it does not, the pool is confidently
                       wrong rather than merely uncertain, which is a different problem.
  by error direction   false positives and false negatives can have different causes
  by flag              why I disagreed: contestable gold, an unanswerable closed-book
                       SQuAD question, an anaphoric CoQA question with no recoverable
                       referent, or an evasive non-answer

Usage: uv run python scripts/label_audit_analyze.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from halluc.io import write_json

OUT = Path("runs/label_audit")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*",
                    default=["main", "gemma3-12b", "gemma3-4b", "llama3.2-3b"])
    args = ap.parse_args()

    key = json.loads((OUT / "key.json").read_text())
    summary = {}

    for run in args.runs:
        vf = OUT / f"verdicts_{run}.json"
        if not vf.exists():
            print(f"{run}: no verdicts yet, skipping")
            continue
        ver = json.loads(vf.read_text())
        rows = [(k, v, key[k]) for k, v in ver.items() if k in key]

        agree = [r for r in rows if r[1]["verdict"] == r[2]["reference_label"]]
        disag = [r for r in rows if r[1]["verdict"] != r[2]["reference_label"]]
        rate = len(disag) / max(len(rows), 1)

        print(f"\n{'=' * 74}\n{run}: {len(rows)} adjudicated answers SAPLMA got wrong")
        print(f"{'=' * 74}")
        print(f"  I agree with the judge pool : {len(agree):3d}  ({1 - rate:.1%})")
        print(f"  I disagree (candidate label error): {len(disag):3d}  ({rate:.1%})")

        for name, keyfn in (("judge unanimity", lambda k: "unanimous 3-0"
                             if k["unanimous"] else f"split {k['n_agreeing']}-1"),
                            ("SAPLMA error direction", lambda k: k["error_type"]),
                            ("dataset", lambda k: k["dataset"])):
            tab = defaultdict(lambda: [0, 0])
            for _, v, k in rows:
                tab[keyfn(k)][v["verdict"] != k["reference_label"]] += 1
            print(f"\n  by {name}")
            for g, (ok, bad) in sorted(tab.items()):
                n = ok + bad
                print(f"    {g:22s} n={n:3d}   disagree {bad:3d}  ({bad / n:.1%})")

        tab = defaultdict(lambda: [0, 0])
        for _, v, k in rows:
            tab[v["flag"]][v["verdict"] != k["reference_label"]] += 1
        print(f"\n  by reason flag (disagreements only shown as a share of that flag)")
        for g, (ok, bad) in sorted(tab.items(), key=lambda x: -x[1][1]):
            n = ok + bad
            print(f"    {g:22s} n={n:3d}   disagree {bad:3d}  ({bad / n:.1%})")

        summary[run] = {
            "n": len(rows), "disagreement_rate": round(rate, 4),
            "disagreements": [{"item": k, "dataset": kk["dataset"],
                               "reference": kk["reference_label"],
                               "mine": v["verdict"], "flag": v["flag"],
                               "unanimous": kk["unanimous"],
                               "error_type": kk["error_type"]}
                              for k, v, kk in disag]}

    write_json(OUT / "audit_summary.json", summary)
    if len(summary) > 1:
        print(f"\n{'=' * 74}\n  pooled across models")
        tot = sum(s["n"] for s in summary.values())
        bad = sum(len(s["disagreements"]) for s in summary.values())
        print(f"    {tot} adjudicated, {bad} disagreements ({bad / tot:.1%})")
    print(f"\nwrote {OUT}/audit_summary.json")


if __name__ == "__main__":
    main()
