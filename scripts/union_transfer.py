"""What does each union keep, lose, fix and inherit from each individual method?

For every (method M, union U) pair this splits the items by whether M was right and
whether U was right:

  retained  M right, U right   - U kept what M got right
  lost      M right, U wrong   - U threw away a correct prediction M had
  fixed     M wrong, U right   - U repaired an error M made
  inherited M wrong, U wrong   - U copied M's mistake

Retention alone is not informative: a union that agreed with everything would retain
100% and fix nothing. The pair that matters is `lost` against `fixed` — whether
combining pays for the correct predictions it discards.

Counts are means per seed over the 8,000 evaluated items.

Usage: uv run python scripts/union_transfer.py [--runs main gemma3-4b llama3.2-3b]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluc.config import Config
from halluc.io import write_json
from halluc.pipeline.stage5_posthoc import load_seed

LABELS = {"main": "Qwen3-14B", "gemma3-4b": "gemma-3-4b", "llama3.2-3b": "Llama-3.2-3B"}
UNIONS = ["union_equal", "union_raw"]


def transfer_for_run(run_id: str, seeds: list[int]) -> dict:
    cfg = Config(); cfg.run_id = run_id
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        sd = load_seed(cfg, seed)
        if sd is None:
            continue
        y = sd["y"]
        correct = {m: sd["preds"][m] == y for m in sd["methods"]}
        base = [m for m in sd["methods"] if not m.startswith("union")]

        for u in UNIONS:
            if u not in correct:
                continue
            cu = correct[u]
            for m in base:
                cm = correct[m]
                key = f"{u}|{m}"
                a = acc[key]
                a["m_right"].append(int(cm.sum()))
                a["m_wrong"].append(int((~cm).sum()))
                a["retained"].append(int((cm & cu).sum()))
                a["lost"].append(int((cm & ~cu).sum()))
                a["fixed"].append(int((~cm & cu).sum()))
                a["inherited"].append(int((~cm & ~cu).sum()))
    return {
        k: {stat: round(float(np.mean(v)), 1) for stat, v in d.items()}
        for k, d in acc.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main", "gemma3-4b", "llama3.2-3b"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    results = {r: transfer_for_run(r, args.seeds) for r in args.runs}
    write_json(Path("runs/union_transfer.json"), results)

    for run in args.runs:
        print(f"\n=== {LABELS.get(run, run)} ===")
        for u in UNIONS:
            rows = {k.split("|")[1]: v for k, v in results[run].items() if k.startswith(u + "|")}
            if not rows:
                continue
            print(f"\n  {u}")
            print(f"    {'method':16s}{'M right':>9s}{'retained':>19s}{'lost':>16s}"
                  f"{'M wrong':>10s}{'fixed':>17s}{'inherited':>18s}{'net':>8s}")
            for m, v in sorted(rows.items(), key=lambda kv: -kv[1]["fixed"] + kv[1]["lost"]):
                net = v["fixed"] - v["lost"]
                print(f"    {m:16s}{v['m_right']:9.0f}"
                      f"{v['retained']:11.0f} ({v['retained'] / v['m_right']:5.1%})"
                      f"{v['lost']:8.0f} ({v['lost'] / v['m_right']:5.1%})"
                      f"{v['m_wrong']:10.0f}"
                      f"{v['fixed']:9.0f} ({v['fixed'] / v['m_wrong']:5.1%})"
                      f"{v['inherited']:10.0f} ({v['inherited'] / v['m_wrong']:5.1%})"
                      f"{net:+8.0f}")
    print("\nwrote runs/union_transfer.json")


if __name__ == "__main__":
    main()
