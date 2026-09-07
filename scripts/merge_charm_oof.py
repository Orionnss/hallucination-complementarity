"""Fold CHARM's out-of-fold predictions into block_oof so it joins the analyses.

CHARM ran as a separate stage and writes its own predictions.npz per seed. Every
complementarity result in this study — Cohen's kappa against the independence null,
voting, risk-coverage, the oracle bound, the confidence hand-off — reads `preds__<method>`
and `scores__<method>` from block_oof, so CHARM was scored in the detector tables but
absent from all of them. Since it is the strongest published method, that is the one place
its absence actually matters.

The two files share a schema and a draw, so merging is a reindex rather than a join: item
order is not assumed to match, and the labels are re-checked per item after alignment. A
label disagreement would mean the two are describing different data and the merge aborts
rather than silently producing a method that appears to disagree with the reference for the
wrong reason.

Writes to block_oof in place (atomically, via a temp file), preserving every existing key,
because rewriting the analyses to special-case one method would leave the same gap open for
the next one. `--dry-run` reports what would change without touching anything.

Usage: uv run python scripts/merge_charm_oof.py --run main
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bo_files = sorted(glob.glob(f"runs/{args.run}/stage5_posthoc/block_oof/*.npz"))
    merged = 0
    for si, bo in enumerate(bo_files):
        cp = Path(f"runs/{args.run}/stage6_charm/pooled/seed{si}/predictions.npz")
        if not cp.exists():
            print(f"  seed {si}: no CHARM predictions, skipping"); continue

        b = dict(np.load(bo, allow_pickle=True))
        c = np.load(cp, allow_pickle=True)
        b_ids = b["item_ids"].astype(str)
        c_ids = c["item_ids"].astype(str)

        if "preds__charm" in b:
            print(f"  seed {si}: already merged, skipping"); continue

        pos = {i: k for k, i in enumerate(c_ids)}
        missing = [i for i in b_ids if i not in pos]
        if missing:
            print(f"  seed {si}: ABORT — {len(missing)} block_oof items absent from CHARM "
                  f"(e.g. {missing[:3]})")
            continue
        idx = np.array([pos[i] for i in b_ids])

        # Alignment is only trustworthy if the labels agree once reindexed.
        if not np.array_equal(c["y"][idx], b["y"]):
            n = int((c["y"][idx] != b["y"]).sum())
            print(f"  seed {si}: ABORT — labels differ on {n} items after alignment")
            continue

        same_order = bool(np.array_equal(c_ids, b_ids))
        b["preds__charm"] = c["preds__charm"][idx]
        b["scores__charm"] = c["scores__charm"][idx]
        print(f"  seed {si}: merged {len(idx)} items "
              f"(order {'identical' if same_order else 'REINDEXED'}), "
              f"charm positive rate {b['preds__charm'].mean():.3f}")

        if not args.dry_run:
            tmp = bo + ".tmp"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **b)
            os.replace(tmp, bo)
        merged += 1

    print(f"\n{'would merge' if args.dry_run else 'merged'} {merged} seeds")
    if not args.dry_run and merged:
        d = np.load(bo_files[0], allow_pickle=True)
        print("  keys now present:", [k for k in d.files if k.startswith("preds__")])


if __name__ == "__main__":
    main()
