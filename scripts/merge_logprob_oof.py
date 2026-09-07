"""Fit token-confidence out-of-fold predictions and fold them into block_oof.

CHARM shipped its own predictions.npz, so merging it was a reindex. Logprob has no such
file: the features were extracted after stage 1 and the probe has only ever been fitted
inline, inside whichever analysis needed it. That made it invisible to every analysis that
reads block_oof — voting, the oracle bound, risk-coverage, the confidence hand-off.

It is also the method most worth having there. Logprob is the only signal in this study
that escapes the accuracy-independence line, rescuing SAPLMA's errors at 0.66x independence
where its accuracy predicts 0.52x, because it reads the output distribution rather than the
residual stream. The mechanism therefore predicts something specific and testable: adding
it to a ballot should help, where adding CHARM - the most redundant method - made the
ballot worse.

The probe is fitted on the same folds and under the same search as everywhere else, so its
predictions are comparable with the stored ones rather than merely adjacent to them. Items
whose tokenizer round trip failed (~0.1%) are imputed with the column median so the item
set matches block_oof exactly; dropping them would leave a method scored on a different
population.

Usage: uv run python scripts/merge_logprob_oof.py --run main
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from halluc.config import Config
from halluc.pipeline.stage5_posthoc import load_seed
from kappa_logprob import logprob_matrix, oof_logprob  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    files = sorted(glob.glob(f"runs/{args.run}/stage5_posthoc/block_oof/*.npz"))
    merged = 0
    for si, bo in enumerate(files):
        b = dict(np.load(bo, allow_pickle=True))
        if "preds__logprob" in b:
            print(f"  seed {si}: already merged, skipping"); continue

        t0 = time.perf_counter()
        sd = load_seed(cfg, si)
        ids = list(sd["item_ids"])
        if list(b["item_ids"].astype(str)) != [str(i) for i in ids]:
            print(f"  seed {si}: ABORT — block_oof order differs from the seed draw")
            continue

        X = logprob_matrix(cfg, ids)
        score, pred = oof_logprob(cfg, sd, X, si)
        if not np.array_equal(sd["y"], b["y"]):
            print(f"  seed {si}: ABORT — labels differ between seed draw and block_oof")
            continue

        b["scores__logprob"], b["preds__logprob"] = score, pred
        print(f"  seed {si}: fitted {len(pred)} items in "
              f"{(time.perf_counter() - t0) / 60:.1f} min, "
              f"positive rate {pred.mean():.3f}")
        if not args.dry_run:
            tmp = bo + ".tmp"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **b)
            os.replace(tmp, bo)
        merged += 1
        del X

    print(f"\n{'would merge' if args.dry_run else 'merged'} {merged} seeds")
    if not args.dry_run and merged:
        d = np.load(files[0], allow_pickle=True)
        print("  methods now present:",
              [k.replace("preds__", "") for k in d.files if k.startswith("preds__")])


if __name__ == "__main__":
    main()
