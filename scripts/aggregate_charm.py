"""Aggregate CHARM's per-seed metrics into one file, with a comparability check.

Stage 6 writes one metrics.json per seed. Every other method in this study has a single
aggregated file that the report's numbers can be traced to; CHARM did not, so its figures
were the only ones in the report without a source on disk. This closes that.

The check matters as much as the aggregation. CHARM was run separately from the rest of
the study, so before its numbers can sit in the same table as everyone else's it has to be
established that it saw the same items. Each seed's draw is verified against the stored
out-of-fold predictions on sample count and positive rate; a mismatch means the two are
scoring different data and the comparison is void.

Also recorded: that CHARM is scored under its own tuning procedure rather than the shared
config space used by fair_comparison.py, and on one generator only. Both are real limits
on how its row should be read, and belong next to the numbers rather than in prose
somewhere else.

Usage: uv run python scripts/aggregate_charm.py
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

from halluc.io import write_json

SCOPES = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main"])
    args = ap.parse_args()

    out = {}
    for run in args.runs:
        files = sorted(glob.glob(f"runs/{run}/stage6_charm/pooled/seed*/metrics.json"))
        if not files:
            print(f"{run}: no CHARM results"); continue
        bo = sorted(glob.glob(f"runs/{run}/stage5_posthoc/block_oof/*.npz"))

        per = defaultdict(lambda: defaultdict(list))
        seeds, checks = [], []
        for f in files:
            j = json.load(open(f))
            s = j["seed"]; seeds.append(s)
            pm = j["per_method"]["charm"]
            per["pooled"]["auroc"].append(pm["pooled_auroc"])
            per["pooled"]["mcc"].append(pm["pooled_mcc"])
            for ds, v in pm["per_dataset"].items():
                per[ds]["auroc"].append(v["auroc"])
                per[ds]["mcc"].append(v["mcc"])
                per[ds]["positive_rate"].append(v["positive_rate"])

            # Same draw as the rest of the study? Compared on the two quantities a
            # different draw could not coincidentally match.
            ok, detail = None, {}
            if s < len(bo):
                d = np.load(bo[s], allow_pickle=True)
                same_n = int(j["n_samples"]) == len(d["y"])
                same_p = abs(j["positive_rate"] - float(d["y"].mean())) < 2e-3
                ok = bool(same_n and same_p)
                detail = {"charm_n": int(j["n_samples"]), "oof_n": int(len(d["y"])),
                          "charm_positive_rate": round(float(j["positive_rate"]), 4),
                          "oof_positive_rate": round(float(d["y"].mean()), 4)}
            checks.append({"seed": s, "same_draw": ok,
                           "draw_fingerprint": j.get("draw_fingerprint"), **detail})

        agg = {sc: {m: {"mean": round(float(np.mean(v)), 4),
                        "std": round(float(np.std(v)), 4), "n_seeds": len(v)}
                    for m, v in d.items()} for sc, d in per.items()}
        all_ok = all(c["same_draw"] for c in checks if c["same_draw"] is not None)
        out[run] = {
            "method": "charm", "seeds": sorted(seeds), "metrics": agg,
            "draw_verification": {"all_seeds_match_block_oof": all_ok, "per_seed": checks},
            "caveats": {
                "tuning": "CHARM selects its own architecture and optimiser settings "
                          "(hidden size, depth, lr, scheduler, dropout, weight decay, "
                          "batch norm, residual), not the shared config space used by "
                          "scripts/fair_comparison.py. Its grid is if anything richer, so "
                          "the gap to SAPLMA is unlikely to be a tuning artefact, but the "
                          "search that produced it is not the one the other rows went "
                          "through.",
                "coverage": "One generator only; not run on gemma3-12b, gemma3-4b or "
                            "llama3.2-3b.",
                "metrics": "Stage 6 records AUROC and MCC only — no accuracy, balanced "
                           "accuracy or F1, so CHARM cannot fill those columns.",
            },
        }

        print(f"\n=== {run}: CHARM, {len(seeds)} seeds ===")
        print(f"  draws match the rest of the study: {'YES' if all_ok else 'NO'}")
        print(f"  {'metric':8s}" + "".join(f"{s[:11]:>13s}" for s in SCOPES))
        for m in ("mcc", "auroc"):
            print(f"  {m:8s}" + "".join(
                f"{agg[s][m]['mean']:13.4f}" if s in agg else f"{'-':>13s}"
                for s in SCOPES))
        print(f"  {'  std':8s}" + "".join(
            f"{agg[s]['mcc']['std']:13.4f}" if s in agg else f"{'-':>13s}"
            for s in SCOPES) + "   (mcc)")

    write_json(Path("runs/charm_metrics.json"), out)
    print("\nwrote runs/charm_metrics.json")


if __name__ == "__main__":
    main()
