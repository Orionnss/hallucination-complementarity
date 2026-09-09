"""One table over every generator and every method run so far.

Results accumulated across instruct and base generators, several stages and a few
side-runs, each writing its own file. This pulls them into one place so a number can be
compared with the one beside it rather than with a number from a different protocol.

Everything here comes from stage 3's per-seed metrics, which all six generators share, so
the rows are directly comparable. Three additions are marked because they are not:

  saplma (PCA+LR)  the same features under the union's classifier — measured only on the
                   four instruct generators
  charm            its own stage, its own tuning; ran on two generators, failed on a third
  logprob          token confidence, extracted after stage 1; one generator

Reported per generator alongside its hallucination rate and INVALID rate, because a method
comparison across generators means little without both: detectability tracks class balance,
and the INVALID rate decides how filtered the scored pool is.

Usage: uv run python scripts/master_table.py
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

#: (run-id, display name, short column label, instruct/base). Short labels are explicit
#: rather than truncated: two generators share the "gemma" prefix and collapsed into one
#: another in an earlier version of this table.
RUNS = [
    ("main", "Qwen3-14B", "Qwen14/it", "it"),
    ("gemma3-12b", "gemma-3-12b", "G12b/it", "it"),
    ("gemma3-4b", "gemma-3-4b", "G4b/it", "it"),
    ("llama3.2-3b", "Llama-3.2-3B", "L3b/it", "it"),
    ("gemma3-12b-pt", "gemma-3-12b", "G12b/BASE", "BASE"),
    ("llama3.2-3b-base", "Llama-3.2-3B", "L3b/BASE", "BASE"),
]

#: CHARM was only ever launched on three runs; "not run" and "failed" are different
#: facts and the table must not present the first as the second.
CHARM_ATTEMPTED = {"main", "llama3.2-3b-base", "gemma3-12b-pt"}
METHODS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline",
           "union_raw", "union_equal"]
DS = ["triviaqa", "nq_open", "squad_v2", "coqa"]


def stage3(run):
    """Per-method pooled and per-dataset metrics, averaged over seeds."""
    agg = defaultdict(lambda: defaultdict(list))
    pr, n = [], 0
    for f in sorted(glob.glob(f"runs/{run}/stage3_train/pooled/seed*/metrics.json")):
        j = json.load(open(f)); pr.append(j["positive_rate"]); n += 1
        for m, v in j["per_method"].items():
            agg[(m, "pooled")]["mcc"].append(v["pooled_mcc"])
            agg[(m, "pooled")]["auroc"].append(v["pooled_auroc"])
            for ds, dv in v.get("per_dataset", {}).items():
                agg[(m, ds)]["mcc"].append(dv["mcc"])
                agg[(m, ds)]["auroc"].append(dv["auroc"])
    return agg, (float(np.mean(pr)) if pr else float("nan")), n


def charm(run):
    out = defaultdict(lambda: defaultdict(list))
    fs = sorted(glob.glob(f"runs/{run}/stage6_charm/pooled/seed*/metrics.json"))
    for f in fs:
        pm = json.load(open(f))["per_method"]["charm"]
        out[("charm", "pooled")]["mcc"].append(pm["pooled_mcc"])
        out[("charm", "pooled")]["auroc"].append(pm["pooled_auroc"])
        for ds, dv in pm["per_dataset"].items():
            out[("charm", ds)]["mcc"].append(dv["mcc"])
            out[("charm", ds)]["auroc"].append(dv["auroc"])
    return out, len(fs)


def judge_stats(run):
    tot = inv = h = sc = 0
    for ds in DS:
        try:
            j = json.load(open(f"runs/{run}/stage2_judge/{ds}/labels.json"))
        except FileNotFoundError:
            continue
        tot += j["n_items"]; inv += j["n_dropped_invalid"]; sc += j["n_scored"]
        h += j["n_scored"] * j["hallucination_rate_scored"]
    return tot, inv, sc, (h / sc if sc else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="both", choices=("mcc", "auroc", "both"))
    args = ap.parse_args()

    data, meta = {}, {}
    pcalr = json.load(open("runs/saplma_pcalr_metrics.json")) \
        if Path("runs/saplma_pcalr_metrics.json").exists() else {}
    for run, name, short, kind in RUNS:
        a, pr, n = stage3(run)
        c, nc = charm(run)
        a.update(c)
        if run in pcalr:
            for scope, v in pcalr[run].items():
                a[("saplma (PCA+LR)", scope)] = {k: [v[k]["mean"]] for k in ("mcc", "auroc")
                                                 if k in v}
        data[run] = a
        tot, inv, sc, hr = judge_stats(run)
        meta[run] = dict(name=name, short=short, kind=kind, seeds=n, charm_seeds=nc,
                         answers=tot, invalid=inv, scored=sc, halluc=hr)

    order = ["saplma (PCA+LR)"] + METHODS + ["charm"]
    metrics = ["mcc", "auroc"] if args.metric == "both" else [args.metric]

    print(f"\n{'=' * 118}\n  GENERATORS\n{'=' * 118}")
    print(f"  {'generator':22s}{'seeds':>7s}{'answers':>9s}{'INVALID':>9s}"
          f"{'scored':>9s}{'halluc rate':>13s}{'CHARM':>8s}")
    for run, _, _, _ in RUNS:
        m = meta[run]
        cs = (f"{m['charm_seeds']} sd" if m["charm_seeds"]
              else ("FAILED" if run in CHARM_ATTEMPTED else "not run"))
        print(f"  {m['name'] + ' [' + m['kind'] + ']':22s}{m['seeds']:>7d}"
              f"{m['answers']:>9d}{m['invalid']:>9d}{m['scored']:>9d}"
              f"{m['halluc']:>12.1%}{cs:>9s}")

    for metric in metrics:
        print(f"\n{'=' * 118}\n  POOLED {metric.upper()}\n{'=' * 118}")
        print(f"  {'method':20s}" + "".join(f"{meta[r]['short']:>14s}" for r, _, _, _ in RUNS))
        for m in order:
            row = ""
            for run, _, _, _ in RUNS:
                v = data[run].get((m, "pooled"))
                row += f"{np.mean(v[metric]):14.4f}" if v and v.get(metric) else f"{'—':>14s}"
            print(f"  {m:20s}{row}")

    print(f"\n{'=' * 118}\n  PER DATASET (MCC)\n{'=' * 118}")
    for run, name, _, kind in RUNS:
        print(f"\n  {name} [{kind}]")
        print(f"    {'method':20s}" + "".join(f"{d[:11]:>13s}" for d in DS))
        for m in order:
            if not any((m, d) in data[run] for d in DS):
                continue
            print(f"    {m:20s}" + "".join(
                f"{np.mean(data[run][(m, d)]['mcc']):13.4f}" if (m, d) in data[run]
                else f"{'—':>13s}" for d in DS))

    Path("runs").mkdir(exist_ok=True)
    json.dump({"meta": meta,
               "results": {r: {f"{m}|{s}": {k: round(float(np.mean(v)), 4)
                                            for k, v in d.items() if v}
                               for (m, s), d in data[r].items()} for r, _, _, _ in RUNS}},
              open("runs/master_table.json", "w"), indent=1)
    print("\nwrote runs/master_table.json")


if __name__ == "__main__":
    main()
