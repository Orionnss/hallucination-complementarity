"""Do dataset experts disagree with each other more than different methods do?

The mixture of dataset experts failed, but for an ambiguous reason: each expert saw a
quarter of the training data, so "specialisation does not help" and "specialisation costs
4x the data" both predict the observed loss. Agreement structure separates them.

If the four experts — identical features, identical estimator, disjoint training sets —
agree with each other about as much as five *different methods* do, then diversity of
training distribution buys no more independence than diversity of representation, and the
accuracy-independence tradeoff is not about the substrate at all. If they agree far more,
the substrate account survives and the experts are simply four noisy copies of one probe.

Three measurements, on out-of-fold predictions from the same folds as everything else:

  pairwise kappa among the four experts, against the same statistic among the five methods
  each expert against the pooled probe, which is the thing they would have to beat
  each expert on the pooled probe's errors, against the independence null used elsewhere
    (per-class accuracy over all data, reweighted to the class mix of the error slice)

Predictions are cached per seed so the analysis can be rerun without refitting.

Usage: uv run python scripts/moe_kappa.py --run main --seeds 0 1
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold

from halluc.config import Config
from halluc.io import write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_experts import LAYER, Probe, load_saplma  # noqa: E402

METHODS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline"]


def build(run, seed, cache_dir):
    """Out-of-fold predictions for the pooled probe and each dataset expert."""
    f = cache_dir / f"seed{seed}.npz"
    if f.exists():
        return dict(np.load(f, allow_pickle=True))

    cfg = Config(); cfg.run_id = run
    sd = load_seed(cfg, seed)
    ids, y = list(sd["item_ids"]), sd["y"]
    groups, ds_arr = sd["groups"], sd["dataset"].astype(str)
    strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
    X = load_saplma(cfg, ids, LAYER.get(run, 24))
    dsets = sorted(set(ds_arr))

    pred = {k: np.full(len(y), -1, dtype=int) for k in ["pooled"] + dsets}
    for tr, te in _folds(sd, seed, cfg.n_folds):
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
        i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
        a_tr, a_va = tr[i_tr], tr[i_va]

        p = Probe().fit(X, y, a_tr, a_va, seed)
        pred["pooled"][te] = (p.score(X[te]) >= p.thr).astype(int)
        for d in dsets:
            m_tr, m_va = a_tr[ds_arr[a_tr] == d], a_va[ds_arr[a_va] == d]
            if len(m_tr) < 50 or len(np.unique(y[m_tr])) < 2:
                continue
            e = Probe().fit(X, y, m_tr, m_va, seed)
            pred[d][te] = (e.score(X[te]) >= e.thr).astype(int)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(open(f, "wb"), y=y, dataset=ds_arr,
                        **{f"preds__{k}": v for k, v in pred.items()})
    del X
    return dict(np.load(f, allow_pickle=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    cache = Path(f"runs/{args.run}/stage5_posthoc/moe_oof")
    pair_e, pair_m, vs_pool, resc = (defaultdict(list) for _ in range(4))
    mcc_e = defaultdict(list)

    for seed in args.seeds:
        t0 = time.perf_counter()
        d = build(args.run, seed, cache)
        y = d["y"]
        dsets = sorted(set(d["dataset"].astype(str)))
        E = {k: d[f"preds__{k}"] for k in dsets}
        pooled = d["preds__pooled"]

        for a, b in itertools.combinations(dsets, 2):
            pair_e[f"{a} / {b}"].append(float(cohen_kappa_score(E[a], E[b])))
        for k in dsets:
            vs_pool[k].append(float(cohen_kappa_score(E[k], pooled)))
            mcc_e[k].append(float(matthews_corrcoef(y, E[k])))

        wrong = pooled != y
        for k in dsets:
            p = E[k]
            per_c = {c: (p[y == c] == c).mean() for c in (0, 1)}
            mix = {c: (y[wrong] == c).mean() for c in (0, 1)}
            null = sum(per_c[c] * mix[c] for c in (0, 1))
            resc[k].append((float((p[wrong] == y[wrong]).mean()), float(null),
                            float(cohen_kappa_score(y[wrong], p[wrong]))))

        # the five methods, on the same items, for the comparison that matters
        bo = sorted(glob.glob(f"runs/{args.run}/stage5_posthoc/block_oof/*.npz"))[seed]
        b = np.load(bo, allow_pickle=True)
        for x, z in itertools.combinations(METHODS, 2):
            pair_m[f"{x} / {z}"].append(
                float(cohen_kappa_score(b[f"preds__{x}"], b[f"preds__{z}"])))
        print(f"  seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    ke = [np.mean(v) for v in pair_e.values()]
    km = [np.mean(v) for v in pair_m.values()]
    print(f"\n=== {args.run}: agreement structure, {len(args.seeds)} seeds ===")
    print(f"\n  pairwise Cohen's kappa")
    print(f"    four DATASET EXPERTS (same features, disjoint training data)")
    for k, v in sorted(pair_e.items(), key=lambda x: -np.mean(x[1])):
        print(f"      {k:28s}{np.mean(v):8.3f}")
    print(f"      {'-> range':28s}{min(ke):8.3f} – {max(ke):.3f}   mean {np.mean(ke):.3f}")
    print(f"    five METHODS (different features, same training data)")
    for k, v in sorted(pair_m.items(), key=lambda x: -np.mean(x[1])):
        print(f"      {k:28s}{np.mean(v):8.3f}")
    print(f"      {'-> range':28s}{min(km):8.3f} – {max(km):.3f}   mean {np.mean(km):.3f}")

    print(f"\n  each expert vs the pooled probe")
    print(f"    {'expert':14s}{'MCC':>9s}{'kappa vs pooled':>18s}")
    for k in vs_pool:
        print(f"    {k:14s}{np.mean(mcc_e[k]):9.4f}{np.mean(vs_pool[k]):18.3f}")

    print(f"\n  on the POOLED probe's errors")
    print(f"    {'expert':14s}{'acc there':>11s}{'indep. null':>13s}{'ratio':>8s}{'kappa':>9s}")
    for k, v in resc.items():
        a = np.array(v).mean(0)
        print(f"    {k:14s}{a[0]:11.1%}{a[1]:13.1%}{a[0] / a[1]:8.2f}{a[2]:9.3f}")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/moe_kappa.json"),
               {"expert_pairs": {k: round(float(np.mean(v)), 4) for k, v in pair_e.items()},
                "method_pairs": {k: round(float(np.mean(v)), 4) for k, v in pair_m.items()},
                "vs_pooled": {k: round(float(np.mean(v)), 4) for k, v in vs_pool.items()},
                "on_pooled_errors": {k: [round(float(x), 4) for x in np.array(v).mean(0)]
                                     for k, v in resc.items()}})
    print(f"\nwrote runs/{args.run}/stage5_posthoc/moe_kappa.json")


if __name__ == "__main__":
    main()
