"""Do a method's rescues of SAPLMA's errors occupy dense regions of its feature space?

The separability probes (linear, boosted, at several reduction widths) all landed near
AUROC 0.55, but a classifier can only find structure it can carve with a boundary.
Density clustering asks a weaker and more forgiving question: do the rescued answers
concentrate anywhere at all?

Pipeline: standardise -> PCA 128 -> UMAP -> HDBSCAN, over all 8,000 answers in the
rescuing method's own feature space. Then, per cluster, compare the rescue rate against
the global base rate. UMAP precedes HDBSCAN because density estimation degrades badly in
128 dimensions; PCA alone leaves nearly everything labelled noise.

Two null models make "some cluster looks enriched" interpretable:
  * label permutation - shuffle the rescue flag, recluster nothing, recompute the same
    enrichment statistic. Any real structure must beat this.
  * the statistic itself is the rescue-rate spread across clusters, weighted by size.

Rescue is defined only over SAPLMA's errors; answers SAPLMA gets right are excluded from
the rate (they are still clustered, since the geometry should not depend on the label).

Usage: uv run python scripts/rescue_clusters.py --run main --method icr
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import umap
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.io import load_features, write_json

BLOCKS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline"]


def load_block(cfg, item_ids, name):
    pos = {i: k for k, i in enumerate(item_ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in item_ids if i.startswith(ds + ":")]
        if not sub:
            continue
        arr = load_features(cfg.stage_dir("stage1_extract", ds), name, sub)
        arr = arr.reshape(len(sub), -1).astype(np.float32)
        if out is None:
            out = np.zeros((len(item_ids), arr.shape[1]), np.float32)
        for k, i in enumerate(sub):
            out[pos[i]] = arr[k]
    return out


def enrichment(labels, rescued, eligible):
    """Size-weighted spread of per-cluster rescue rate around the global rate."""
    base = rescued[eligible].mean()
    stat, rows = 0.0, []
    for c in sorted(set(labels)):
        m = (labels == c) & eligible
        if m.sum() < 20:
            continue
        rate = rescued[m].mean()
        stat += m.sum() * abs(rate - base)
        rows.append({"cluster": int(c), "n_eligible": int(m.sum()),
                     "n_total": int((labels == c).sum()),
                     "rescue_rate": round(float(rate), 4),
                     "lift": round(float(rate / base), 3)})
    return stat / max(eligible.sum(), 1), base, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--method", default="icr")
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--umap-dim", type=int, default=10)
    ap.add_argument("--min-cluster-size", type=int, default=50)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed-file", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    f = sorted(glob.glob(f"runs/{args.run}/stage5_posthoc/block_oof/*.npz"))[args.seed_file]
    d = np.load(f, allow_pickle=True)
    y, ids = d["y"], d["item_ids"].astype(str)
    sap_wrong = d["preds__saplma"] != y
    rescued = (d[f"preds__{args.method}"] == y) & sap_wrong

    X = load_block(cfg, list(ids), args.method)
    X = StandardScaler().fit_transform(X)
    raw_dim = X.shape[1]
    if raw_dim > args.pca_dim:
        X = PCA(n_components=args.pca_dim, random_state=0).fit_transform(X)
    emb = umap.UMAP(n_components=args.umap_dim, n_neighbors=30, min_dist=0.0,
                    random_state=0).fit_transform(X)
    labels = HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(emb)

    noise = float((labels == -1).mean())
    n_clusters = len({c for c in labels if c != -1})
    stat, base, rows = enrichment(labels, rescued, sap_wrong)

    # Null: permute the rescue flag among SAPLMA's errors, keeping the clustering fixed.
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(sap_wrong)
    null = []
    for _ in range(args.n_perm):
        r = np.zeros(len(y), bool)
        r[rng.permutation(idx)[: rescued.sum()]] = True
        null.append(enrichment(labels, r, sap_wrong)[0])
    null = np.array(null)
    p = float((null >= stat).mean())

    print(f"=== {args.run} / {args.method}: HDBSCAN over all {len(y)} answers ===")
    print(f"  feature dims {raw_dim} -> PCA {min(raw_dim, args.pca_dim)} -> UMAP {args.umap_dim}")
    print(f"  clusters found: {n_clusters}   noise: {noise:.1%}")
    print(f"  global rescue rate among SAPLMA's {int(sap_wrong.sum())} errors: {base:.3f}")
    print(f"\n  {'cluster':>8s}{'n_total':>9s}{'n_err':>7s}{'rescue rate':>13s}{'lift':>7s}")
    for r in sorted(rows, key=lambda r: -r["lift"]):
        print(f"  {r['cluster']:8d}{r['n_total']:9d}{r['n_eligible']:7d}"
              f"{r['rescue_rate']:13.3f}{r['lift']:7.2f}")
    print(f"\n  enrichment statistic : {stat:.4f}")
    print(f"  permutation null     : {null.mean():.4f}  95% [{np.percentile(null,2.5):.4f}, "
          f"{np.percentile(null,97.5):.4f}]")
    print(f"  p-value              : {p:.3f}   "
          f"({'structure beyond chance' if p < 0.05 else 'NOT distinguishable from chance'})")

    write_json(Path(f"runs/{args.run}/stage5_posthoc/rescue_clusters_{args.method}.json"),
               {"run": args.run, "method": args.method, "raw_dim": raw_dim,
                "pca_dim": args.pca_dim, "umap_dim": args.umap_dim,
                "min_cluster_size": args.min_cluster_size,
                "n_clusters": n_clusters, "noise_fraction": noise,
                "base_rescue_rate": float(base), "clusters": rows,
                "enrichment": float(stat), "null_mean": float(null.mean()),
                "p_value": p})


if __name__ == "__main__":
    main()
