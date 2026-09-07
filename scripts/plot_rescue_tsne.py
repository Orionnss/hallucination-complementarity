"""Interactive 3D t-SNE of SAPLMA's errors, one panel per rescuing method.

Each panel embeds the same population — the answers SAPLMA gets wrong — in a *different*
method's feature space, and highlights the answers that method recovers. If a method's
rescues occupied a region of its own representation, that panel would show a coloured
lobe. The statistical tests say they do not; this is the visual check, and 3-D structure
is only really judgeable when you can rotate it.

Positions are not comparable across panels: each is its own embedding. Only the
within-panel mixing carries meaning.

Colours are the dataviz reference palette's categorical slots in fixed order, used to
identify the method, with non-rescued answers in recessive grey. Identity is never
colour-alone — every panel is titled with its method and every trace is named.

Embeddings are cached to rescue_tsne_embeddings.npz; re-runs reuse them unless --refit.

Usage: uv run python scripts/plot_rescue_tsne.py --run main
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.io import load_features

# dataviz reference palette, categorical slots 1-4, fixed order.
SERIES = {"lapeigvals": "#2a78d6", "icr": "#eb6834",
          "attn_baseline": "#1baf7a", "svd_baseline": "#eda100"}
GREY = "#b8b7b2"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


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


def embed(cfg, ids, sel_idx, method, pca_dim, seed=0):
    """Standardise, reduce, and embed one subset in one method's feature space."""
    X = load_block(cfg, list(ids), method)[sel_idx]
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > pca_dim:
        X = PCA(n_components=min(pca_dim, len(sel_idx) - 1), random_state=seed).fit_transform(X)
    perp = min(30, max(5, (len(sel_idx) - 1) // 3))
    return TSNE(n_components=3, perplexity=perp, init="pca", random_state=seed).fit_transform(X)


def per_dataset_figure(cfg, args, d, y, ids, ds_arr, sel):
    """4x4 grid: one embedding per (method, dataset), fitted independently.

    Pooling four datasets with different base rates and sequence lengths could mask
    structure that exists inside a dataset. Fitting each cell separately removes that
    possibility: any clustering here is within-dataset by construction.
    """
    datasets = sorted(set(ds_arr[sel]))
    methods = list(SERIES)
    titles = []
    for m in methods:
        for x in datasets:
            cell = sel[ds_arr[sel] == x]
            r = (d[f"preds__{m}"] == y)[cell]
            titles.append(f"<b>{m}</b> · {x}<br>"
                          f"<span style='font-size:11px'>n={len(cell)}, rescues {r.mean():.0%}</span>")

    fig = make_subplots(rows=len(methods), cols=len(datasets),
                        specs=[[{"type": "scatter3d"}] * len(datasets)] * len(methods),
                        subplot_titles=titles,
                        horizontal_spacing=0.015, vertical_spacing=0.045)

    for i, method in enumerate(methods):
        colour = SERIES[method]
        for j, x in enumerate(datasets):
            cell = sel[ds_arr[sel] == x]
            emb = embed(cfg, ids, cell, method, args.pca_dim)
            rescued = (d[f"preds__{method}"] == y)[cell]
            print(f"  {method:14s} {x:10s} n={len(cell):4d} rescued={rescued.sum():4d}", flush=True)
            for mask, name, cl, size, op in (
                (~rescued, "not rescued", GREY, 2.8, 0.5),
                (rescued, f"{method} correct", colour, 3.6, 0.9),
            ):
                fig.add_trace(go.Scatter3d(
                    x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2], mode="markers",
                    marker=dict(size=size, color=cl, opacity=op),
                    name=name, legendgroup=method, showlegend=(j == 0),
                    legendgrouptitle_text=method if (j == 0 and not mask[0]) else None,
                    text=ids[cell][mask], hovertemplate="%{text}<extra></extra>",
                ), row=i + 1, col=j + 1)

    axis = dict(showticklabels=False, showbackground=False, title="",
                gridcolor="#e6e5e1", zerolinecolor="#e6e5e1")
    n_scenes = len(methods) * len(datasets)
    fig.update_layout(
        title=dict(text=f"<b>Rescues of SAPLMA's errors, per dataset — {args.run}</b>"
                        f"<br><span style='font-size:13px;color:{INK_2}'>"
                        "a separate scaler, PCA and t-SNE per cell: rows are methods "
                        "(in their own feature space), columns are datasets. "
                        "Positions are comparable within a cell only.</span>",
                   x=0.5, xanchor="center", font=dict(size=19, color=INK)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK_2, size=11),
        height=1750, width=1650,
        legend=dict(groupclick="togglegroup", itemsizing="constant",
                    bgcolor="rgba(252,252,251,0.85)", bordercolor="#e6e5e1", borderwidth=1),
        margin=dict(l=10, r=10, t=115, b=10),
        **{f"scene{n or ''}": dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube")
           for n in [None] + list(range(2, n_scenes + 1))},
    )
    for a in fig.layout.annotations:
        a.font.size, a.font.color = 11, INK

    out = args.out or f"runs/{args.run}/stage5_posthoc/rescue_tsne3d_per_dataset.html"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="inline", full_html=True)
    print(f"wrote {out}  ({Path(out).stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seed-file", type=int, default=0)
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--refit", action="store_true", help="recompute t-SNE, ignoring cache")
    ap.add_argument("--per-dataset", action="store_true",
                    help="fit a separate scaler+PCA+t-SNE per (method, dataset) cell, so "
                         "pooling across datasets cannot mask within-dataset structure")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    f = sorted(glob.glob(f"runs/{args.run}/stage5_posthoc/block_oof/*.npz"))[args.seed_file]
    d = np.load(f, allow_pickle=True)
    y, ids = d["y"], d["item_ids"].astype(str)
    ds_arr = d["dataset"].astype(str)
    sel = np.flatnonzero(d["preds__saplma"] != y)
    print(f"{args.run}: SAPLMA wrong on {len(sel)} answers", flush=True)

    if args.per_dataset:
        return per_dataset_figure(cfg, args, d, y, ids, ds_arr, sel)

    cache = Path(f"runs/{args.run}/stage5_posthoc/rescue_tsne_embeddings.npz")
    embeds = {}
    if cache.exists() and not args.refit:
        z = np.load(cache)
        if len(z["sel"]) == len(sel):
            embeds = {m: z[m] for m in SERIES if m in z.files}
            print(f"  reusing cached embeddings for {sorted(embeds)}", flush=True)

    for method in SERIES:
        if method in embeds:
            continue
        X = load_block(cfg, list(ids), method)[sel]
        X = StandardScaler().fit_transform(X)
        if X.shape[1] > args.pca_dim:
            X = PCA(n_components=args.pca_dim, random_state=0).fit_transform(X)
        print(f"  t-SNE for {method} ({X.shape[1]} dims) ...", flush=True)
        embeds[method] = TSNE(n_components=3, perplexity=30, init="pca",
                              random_state=0).fit_transform(X)
    np.savez_compressed(cache, sel=sel, **embeds)

    titles = []
    for m in SERIES:
        r = (d[f"preds__{m}"] == y)[sel]
        titles.append(f"<b>{m}</b> — rescues {r.mean():.1%} of SAPLMA's errors")
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scatter3d"}] * 2] * 2,
                        subplot_titles=titles, horizontal_spacing=0.03,
                        vertical_spacing=0.08)

    for i, (method, colour) in enumerate(SERIES.items()):
        row, col = i // 2 + 1, i % 2 + 1
        emb = embeds[method]
        rescued = (d[f"preds__{method}"] == y)[sel]
        hover = np.array([f"{a}<br>{b}<br>label={'HALLUC' if c else 'NOT'}"
                          for a, b, c in zip(ids[sel], ds_arr[sel], y[sel])])
        for mask, name, cl, size, opacity in (
            (~rescued, f"not rescued ({(~rescued).sum()})", GREY, 2.6, 0.55),
            (rescued, f"{method} correct ({rescued.sum()})", colour, 3.4, 0.9),
        ):
            fig.add_trace(go.Scatter3d(
                x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2], mode="markers",
                marker=dict(size=size, color=cl, opacity=opacity),
                name=name, legendgroup=method, showlegend=True,
                legendgrouptitle_text=method if mask is not rescued else None,
                text=hover[mask], hovertemplate="%{text}<extra></extra>",
            ), row=row, col=col)

    axis = dict(showticklabels=False, showbackground=False, title="",
                gridcolor="#e6e5e1", zerolinecolor="#e6e5e1")
    fig.update_layout(
        title=dict(text=f"<b>Where each method rescues SAPLMA's errors — {args.run}</b>"
                        f"<br><span style='font-size:13px;color:{INK_2}'>"
                        f"3-D t-SNE of the same {len(sel)} answers SAPLMA gets wrong, "
                        "embedded separately in each method's own feature space "
                        "(positions are not comparable between panels)</span>",
                   x=0.5, xanchor="center", font=dict(size=19, color=INK)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK_2, size=12), height=1000, width=1500,
        legend=dict(groupclick="togglegroup", itemsizing="constant",
                    bgcolor="rgba(252,252,251,0.85)", bordercolor="#e6e5e1", borderwidth=1),
        margin=dict(l=10, r=10, t=110, b=10),
        **{f"scene{n or ''}": dict(xaxis=axis, yaxis=axis, zaxis=axis,
                                   aspectmode="cube") for n in (None, 2, 3, 4)},
    )
    for a in fig.layout.annotations:
        a.font.size, a.font.color = 13, INK

    out = args.out or f"runs/{args.run}/stage5_posthoc/rescue_tsne3d.html"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # Inline the library: the viewer must work offline and under a strict CSP.
    fig.write_html(out, include_plotlyjs="inline", full_html=True)
    print(f"wrote {out}  ({Path(out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
