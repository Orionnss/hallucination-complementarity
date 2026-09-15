"""Cluster SAPLMA's PLS-4 projection, and render it as an inspectable 3D scatter.

The supervised-reduction sweep showed PLS-4 lands within 0.0024 AUROC of PCA-128 on
Qwen3-14B: four supervised directions carry essentially all of the probe's signal. That
makes the PLS-4 space small enough to look at directly, and the question this answers is
whether the items group into anything interpretable inside it — dataset, difficulty,
error type — or whether it is one undifferentiated cloud with a label gradient across it.

Protocol, and the one thing that would invalidate it:

  PLS consumes the label. Fitted on all 8,000 rows it would arrange them by label by
  construction, and any cluster structure would be an artefact of that fit. So the
  projection is fitted on stage 3 fold 0's TRAINING rows only and then applied to every
  row, giving one coordinate system in which 1,600 of the points were never seen. Each
  point carries its split, and the held-out subset is the honest view -- if the structure
  is real it survives filtering to those.

  Clustering is unsupervised (k-means on the 4-D scores, k = 2 x dims), so it never sees
  the label at all. The 3-D scatter is PCA-3 of those same 4-D scores: a rotation for
  display only, with the variance it keeps printed so the reader knows what the view is
  dropping.

Clicking a point shows the whole tuple behind it -- question, the answer the generator
produced, the gold answers, what the judges ruled and what SAPLMA predicted -- because
the useful output here is the items themselves, not the geometry.

Usage: uv run python scripts/pls_clusters.py --run main --seed 0
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.io import read_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14,
         "llama3.2-3b-base": 14, "gemma3-12b-pt": 29}

#: Colour-blind safe qualitative set (Okabe-Ito), extended for k up to 12.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
           "#F0E442", "#8C6BB1", "#117733", "#882255", "#44AA99", "#999933"]


def load_saplma(cfg, ids, layer):
    from halluc.io import load_features
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        a = load_features(cfg.stage_dir("stage1_extract", ds), "saplma", sub)
        a = a[:, min(layer, a.shape[1] - 1), :].astype(np.float32)
        if out is None:
            out = np.zeros((len(ids), a.shape[1]), np.float32)
        out[[pos[i] for i in sub]] = a
        del a
    return out


def load_text(cfg) -> dict[str, dict]:
    """Question, generated answer, gold answers and judge verdict, per item."""
    meta: dict[str, dict] = {}
    for ds in cfg.datasets:
        for r in read_json(cfg.stage_dir("stage1_extract", ds) / "manifest.json")["items"]:
            meta[r["item_id"]] = {
                "question": r["question"],
                "answer": r["answer"],
                "gold": r["gold_answers"],
                "has_context": r["has_context"],
                "dataset": ds,
            }
        for e in read_json(cfg.stage_dir("stage2_judge", ds) / "labels.json")["labels"]:
            if e["item_id"] in meta:
                meta[e["item_id"]].update(judge=e["label"], n_agreeing=e["n_agreeing"],
                                          unanimous=bool(e["unanimous"]))
    return meta


def build(run: str, seed: int, dims: int, k: int | None):
    cfg = Config(); cfg.run_id = run
    sd = load_seed(cfg, seed)
    if sd is None:
        raise SystemExit(f"no stage-3 predictions for {run} seed {seed}")
    ids, y = list(sd["item_ids"]), sd["y"]
    ds_arr = sd["dataset"].astype(str)
    saplma_pred = sd["preds"]["saplma"]
    saplma_score = sd["scores"]["saplma"]

    X = load_saplma(cfg, ids, LAYER.get(run, 24))
    tr, te = _folds(sd, seed, cfg.n_folds)[0]
    print(f"  {run} seed {seed}: {len(y)} items, {X.shape[1]} dims, "
          f"PLS fitted on {len(tr)} training rows, {len(te)} held out")

    # PLS consumes the label, so it is fitted on the training rows only and then applied
    # to everything -- one basis, and a held-out subset that the basis never saw.
    scaler = StandardScaler().fit(X[tr])
    pls = PLSRegression(n_components=dims, scale=False).fit(scaler.transform(X[tr]), y[tr])
    Z = pls.transform(scaler.transform(X)).astype(np.float64)

    k = k or 2 * dims
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    labels = km.labels_

    pca = PCA(n_components=3, random_state=seed).fit(Z)
    coords = pca.transform(Z)
    kept = float(pca.explained_variance_ratio_.sum())
    print(f"  k-means k={k} on the {dims}-D scores; PCA-3 view keeps "
          f"{kept:.1%} of their variance ({', '.join(f'{v:.1%}' for v in pca.explained_variance_ratio_)})")

    text = load_text(cfg)
    held = np.zeros(len(y), bool); held[te] = True
    return dict(cfg=cfg, run=run, seed=seed, dims=dims, k=k, ids=ids, y=y, ds=ds_arr,
                pred=saplma_pred, score=saplma_score, Z=Z, labels=labels, coords=coords,
                kept=kept, evr=pca.explained_variance_ratio_, text=text, held=held)


def report(b) -> list[dict]:
    """Per-cluster composition. This is the experiment; the plot is how you read it."""
    y, lab, ds, pred = b["y"], b["labels"], b["ds"], b["pred"]
    rows = []
    print(f"\n  {'cluster':>8s}{'n':>7s}{'halluc':>9s}{'saplma acc':>12s}"
          f"{'flagged':>9s}   dominant dataset")
    for c in range(b["k"]):
        m = lab == c
        counts = {d: int((ds[m] == d).sum()) for d in sorted(set(ds))}
        top = max(counts, key=counts.get)
        row = dict(cluster=c, n=int(m.sum()), halluc=float(y[m].mean()),
                   saplma_acc=float((pred[m] == y[m]).mean()),
                   flagged=float((pred[m] == 1).mean()),
                   datasets=counts, top_dataset=top,
                   top_share=counts[top] / max(int(m.sum()), 1))
        rows.append(row)
        print(f"  {c:>8d}{row['n']:>7d}{row['halluc']:>9.3f}{row['saplma_acc']:>12.3f}"
              f"{row['flagged']:>9.3f}   {top} {row['top_share']:.0%}")
    print(f"  {'overall':>8s}{len(y):>7d}{y.mean():>9.3f}{(pred == y).mean():>12.3f}"
          f"{(pred == 1).mean():>9.3f}")

    # Does the partition track anything the study already names? ARI against the label is
    # near zero for a clustering that has merely found the label gradient's noise, and
    # high against the dataset if the clusters are really just "which corpus is this".
    print(f"\n  adjusted Rand index vs the hallucination label : {adjusted_rand_score(y, lab):+.4f}")
    print(f"  adjusted Rand index vs the source dataset       : "
          f"{adjusted_rand_score(ds, lab):+.4f}")
    spread = max(r['halluc'] for r in rows) - min(r['halluc'] for r in rows)
    print(f"  hallucination rate spread across clusters       : {spread:.3f} "
          f"(overall {y.mean():.3f})")
    return rows


def figure(b, rows):
    coords, lab = b["coords"], b["labels"]
    fig = go.Figure()
    for c in range(b["k"]):
        m = lab == c
        idx = np.where(m)[0]
        r = rows[c]
        fig.add_trace(go.Scatter3d(
            x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
            mode="markers", name=f"cluster {c} · n={r['n']} · halluc {r['halluc']:.0%}",
            marker=dict(size=2.6, color=PALETTE[c % len(PALETTE)], opacity=0.78,
                        line=dict(width=0)),
            customdata=idx.reshape(-1, 1),
            hovertemplate=("<b>cluster %{marker.color}</b><br>click for the full item"
                           "<extra></extra>"),
        ))
    ax = dict(showbackground=True, backgroundcolor="rgba(0,0,0,0.02)",
              gridcolor="rgba(0,0,0,0.12)", zerolinecolor="rgba(0,0,0,0.25)")
    fig.update_layout(
        scene=dict(xaxis=dict(title=f"PC1 ({b['evr'][0]:.0%})", **ax),
                   yaxis=dict(title=f"PC2 ({b['evr'][1]:.0%})", **ax),
                   zaxis=dict(title=f"PC3 ({b['evr'][2]:.0%})", **ax)),
        margin=dict(l=0, r=0, t=0, b=0), height=720,
        legend=dict(itemsizing="constant", y=0.99, x=0.01,
                    bgcolor="rgba(255,255,255,0.75)", font=dict(size=11)),
        paper_bgcolor="white",
    )
    return fig


def payload(b) -> list[dict]:
    """One record per point, read by the click handler."""
    out = []
    for i, iid in enumerate(b["ids"]):
        t = b["text"].get(iid, {})
        out.append({
            "id": iid,
            "ds": str(b["ds"][i]),
            "q": t.get("question", "(missing)"),
            "a": t.get("answer", "(missing)"),
            "g": t.get("gold", []),
            "judge": t.get("judge", "?"),
            "agree": t.get("n_agreeing", 0),
            "ctx": bool(t.get("has_context", False)),
            "saplma": int(b["pred"][i]),
            "score": round(float(b["score"][i]), 4),
            "truth": int(b["y"][i]),
            "cluster": int(b["labels"][i]),
            "held": bool(b["held"][i]),
        })
    return out


PAGE_CSS = """
:root{--paper:#F7F8FA;--surface:#FFF;--ink:#18212B;--soft:#4E5A66;--faint:#7B8792;
      --rule:#DBE1E7;--accent:#0E7C86;--crit:#A8322D;--ok:#1F7A4D;--warn:#B8730A}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
     font:15px/1.6 "IBM Plex Sans",system-ui,-apple-system,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:1.6rem 1.4rem 3rem}
h1{font-size:1.42rem;margin:0 0 .3rem;font-weight:600;letter-spacing:-.01em}
.sub{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.71rem;
     letter-spacing:.09em;text-transform:uppercase;color:var(--faint);margin:0 0 1rem}
.lede{max-width:56rem;color:var(--soft);font-size:.93rem;margin:0 0 1.3rem}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 25rem;gap:1rem;align-items:start}
@media(max-width:1100px){.grid{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:4px}
.plot{padding:.4rem}
.panel{padding:1rem 1.1rem;position:sticky;top:1rem;max-height:88vh;overflow-y:auto}
.panel h2{font-size:.72rem;font-family:"IBM Plex Mono",ui-monospace,monospace;
          letter-spacing:.1em;text-transform:uppercase;color:var(--faint);
          margin:0 0 .8rem;font-weight:500}
.empty{color:var(--faint);font-size:.88rem}
.fld{margin-bottom:.85rem}
.fld .k{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.63rem;
        letter-spacing:.09em;text-transform:uppercase;color:var(--faint);
        display:block;margin-bottom:.2rem}
.fld .v{font-size:.9rem;white-space:pre-wrap;overflow-wrap:anywhere}
ul.gold{margin:.1rem 0 0;padding-left:1.1rem;font-size:.88rem}
.pill{display:inline-block;font-family:"IBM Plex Mono",ui-monospace,monospace;
      font-size:.66rem;letter-spacing:.05em;padding:.14rem .45rem;border-radius:3px;
      margin-right:.35rem;border:1px solid transparent}
.p-h{background:#A8322D14;color:var(--crit);border-color:#A8322D44}
.p-o{background:#1F7A4D14;color:var(--ok);border-color:#1F7A4D44}
.p-n{background:#0E7C8614;color:var(--accent);border-color:#0E7C8644}
.p-w{background:#B8730A14;color:var(--warn);border-color:#B8730A44}
table{border-collapse:collapse;width:100%;font-size:.78rem;
      font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
th,td{padding:.34rem .5rem;text-align:right;border-bottom:1px solid var(--rule)}
th{font-size:.63rem;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);
   font-weight:500}
th:first-child,td:first-child{text-align:left}
.sw{display:inline-block;width:.62rem;height:.62rem;border-radius:2px;margin-right:.45rem}
.note{margin-top:1.1rem;padding:.85rem 1rem;border-left:3px solid var(--accent);
      background:#0E7C8611;font-size:.87rem;color:var(--soft)}
.note b{color:var(--ink)}
"""

PAGE_JS = """
const PTS = __PAYLOAD__;
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pill = (t, k) => `<span class="pill p-${k}">${esc(t)}</span>`;
function show(i){
  const p = PTS[i];
  const judge = p.truth === 1 ? pill('judges: HALLUCINATED','h') : pill('judges: OK','o');
  const sap = p.saplma === 1 ? pill('saplma: HALLUCINATED','h') : pill('saplma: OK','o');
  const verdict = p.saplma === p.truth ? pill('correct','o') : pill('SAPLMA WRONG','w');
  const gold = p.g.length ? `<ul class="gold">${p.g.map(g=>`<li>${esc(g)}</li>`).join('')}</ul>`
                          : '<div class="v">(none)</div>';
  document.getElementById('detail').innerHTML = `
    <div class="fld">${judge}${sap}${verdict}</div>
    <div class="fld"><span class="k">question</span><div class="v">${esc(p.q)}</div></div>
    <div class="fld"><span class="k">generated answer</span><div class="v">${esc(p.a)}</div></div>
    <div class="fld"><span class="k">expected answer${p.g.length>1?'s':''}</span>${gold}</div>
    <div class="fld"><span class="k">saplma score</span><div class="v">${p.score} → ${p.saplma===1?'HALLUCINATED':'OK'}</div></div>
    <div class="fld"><span class="k">judges</span><div class="v">${esc(p.judge)} (${p.agree}/3 agreeing)</div></div>
    <div class="fld"><span class="k">item</span><div class="v">${esc(p.id)} · ${esc(p.ds)}${p.ctx?' · passage supplied':''}
 · cluster ${p.cluster} · ${p.held?'held out':'used to fit PLS'}</div></div>`;
}
const gd = document.getElementById('__DIV__');
gd.on('plotly_click', e => { if (e.points.length) show(e.points[0].customdata[0]); });
"""


def page(b, rows, fig) -> str:
    div = "plsplot"
    plot = pio.to_html(fig, include_plotlyjs=True, full_html=False, div_id=div,
                       config={"displaylogo": False, "responsive": True})
    tbl = "".join(
        f"<tr><td><span class='sw' style='background:{PALETTE[r['cluster'] % len(PALETTE)]}'></span>"
        f"{r['cluster']}</td><td>{r['n']}</td><td>{r['halluc']:.3f}</td>"
        f"<td>{r['saplma_acc']:.3f}</td><td>{html.escape(r['top_dataset'])} "
        f"{r['top_share']:.0%}</td></tr>" for r in rows)
    y = b["y"]; spread = max(r['halluc'] for r in rows) - min(r['halluc'] for r in rows)
    ari_y = adjusted_rand_score(y, b["labels"])
    ari_d = adjusted_rand_score(b["ds"], b["labels"])
    js = PAGE_JS.replace("__PAYLOAD__", json.dumps(payload(b))).replace("__DIV__", div)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAPLMA PLS-4 clusters</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{PAGE_CSS}</style></head><body><div class="wrap">
<h1>SAPLMA's PLS-4 space, clustered</h1>
<p class="sub">{html.escape(b['run'])} · seed {b['seed']} · layer {LAYER.get(b['run'], 24)} ·
 PLS {b['dims']} dims · k-means k={b['k']} · PCA-3 view keeps {b['kept']:.1%}</p>
<p class="lede">PLS is fitted on stage&nbsp;3 fold&nbsp;0's training rows only and then applied to all
{len(y):,} items, so the {int((~b['held']).sum()):,} fitting rows and {int(b['held'].sum()):,} held-out rows
share one basis and the held-out points sit in a space that never saw their labels.
k-means runs on the {b['dims']}-D scores and never sees a label at all. The axes below are
PCA-3 of those same scores — a rotation for display, dropping {1 - b['kept']:.1%} of the variance.
<b>Click any point</b> to read the item behind it.</p>
<div class="grid">
  <div class="card plot">{plot}</div>
  <div>
    <div class="card panel">
      <h2>selected item</h2>
      <div id="detail"><p class="empty">Click a point in the scatter to see its question,
      the answer the generator produced, what SAPLMA called it, and the expected answers.</p></div>
    </div>
    <div class="card panel" style="margin-top:1rem;position:static">
      <h2>cluster composition</h2>
      <table><thead><tr><th>cluster</th><th>n</th><th>halluc</th><th>saplma acc</th>
      <th>dominant set</th></tr></thead><tbody>{tbl}</tbody></table>
      <div class="note">Hallucination rate ranges <b>{min(r['halluc'] for r in rows):.3f}</b> to
      <b>{max(r['halluc'] for r in rows):.3f}</b> across clusters against <b>{y.mean():.3f}</b> overall,
      a spread of {spread:.3f}. Adjusted Rand index is <b>{ari_y:+.3f}</b> against the
      hallucination label and <b>{ari_d:+.3f}</b> against the source dataset — the second is
      what says whether these clusters are really just "which corpus is this".</div>
    </div>
  </div>
</div></div><script>{js}</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dims", type=int, default=4, help="PLS components")
    ap.add_argument("--clusters", type=int, default=None, help="default 2 x dims")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b = build(args.run, args.seed, args.dims, args.clusters)
    rows = report(b)
    fig = figure(b, rows)
    out = Path(args.out or f"runs/{args.run}/stage5_posthoc/pls_clusters.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(b, rows, fig))
    js = out.with_suffix(".json")
    js.write_text(json.dumps({"run": args.run, "seed": args.seed, "dims": args.dims,
                              "k": b["k"], "pca3_variance_kept": b["kept"],
                              "clusters": rows,
                              "ari_label": adjusted_rand_score(b["y"], b["labels"]),
                              "ari_dataset": adjusted_rand_score(b["ds"], b["labels"])},
                             indent=1))
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB, plotly inlined)")
    print(f"wrote {js}")


if __name__ == "__main__":
    main()
