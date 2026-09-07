"""Does SAPLMA over k sampled generations beat SAPLMA over the greedy one?

The k sampled hidden states are read as a sorted vector rather than collapsed into a
consistency score, so the probe sees the shape of the scatter and not only its centre.
Everything else is held to the study's protocol: same folds, PCA-128 per block, logistic
regression with C and threshold tuned on an inner split.

Variants, each adding one thing to the one above:

  greedy          the established baseline: the greedy answer's state alone
  sampled         the k sampled states only, each block projected separately. Tests
                  whether sampled states carry the signal at all, without the greedy one
  greedy+sampled  both. The question the experiment exists to answer
  greedy+spread   greedy plus a compact summary of the scatter — distance of each sample
                  from greedy, pairwise sample distances, and their spread. This is the
                  representation-space analogue of a consistency score, and if it matches
                  greedy+sampled then the k-vector's extra width buys nothing over a
                  handful of summary statistics

Items whose k samples were all identical contribute a degenerate (zero-variance) scatter;
their share is reported per slice, because it caps how much the sampled blocks can
possibly add on that slice.

Evaluated only on items that have sampled features, so the comparison is like-for-like;
pass --datasets to restrict further.

Usage: uv run python scripts/eval_sampled_saplma.py --run main
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}
C_GRID = (0.003, 0.03, 0.3, 3.0)
DIM = 128


def load_blocks(cfg, run, tag, datasets):
    """Greedy state, sampled states and sample logprobs, for items that have all three."""
    layer = LAYER.get(run, 24)
    G, S, L, ids, ds = [], [], [], [], []
    for d in datasets:
        base = cfg.stage_dir("stage1_extract", d)
        p = base / tag
        if not (p / "checkpoint.jsonl").exists():
            continue
        have = []
        for line in (p / "checkpoint.jsonl").open():
            r = json.loads(line)
            if "shard" in r:
                have.append(r.get("item_id") or r.get("id"))
        if not have:
            continue
        g = load_features(base, "saplma", have)[:, min(layer, 1000), :]
        s = load_features(p, "sampled_saplma", have)
        lp = load_features(p, "sample_logp", have)
        G.append(g.astype(np.float32)); S.append(s.astype(np.float32))
        L.append(lp.astype(np.float32)); ids += have; ds += [d] * len(have)
    return (np.vstack(G), np.concatenate(S, 0), np.vstack(L),
            np.array(ids), np.array(ds))


def scatter_feats(G, S):
    """Compact description of how the k sampled states sit relative to each other and to
    the greedy state — the representation-space analogue of a consistency score."""
    k = S.shape[1]
    gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    sn = S / (np.linalg.norm(S, axis=2, keepdims=True) + 1e-8)
    cos_g = np.einsum("nd,nkd->nk", gn, sn)                     # sample vs greedy
    pair = np.stack([np.einsum("nd,nd->n", sn[:, i], sn[:, j])
                     for i in range(k) for j in range(i + 1, k)], axis=1)
    d_g = np.linalg.norm(S - G[:, None, :], axis=2)
    return np.hstack([np.sort(cos_g, axis=1), np.sort(pair, axis=1),
                      np.sort(d_g, axis=1),
                      cos_g.mean(1, keepdims=True), cos_g.std(1, keepdims=True),
                      pair.mean(1, keepdims=True), pair.std(1, keepdims=True),
                      d_g.mean(1, keepdims=True), d_g.std(1, keepdims=True)])


def reduce_fit(block, tr, te, seed, dim=DIM):
    s = StandardScaler().fit(block[tr])
    a, b = s.transform(block[tr]), s.transform(block[te])
    k = min(dim, a.shape[1], len(tr) - 1)
    if k < a.shape[1]:
        p = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(a)
        a, b = p.transform(a), p.transform(b)
    return a, b


def fit_eval(A, B, y, tr, i_tr, i_va, seed):
    best_c, best_a = C_GRID[0], -1.0
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(
            A[i_tr], y[tr][i_tr])
        a = roc_auc_score(y[tr][i_va], m.predict_proba(A[i_va])[:, 1])
        if a > best_a:
            best_a, best_c = float(a), C
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(
        A[i_tr], y[tr][i_tr])
    thr, _ = best_threshold(y[tr][i_va], m.predict_proba(A[i_va])[:, 1])
    m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(A, y[tr])
    s = m.predict_proba(B)[:, 1]
    return s, (s >= thr).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--tag", default="sampled_k3_t0.5")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    dsets = args.datasets or cfg.datasets
    G, S, LP, ids, ds_all = load_blocks(cfg, args.run, args.tag, dsets)
    pos = {i: k for k, i in enumerate(ids)}
    print(f"{args.run}: {len(ids)} items with k={S.shape[1]} sampled states", flush=True)
    ident = np.mean(S.std(axis=1).max(axis=1) < 1e-6)
    print(f"  all-{S.shape[1]}-states-identical: {ident:.1%}", flush=True)

    variants = ["greedy", "sampled", "greedy+sampled", "greedy+spread",
                "sampled_joint", "greedy+sampled_joint", "sampled_mean",
                "greedy+mean", "greedy+diff"]
    acc = defaultdict(lambda: defaultdict(list))
    for seed in args.seeds:
        sd = load_seed(cfg, seed)
        sel = [k for k, i in enumerate(sd["item_ids"]) if i in pos]
        rows = np.array([pos[sd["item_ids"][k]] for k in sel])
        y = sd["y"][sel]
        groups, ds_arr = sd["groups"][sel], sd["dataset"][sel].astype(str)
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
        Gs, Ss = G[rows], S[rows]
        SP = scatter_feats(Gs, Ss)

        sc = {v: np.full(len(y), np.nan) for v in variants}
        pr = {v: np.full(len(y), -1, dtype=int) for v in variants}
        outer = StratifiedGroupKFold(cfg.n_folds, shuffle=True, random_state=seed)
        for tr, te in outer.split(np.zeros(len(y)), strat, groups):
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
            g_tr, g_te = reduce_fit(Gs, tr, te, seed)
            samp = [reduce_fit(Ss[:, j], tr, te, seed) for j in range(Ss.shape[1])]
            sp_tr, sp_te = reduce_fit(SP, tr, te, seed, dim=64)
            # One PCA over the concatenated k states, so the sampled block is 128 dims
            # like greedy rather than k*128 — the earlier loss may have been width, not
            # content. Also the elementwise mean (pure noise reduction) and the
            # deviations from greedy (the scatter alone, at full dimensionality rather
            # than in 20 summary statistics).
            cat = Ss.reshape(len(y), -1)
            j_tr, j_te = reduce_fit(cat, tr, te, seed)
            m_tr, m_te = reduce_fit(Ss.mean(axis=1), tr, te, seed)
            d_tr, d_te = reduce_fit((Ss - Gs[:, None, :]).reshape(len(y), -1), tr, te, seed)
            blocks = {
                "greedy": (g_tr, g_te),
                "sampled": (np.hstack([a for a, _ in samp]),
                            np.hstack([b for _, b in samp])),
                "greedy+sampled": (np.hstack([g_tr] + [a for a, _ in samp]),
                                   np.hstack([g_te] + [b for _, b in samp])),
                "greedy+spread": (np.hstack([g_tr, sp_tr]), np.hstack([g_te, sp_te])),
                "sampled_joint": (j_tr, j_te),
                "greedy+sampled_joint": (np.hstack([g_tr, j_tr]), np.hstack([g_te, j_te])),
                "sampled_mean": (m_tr, m_te),
                "greedy+mean": (np.hstack([g_tr, m_tr]), np.hstack([g_te, m_te])),
                "greedy+diff": (np.hstack([g_tr, d_tr]), np.hstack([g_te, d_te])),
            }
            for v, (A, B) in blocks.items():
                s_, p_ = fit_eval(A, B, y, tr, i_tr, i_va, seed)
                sc[v][te], pr[v][te] = s_, p_

        for v in variants:
            for scope in ["pooled"] + sorted(set(ds_arr)):
                m = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                if len(np.unique(y[m])) < 2:
                    continue
                acc[(v, scope)]["mcc"].append(float(matthews_corrcoef(y[m], pr[v][m])))
                acc[(v, scope)]["auroc"].append(float(roc_auc_score(y[m], sc[v][m])))
        print(f"  seed {seed} done", flush=True)

    scopes = ["pooled"] + sorted(set(ds_all))
    for metric in ("mcc", "auroc"):
        print(f"\n### {metric.upper()}  ({len(args.seeds)} seeds)")
        print(f"  {'variant':18s}" + "".join(f"{s[:11]:>13s}" for s in scopes))
        for v in variants:
            print(f"  {v:18s}" + "".join(
                f"{np.mean(acc[(v, s)][metric]):13.4f}" if (v, s) in acc
                else f"{'-':>13s}" for s in scopes))
        b = {s: np.mean(acc[("greedy", s)][metric]) for s in scopes if ("greedy", s) in acc}
        for v in variants[1:]:
            print(f"  {'  vs greedy: ' + v:18s}" + "".join(
                f"{np.mean(acc[(v, s)][metric]) - b[s]:+13.4f}" if (v, s) in acc
                else f"{'-':>13s}" for s in scopes))

    write_json(Path(f"runs/{args.run}/stage5_posthoc/sampled_saplma.json"),
               {f"{v}|{s}": {m: round(float(np.mean(d[m])), 4) for m in d}
                for (v, s), d in acc.items()})
    print(f"\nwrote runs/{args.run}/stage5_posthoc/sampled_saplma.json")


if __name__ == "__main__":
    main()
