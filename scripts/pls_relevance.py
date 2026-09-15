"""How much output-relevant linear signal does each detector's features carry?

One PLS model per source, each predicting the same target, then the sources compared by
how much of that target their components capture. PLS's own objective is
max cov(Xw, y)^2, so the per-component covariance is a direct read on "how much
linearly-decodable, output-relevant information is in this block alone".

The reason this cannot be done with covariance by itself:

    cov(Xw, y) grows with the number of columns in X. attn_baseline and lapeigvals carry
    16,000 dims on Qwen3-14B, icr carries 40. A wide block reaches a high covariance by
    combining noise across many variables, with no more real signal than a narrow one.
    Ranking blocks on raw covariance would therefore rank them substantially by width.

So three measures are reported side by side, and they disagree in exactly the way that
makes the confound visible:

  corr(t1, y)   covariance normalised by both standard deviations -- the same quantity,
                made scale-free and bounded. Comparable across blocks of any width.
  R2 out-of-fold  PLS fitted on the training fold, scored on the held-out one. Capacity
                that only fits noise scores ~0 here, so this is decodable signal rather
                than apparent signal. This is the measure to rank on.
  R2 permuted   the same fit with the labels shuffled inside the training fold. This is
                what the block's width alone buys, and it is the baseline every in-sample
                number has to be read against.

AUROC of the PLS scores is reported alongside because it is the currency the rest of this
study uses, and it makes the ranking directly comparable with the detector tables.

Everything is fitted strictly inside the training fold -- PLS consumes the label, so a
projection fitted before the split would leak it into the representation itself.

Usage: uv run python scripts/pls_relevance.py --run main --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.io import write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fair_comparison import LAYER, load_block

METHODS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline", "logprob"]
N_COMP = 8
AT = (1, 2, 4, 8)


def drop_degenerate(X_tr, X_te):
    """Drop columns with no usable variance in the training fold.

    Gemma's attention blocks contain columns that are constant, or near enough, over a
    training fold: 827 of attn_baseline's 7,680 are exactly constant on gemma-3-12b and
    955 sit below 1e-8. StandardScaler divides each by its own tiny standard deviation,
    which does two damaging things at once -- it hands a noise-only column unit variance,
    so a variance-ranked method treats it as equal to real signal, and it amplifies any
    held-out value that differs at all: column 349 of lapeigvals has training variance
    5.4e-16 and reaches 7.5e4 after scaling on the test fold, which is enough to blow the
    regression's held-out R2 to -1.6e7 while leaving rank-based AUROC untouched.

    The floor is relative to the block's own median variance, since the blocks differ by
    orders of magnitude in scale.
    """
    v = X_tr.var(axis=0)
    med = float(np.median(v[v > 0])) if (v > 0).any() else 1.0
    keep = v > max(1e-12, 1e-8 * med)
    return X_tr[:, keep], X_te[:, keep], int((~keep).sum())


def r2(y_true, y_hat) -> float:
    """R^2 against the target's own mean. Negative when the fit is worse than that mean."""
    ss_res = float(((y_true - y_hat) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fold_stats(A, B, y_tr, y_te, n_comp, seed, rng):
    """PLS on one fold: covariance/correlation of the components, and held-out fit."""
    k = min(n_comp, A.shape[1])
    pls = PLSRegression(n_components=k, scale=False).fit(A, y_tr)
    T, U = pls.transform(A), pls.transform(B)
    out = {}

    # The PLS objective itself, and its scale-free form. Both on the training fold,
    # because that is where the components were chosen to maximise it.
    yc = y_tr - y_tr.mean()
    out["cov2_c1"] = float(np.cov(T[:, 0], yc)[0, 1] ** 2)
    out["corr_c1"] = abs(float(np.corrcoef(T[:, 0], y_tr)[0, 1]))
    out["corr_c1_oof"] = abs(float(np.corrcoef(U[:, 0], y_te)[0, 1]))
    out["cov2_total"] = float(sum(np.cov(T[:, a], yc)[0, 1] ** 2 for a in range(k)))

    # Per-component decay, and the cumulative sums at the two widths asked for. PLS
    # extracts components sequentially against a deflated X, so cov(t_a, y) falls with a;
    # how fast it falls says whether a source's relevance sits in one direction or is
    # spread over several. Correlation is given per component too, since the covariances
    # are not comparable between sources of different width.
    for a in range(k):
        out[f"cov2_c{a + 1}"] = float(np.cov(T[:, a], yc)[0, 1] ** 2)
        out[f"corr_c{a + 1}"] = abs(float(np.corrcoef(T[:, a], y_tr)[0, 1]))
        out[f"corr_c{a + 1}_oof"] = (abs(float(np.corrcoef(U[:, a], y_te)[0, 1]))
                                     if len(np.unique(y_te)) > 1 else float("nan"))
    for a in AT:
        if a <= k:
            out[f"cov2_cum@{a}"] = float(sum(np.cov(T[:, j], yc)[0, 1] ** 2
                                             for j in range(a)))

    # Cumulative fit at each width, in-sample and held out. A refit per width rather than
    # a prefix, because PLS's regression coefficients are re-solved for each rank.
    for a in AT:
        if a > k:
            continue
        p = PLSRegression(n_components=a, scale=False).fit(A, y_tr)
        out[f"r2_train@{a}"] = r2(y_tr, p.predict(A).ravel())
        out[f"r2_oof@{a}"] = r2(y_te, p.predict(B).ravel())
        if len(np.unique(y_te)) > 1:
            out[f"auroc_oof@{a}"] = float(roc_auc_score(y_te, p.predict(B).ravel()))

    # The null: identical fit, labels destroyed. Whatever this scores is what the block's
    # width buys on its own, and it is the baseline the in-sample numbers sit above.
    y_perm = rng.permutation(y_tr)
    p = PLSRegression(n_components=k, scale=False).fit(A, y_perm)
    out[f"r2_train_perm@{k}"] = r2(y_perm, p.predict(A).ravel())
    out[f"r2_oof_perm@{k}"] = r2(y_te, p.predict(B).ravel())
    out["corr_c1_perm"] = abs(float(np.corrcoef(
        PLSRegression(n_components=1, scale=False).fit(A, y_perm).transform(A)[:, 0],
        y_perm)[0, 1]))
    return out


def run_method(cfg, run, method, seeds):
    acc = defaultdict(list)
    dropped: list[int] = []
    width = None
    for seed in seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"].astype(float)
        X = load_block(cfg, ids, method, LAYER.get(run, 24))
        width = int(X.shape[1])
        rng = np.random.default_rng(seed)
        for tr, te in _folds(sd, seed, cfg.n_folds):
            A_raw, B_raw, n_dropped = drop_degenerate(X[tr], X[te])
            dropped.append(n_dropped)
            s = StandardScaler().fit(A_raw)
            for k, v in fold_stats(s.transform(A_raw), s.transform(B_raw),
                                   y[tr], y[te], N_COMP, seed, rng).items():
                acc[k].append(v)
        print(f"    {run}/{method} seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min",
              flush=True)
        del X
    out = {k: {"mean": round(float(np.mean(v)), 6), "std": round(float(np.std(v)), 6)}
           for k, v in acc.items()}
    out["n_dropped_degenerate"] = {"mean": round(float(np.mean(dropped)), 1),
                                   "std": round(float(np.std(dropped)), 1)}
    return out, width


def report(run, res):
    print(f"\n=== {run}: output-relevant linear signal per source "
          f"(PLS, one model per source, same target) ===")
    order = sorted(res, key=lambda m: -res[m]["stats"].get("r2_oof@8", {}).get("mean", -9))
    print(f"\n  {'source':16s}{'dims':>7s}{'corr(t1,y)':>12s}{'oof':>8s}{'perm':>8s}"
          f"{'R2 tr@8':>10s}{'R2 oof@8':>10s}{'perm':>9s}{'AUROC oof@8':>13s}")
    for m in order:
        s, w = res[m]["stats"], res[m]["n_features"]
        pk = next((k for k in s if k.startswith("r2_train_perm@")), None)
        pko = next((k for k in s if k.startswith("r2_oof_perm@")), None)
        g = lambda k: s[k]["mean"] if k in s else float("nan")
        print(f"  {m:16s}{w:>7d}{g('corr_c1'):>12.4f}{g('corr_c1_oof'):>8.4f}"
              f"{g('corr_c1_perm'):>8.4f}{g('r2_train@8'):>10.4f}{g('r2_oof@8'):>10.4f}"
              f"{g(pko):>9.4f}{g('auroc_oof@8'):>13.4f}")

    print(f"\n  held-out R2 by number of components")
    print(f"  {'source':16s}" + "".join(f"{'@' + str(a):>10s}" for a in AT)
          + f"{'AUROC@1':>10s}{'AUROC@8':>10s}")
    for m in order:
        s = res[m]["stats"]
        g = lambda k: s[k]["mean"] if k in s else float("nan")
        print(f"  {m:16s}" + "".join(f"{g(f'r2_oof@{a}'):>10.4f}" for a in AT)
              + f"{g('auroc_oof@1'):>10.4f}{g('auroc_oof@8'):>10.4f}")

    print(f"\n  cov^2(t_a, y) of the PLS projections -- the objective itself.")
    print(f"  Rises with source width even at equal correlation, so it ranks by capacity")
    print(f"  as much as by signal; the correlation table below is the comparable form.")
    print(f"  {'source':16s}{'dims':>7s}{'cum@4':>11s}{'cum@8':>11s}"
          + "".join(f"{'c' + str(a):>9s}" for a in range(1, 9)))
    for m in sorted(res, key=lambda m: -res[m]["stats"].get("cov2_cum@8", {}).get("mean", -9)):
        st, w = res[m]["stats"], res[m]["n_features"]
        g = lambda k: st[k]["mean"] if k in st else float("nan")
        print(f"  {m:16s}{w:>7d}{g('cov2_cum@4'):>11.3f}{g('cov2_cum@8'):>11.3f}"
              + "".join(f"{g(f'cov2_c{a}'):>9.3f}" for a in range(1, 9)))

    print(f"\n  corr(t_a, y) per component -- scale-free, comparable across sources")
    print(f"  {'source':16s}{'dims':>7s}" + "".join(f"{'c' + str(a):>9s}" for a in range(1, 9)))
    for m in sorted(res, key=lambda m: -res[m]["stats"].get("corr_c1", {}).get("mean", -9)):
        st, w = res[m]["stats"], res[m]["n_features"]
        g = lambda k: st[k]["mean"] if k in st else float("nan")
        print(f"  {m:16s}{w:>7d}" + "".join(f"{g(f'corr_c{a}'):>9.3f}" for a in range(1, 9)))

    infl = [(res[m]["stats"]["r2_train@8"]["mean"]
             - res[m]["stats"].get("r2_oof@8", {}).get("mean", 0), m) for m in res]
    print(f"\n  in-sample minus held-out R2 at 8 components (the width premium):")
    for d, m in sorted(infl, reverse=True):
        print(f"    {m:16s}{d:+.4f}   ({res[m]['n_features']} dims)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="main")
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    args = ap.parse_args()

    if args.merge is not None:
        out = {}
        for f in args.merge:
            out.update(json.loads(Path(f).read_text()))
        write_json(Path("runs/pls_relevance.json"), out)
        for run, res in out.items():
            report(run, res)
        print(f"\nmerged {len(args.merge)} files -> runs/pls_relevance.json")
        return

    cfg = Config(); cfg.run_id = args.run
    res = {}
    for method in args.methods:
        try:
            stats, width = run_method(cfg, args.run, method, args.seeds)
        except (FileNotFoundError, KeyError) as exc:
            print(f"    {args.run}/{method}: unavailable ({type(exc).__name__}), skipped",
                  flush=True)
            continue
        res[method] = {"stats": stats, "n_features": width, "seeds": args.seeds}
    if not res:
        print(f"nothing computed for {args.run}")
        return
    dest = Path(args.out or f"runs/pls_relevance_parts/{args.run}.json")
    write_json(dest, {args.run: res})
    report(args.run, res)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
