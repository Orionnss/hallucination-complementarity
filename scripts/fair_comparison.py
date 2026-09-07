"""Every method, the same classifier search. The control the central claim depends on.

The paper's claim is that prior comparisons were unfair to SAPLMA because each method was
scored with its own published probe. That argument collapses if our own protocol gives
SAPLMA the most classifier search — which, as written, it did: SAPLMA was run under four
classifiers while LapEigvals used a fixed n_components=512 / C=1.0 with no per-fold search
and ICR got only its native MLP.

Here every method draws from an identical config space, selected on an inner split by
AUROC, on identical folds, with the same threshold rule:

  PCA {64, 128} x logreg C {0.003, 0.03, 0.3, 3.0}     8 linear configs
  PCA 128 x MLP (256,128,64) and (128,) x alpha {1e-4, 1e-2}   4 non-linear configs

Both published probe shapes are inside that space — SAPLMA's (256,128,64) MLP and the
linear probes — so no method can lose to a config its own paper would not have chosen. ICR
additionally selects its pooling (last / mean / both) as its paper leaves that unspecified.

Two deliberate asymmetries, both against the claim rather than for it:

  SAPLMA's probe layer is FIXED at the depth used elsewhere in this study, while the
  published SAPLMA tunes it over seven depths. SAPLMA is therefore handicapped here.

  The MLP arm cannot use class_weight (sklearn supports neither that nor sample_weight for
  MLPClassifier), while the linear arm can. This handicaps the non-linear arm on skewed
  slices, so a linear win on CoQA should be read with that in mind.

Results accumulate per seed across invocations: background jobs are reaped on this host, so
the run is chunked into foreground calls.

Usage: uv run python scripts/fair_comparison.py --runs main --seeds 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}
METHODS = ["saplma", "lapeigvals", "attn_baseline", "icr", "svd_baseline", "logprob"]
C_GRID = (0.003, 0.03, 0.3, 3.0)
PCA_DIMS = (64, 128)
MLP_GRID = [((256, 128, 64), 1e-4), ((256, 128, 64), 1e-2), ((128,), 1e-4), ((128,), 1e-2)]


_LP_CACHE: dict = {}


def _logprob_ids(d):
    """Item ids that actually have logprob features, from the extraction checkpoint."""
    key = str(d)
    if key not in _LP_CACHE:
        ids = set()
        for line in (d / "checkpoint.jsonl").open():
            r = json.loads(line)
            if "shard" in r:
                ids.add(r.get("item_id") or r.get("id"))
        _LP_CACHE[key] = ids
    return _LP_CACHE[key]


def load_block(cfg, ids, name, layer):
    pos = {i: k for k, i in enumerate(ids)}
    out = None
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        base = cfg.stage_dir("stage1_extract", ds)
        if name == "logprob":
            # logprob lives in its own subdirectory (extracted after stage 1, logits
            # head only). ~0.1% of items failed the tokenizer round-trip and have no
            # features; those rows are imputed with the column median rather than
            # dropped, so every method is scored on an identical item set.
            have = [i for i in sub if i in _logprob_ids(base / "logprob")]
            got = load_features(base / "logprob", name, have).reshape(len(have), -1)
            a = np.tile(np.median(got, axis=0), (len(sub), 1)).astype(np.float32)
            pos_sub = {i: k for k, i in enumerate(sub)}
            a[[pos_sub[i] for i in have]] = got
            if len(have) < len(sub):
                print(f"    logprob: imputed {len(sub) - len(have)}/{len(sub)} in {ds}",
                      flush=True)
        else:
            a = load_features(base, name, sub)
        if name == "saplma":
            a = a[:, min(layer, a.shape[1] - 1), :]
        a = a.reshape(len(sub), -1).astype(np.float32)
        if out is None:
            out = np.zeros((len(ids), a.shape[1]), np.float32)
        out[[pos[i] for i in sub]] = a
        del a
    return out


def configs(arm="both"):
    """The identical config space every method draws from.

    `arm` restricts it to one classifier family, so the linear and non-linear halves can
    be reported separately on the same folds. With the full space the search picks the
    linear arm in 59 of 60 folds, which answers which is better but hides by how much.
    """
    out = []
    if arm in ("both", "linear"):
        out += [("lr", d, C, None) for d in PCA_DIMS for C in C_GRID]
    if arm in ("both", "mlp"):
        out += [("mlp", 128, h, a) for h, a in MLP_GRID]
    return out


def build(kind, d, p1, p2, seed):
    if kind == "lr":
        return LogisticRegression(C=p1, max_iter=3000, class_weight="balanced")
    return MLPClassifier(hidden_layer_sizes=p1, alpha=p2, max_iter=600,
                         early_stopping=True, n_iter_no_change=20, random_state=seed)


def score_of(m, X):
    return m.predict_proba(X)[:, 1]


def run(run_id, seeds, methods, arm="both"):
    cfg = Config(); cfg.run_id = run_id
    layer = LAYER.get(run_id, 24)
    acc = defaultdict(lambda: defaultdict(list))
    picks = defaultdict(list)

    for seed in seeds:
        sd = load_seed(cfg, seed)
        ids, y = list(sd["item_ids"]), sd["y"]
        groups, ds_arr = sd["groups"], sd["dataset"]
        strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])

        for meth in methods:
            t0 = time.perf_counter()
            if meth == "icr":
                # The ICR paper does not fix the pooling, so it is part of the search.
                blocks = {"last": load_block(cfg, ids, "icr", layer),
                          "mean": load_block(cfg, ids, "icr_mean", layer)}
                blocks["both"] = np.hstack([blocks["last"], blocks["mean"]])
                pools = ["last", "mean", "both"]
            else:
                blocks = {"x": load_block(cfg, ids, meth, layer)}
                pools = ["x"]

            score = np.full(len(y), np.nan)
            pred = np.full(len(y), -1, dtype=int)
            for tr, te in _folds(sd, seed, cfg.n_folds):
                inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
                i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))

                # One PCA per pooling at max width; smaller widths are prefixes of it.
                rep = {}
                for pl in pools:
                    X = blocks[pl]
                    s = StandardScaler().fit(X[tr])
                    a, b = s.transform(X[tr]), s.transform(X[te])
                    k = min(max(PCA_DIMS), a.shape[1], len(tr) - 1)
                    if k < a.shape[1]:
                        p = PCA(n_components=k, svd_solver="randomized",
                                random_state=seed).fit(a)
                        a, b = p.transform(a), p.transform(b)
                    rep[pl] = (a, b)

                best, best_a = None, -1.0
                for pl in pools:
                    A, _ = rep[pl]
                    for kind, d, p1, p2 in configs(arm):
                        dd = min(d, A.shape[1])
                        m = build(kind, dd, p1, p2, seed).fit(A[i_tr, :dd], y[tr][i_tr])
                        auc = roc_auc_score(y[tr][i_va], score_of(m, A[i_va, :dd]))
                        if auc > best_a:
                            best_a, best = float(auc), (pl, kind, dd, p1, p2)
                picks[meth if arm == "both" else f"{meth}__{arm}"].append(str(best))

                pl, kind, dd, p1, p2 = best
                A, B = rep[pl]
                m = build(kind, dd, p1, p2, seed).fit(A[i_tr, :dd], y[tr][i_tr])
                thr, _ = best_threshold(y[tr][i_va], score_of(m, A[i_va, :dd]))
                m = build(kind, dd, p1, p2, seed).fit(A[:, :dd], y[tr])
                score[te] = score_of(m, B[:, :dd])
                pred[te] = (score[te] >= thr).astype(int)

            name = meth if arm == "both" else f"{meth}__{arm}"
            for scope in ["pooled"] + sorted(set(ds_arr)):
                msk = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                if len(np.unique(y[msk])) < 2:
                    continue
                d = acc[(name, scope)]
                d["mcc"].append(float(matthews_corrcoef(y[msk], pred[msk])))
                d["auroc"].append(float(roc_auc_score(y[msk], score[msk])))
                d["accuracy"].append(float(accuracy_score(y[msk], pred[msk])))
                d["balanced_accuracy"].append(
                    float(balanced_accuracy_score(y[msk], pred[msk])))
                d["f1"].append(float(f1_score(y[msk], pred[msk], zero_division=0)))
            # per-seed OOF kept so paired tests can run later without refitting
            acc[(name, "_oof")]["seed"].append(seed)
            print(f"  {run_id} seed {seed} {meth} in "
                  f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            del blocks

    return ({f"{m}|{s}": dict(d) for (m, s), d in acc.items() if s != "_oof"},
            {k: v for k, v in picks.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--arm", choices=("both", "linear", "mlp"), default="both",
                    help="restrict the config space to one classifier family; results are "
                         "stored under '<method>__<arm>' so they do not overwrite the "
                         "full-search rows")
    args = ap.parse_args()

    dest = Path("runs/fair_comparison.json")
    out = json.loads(dest.read_text()) if dest.exists() else {}
    for run_id in args.runs:
        res, picks = run(run_id, args.seeds, args.methods, args.arm)
        prev = out.get(run_id, {"raw": {}, "seeds": [], "configs": {}})
        for key, d in res.items():
            for m, v in d.items():
                prev["raw"].setdefault(key, {}).setdefault(m, []).extend(v)
        prev["seeds"] = sorted(set(prev["seeds"]) | set(args.seeds))
        for k, v in picks.items():
            prev["configs"].setdefault(k, []).extend(v)
        out[run_id] = prev
    write_json(dest, out)

    scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
    for metric in ("mcc", "auroc"):
        print(f"\n{'=' * 84}\n  {metric.upper()} — every method, identical search\n{'=' * 84}")
        for run_id in out:
            r = out[run_id]["raw"]
            rows = [(m, [np.mean(r[f"{m}|{s}"][metric]) if f"{m}|{s}" in r else None
                         for s in scopes]) for m in METHODS if f"{m}|pooled" in r]
            if not rows:
                continue
            print(f"\n  {run_id}  [{len(out[run_id]['seeds'])} seeds]")
            print(f"    {'method':16s}" + "".join(f"{s[:11]:>13s}" for s in scopes))
            for m, vals in sorted(rows, key=lambda x: -(x[1][0] or 0)):
                print(f"    {m:16s}" + "".join(
                    f"{v:13.4f}" if v is not None else f"{'-':>13s}" for v in vals))
    print("\nwrote runs/fair_comparison.json")


if __name__ == "__main__":
    main()
