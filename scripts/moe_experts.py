"""Mixture of dataset experts: same features, different training data.

Every combination tested so far varied the representation and held the training set fixed
— five methods, one pooled probe each — and every one failed. The mechanism proposed for
that failure is that the methods read a common substrate, so their disagreement is
estimation noise rather than different information.

This varies the other axis. One SAPLMA probe (features + PCA + logreg) is trained per
dataset, and the four specialists are then combined. The representation is identical
across experts; only the data they saw differs. If diversity of *training distribution*
buys what diversity of *representation* did not, the negative result is narrower than
stated; if it also fails, the result generalises across both axes.

There is a real reason to expect a gain here: the datasets differ in base rate
(0.15-0.66), in whether a passage is supplied, and in answer length, so a pooled probe
must compromise on decision geometry that a specialist need not.

Variants, in increasing order of what they are allowed to know:

  pooled          one probe on all training data. The baseline.
  expert_oracle   each dataset scored by its own expert. Uses dataset identity at test
                  time, which deployment usually does not have — reported as the ceiling
                  on what routing could achieve, not as a competitor.
  moe_vote_hard   majority of the four experts' thresholded votes. Blind.
  moe_vote_soft   mean of the four experts' probabilities. Blind. Experts are fitted with
                  balanced class weights, which calibrates each to a 50/50 prior and makes
                  their outputs comparable despite very different base rates.
  moe_max         max over experts — "any specialist calls this risky". Blind.
  moe_gated       the actual mixture of experts: a gate predicts the dataset from the same
                  features, and expert scores are averaged weighted by gate posterior.

Gate accuracy is reported, because it decides how to read moe_gated: if the hidden state
trivially encodes which dataset an item came from, the gate collapses onto expert_oracle
and the two rows answer the same question.

Usage: uv run python scripts/moe_experts.py --runs main
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
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _folds, load_seed

LAYER = {"main": 24, "gemma3-12b": 29, "gemma3-4b": 17, "llama3.2-3b": 14}
C_GRID = (0.003, 0.03, 0.3, 3.0)
DIM = 128


def load_saplma(cfg, ids, layer):
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


class Probe:
    """Scaler + PCA + logistic regression, C chosen on an inner split."""

    def fit(self, X, y, i_tr, i_va, seed):
        self.s = StandardScaler().fit(X[i_tr])
        A = self.s.transform(X[i_tr])
        k = min(DIM, A.shape[1], len(i_tr) - 1)
        self.p = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(A)
        A = self.p.transform(A)
        V = self.p.transform(self.s.transform(X[i_va]))
        best_c, best_a = C_GRID[0], -1.0
        for C in C_GRID:
            m = LogisticRegression(C=C, max_iter=3000,
                                   class_weight="balanced").fit(A, y[i_tr])
            a = (roc_auc_score(y[i_va], m.predict_proba(V)[:, 1])
                 if len(np.unique(y[i_va])) > 1 else 0.5)
            if a > best_a:
                best_a, best_c = float(a), C
        self.m = LogisticRegression(C=best_c, max_iter=3000,
                                    class_weight="balanced").fit(A, y[i_tr])
        self.thr, _ = best_threshold(y[i_va], self.m.predict_proba(V)[:, 1])
        # Refit on train+val once the threshold is set, as elsewhere in this study.
        allX = np.vstack([X[i_tr], X[i_va]])
        ally = np.concatenate([y[i_tr], y[i_va]])
        self.m = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(
            self.p.transform(self.s.transform(allX)), ally)
        return self

    def score(self, X):
        return self.m.predict_proba(self.p.transform(self.s.transform(X)))[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=["main"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    dest = Path("runs/moe_experts.json")
    out = json.loads(dest.read_text()) if dest.exists() else {}

    for run in args.runs:
        cfg = Config(); cfg.run_id = run
        acc = defaultdict(lambda: defaultdict(list))
        gate_acc = []
        for seed in args.seeds:
            t0 = time.perf_counter()
            sd = load_seed(cfg, seed)
            ids, y = list(sd["item_ids"]), sd["y"]
            groups, ds_arr = sd["groups"], sd["dataset"].astype(str)
            strat = np.array([f"{a}_{b}" for a, b in zip(ds_arr, y)])
            X = load_saplma(cfg, ids, LAYER.get(run, 24))
            dsets = sorted(set(ds_arr))

            variants = ["pooled", "expert_oracle", "moe_vote_hard", "moe_vote_soft",
                        "moe_max", "moe_gated"]
            sc = {v: np.full(len(y), np.nan) for v in variants}
            pr = {v: np.full(len(y), -1, dtype=int) for v in variants}

            for tr, te in _folds(sd, seed, cfg.n_folds):
                inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
                i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
                a_tr, a_va = tr[i_tr], tr[i_va]

                pooled = Probe().fit(X, y, a_tr, a_va, seed)
                s_va, s_te = pooled.score(X[a_va]), pooled.score(X[te])
                sc["pooled"][te] = s_te
                pr["pooled"][te] = (s_te >= pooled.thr).astype(int)

                # one expert per dataset, trained only on that dataset's training rows
                experts = {}
                for dnm in dsets:
                    m_tr = a_tr[ds_arr[a_tr] == dnm]
                    m_va = a_va[ds_arr[a_va] == dnm]
                    if len(m_tr) < 50 or len(np.unique(y[m_tr])) < 2:
                        continue
                    experts[dnm] = Probe().fit(X, y, m_tr, m_va, seed)

                E_te = np.vstack([experts[d].score(X[te]) for d in dsets if d in experts])
                E_va = np.vstack([experts[d].score(X[a_va]) for d in dsets if d in experts])
                T = np.array([experts[d].thr for d in dsets if d in experts])[:, None]
                names = [d for d in dsets if d in experts]

                # oracle routing: each item scored by the expert for its own dataset
                own = np.full(len(te), np.nan)
                own_p = np.zeros(len(te), dtype=int)
                for k, dnm in enumerate(names):
                    m = ds_arr[te] == dnm
                    own[m] = E_te[k][m]
                    own_p[m] = (E_te[k][m] >= experts[dnm].thr).astype(int)
                sc["expert_oracle"][te], pr["expert_oracle"][te] = own, own_p

                # blind combinations
                votes_te = (E_te >= T).sum(0)
                votes_va = (E_va >= T).sum(0)
                for nm, v_te, v_va in (
                        ("moe_vote_hard", votes_te / len(names), votes_va / len(names)),
                        ("moe_vote_soft", E_te.mean(0), E_va.mean(0)),
                        ("moe_max", E_te.max(0), E_va.max(0))):
                    t, _ = best_threshold(y[a_va], v_va)
                    sc[nm][te], pr[nm][te] = v_te, (v_te >= t).astype(int)

                # learned gate over the same features
                g = Probe()
                g.s = StandardScaler().fit(X[a_tr])
                A = g.s.transform(X[a_tr])
                g.p = PCA(n_components=min(DIM, A.shape[1], len(a_tr) - 1),
                          svd_solver="randomized", random_state=seed).fit(A)
                gate = LogisticRegression(max_iter=3000).fit(
                    g.p.transform(A), ds_arr[a_tr])
                P_te = gate.predict_proba(g.p.transform(g.s.transform(X[te])))
                P_va = gate.predict_proba(g.p.transform(g.s.transform(X[a_va])))
                order = [list(gate.classes_).index(d) for d in names]
                gm_te = (E_te * P_te[:, order].T).sum(0)
                gm_va = (E_va * P_va[:, order].T).sum(0)
                t, _ = best_threshold(y[a_va], gm_va)
                sc["moe_gated"][te], pr["moe_gated"][te] = gm_te, (gm_te >= t).astype(int)
                gate_acc.append(float((gate.predict(
                    g.p.transform(g.s.transform(X[te]))) == ds_arr[te]).mean()))

            for v in variants:
                for scope in ["pooled"] + dsets:
                    m = np.ones(len(y), bool) if scope == "pooled" else (ds_arr == scope)
                    if len(np.unique(y[m])) < 2:
                        continue
                    acc[(v, scope)]["mcc"].append(float(matthews_corrcoef(y[m], pr[v][m])))
                    acc[(v, scope)]["auroc"].append(float(roc_auc_score(y[m], sc[v][m])))
            print(f"  {run} seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min",
                  flush=True)
            del X

        prev = out.get(run, {"raw": {}, "seeds": [], "gate_accuracy": []})
        for (v, s), d in acc.items():
            for k, vals in d.items():
                prev["raw"].setdefault(f"{v}|{s}", {}).setdefault(k, []).extend(vals)
        prev["seeds"] = sorted(set(prev["seeds"]) | set(args.seeds))
        prev["gate_accuracy"] += gate_acc
        out[run] = prev
    write_json(dest, out)

    order = ["pooled", "expert_oracle", "moe_gated", "moe_vote_soft", "moe_vote_hard",
             "moe_max"]
    for metric in ("mcc", "auroc"):
        print(f"\n{'=' * 84}\n  {metric.upper()}\n{'=' * 84}")
        for run in out:
            r = out[run]["raw"]
            scopes = ["pooled", "triviaqa", "nq_open", "squad_v2", "coqa"]
            print(f"\n  {run}  [{len(out[run]['seeds'])} seeds]  "
                  f"gate accuracy {np.mean(out[run]['gate_accuracy']):.1%}")
            print(f"    {'variant':16s}" + "".join(f"{s[:11]:>13s}" for s in scopes))
            for v in order:
                if f"{v}|pooled" not in r:
                    continue
                print(f"    {v:16s}" + "".join(
                    f"{np.mean(r[f'{v}|{s}'][metric]):13.4f}" if f"{v}|{s}" in r
                    else f"{'-':>13s}" for s in scopes))
            base = np.mean(r["pooled|pooled"][metric])
            for v in order[1:]:
                if f"{v}|pooled" in r:
                    d = np.mean(r[f"{v}|pooled"][metric]) - base
                    print(f"    {'  vs pooled: ' + v:30s}{d:+.4f}")
    print("\nwrote runs/moe_experts.json")


if __name__ == "__main__":
    main()
