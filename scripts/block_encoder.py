"""Learned per-block encoders instead of PCA, then a shared classification head.

PCA equalises block widths but is *unsupervised*: it keeps directions of maximum
variance, which need not be the directions carrying label information. The subset
ablation showed the cost of that — adding blocks under PCA+concat monotonically hurt
AUROC, i.e. the extra 128 dims per block were mostly noise.

This replaces each block's PCA with a small MLP encoder trained end-to-end with the
classifier, so every block learns a label-relevant projection to a common width:

    x_b  ->  [dropout -> Linear(d_b, hidden) -> GELU -> dropout -> Linear(hidden, k)]  ->  z_b
    [z_1 ... z_5]  ->  head (linear or MLP)  ->  logit

Both heads are run. Compared against two references fitted on the identical folds:
PCA+logreg on all blocks (stage 3's `union_equal`) and PCA+logreg on the SAPLMA block
alone, which is the strongest single-block configuration found so far.

Trained with BCE (positive class weighted), AdamW, early stopping on inner-split AUROC.
The decision threshold is tuned on that same inner split, never on test.

Usage: uv run python scripts/block_encoder.py --run main --saplma-layer 24 --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from halluc.config import Config
from halluc.eval.metrics import best_threshold
from halluc.io import load_features, write_json
from halluc.pipeline.stage5_posthoc import _fast_mcc, _folds, load_seed

BLOCKS = ["lapeigvals", "attn_baseline", "saplma", "svd_baseline", "icr"]


class BlockEncoder(nn.Module):
    """One encoder per block to a shared width, then a head over the concatenation."""

    def __init__(self, dims: dict[str, int], k: int = 64, hidden: int = 128,
                 dropout: float = 0.3, head: str = "linear"):
        super().__init__()
        self.names = list(dims)
        self.encoders = nn.ModuleDict({
            b: nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, k),
            )
            for b, d in dims.items()
        })
        width = k * len(dims)
        if head == "linear":
            self.head = nn.Linear(width, 1)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(width), nn.Dropout(dropout),
                nn.Linear(width, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

    def forward(self, xs: dict[str, torch.Tensor]) -> torch.Tensor:
        z = torch.cat([self.encoders[b](xs[b]) for b in self.names], dim=1)
        return self.head(z).squeeze(-1)


def train_encoder(Xtr, ytr, Xva, yva, Xte, dims, device, seed, head, epochs=200,
                  k=64, hidden=128, dropout=0.3, lr=1e-3, wd=1e-2, patience=25):
    torch.manual_seed(seed)
    model = BlockEncoder(dims, k=k, hidden=hidden, dropout=dropout, head=head).to(device)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    n = len(ytr)
    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            out = model({b: Xtr[b][idx] for b in dims})
            loss_fn(out, yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = model({b: Xva[b] for b in dims}).cpu().numpy()
        auc = roc_auc_score(yva, va) if len(np.unique(yva)) > 1 else 0.5
        if auc > best_auc:
            best_auc, bad = float(auc), 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va = torch.sigmoid(model({b: Xva[b] for b in dims})).cpu().numpy()
        te = torch.sigmoid(model({b: Xte[b] for b in dims})).cpu().numpy()
    return va, te, best_auc


def pca_logreg(blocks_tr, blocks_te, ytr, subset, seed, n_comp=128):
    """Reference pipeline: standardise, PCA per block, concatenate, logistic regression."""
    parts_tr, parts_te = [], []
    for b in subset:
        s = StandardScaler().fit(blocks_tr[b])
        a, c = s.transform(blocks_tr[b]), s.transform(blocks_te[b])
        kk = min(n_comp, a.shape[1], len(a) - 1)
        if kk < a.shape[1]:
            p = PCA(n_components=kk, random_state=seed).fit(a)
            a, c = p.transform(a), p.transform(c)
        parts_tr.append(a); parts_te.append(c)
    Xtr, Xte = np.hstack(parts_tr), np.hstack(parts_te)
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    return m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1], m


def load_blocks(cfg, sd, saplma_layer):
    ids = list(sd["item_ids"])
    pos = {i: k for k, i in enumerate(ids)}
    out: dict[str, np.ndarray] = {}
    for ds in cfg.datasets:
        sub = [i for i in ids if i.startswith(ds + ":")]
        if not sub:
            continue
        d = cfg.stage_dir("stage1_extract", ds)
        rows = [pos[i] for i in sub]
        for b in BLOCKS:
            arr = load_features(d, b, sub)
            if b == "saplma":
                arr = arr[:, min(saplma_layer, arr.shape[1] - 1), :]
            arr = arr.reshape(len(sub), -1).astype(np.float32)
            if b not in out:
                out[b] = np.zeros((len(ids), arr.shape[1]), np.float32)
            out[b][rows] = arr
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--saplma-layer", type=int, required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    ap.add_argument("--exclude-datasets", nargs="*", default=[],
                    help="datasets to drop from training and evaluation, e.g. coqa")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="torch intra-op threads (0 = leave default); set when running "
                         "several models concurrently on CPU")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    cfg = Config(); cfg.run_id = args.run
    dev = torch.device(args.device)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    if args.exclude_datasets:
        # Drop the datasets from the *pooled* problem while keeping stage 3's exact fold
        # partition: folds are computed over all items, then each train/test index set is
        # intersected with the kept rows. Re-deriving folds on the subset instead would
        # change the grouping and break comparability with every other result.
        print(f"excluding datasets: {args.exclude_datasets}", flush=True)
    variants = ["enc_linear", "enc_mlp", "pca_logreg_all", "pca_logreg_saplma"]
    acc = {v: {"auroc": [], "mcc": []} for v in variants}

    for seed in args.seeds:
        t0 = time.perf_counter()
        sd = load_seed(cfg, seed)
        print(f"seed {seed}: loading blocks ...", flush=True)
        blocks = load_blocks(cfg, sd, args.saplma_layer)
        y, groups = sd["y"], sd["groups"]
        strat = np.array([f"{d}_{v}" for d, v in zip(sd["dataset"], y)])
        keep = ~np.isin(sd["dataset"], args.exclude_datasets)
        oof = {v: np.full(len(y), np.nan) for v in variants}
        preds = {v: np.full(len(y), -1, dtype=int) for v in variants}

        for fold, (tr, te) in enumerate(_folds(sd, seed, cfg.n_folds)):
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            i_tr, i_va = next(inner.split(np.zeros(len(tr)), strat[tr], groups[tr]))
            a_tr, a_va = tr[i_tr], tr[i_va]
            # Restrict to the kept datasets; the partition itself is unchanged.
            a_tr, a_va, te = a_tr[keep[a_tr]], a_va[keep[a_va]], te[keep[te]]
            if len(a_tr) < 100 or len(te) < 20:
                continue

            # Standardise per block on the encoder's training rows only.
            scalers = {b: StandardScaler().fit(blocks[b][a_tr]) for b in BLOCKS}
            def to_t(idx):
                return {b: torch.tensor(scalers[b].transform(blocks[b][idx]),
                                        dtype=torch.float32, device=dev) for b in BLOCKS}
            Xtr, Xva, Xte = to_t(a_tr), to_t(a_va), to_t(te)
            dims = {b: blocks[b].shape[1] for b in BLOCKS}

            for head, name in (("linear", "enc_linear"), ("mlp", "enc_mlp")):
                va, tep, _ = train_encoder(Xtr, y[a_tr], Xva, y[a_va], Xte, dims, dev,
                                           seed, head, k=args.k, hidden=args.hidden,
                                           dropout=args.dropout)
                thr, _ = best_threshold(y[a_va], va)
                oof[name][te] = tep
                preds[name][te] = (tep >= thr).astype(int)
            del Xtr, Xva, Xte
            torch.cuda.empty_cache()

            for name, subset in (("pca_logreg_all", BLOCKS), ("pca_logreg_saplma", ["saplma"])):
                btr = {b: blocks[b][a_tr] for b in BLOCKS}
                bva = {b: blocks[b][a_va] for b in BLOCKS}
                bte = {b: blocks[b][te] for b in BLOCKS}
                _, vap, m = pca_logreg(btr, bva, y[a_tr], subset, seed)
                _, tep, _ = pca_logreg(btr, bte, y[a_tr], subset, seed)
                thr, _ = best_threshold(y[a_va], vap)
                oof[name][te] = tep
                preds[name][te] = (tep >= thr).astype(int)
            print(f"  seed {seed} fold {fold + 1}/{cfg.n_folds} done", flush=True)

        scored = keep & (preds[variants[0]] >= 0)
        for v in variants:
            acc[v]["auroc"].append(float(roc_auc_score(y[scored], oof[v][scored])))
            acc[v]["mcc"].append(_fast_mcc(y[scored], preds[v][scored]))
        print(f"  seed {seed} in {(time.perf_counter() - t0) / 60:.1f} min: " +
              "  ".join(f"{v}={acc[v]['auroc'][-1]:.4f}" for v in variants), flush=True)
        del blocks

    out = {"run": args.run, "seeds": args.seeds, "device": args.device,
           "excluded_datasets": args.exclude_datasets,
           "config": {"k": args.k, "hidden": args.hidden, "dropout": args.dropout},
           "results": {v: {"auroc_mean": round(float(np.mean(d["auroc"])), 4),
                           "auroc_std": round(float(np.std(d["auroc"])), 4),
                           "mcc_mean": round(float(np.mean(d["mcc"])), 4),
                           "mcc_std": round(float(np.std(d["mcc"])), 4)}
                       for v, d in acc.items()}}
    tag = "_no" + "".join(args.exclude_datasets) if args.exclude_datasets else ""
    write_json(Path(f"runs/{args.run}/stage5_posthoc/block_encoder{tag}.json"), out)
    print(f"\n=== {args.run} ===")
    print(f"{'variant':22s}{'AUROC':>16s}{'MCC':>16s}")
    for v, d in sorted(out["results"].items(), key=lambda kv: -kv[1]["auroc_mean"]):
        print(f"  {v:20s}{d['auroc_mean']:9.4f} ±{d['auroc_std']:.4f}"
              f"{d['mcc_mean']:9.4f} ±{d['mcc_std']:.4f}")


if __name__ == "__main__":
    main()
