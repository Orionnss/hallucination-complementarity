"""Fitting CHARM: AdamW, the paper's schedulers, early stopping on a validation split.

Optimiser and schedule follow Appendix D.1.3 — AdamW, with either "reduce on plateau" or
cosine annealing under a warmup spanning 10% of training steps, whichever validates
better. The hyperparameter grid is Table 9's.

Early stopping is on validation AUROC. The rest of this repo tunes with
`sklearn`'s `early_stopping=True` on its MLP probes, so stopping on a held-out split
rather than a fixed epoch count is the established convention here; it also means the
number of epochs never has to be tuned as a grid axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .data import GraphCache, batch_indices, collate
from .graph import AttentionGraph, GraphSpec
from .model import CharmNet


@dataclass(frozen=True)
class CharmParams:
    """One point in Table 9's search space."""

    hidden: int = 64
    n_layers: int = 2
    dropout: float = 0.25
    lr: float = 5e-4
    weight_decay: float = 1e-3
    act_weight_decay: float = 0.05
    batch_norm: bool = True
    residual: bool = True
    scheduler: str = "cosine"  # "cosine" | "plateau"
    batch_size: int = 32
    max_epochs: int = 60
    patience: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """Everything that is a property of the machine rather than of the method."""

    device: str = "cuda:1"
    #: Edge budget per batch. The dense per-batch message tensor is [E, hidden], and the
    #: EmbeddingBag over the CSR features scales with the same E, so this is the knob
    #: that bounds GPU memory when a batch collects several long CoQA graphs.
    max_edges_per_batch: int = 400_000
    #: Graph-building work is on the extraction device; training can use the same GPU
    #: because the generator has already been unloaded by then.
    seed: int = 0
    #: Pin the fit to a single thread. Message passing aggregates with `index_add_` and
    #: `EmbeddingBag`, whose backward passes sum gradients in whatever order the threads
    #: (or CUDA atomics) happen to finish in, so a repeated fit at the same seed lands
    #: within ~1e-4 on the output probabilities rather than bitwise identical. That is
    #: two orders of magnitude below the across-seed spread this study reports, so the
    #: default keeps the parallelism; set this when a byte-exact rerun is needed.
    deterministic: bool = False


def _make_batches(graphs, order, params, cfg):
    return batch_indices(graphs, order, params.batch_size, cfg.max_edges_per_batch)


def _scheduler(name: str, optimiser, total_steps: int):
    """Either of the two schedules in Appendix D.1.3."""
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", factor=0.5, patience=3
        ), "epoch"
    warmup = max(1, int(0.1 * total_steps))  # "warmup spanned 10% of the total training steps"

    def curve(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimiser, curve), "step"


@torch.no_grad()
def _predict(model, graphs, order, params, cfg, device) -> np.ndarray:
    model.eval()
    scores = np.zeros(len(order), dtype=np.float64)
    position = {int(idx): slot for slot, idx in enumerate(order)}
    for chunk in _make_batches(graphs, order, params, cfg):
        batch = collate([graphs[i] for i in chunk], device)
        logits = model(batch).float().cpu().numpy()
        for value, idx in zip(logits, chunk):
            scores[position[int(idx)]] = value
    # Logits are monotone in probability, but the rest of the pipeline thresholds and
    # reports these as probabilities, so map them through the sigmoid.
    return 1.0 / (1.0 + np.exp(-scores))


def fit_predict(
    graphs: list[AttentionGraph],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    spec: GraphSpec,
    params: CharmParams,
    cfg: TrainConfig,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Train on `train_idx`, early-stop on `val_idx`, score `val_idx` and `eval_idx`.

    Returns (validation scores, evaluation scores, training record, weights). The
    validation scores are what the caller tunes the decision threshold on, exactly as
    stage 3 does for every other detector; the weights are the early-stopped best epoch,
    on CPU, for `charm.checkpoint` to persist.
    """
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)

    model = CharmNet(
        n_channels=spec.n_channels,
        d_act=spec.d_act,
        hidden=params.hidden,
        n_layers=params.n_layers,
        dropout=params.dropout,
        batch_norm=params.batch_norm,
        residual=params.residual,
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameter_groups(params.weight_decay, params.act_weight_decay), lr=params.lr
    )
    rng = np.random.default_rng(cfg.seed)
    steps_per_epoch = max(1, len(_make_batches(graphs, train_idx, params, cfg)))
    scheduler, schedule_on = _scheduler(
        params.scheduler, optimiser, steps_per_epoch * params.max_epochs
    )

    # Class balance, matching `class_weight="balanced"` on every other detector here.
    positives = float(y[train_idx].sum())
    pos_weight = torch.tensor(
        [(len(train_idx) - positives) / max(positives, 1.0)], device=device, dtype=torch.float32
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    y_train = torch.from_numpy(y.astype(np.float32)).to(device)

    best_auroc, best_state, best_epoch, stale = -1.0, None, -1, 0
    history = []
    for epoch in range(params.max_epochs):
        model.train()
        order = train_idx[rng.permutation(len(train_idx))]
        total_loss, n_seen = 0.0, 0
        for chunk in _make_batches(graphs, order, params, cfg):
            batch = collate([graphs[i] for i in chunk], device)
            # BatchNorm cannot compute statistics over a single sample; a trailing
            # batch of one would abort the epoch.
            if batch.n_graphs < 2 and params.batch_norm:
                continue
            loss = criterion(model(batch), y_train[torch.from_numpy(chunk).to(device)])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            if schedule_on == "step":
                scheduler.step()
            total_loss += float(loss.detach()) * len(chunk)
            n_seen += len(chunk)

        val_scores = _predict(model, graphs, val_idx, params, cfg, device)
        auroc = (
            float(roc_auc_score(y[val_idx], val_scores))
            if len(np.unique(y[val_idx])) > 1
            else 0.5
        )
        if schedule_on == "epoch":
            scheduler.step(auroc)
        history.append({"epoch": epoch, "loss": round(total_loss / max(n_seen, 1), 4),
                        "val_auroc": round(auroc, 4)})

        if auroc > best_auroc:
            best_auroc, best_epoch, stale = auroc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= params.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores = _predict(model, graphs, val_idx, params, cfg, device)
    eval_scores = _predict(model, graphs, eval_idx, params, cfg, device)

    # Detached to CPU so the caller can checkpoint it without pinning GPU memory.
    weights = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    record = {
        "val_auroc": round(best_auroc, 4),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "history": history,
    }
    del model, best_state
    torch.cuda.empty_cache()
    return val_scores, eval_scores, record, weights


#: Default grid. Table 9's full space is 576 points (lr x scheduler x dropout x hidden
#: x depth x weight-decay x batchnorm x residual), which at 5 seeds x 5 outer folds
#: would mean 14,400 network fits — weeks of GPU. The repo
#: already precedents pinning a grid after a sweep (see `lapeigvals` in detectors.py),
#: so the default varies the two axes the paper's own ablations show matter most
#: (capacity and depth) and pins the rest at Table 9 values. `--grid full` restores the
#: published search space.
DEFAULT_GRID = [
    CharmParams(hidden=64, n_layers=2),
    CharmParams(hidden=128, n_layers=2),
    CharmParams(hidden=64, n_layers=3),
    CharmParams(hidden=128, n_layers=3),
]


def full_grid() -> list[CharmParams]:
    """Table 9's published search space, in full."""
    from itertools import product

    return [
        CharmParams(
            lr=lr, scheduler=sched, dropout=dropout, hidden=hidden,
            n_layers=layers, weight_decay=wd, batch_norm=bn, residual=res,
        )
        for lr, sched, dropout, hidden, layers, wd, bn, res in product(
            (1e-3, 5e-4), ("plateau", "cosine"), (0.25, 0.5), (32, 64, 128),
            (1, 2, 3), (0.0, 1e-3), (True, False), (True, False),
        )
    ]


GRIDS = {"default": lambda: list(DEFAULT_GRID), "full": full_grid}
