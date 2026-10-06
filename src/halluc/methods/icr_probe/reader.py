"""Native reader and grid registration for the official ICR Probe blocks.

The official probe is `src.utils.ICRProbe`: an MLP [L] -> 128 -> 64 -> 32 -> 1 with batch
norm, dropout 0.3, leaky ReLU and a sigmoid. `src.icr_probe.ICRProbeTrainer` trains it:
BCE, Adam (lr 1e-3, weight decay 1e-5), ReduceLROnPlateau on the validation loss
(factor 0.5, patience 5), 100 epochs, batch size 16 (`src.config.Config`), and keeps the
weights of the epoch with the lowest validation loss (`save_model`).

The upstream trainer is incomplete: `setup_data` calls `_load_data` and
`_create_data_loaders`, which do not exist, and `_validate_epoch` reads
`config.halu_threshold`, which `Config` does not define. So this reader supplies only
those missing parts and calls the rest unchanged:

  * the loaders: a validation split of `Config.test_size` (0.2) taken from the training
    rows, stratified on the label (rule: a reader takes its validation set from its own
    training rows); training batches of 16, shuffled; the last incomplete training batch
    dropped, because BatchNorm cannot train on one row; validation in one batch, which
    eval-mode BatchNorm makes equivalent to any batching
  * patch P2: `halu_threshold = 0.5` on the config object; it only feeds the logged
    precision/recall, never the fit or the model choice
  * `setup_model`, `train`, `_train_epoch`, `_validate_epoch`, `save_model` are upstream
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ...config import Config
from ...grid import BLOCKS, BlockSpec, Reader, READERS
from .adapter import ICR_BLOCK
from .upstream_spec import load


class OfficialIcrProbe(BaseEstimator, ClassifierMixin):
    def __init__(self, random_seed: int = 0, device: str | None = None) -> None:
        self.random_seed = random_seed
        self.device = device

    def _config(self, save_dir: str):
        cfg = load()["config"].Config()
        cfg.save_dir = save_dir
        cfg.halu_threshold = 0.5  # P2: missing upstream attribute, logging only
        return cfg

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        up = load()
        logging.getLogger(up["icr_probe"].__name__).setLevel(logging.WARNING)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(tmp)
            tr, va = train_test_split(np.arange(len(y)), test_size=cfg.test_size,
                                      stratify=y, random_state=self.random_seed)
            torch.manual_seed(self.random_seed)
            gen = torch.Generator().manual_seed(self.random_seed)
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                batch_size=cfg.batch_size, shuffle=True, drop_last=True, generator=gen)
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X[va]), torch.from_numpy(y[va])),
                batch_size=len(va))
            trainer = up["icr_probe"].ICRProbeTrainer(None, train_loader, val_loader, cfg)
            if self.device is not None:
                trainer.device = torch.device(self.device)
            trainer.setup_model()
            trainer.train()
            # The upstream trainer saves the best epoch to save_dir; read it back.
            state = torch.load(Path(tmp) / "model.pth", map_location=trainer.device,
                               weights_only=True)
        trainer.model.load_state_dict(state)
        self.model_ = trainer.model.eval()
        self.device_ = trainer.device
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device_)
        p = self.model_(X).squeeze(-1).double().cpu().numpy()
        return np.stack([1.0 - p, p], axis=1)


def native_round(scores: np.ndarray) -> np.ndarray:
    """`outputs.round()` in `_validate_epoch`: a tie at 0.5 rounds to 0 (half to even)."""
    return np.round(scores).astype(int)


READERS.register("icr_official")(lambda: Reader(
    name="icr_official",
    grid=[{}],
    build=lambda p, s: OfficialIcrProbe(random_seed=s),
    native_decision=native_round,
    note="official ICRProbe MLP, upstream trainer, best epoch by validation loss",
))


def _source(cfg: Config, dataset: str):
    return cfg.stage_dir("methods", "icr_official", dataset)


def _view(data: dict[str, np.ndarray], params: dict) -> np.ndarray:
    # An empty answer has no response token, so its ICR is NaN (adapter). The upstream
    # probe asserts finite input; such rows read 0, i.e. "no divergence" (METHOD_CARD D4).
    return np.nan_to_num(data[ICR_BLOCK].astype(np.float64), nan=0.0)


BLOCKS.register(ICR_BLOCK)(lambda: BlockSpec(
    name=ICR_BLOCK, ftype="vector", arrays=(ICR_BLOCK,), source=_source, view=_view))
