"""Native reader and grid registration for the official LapEigvals blocks.

The official probe is `hallucinations.probe_models.lr.train_logistic_regression`: PCA(512)
then logistic regression (C=1.0, class_weight="balanced", max_iter=2000), with **no
standardisation** before the PCA, and a decision by argmax, i.e. p > 0.5. Our earlier
`PCALinearDetector` standardised first, so it was not the published probe.

The upstream function fits and scores in one call, from a feature tensor and an index
split. `OfficialProbe` adapts that to the sklearn fit / predict_proba contract that stage
3's `run_fold` drives. The upstream function also computes test metrics from the labels it
is given. The adapter has no test labels (and must not have them), so it passes
placeholder labels and discards those metrics. The probabilities do not depend on them:
the model is fitted on the training rows only. Test F6b checks this.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from ...config import Config
from ...grid import BLOCKS, BlockSpec, Reader, READERS
from .adapter import ATTN_BLOCK, LAP_BLOCK, LEN_BLOCK
from .upstream_spec import PCA_DIM, TOP_K_EIGVALS, load


class OfficialProbe(BaseEstimator, ClassifierMixin):
    def __init__(self, pca_dim: int = PCA_DIM, random_seed: int = 42) -> None:
        self.pca_dim = pca_dim
        self.random_seed = random_seed

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y, dtype=np.int64)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_train, n_eval = len(self.X_), len(X)
        # Placeholder labels for the evaluation rows: both classes present, so the
        # upstream metric code (roc_auc_score) runs. Never seen by the fit.
        placeholder = np.arange(n_eval) % 2
        features = torch.from_numpy(np.concatenate([self.X_, X]))
        labels = torch.from_numpy(np.concatenate([self.y_, placeholder]))
        split = {"train_idx": torch.arange(n_train),
                 "test_idx": torch.arange(n_train, n_train + n_eval)}
        lr = load()["lr"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lr.train_logistic_regression(
                features=features, labels=labels, split=split,
                pca_dim=self.pca_dim, use_cuda=False, random_seed=self.random_seed,
            )
        return np.asarray(result["test_proba"], dtype=np.float64)


def native_argmax(scores: np.ndarray) -> np.ndarray:
    """`test_proba.argmax(axis=-1)` on [1-p, p]: a tie at 0.5 goes to class 0."""
    return (scores > 0.5).astype(int)


READERS.register("lapeigvals_official")(lambda: Reader(
    name="lapeigvals_official",
    grid=[{}],
    build=lambda p, s: OfficialProbe(pca_dim=PCA_DIM, random_seed=s),
    native_decision=native_argmax,
    note="official PCA(512) + logistic regression, no standardisation, argmax decision",
))


def _source(cfg: Config, dataset: str):
    return cfg.stage_dir("methods", "lapeigvals_official", dataset)


def _k_grid(data: dict[str, np.ndarray]) -> list[dict]:
    """train_attn_vs_laplacian.py: keep every k no larger than the shortest item.

    Upstream takes the minimum over its whole dataset directory, train and test. Here it
    is the minimum over the drawn sample, train and test, which is the same rule. It reads
    sequence lengths only, never labels.
    """
    shortest = int(data[LEN_BLOCK].min())
    return [{"k": k} for k in TOP_K_EIGVALS if k <= shortest]


def _topk_view(block: str):
    def view(data: dict[str, np.ndarray], params: dict) -> np.ndarray:
        x = data[block][..., : params["k"]]  # [n, L, H, k]: layer_idx=None, all layers
        return x.reshape(len(x), -1)
    return view


for _block in (LAP_BLOCK, ATTN_BLOCK):
    BLOCKS.register(_block)(lambda _b=_block: BlockSpec(
        name=_b, ftype="vector", arrays=(_b, LEN_BLOCK), source=_source,
        view_grid=_k_grid, view=_topk_view(_b),
    ))
