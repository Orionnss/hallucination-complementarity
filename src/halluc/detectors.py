"""Detection methods, as feature-block selection plus an estimator.

Every detector reads the blocks stage 1 already computed, so adding one means declaring
which blocks it wants and how to turn them into a design matrix — no re-extraction.

PCA policy (locked): LapEigvals uses PCA because that is part of the published method
and 16,000 dims would otherwise swamp a logistic regression. The attention baseline does
not, so the pair isolates the Laplacian transform rather than the dimensionality
reduction. Every transform is fit on training folds only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .registry import Registry

#: Candidate probe depths for SAPLMA, as fractions of model depth. The original paper
#: picks a middle layer; the exact one is model-specific, so it is tuned on the inner
#: fold. Fractions rather than absolute indices because generators differ in depth —
#: Llama-3.2-3B has 28 layers, Gemma-3-12B has 48 — and a hardcoded index either crashes
#: or silently probes a different relative depth.
SAPLMA_DEPTH_FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
#: Depths the union detectors probe, likewise relative.
UNION_DEPTH_FRACTIONS = (0.5, 0.6, 0.7)


def depth_layers(n_layers: int, fractions=SAPLMA_DEPTH_FRACTIONS) -> tuple[int, ...]:
    """Absolute layer indices for the given depth fractions.

    At n_layers=40 (Qwen3-14B) this reproduces the original (12, 16, 20, 24, 28, 32, 40).
    """
    return tuple(sorted({max(1, min(n_layers, round(f * n_layers))) for f in fractions}))


def _flatten(array: np.ndarray) -> np.ndarray:
    return array.reshape(array.shape[0], -1).astype(np.float64)


@dataclass
class Detector:
    """A named detector: which blocks it reads, how it builds X, and its estimator."""

    name: str
    blocks: tuple[str, ...]
    grid: list[dict] = field(default_factory=lambda: [{}])

    def matrix(self, data: dict[str, np.ndarray], params: dict) -> np.ndarray:
        return _flatten(data[self.blocks[0]])

    def estimator(self, params: dict, seed: int):
        raise NotImplementedError


def _logreg(C: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=C, max_iter=3000, class_weight="balanced", random_state=seed
    )


_C_GRID = (0.003, 0.03, 0.3, 3.0)


@dataclass
class LinearDetector(Detector):
    """Standardise then logistic regression."""

    def estimator(self, params: dict, seed: int):
        return Pipeline(
            [("scale", StandardScaler()), ("clf", _logreg(params["C"], seed))]
        )


@dataclass
class PCALinearDetector(Detector):
    """Standardise, reduce with PCA, then logistic regression (LapEigvals)."""

    def estimator(self, params: dict, seed: int):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=params["n_components"], random_state=seed)),
                ("clf", _logreg(params["C"], seed)),
            ]
        )


@dataclass
class SaplmaDetector(Detector):
    """MLP probe on one layer's last-token hidden state."""

    def matrix(self, data: dict[str, np.ndarray], params: dict) -> np.ndarray:
        # Clamp defensively: hidden_states has n_layers+1 entries, and a grid built for
        # a deeper model must not index past the end.
        layer = min(params["layer"], data["saplma"].shape[1] - 1)
        return data["saplma"][:, layer, :].astype(np.float64)

    def estimator(self, params: dict, seed: int):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128, 64),
                        max_iter=600,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=seed,
                    ),
                ),
            ]
        )


@dataclass
class IcrDetector(Detector):
    """Four-layer MLP on the pooled ICR layer profile."""

    def matrix(self, data: dict[str, np.ndarray], params: dict) -> np.ndarray:
        # Both poolings are persisted; which one the paper uses is unconfirmed, so it is
        # selected on the inner fold rather than assumed.
        if params["pooling"] == "both":
            return np.concatenate([_flatten(data["icr"]), _flatten(data["icr_mean"])], axis=1)
        return _flatten(data[params["pooling"]])

    def estimator(self, params: dict, seed: int):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32),
                        max_iter=1500,
                        early_stopping=True,
                        n_iter_no_change=30,
                        random_state=seed,
                    ),
                ),
            ]
        )


@dataclass
class UnionDetector(Detector):
    """Concatenation of every method's features.

    `equal=False` z-scores each block and concatenates raw, so LapEigvals contributes
    16,000 of ~21,200 columns. `equal=True` reduces each block to the same width first,
    so a win cannot be explained by one block's dimensionality alone. Both are reported.
    """

    equal: bool = False

    def matrix(self, data: dict[str, np.ndarray], params: dict) -> np.ndarray:
        layer = min(params["layer"], data["saplma"].shape[1] - 1)
        parts = [_flatten(data["lapeigvals"]), _flatten(data["attn_baseline"]),
                 data["saplma"][:, layer, :].astype(np.float64),
                 _flatten(data["svd_baseline"]), _flatten(data["icr"])]
        return np.concatenate(parts, axis=1)

    def block_slices(self, data: dict[str, np.ndarray], params: dict) -> list[slice]:
        widths = [
            _flatten(data["lapeigvals"]).shape[1],
            _flatten(data["attn_baseline"]).shape[1],
            data["saplma"].shape[2],
            _flatten(data["svd_baseline"]).shape[1],
            _flatten(data["icr"]).shape[1],
        ]
        edges = np.cumsum([0, *widths])
        return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]

    def estimator(self, params: dict, seed: int):
        if not self.equal:
            return Pipeline(
                [("scale", StandardScaler()), ("clf", _logreg(params["C"], seed))]
            )
        from .eval.blockpca import BlockPCA

        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("blockpca", BlockPCA(slices=params["slices"], n_components=params["per_block"],
                                      random_state=seed)),
                ("clf", _logreg(params["C"], seed)),
            ]
        )


DETECTORS: Registry[Detector] = Registry("detector")


def _grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, values)) for values in product(*(axes[k] for k in keys))]


def build_detectors(n_layers: int = 40) -> dict[str, Detector]:
    """All detectors for a run. Registered lazily so the grids stay in one place.

    `n_layers` is the generator's depth, so SAPLMA's probe-layer grid adapts to the
    model instead of assuming Qwen3-14B's 40 layers.
    """
    saplma_layers = depth_layers(n_layers)
    union_layers = depth_layers(n_layers, UNION_DEPTH_FRACTIONS)
    return {
        # Fixed rather than searched: these values come from a prior experiment. The
        # earlier grid selected its own boundary on both axes (n_components=256 was the
        # maximum offered, C=0.003 the minimum), so it was under-tuned; 512 components
        # with lighter regularisation supersedes it.
        # NOTE: k stays at 10. k is fixed at extraction time — stage 1 persists only the
        # top 10 eigenvalues per head — so changing it requires re-extraction, not a
        # change here.
        "lapeigvals": PCALinearDetector(
            name="lapeigvals",
            blocks=("lapeigvals",),
            grid=[{"n_components": 512, "C": 1.0}],
        ),
        "attn_baseline": LinearDetector(
            name="attn_baseline", blocks=("attn_baseline",), grid=_grid(C=_C_GRID)
        ),
        "svd_baseline": LinearDetector(
            name="svd_baseline", blocks=("svd_baseline",), grid=_grid(C=_C_GRID)
        ),
        "saplma": SaplmaDetector(
            name="saplma", blocks=("saplma",), grid=_grid(layer=saplma_layers)
        ),
        "icr": IcrDetector(
            name="icr", blocks=("icr", "icr_mean"), grid=_grid(pooling=("icr", "icr_mean", "both"))
        ),
        "union_raw": UnionDetector(
            name="union_raw",
            blocks=("lapeigvals", "attn_baseline", "saplma", "svd_baseline", "icr"),
            grid=_grid(layer=union_layers, C=_C_GRID),
            equal=False,
        ),
        "union_equal": UnionDetector(
            name="union_equal",
            blocks=("lapeigvals", "attn_baseline", "saplma", "svd_baseline", "icr"),
            grid=_grid(layer=union_layers, per_block=(128,), C=_C_GRID),
            equal=True,
        ),
    }
