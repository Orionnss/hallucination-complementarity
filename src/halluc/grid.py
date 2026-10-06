"""Feature blocks x readers: the grid of PROTOCOL.md step 3 (ADDING_A_METHOD.md §6).

`detectors.Detector` binds one feature matrix to one estimator, so a method can only ever
be scored with the probe it was registered with. Here the two halves are registered
separately:

  BlockSpec  where a feature block lives on disk, its type, and its *views*: the
             label-free choices that turn the stored array into a design matrix (top-k for
             LapEigvals, the layer for SAPLMA). Views are hyperparameters, selected on the
             inner split like any other.
  Reader     a classifier family and its search grid. A method's native probe is a reader
             like any other, so it can also read every other block.

`CellDetector` joins one of each back into the `Detector` interface, so a cell runs
through stage 3's `run_fold` unchanged: same inner split, same selection by AUROC, same
MCC threshold, same refit. Nothing in this module changes how existing detectors run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .detectors import Detector, SAPLMA_DEPTH_FRACTIONS, _C_GRID, _grid, depth_layers
from .registry import Registry

#: Feature types a block can declare (ADDING_A_METHOD.md §5.1). A view always produces a
#: 2-D matrix, so readers here accept "vector", "layered" and "scalar" alike; "sequence"
#: and "graph" need a named pooling adapter first.
FEATURE_TYPES = ("vector", "layered", "scalar", "sequence", "graph")
MATRIX_TYPES = frozenset({"vector", "layered", "scalar"})


@dataclass
class BlockSpec:
    name: str
    ftype: str
    #: Arrays to load for this block (the block itself plus any helper arrays its view
    #: needs, such as per-item sequence lengths).
    arrays: tuple[str, ...]
    #: Directory holding the shards, given (config, dataset).
    source: Callable[[Config, str], Path]
    #: Label-free view grid, computed from the drawn data. Must not read labels.
    view_grid: Callable[[dict[str, np.ndarray]], list[dict]] = lambda data: [{}]
    view: Callable[[dict[str, np.ndarray], dict], np.ndarray] | None = None

    def matrix(self, data: dict[str, np.ndarray], params: dict) -> np.ndarray:
        X = (self.view(data, params) if self.view is not None
             else data[self.arrays[0]].reshape(len(data[self.arrays[0]]), -1))
        X = np.asarray(X, dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError(f"block {self.name} view {params} has non-finite values")
        return X


@dataclass
class Reader:
    name: str
    grid: list[dict]
    build: Callable[[dict, int], object]
    accepts: frozenset[str] = MATRIX_TYPES
    #: Decision the method's own code makes from its score, if it has one. Recorded next
    #: to the harness decision, never used for the main results.
    native_decision: Callable[[np.ndarray], np.ndarray] | None = None
    note: str = ""

    def estimator(self, params: dict, seed: int):
        return self.build(params, seed)


BLOCKS: Registry[BlockSpec] = Registry("block")
READERS: Registry[Reader] = Registry("reader")


@dataclass
class CellDetector(Detector):
    """One (block, reader) cell, exposed through the Detector interface for run_fold."""

    spec: BlockSpec | None = None
    reader: Reader | None = None

    def matrix(self, data: dict[str, np.ndarray], params: dict) -> np.ndarray:
        return self.spec.matrix(data, params)

    def estimator(self, params: dict, seed: int):
        return self.reader.estimator(params, seed)

    def budget(self) -> dict:
        axes = sorted({k for p in self.grid for k in p})
        return {"n_configs": len(self.grid), "axes": axes}


def make_cell(block: str, reader: str, data: dict[str, np.ndarray]) -> CellDetector:
    spec, rd = BLOCKS.create(block), READERS.create(reader)
    if spec.ftype not in rd.accepts:
        raise ValueError(f"reader {reader} does not accept {spec.ftype} block {block}")
    views = spec.view_grid(data)
    grid = [{**v, **r} for v, r in product(views, rd.grid)]
    clash = set().union(*views) & set().union(*rd.grid)
    if clash:
        raise ValueError(f"view and reader parameters overlap: {sorted(clash)}")
    return CellDetector(name=f"{block}__{reader}", blocks=spec.arrays, grid=grid,
                        spec=spec, reader=rd)


# --------------------------------------------------------------------------- readers
# Shared readers. Each mirrors a probe already used in this study, so cells built from
# them are comparable with the existing tables.

def _logreg(C: float, seed: int) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)


def _register(name: str, **kwargs) -> Reader:
    reader = Reader(name=name, **kwargs)
    READERS.register(name)(lambda: reader)
    return reader


_register(
    "logreg",
    grid=_grid(C=_C_GRID),
    build=lambda p, s: Pipeline([("scale", StandardScaler()), ("clf", _logreg(p["C"], s))]),
    note="standardise + logistic regression; detectors.LinearDetector",
)
_register(
    "pca_logreg",
    grid=_grid(n_components=(128, 256), C=_C_GRID),
    build=lambda p, s: Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=p["n_components"], random_state=s)),
        ("clf", _logreg(p["C"], s)),
    ]),
    note="standardise + PCA + logistic regression; the reader under which SAPLMA wins",
)
_register(
    "mlp_saplma",
    grid=[{}],
    build=lambda p, s: Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=600,
                              early_stopping=True, n_iter_no_change=20, random_state=s)),
    ]),
    note="the published SAPLMA probe; detectors.SaplmaDetector",
)


def _stage3_lapeigvals(params: dict, seed: int):
    # Built by the stage-3 detector itself, so this reader cannot drift from the probe
    # behind the existing "lapeigvals" results.
    from .detectors import build_detectors

    return build_detectors()["lapeigvals"].estimator(params, seed)


_register(
    "lapeigvals_stage3",
    grid=[{"n_components": 512, "C": 1.0}],
    build=_stage3_lapeigvals,
    note="our stage-3 LapEigvals probe: standardise + PCA(512) + logistic regression (C=1)",
)


# ---------------------------------------------------------------------- stage-1 blocks
# The blocks stage 1 already wrote, so every new reader can also read them, and every new
# block can be compared with our earlier reimplementation under the same reader.

def _stage1(cfg: Config, dataset: str) -> Path:
    return cfg.stage_dir("stage1_extract", dataset)


for _name in ("lapeigvals", "icr"):
    BLOCKS.register(_name)(
        lambda _n=_name: BlockSpec(name=_n, ftype="vector", arrays=(_n,), source=_stage1)
    )


def _saplma_layers(data: dict[str, np.ndarray]) -> list[dict]:
    n_layers = data["saplma"].shape[1] - 1
    return [{"layer": layer} for layer in depth_layers(n_layers, SAPLMA_DEPTH_FRACTIONS)]


BLOCKS.register("saplma")(lambda: BlockSpec(
    name="saplma", ftype="layered", arrays=("saplma",), source=_stage1,
    view_grid=_saplma_layers,
    view=lambda data, p: data["saplma"][:, p["layer"], :],
))
