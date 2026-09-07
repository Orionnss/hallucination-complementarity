"""CHARM: Catching HAllucinated Responses via learnable Message-passing.

arXiv 2509.24770 (ICLR 2026), Frasca et al., "Neural Message-Passing on Attention
Graphs for Hallucination Detection".

CHARM is the one detector in this repo that is **not** fed from stage 1's feature
shards. Its input is a whole attributed graph per sample — every surviving
(token, token) attention edge carrying an L*H-dimensional feature vector — which is
both far too large to persist and ragged in shape, so it cannot go in an `.npz`
sidecar the way `lapeigvals` or `saplma` do. Instead the graphs are built inside an
extraction loop, held in RAM, and the GNN is fitted from that cache.

Everything downstream of the graph is deliberately identical to stage 3: the same
per-seed draw, the same grouped, (dataset, label)-stratified folds, the same inner
split for hyperparameters and threshold. Only the feature pipeline differs, so a
CHARM-vs-other-method comparison isolates the method and not the protocol.
"""

from .checkpoint import FitStore, seed_fingerprint
from .graph import AttentionGraph, GraphSpec, build_graph
from .model import CharmNet
from .train import CharmParams, fit_predict

__all__ = [
    "AttentionGraph",
    "FitStore",
    "seed_fingerprint",
    "GraphSpec",
    "build_graph",
    "CharmNet",
    "CharmParams",
    "fit_predict",
]
