"""Pinned identity of the official LapEigvals code, and the modules the adapter runs."""

from __future__ import annotations

import numpy as np

from ..upstream import UpstreamSpec, load_upstream

SPEC = UpstreamSpec(
    name="lapeigvals",
    url="https://github.com/graphml-lab-pwr/lapeigvals",
    commit="74f885c69399ed525f4eecd2574c218fac7235d7",
    files={
        # Feature path, as called by hallucinations/llm/feature_storage.py and
        # scripts/probes/train_attn_vs_laplacian.py.
        "hallucinations/features/attention_weights.py":
            "83f57708353e9112e82064db8b02aaf018fe340192b4af1ac037b9c2d88f9846",
        "hallucinations/features/processing.py":
            "797f778b74008f6d07ffbdb696e5da7367b79bcd728bf68a1288f4fa1474106e",
        "hallucinations/features/attn_feats.py":
            "b6cf876926004bef6d1c9ba61c10a6e0a3e7558cfb0cd5c604a3f9d40bc1b18f",
        # Probe.
        "hallucinations/probe_models/lr.py":
            "3e354ca6f3140f61be4d35dfb782b223eb16f056f501c56efc442c99191b5978",
        "hallucinations/probe_models/commons.py":
            "ad8038bff046078470fb3a33bc65d7ce95f6095914b6fe12ce17c882cea2722c",
        # Not imported, but its constants (TOP_K_EIGVALS, RANDOM_SEED) and its call
        # sequence are what the adapter reproduces, so a change to it must be noticed.
        "scripts/probes/train_attn_vs_laplacian.py":
            "981bab6c5cf123dd09c0b4b2a3c1acbe0991d53365f7ac7c0ebb8f1821ee958a",
    },
    # `hallucinations/__init__.py` sets torch.set_float32_matmul_precision("high"), a
    # process-wide side effect on every other method; `features/__init__.py` imports the
    # hidden-state stack. Neither is needed by the functions the adapter calls.
    packages=(
        "hallucinations",
        "hallucinations.features",
        "hallucinations.probe_models",
        "hallucinations.utils",
    ),
    # Imported at module level by attention_weights.py, used only by its shard-loading
    # helpers (yield_stacked_attentions), which read the upstream directory layout.
    stubs={
        "hallucinations.dirs": ("DatasetDir",),
        "hallucinations.utils.misc": ("load_and_resolve_config",),
    },
)

#: Constants from scripts/probes/train_attn_vs_laplacian.py and dvc.yaml at SPEC.commit.
#: The paper (arXiv v2, Section 4.2) lists k in {5, 10, 20, 50, 100}; the code sweeps 25,
#: not 20. The code produced the published numbers, so it is the default.
TOP_K_EIGVALS = (5, 10, 25, 50, 100)
PCA_DIM = 512  # dvc.yaml, stage train_attn_vs_laplacian_pca: --pca-dim 512
RANDOM_SEED = 42


def _patch_p1_sklearn_scalar_metrics(lr_module) -> None:
    """Patch P1 (METHOD_CARD.md): dependency version, not computation.

    `train_logistic_regression` calls `.item()` on the results of `roc_auc_score` and
    `average_precision_score`. Under the pinned scikit-learn 1.6.1 they return
    np.float64; under ours (1.9) they return a Python float, which has no `.item()`. The
    names are rebound inside the upstream module's namespace only, to wrappers that
    restore the 1.6.1 return type. The file on disk is unchanged (its hash still
    verifies), and the wrapped calls compute train/test metrics that the adapter
    discards; the fit and the probabilities do not pass through them.
    """
    if getattr(lr_module, "_halluc_patched", False):
        return
    for name in ("roc_auc_score", "average_precision_score"):
        original = getattr(lr_module, name)
        setattr(lr_module, name, lambda *a, _f=original, **k: np.float64(_f(*a, **k)))
    lr_module._halluc_patched = True


def load() -> dict:
    """Import the official modules. Returns {short name: module}."""
    mods = load_upstream(
        SPEC,
        [
            "hallucinations.features.processing",
            "hallucinations.features.attention_weights",
            "hallucinations.features.attn_feats",
            "hallucinations.probe_models.lr",
        ],
    )
    out = {name.rsplit(".", 1)[1]: module for name, module in mods.items()}
    _patch_p1_sklearn_scalar_metrics(out["lr"])
    return out
