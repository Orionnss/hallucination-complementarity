"""Pinned identity of the official ICR Probe code, the modules the adapter runs, and patches."""

from __future__ import annotations

import contextlib

import torch

from ..upstream import UpstreamSpec, load_upstream

SPEC = UpstreamSpec(
    name="icr_probe",
    url="https://github.com/XavierZhang2002/ICR_Probe",
    commit="40ec490e762cadbac6bcefdc24a8f0d5974e8448",
    files={
        "src/icr_score.py": "e0af974d382f7219d5d1fbbc7b465a66a8425cc12fb5bbc8d82bea086ce90ce2",
        "src/utils.py": "e87d0e360dd01efa49dc4bcb852f6b1b2c88dfe5f18897c762741bcdd2bc5406",
        "src/icr_probe.py": "fed2e85beed91715af61adf5cef3c339c8495114bd77c10b6e1c958a59fe6ca2",
        "src/config.py": "4b6785659913fdc7b42ac05440f374152007b46dc8a3334f7bc24b24fa5b4cfb",
    },
    # `src/__init__.py` is empty; created as a namespace so nothing else in the checkout
    # can be imported by accident.
    packages=("src",),
)

#: `compute_icr` settings and ICRScore thresholds of the README usage example
#: (README.md, "1. Compute ICR Scores"). top_p overrides top_k when both are set
#: (icr_score.py:226), so k = int(0.1 * number of tokens).
ICR_SCORE_KWARGS = {"skew_threshold": 0, "entropy_threshold": 1e5}
COMPUTE_ICR_KWARGS = {"top_k": 20, "top_p": 0.1, "pooling": "mean",
                      "attention_uniform": False, "hidden_uniform": False,
                      "use_induction_head": True}


class _CudaShim:
    """Patch P1: `torch.cuda` as seen by icr_score.py, safe when icr_device is the CPU.

    icr_score.py wraps every cache release in `with torch.cuda.device(self.icr_device)`,
    which raises for a CPU device. On a CUDA device every call goes to the real
    torch.cuda, unchanged. On CPU the context is a no-op and empty_cache is skipped.
    Device placement only; no computation passes through here.
    """

    def __getattr__(self, name):
        return getattr(torch.cuda, name)

    @staticmethod
    def device(device):
        if device is None or torch.device(device).type != "cuda":
            return contextlib.nullcontext()
        return torch.cuda.device(device)

    @staticmethod
    def empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _TorchShim:
    def __init__(self) -> None:
        self.cuda = _CudaShim()

    def __getattr__(self, name):
        return getattr(torch, name)


def load() -> dict:
    """Import the official modules and apply the load-time patches."""
    mods = load_upstream(SPEC, ["src.icr_score", "src.utils", "src.icr_probe", "src.config"])
    out = {name.rsplit(".", 1)[1]: module for name, module in mods.items()}
    if not isinstance(out["icr_score"].torch, _TorchShim):
        out["icr_score"].torch = _TorchShim()          # P1, module namespace only
    return out
