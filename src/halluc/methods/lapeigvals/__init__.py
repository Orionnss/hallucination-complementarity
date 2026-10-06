"""LapEigvals (Binkowski et al., EMNLP 2025, arXiv 2502.17598), adapted from the official code.

See METHOD_CARD.md in this directory for the paper-to-code mapping and the deviations.
"""

from .adapter import K_MAX, OfficialSpectralFeatures, official_input
from .upstream_spec import SPEC, load

__all__ = ["K_MAX", "OfficialSpectralFeatures", "official_input", "SPEC", "load"]
