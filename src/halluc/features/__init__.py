"""Feature extractors. Importing the submodules populates the FEATURES registry."""

from .base import FEATURES, FeatureExtractor, ForwardTrace
from . import hidden, icr, spectral  # noqa: F401  (registration side effects)

__all__ = ["FEATURES", "FeatureExtractor", "ForwardTrace"]
