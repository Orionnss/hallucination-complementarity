"""QA datasets. Importing the submodules populates the DATASETS registry."""

from .base import DATASETS, QADataset, QAItem
from . import qa  # noqa: F401  (registration side effects)

__all__ = ["DATASETS", "QADataset", "QAItem"]
