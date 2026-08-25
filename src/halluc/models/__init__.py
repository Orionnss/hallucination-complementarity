"""Generators. Importing the submodules populates the GENERATORS registry."""

from .base import GENERATORS, Generation, Generator
from . import hf  # noqa: F401  (registration side effects)

__all__ = ["GENERATORS", "Generation", "Generator"]
