"""Judges. Importing the submodules populates the JUDGES registry."""

from .base import JUDGES, Judge, Label, Verdict, parse_label
from . import hf  # noqa: F401  (registration side effects)

__all__ = ["JUDGES", "Judge", "Label", "Verdict", "parse_label"]
