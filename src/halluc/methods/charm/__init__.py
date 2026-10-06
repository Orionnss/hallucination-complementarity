"""CHARM (Frasca et al., ICLR 2026, arXiv 2509.24770), adapted from the official code.

Importing this package registers the stage-1 extractor `charm_official`. See
METHOD_CARD.md in this directory for the paper-to-code mapping and the deviations.
"""

from .adapter import BLOCKS, OfficialCharmGraph
from .upstream_spec import SPEC, load

__all__ = ["BLOCKS", "OfficialCharmGraph", "SPEC", "load"]
