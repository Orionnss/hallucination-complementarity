"""ICR Probe (Zhang et al., ACL 2025, arXiv 2507.16488), adapted from the official code.

Importing this package registers the stage-1 extractor `icr_official`. See
METHOD_CARD.md in this directory for the paper-to-code mapping and the deviations.
"""

from .adapter import ICR_BLOCK, OfficialIcrFeatures
from .upstream_spec import SPEC, load

__all__ = ["ICR_BLOCK", "OfficialIcrFeatures", "SPEC", "load"]
