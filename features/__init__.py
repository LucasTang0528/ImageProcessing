"""Feature extraction techniques for the FreshSight comparative study.

Each technique implements the :class:`~features.base.FeatureExtractor`
interface and nothing else. The concrete techniques (``t1_colour``,
``t2_glcm``, ``t3_lbp_blemish``) arrive in Phase 2; Phase 1 defines only the
interface they must satisfy.
"""

from features.base import (
    FeatureExtractionError,
    FeatureExtractor,
    require_non_empty_mask,
    validate_vector,
)

__all__ = [
    "FeatureExtractor",
    "FeatureExtractionError",
    "validate_vector",
    "require_non_empty_mask",
]
