"""Feature extraction techniques for the FreshSight comparative study.

Each technique implements the :class:`~features.base.FeatureExtractor`
interface and nothing else, so the harness can call all three
polymorphically without knowing which is which.

* ``t1_dominant_colour`` - MPEG-7 dominant colour descriptor (37 dimensions)
* ``t2_glcm`` - GLCM texture descriptors (40 dimensions, pending)
* ``t3_morphology`` - multiscale morphological descriptors (36, pending)
"""

from features.base import (
    FeatureExtractionError,
    FeatureExtractor,
    require_non_empty_mask,
    validate_vector,
)
from features.t1_dominant_colour import DominantColourExtractor

__all__ = [
    "FeatureExtractor",
    "FeatureExtractionError",
    "validate_vector",
    "require_non_empty_mask",
    "DominantColourExtractor",
]
