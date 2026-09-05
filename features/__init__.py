"""Feature extraction techniques for the FreshSight comparative study.

Each technique implements the :class:`~features.base.FeatureExtractor`
interface and nothing else, so the harness can call all three
polymorphically without knowing which is which.

* ``t1_dominant_colour`` - MPEG-7 dominant colour descriptor (37 dimensions)
* ``t2_glcm`` - GLCM texture descriptors (40 dimensions)
* ``t3_morphological`` - multiscale morphological descriptors (36 dimensions)
"""

from features.base import (
    FeatureExtractionError,
    FeatureExtractor,
    require_non_empty_mask,
    validate_vector,
)
from features.t1_dominant_colour import DominantColourExtractor
from features.t2_glcm import GLCMExtractor
from features.t3_morphological import T3MorphologicalExtractor

__all__ = [
    "FeatureExtractor",
    "FeatureExtractionError",
    "validate_vector",
    "require_non_empty_mask",
    "DominantColourExtractor",
    "GLCMExtractor",
    "T3MorphologicalExtractor",
]
