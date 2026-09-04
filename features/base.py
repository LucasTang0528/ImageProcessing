"""The shared interface every feature extraction technique must implement.

This module is the architectural guarantee behind the study's central claim.
The harness never imports a concrete technique; it calls objects satisfying
:class:`FeatureExtractor` polymorphically. A technique therefore sees exactly
two things - a preprocessed BGR image and a fruit mask - and returns exactly
one thing: a fixed-length 1-D vector. It has no access to the configuration,
the partition, the classifier, or any other technique's state, so it is not
merely a convention but structurally impossible for a technique to alter any
stage of the experiment other than feature extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class FeatureExtractionError(RuntimeError):
    """Raised when an extractor is handed an input it cannot describe."""


class FeatureExtractor(ABC):
    """Abstract base class for the three descriptor families.

    Concrete subclasses must set :attr:`name`, :attr:`short_name` and
    :attr:`dim`, and implement :meth:`extract_features`.
    """

    #: Full human-readable name, used in tables and plot titles.
    name: str = "unnamed technique"
    #: Short identifier such as ``"T1"``, used in filenames and CSV columns.
    short_name: str = "T?"
    #: Length of the returned vector. Fixed for the lifetime of the instance.
    dim: int = 0

    @abstractmethod
    def extract_features(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> np.ndarray:
        """Describe one segmented fruit as a fixed-length vector.

        Args:
            bgr_image: A preprocessed ``(224, 224, 3)`` ``uint8`` BGR image.
            fruit_mask: A ``(224, 224)`` ``uint8`` mask, 255 inside the fruit
                and 0 outside.

        Returns:
            A 1-D ``float64`` array of length :attr:`dim`, free of NaN and Inf.
        """

    def __call__(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> np.ndarray:
        """Call :meth:`extract_features` and validate what it returned.

        Validation is applied here rather than inside each technique, so that
        all three are held to the same contract by the same code.
        """
        vector = self.extract_features(bgr_image, fruit_mask)
        return validate_vector(vector, expected_dim=self.dim, technique=self.short_name)

    def describe(self) -> str:
        """One-line description of the technique and its dimensionality."""
        return f"{self.short_name}: {self.name} ({self.dim} dimensions)"

    @property
    def feature_names(self) -> Sequence[str]:
        """Names for each dimension, used for reporting.

        The default is positional. Concrete techniques should override this
        with meaningful names.
        """
        return [f"{self.short_name}_{index:03d}" for index in range(self.dim)]


def validate_vector(
    vector: np.ndarray,
    expected_dim: int,
    technique: str = "technique",
) -> np.ndarray:
    """Check a returned feature vector against the shared contract.

    A vector that is the wrong length, or that contains NaN or Inf, is a bug
    that would otherwise be trained on silently, so it is raised immediately.

    Args:
        vector: The candidate feature vector.
        expected_dim: The documented length for this technique.
        technique: Short name, used only in the error messages.

    Returns:
        The vector as a contiguous 1-D ``float64`` array.

    Raises:
        FeatureExtractionError: If any part of the contract is violated.
    """
    array = np.asarray(vector, dtype=np.float64).ravel()

    if array.size != expected_dim:
        raise FeatureExtractionError(
            f"{technique} returned {array.size} features but declares {expected_dim}"
        )
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        positions = np.flatnonzero(~np.isfinite(array))[:10].tolist()
        raise FeatureExtractionError(
            f"{technique} returned {bad} non-finite value(s) at indices {positions}"
        )
    return np.ascontiguousarray(array)


def require_non_empty_mask(fruit_mask: np.ndarray, technique: str = "technique") -> np.ndarray:
    """Return the boolean form of a mask, refusing an empty one.

    The harness already flags images whose mask falls outside the permitted
    coverage bounds, so an empty mask reaching an extractor means the guard was
    bypassed. Failing loudly here prevents a vector of zeros or NaNs from being
    learned from.

    Args:
        fruit_mask: A ``uint8`` or boolean mask.
        technique: Short name, used only in the error message.

    Returns:
        A boolean array of the same shape.

    Raises:
        FeatureExtractionError: If the mask selects no pixels.
    """
    boolean = np.asarray(fruit_mask).astype(bool)
    if not boolean.any():
        raise FeatureExtractionError(
            f"{technique} was given an empty fruit mask; the harness should have "
            f"flagged this image as a segmentation failure before extraction"
        )
    return boolean
