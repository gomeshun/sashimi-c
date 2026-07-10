"""Pinned ITAMAE callable variance adapter for migration CI.

Source commit: 94d58dd2e56ad95e9212dc1f34d5c21cdb4df9a6
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CallableVarianceModel:
    """Wrap existing SASHIMI variance functions behind the ITAMAE interface."""

    identifier: str
    sigma_function: Callable[[Any, Any], Any]
    derivative_function: Callable[[Any, Any], Any]

    def sigma(self, mass: Any, z: Any = 0.0) -> np.ndarray:
        """Return the wrapped rms fluctuation."""
        return np.asarray(self.sigma_function(mass, z), dtype=float)

    def variance(self, mass: Any, z: Any = 0.0) -> np.ndarray:
        """Return the square of the wrapped rms fluctuation."""
        sigma = self.sigma(mass, z)
        return sigma * sigma

    def dvariance_dmass(self, mass: Any, z: Any = 0.0) -> np.ndarray:
        """Return the wrapped derivative of variance with respect to mass."""
        return np.asarray(self.derivative_function(mass, z), dtype=float)


__all__ = ["CallableVarianceModel"]
