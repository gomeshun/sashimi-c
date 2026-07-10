"""Pinned ITAMAE core snapshot for cross-repository migration CI.

Source commit: 0030c5b3fac6fb532986e1eb2da3c1cc10063e57

This module is a temporary CI fixture while the canonical ITAMAE repository is
private and the package is not available from PyPI. Runtime code must never
import it directly. Migration workflows expose its objects through temporary
``itamae`` package namespaces. Remove the fixture after a public package release.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.optimize import brentq

_G_MPC_KMS2_MSUN = 4.30091e-9
_KMS_MPC_TO_GYR_INV = 1.0227121650537077e-3


@dataclass(frozen=True, slots=True)
class NativeFlatLCDM:
    """Flat-LCDM background with NumPy/SciPy only."""

    omega_m0: float = 0.315
    h: float = 0.674

    @property
    def identifier(self) -> str:
        """Return a stable identifier including cosmological parameters."""
        return f"native-flatlcdm:Om={self.omega_m0:.8g}:h={self.h:.8g}"

    @property
    def omega_lambda0(self) -> float:
        """Return the present-day dark-energy density fraction."""
        return 1.0 - self.omega_m0

    def e2(self, z):
        """Return the dimensionless squared expansion rate ``E(z)^2``."""
        z = np.asarray(z, dtype=float)
        return self.omega_m0 * (1.0 + z) ** 3 + self.omega_lambda0

    def H(self, z):
        """Return the Hubble rate in km s^-1 Mpc^-1."""
        return 100.0 * self.h * np.sqrt(self.e2(z))

    def rho_crit(self, z):
        """Return critical density in Msun Mpc^-3."""
        return 3.0 * self.H(z) ** 2 / (8.0 * np.pi * _G_MPC_KMS2_MSUN)

    def rho_m(self, z):
        """Return physical matter density in Msun Mpc^-3."""
        return self.omega_m0 * self.rho_crit(0.0) * (1.0 + np.asarray(z)) ** 3

    def omega_m(self, z):
        """Return the redshift-dependent matter density fraction."""
        return self.omega_m0 * (1.0 + np.asarray(z)) ** 3 / self.e2(z)

    def growth_factor(self, z):
        """Return the Carroll-Press-Turner growth factor normalized at z=0."""
        z = np.asarray(z, dtype=float)
        om = self.omega_m(z)
        ol = 1.0 - om
        g = 2.5 * om / (
            om ** (4.0 / 7.0) - ol + (1.0 + om / 2.0) * (1.0 + ol / 70.0)
        )
        om0 = self.omega_m0
        ol0 = self.omega_lambda0
        g0 = 2.5 * om0 / (
            om0 ** (4.0 / 7.0) - ol0 + (1.0 + om0 / 2.0) * (1.0 + ol0 / 70.0)
        )
        return g / (g0 * (1.0 + z))

    def collapse_threshold(self, z):
        """Return the spherical-collapse threshold scaled by growth."""
        return 1.686 / self.growth_factor(z)

    def cosmic_time(self, z):
        """Return cosmic age in Gyr."""

        def one(zi: float) -> float:
            def integrand(zp: float) -> float:
                return 1.0 / ((1.0 + zp) * self.H(zp) * _KMS_MPC_TO_GYR_INV)

            return quad(integrand, zi, np.inf, epsabs=1e-9, epsrel=1e-9)[0]

        arr = np.asarray(z, dtype=float)
        out = np.vectorize(one, otypes=[float])(arr)
        return float(out) if out.ndim == 0 else out

    def lookback_time(self, z):
        """Return lookback time in Gyr."""
        return self.cosmic_time(0.0) - self.cosmic_time(z)


def gauss_hermite_lognormal(median, sigma_log10, order: int = 5):
    """Return nodes and normalized weights for base-10 log-normal scatter."""
    if order < 1:
        raise ValueError("Quadrature order must be positive.")
    median = np.asarray(median, dtype=float)
    sigma_log10 = np.asarray(sigma_log10, dtype=float)
    if np.any(median <= 0.0):
        raise ValueError("Median must be positive.")
    x, w = hermgauss(order)
    shape = (order,) + (1,) * median.ndim
    nodes = 10.0 ** (
        np.log10(median)[None, ...]
        + np.sqrt(2.0) * sigma_log10[None, ...] * x.reshape(shape)
    )
    weights = np.broadcast_to((w / np.sqrt(np.pi)).reshape(shape), nodes.shape)
    return nodes, weights


def nfw_mass_function(x):
    """Return ``ln(1+x)-x/(1+x)`` for nonnegative ``x``."""
    x = np.asarray(x, dtype=float)
    if np.any(x < 0.0):
        raise ValueError("NFW radius ratio must be nonnegative.")
    return np.log1p(x) - x / (1.0 + x)


def invert_nfw_mass_function(y):
    """Invert the monotonic NFW enclosed-mass function."""
    y = np.asarray(y, dtype=float)
    if np.any(y < 0.0):
        raise ValueError("Enclosed-mass function values must be nonnegative.")

    def one(value: float) -> float:
        if value == 0.0:
            return 0.0
        upper = max(1.0, np.exp(min(value + 1.0, 700.0)))
        while nfw_mass_function(upper) < value:
            upper *= 2.0
        return brentq(lambda x: float(nfw_mass_function(x) - value), 0.0, upper)

    out = np.vectorize(one, otypes=[float])(y)
    return float(out) if out.ndim == 0 else out


def shanks_transform(s0, s1, s2, *, tolerance: float = 1.0e-14):
    """Apply the stable three-term Shanks transformation."""
    s0, s1, s2 = np.broadcast_arrays(
        np.asarray(s0, dtype=float),
        np.asarray(s1, dtype=float),
        np.asarray(s2, dtype=float),
    )
    denominator = s2 - 2.0 * s1 + s0
    safe = np.abs(denominator) > tolerance
    return np.where(safe, s2 - (s2 - s1) ** 2 / denominator, s2)


@dataclass(frozen=True, slots=True)
class WeightedSubhaloCatalog:
    """Store aligned catalog columns and independent statistical weights."""

    columns: Mapping[str, np.ndarray]
    weights: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        shapes = {
            np.asarray(value).shape
            for value in [*self.columns.values(), *self.weights.values()]
        }
        if len(shapes) > 1:
            raise ValueError(f"All catalog arrays must share one shape; got {shapes}.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the common catalog shape."""
        source = next(
            iter(self.columns.values()), next(iter(self.weights.values()), np.array([]))
        )
        return np.asarray(source).shape

    @property
    def weight_final(self) -> np.ndarray:
        """Multiply all independent weight factors."""
        result = np.ones(self.shape, dtype=float)
        for value in self.weights.values():
            result *= np.asarray(value, dtype=float)
        return result

    def select(self, mask) -> "WeightedSubhaloCatalog":
        """Return a selected catalog while preserving metadata."""
        return WeightedSubhaloCatalog(
            columns={key: np.asarray(value)[mask] for key, value in self.columns.items()},
            weights={key: np.asarray(value)[mask] for key, value in self.weights.items()},
            metadata=self.metadata,
        )

    def weighted_sum(self, values) -> float:
        """Return the final-weighted sum of aligned values."""
        values = np.asarray(values, dtype=float)
        if values.shape != self.shape:
            raise ValueError("Values must have the catalog shape.")
        return float(np.sum(values * self.weight_final))
