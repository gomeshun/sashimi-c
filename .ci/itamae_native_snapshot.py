"""Pinned copy of ITAMAE NativeFlatLCDM for cross-repository migration CI.

Source commit: 0030c5b3fac6fb532986e1eb2da3c1cc10063e57

This file is a temporary CI fixture while the canonical ITAMAE repository is
private and the package is not yet available from PyPI.  Runtime code must not
import this module.  Remove it once CI can install the published ITAMAE package.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

_G_MPC_KMS2_MSUN = 4.30091e-9
_KMS_MPC_TO_GYR_INV = 1.0227121650537077e-3


@dataclass(frozen=True, slots=True)
class NativeFlatLCDM:
    """Flat-LCDM background with NumPy/SciPy only.

    Parameters
    ----------
    omega_m0
        Present-day matter density fraction.
    h
        Reduced Hubble constant.
    """

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
        """Return the Carroll-Press-Turner growth approximation normalized at z=0."""
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
        """Return the spherical-collapse threshold scaled by the growth factor."""
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
