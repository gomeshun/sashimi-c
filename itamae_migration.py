"""ITAMAE compatibility layer for incremental SASHIMI-C migration.

This module keeps the legacy SASHIMI-C implementation untouched while allowing
selected classes to use an ITAMAE cosmology backend. It is intentionally an
adapter rather than a replacement: regression tests can compare both execution
paths before shared implementations are removed from ``sashimi_c.py``.
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from itamae.cosmology import NativeFlatLCDM
from sashimi_c import halo_model

_Base = TypeVar("_Base", bound=type)


class ItamaeCosmologyMixin:
    """Delegate background-cosmology calculations to ITAMAE.

    Parameters
    ----------
    *args
        Positional arguments forwarded to the legacy SASHIMI-C class.
    cosmology_backend : object, optional
        ITAMAE-compatible cosmology backend. When omitted, a
        :class:`itamae.cosmology.NativeFlatLCDM` instance is constructed from
        the legacy object's ``OmegaM`` and ``h`` values.
    **kwargs
        Keyword arguments forwarded to the legacy SASHIMI-C class.

    Notes
    -----
    SASHIMI-C uses Mpc, solar mass, and seconds as its floating-point base
    units. ITAMAE returns Hubble rates in km s^-1 Mpc^-1 and densities in
    Msun Mpc^-3, so this mixin performs the explicit conversion at the adapter
    boundary.

    The historical SASHIMI constants use rounded values for the Mpc, solar
    mass, and Newton constant. Their implied critical-density normalization
    differs from ITAMAE by roughly three parts in ten thousand. The adapter
    retains this legacy normalization while taking the redshift dependence from
    ITAMAE. This prevents a purely conventional constant update from changing
    the physical regression baseline during migration.
    """

    def __init__(self, *args: Any, cosmology_backend: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        backend = cosmology_backend or NativeFlatLCDM(omega_m0=self.OmegaM, h=self.h)
        self.itamae_cosmology = backend

        # Many existing formulae access these legacy attributes directly, so
        # synchronize them with the selected backend before deriving H0.
        if hasattr(backend, "omega_m0"):
            self.OmegaM = float(backend.omega_m0)
            self.OmegaL = 1.0 - self.OmegaM
        if hasattr(backend, "h"):
            self.h = float(backend.h)

        self.H0 = float(np.asarray(backend.H(0.0))) * self.km / self.s / self.Mpc

        backend_rho0 = (
            float(np.asarray(backend.rho_crit(0.0))) * self.Msun / self.Mpc**3
        )
        legacy_rho0 = 3.0 * self.H0**2 / (8.0 * np.pi * self.G)
        self._rho_crit_scale = legacy_rho0 / backend_rho0
        self.rhocrit0 = legacy_rho0

    def Hubble(self, z: Any) -> np.ndarray:
        """Return the Hubble rate in the legacy SASHIMI-C unit system.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift.

        Returns
        -------
        numpy.ndarray
            Hubble rate expressed in the legacy inverse-second unit.
        """

        return np.asarray(self.itamae_cosmology.H(z)) * self.km / self.s / self.Mpc

    def rhocrit(self, z: Any) -> np.ndarray:
        """Return critical density in the legacy SASHIMI-C unit system.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift.

        Returns
        -------
        numpy.ndarray
            Critical density in solar masses per cubic Mpc, normalized with the
            historical SASHIMI constant convention.
        """

        density = np.asarray(self.itamae_cosmology.rho_crit(z))
        return density * self.Msun / self.Mpc**3 * self._rho_crit_scale

    def growthD(self, z: Any) -> np.ndarray:
        """Return the linear growth factor normalized to unity at redshift zero.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift.

        Returns
        -------
        numpy.ndarray
            Dimensionless linear growth factor.
        """

        return np.asarray(self.itamae_cosmology.growth_factor(z))


def migrate_class(base_class: _Base) -> _Base:
    """Construct an ITAMAE-backed subclass of a legacy SASHIMI-C class.

    Parameters
    ----------
    base_class : type
        Legacy class whose cosmology methods should be supplied by ITAMAE.

    Returns
    -------
    type
        Dynamically constructed subclass. The legacy class remains unchanged.
    """

    return type(
        f"Itamae{base_class.__name__[0].upper()}{base_class.__name__[1:]}",
        (ItamaeCosmologyMixin, base_class),
        {"__module__": __name__},
    )


ItamaeHaloModel = migrate_class(halo_model)

__all__ = ["ItamaeCosmologyMixin", "ItamaeHaloModel", "migrate_class"]
