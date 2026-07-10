"""Initialized catalog-migration classes for SASHIMI-C.

``TidalStrippingSolver`` builds interpolation tables during its constructor and
therefore invokes cosmology methods before a normal post-``super`` adapter can
attach its backend. The wrappers in this module install a provisional ITAMAE
backend before legacy initialization starts. After initialization, the common
migration mixin recomputes the exact legacy normalization.

Users testing the catalog migration should import classes from this module. The
smaller ``itamae_migration`` module remains useful for classes whose constructors
do not call overridable cosmology methods.
"""

from __future__ import annotations

from typing import Any

from itamae.cosmology import NativeFlatLCDM
import itamae_migration as _migration


class ItamaeTidalStrippingSolver(_migration.ItamaeTidalStrippingSolver):
    """Tidal solver with its ITAMAE backend available during construction."""

    def __init__(
        self,
        *args: Any,
        cosmology_backend: Any | None = None,
        **kwargs: Any,
    ) -> None:
        backend = cosmology_backend or NativeFlatLCDM()

        # Legacy initialization creates perturbative interpolation tables and
        # dispatches to the overridden growth/Hubble methods. These provisional
        # values make those calls valid before the migration mixin finalizes the
        # backend-specific legacy unit normalization.
        self.itamae_cosmology = backend
        self._rho_crit_scale = 1.0
        super().__init__(*args, cosmology_backend=backend, **kwargs)


# The catalog method is implemented in the core migration module and resolves
# this global name when it constructs a solver. Rebinding it here keeps one copy
# of the catalog equations while selecting the safely initialized solver.
_migration.ItamaeTidalStrippingSolver = ItamaeTidalStrippingSolver


class ItamaeSubhaloProperties(_migration.ItamaeSubhaloProperties):
    """ITAMAE-backed SASHIMI-C catalog generator with safe solver startup."""


__all__ = ["ItamaeSubhaloProperties", "ItamaeTidalStrippingSolver"]
