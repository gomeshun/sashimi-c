"""Regression tests for the incremental ITAMAE migration path."""

import numpy as np

from itamae_migration import ItamaeHaloModel
from sashimi_c import halo_model


def test_itamae_cosmology_matches_legacy_background() -> None:
    """ITAMAE and legacy background quantities should agree numerically."""

    legacy = halo_model()
    migrated = ItamaeHaloModel()
    redshift = np.array([0.0, 0.5, 1.0, 3.0, 7.0])

    np.testing.assert_allclose(
        migrated.Hubble(redshift), legacy.Hubble(redshift), rtol=2.0e-12, atol=0.0
    )
    np.testing.assert_allclose(
        migrated.growthD(redshift), legacy.growthD(redshift), rtol=2.0e-12, atol=0.0
    )
    np.testing.assert_allclose(
        migrated.rhocrit(redshift), legacy.rhocrit(redshift), rtol=3.0e-4, atol=0.0
    )


def test_itamae_cosmology_preserves_halo_calculations() -> None:
    """A representative halo calculation should remain regression-equivalent."""

    legacy = halo_model()
    migrated = ItamaeHaloModel()
    mass = np.array([1.0e8, 1.0e10, 1.0e12]) * legacy.Msun
    redshift = np.array([0.0, 1.0, 3.0])

    np.testing.assert_allclose(
        migrated.sigmaMz(mass, redshift),
        legacy.sigmaMz(mass, redshift),
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        migrated.Mvir_from_M200_fit(mass, redshift),
        legacy.Mvir_from_M200_fit(mass, redshift),
        rtol=2.0e-12,
        atol=0.0,
    )
