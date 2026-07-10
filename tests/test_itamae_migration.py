"""Regression tests for the incremental SASHIMI-C ITAMAE migration."""

import numpy as np

from itamae.numerics import gauss_hermite_lognormal
from itamae_migration import (
    ItamaeHaloModel,
    ItamaeSubhaloProperties,
    ItamaeTidalStrippingSolver,
)
from sashimi_c import TidalStrippingSolver, halo_model, subhalo_properties


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
        migrated.rhocrit(redshift), legacy.rhocrit(redshift), rtol=2.0e-12, atol=0.0
    )


def test_itamae_cosmology_preserves_halo_calculations() -> None:
    """Representative halo calculations should remain regression-equivalent."""

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


def test_itamae_concentration_quadrature_is_normalized() -> None:
    """ITAMAE concentration nodes should reproduce log-normal moments."""

    median = np.array([5.0, 10.0, 20.0])
    nodes, weights = gauss_hermite_lognormal(median, 0.128, order=5)

    np.testing.assert_allclose(np.sum(weights, axis=0), 1.0, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        np.sum(np.log10(nodes) * weights, axis=0),
        np.log10(median),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_itamae_shanks_solver_matches_legacy() -> None:
    """The migrated perturbative solver should preserve SASHIMI stabilization."""

    host_mass = 1.0e10
    legacy = TidalStrippingSolver(host_mass, z_min=0.0, z_max=1.5, n_z_interp=32)
    migrated = ItamaeTidalStrippingSolver(
        host_mass, z_min=0.0, z_max=1.5, n_z_interp=32
    )
    mass = np.array([1.0e6, 1.0e7, 1.0e8])

    np.testing.assert_allclose(
        migrated.subhalo_mass_stripped_pert2_shanks(mass, 1.0, 0.0),
        legacy.subhalo_mass_stripped_pert2_shanks(mass, 1.0, 0.0),
        rtol=2.0e-12,
        atol=0.0,
    )


def test_itamae_catalog_preserves_small_legacy_catalog() -> None:
    """A reduced catalog should agree apart from the improved NFW inversion."""

    parameters = {
        "M0": 1.0e10,
        "redshift": 0.0,
        "dz": 0.5,
        "zmax": 1.0,
        "N_ma": 6,
        "sigmalogc": 0.128,
        "N_herm": 3,
        "logmamin": 5.0,
        "logmamax": 7.0,
        "N_hermNa": 3,
        "Na_model": 3,
        "ct_th": 0.0,
        "profile_change": True,
        "method": "pert2_shanks",
    }
    legacy_result = subhalo_properties().subhalo_properties_calc(**parameters)
    catalog = ItamaeSubhaloProperties().subhalo_catalog_calc(**parameters)

    column_names = (
        "m200_acc",
        "z_acc",
        "r_s_acc",
        "rho_s_acc",
        "m_bound",
        "r_s",
        "rho_s",
    )
    for index, name in enumerate(column_names):
        np.testing.assert_allclose(
            catalog.columns[name], legacy_result[index], rtol=5.0e-11, atol=0.0
        )

    legacy_ct = legacy_result[7]
    np.testing.assert_allclose(catalog.columns["c_t"], legacy_ct, rtol=3.0e-3, atol=2.0e-4)
    np.testing.assert_array_equal(catalog.columns["survive"], legacy_result[9])

    legacy_weight = legacy_result[8]
    migrated_weight = (
        np.asarray(catalog.weights["population"])
        * np.asarray(catalog.weights["concentration"])
    )
    np.testing.assert_allclose(migrated_weight, legacy_weight, rtol=5.0e-11, atol=0.0)
    np.testing.assert_allclose(
        catalog.weight_final,
        legacy_weight * legacy_result[9].astype(float),
        rtol=5.0e-11,
        atol=0.0,
    )
