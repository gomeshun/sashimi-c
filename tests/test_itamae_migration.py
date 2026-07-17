"""Regression tests for the incremental SASHIMI-C ITAMAE migration."""

import numpy as np
import pytest

import sashimi_c
import sashimi_c_itamae
from itamae.cosmology import NativeFlatLCDM
from itamae.halo import nfw_mass_function
from itamae.numerics import gauss_hermite_lognormal
from itamae_catalog_migration import (
    ItamaeSubhaloProperties,
    ItamaeTidalStrippingSolver,
)
from itamae_migration import ItamaeHaloModel, ItamaeSubhaloObservables
from sashimi_c import (
    TidalStrippingSolver,
    halo_model,
    subhalo_observables,
    subhalo_properties,
)


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
    np.testing.assert_allclose(
        catalog.columns["c_t"], legacy_ct, rtol=3.0e-3, atol=2.0e-4
    )
    np.testing.assert_array_equal(catalog.columns["survive"], legacy_result[9])

    legacy_weight = legacy_result[8]
    migrated_weight = np.asarray(catalog.weights["weight_base"]) * np.asarray(
        catalog.weights["weight_concentration"]
    )
    np.testing.assert_allclose(migrated_weight, legacy_weight, rtol=5.0e-11, atol=0.0)
    np.testing.assert_allclose(
        catalog.weight_final,
        legacy_weight * legacy_result[9].astype(float),
        rtol=5.0e-11,
        atol=0.0,
    )
    assert set(catalog.weights) == {
        "weight_base",
        "weight_concentration",
        "weight_survival",
    }
    assert catalog.metadata["schema_version"] == "1.0"
    assert catalog.metadata["model_identifier"] == "sashimi-c:cdm:itamae-migration:v1"


def test_catalog_satisfies_mass_profile_and_weight_invariants() -> None:
    """Migrated catalogs must preserve basic physical consistency."""

    migrated = ItamaeSubhaloProperties()
    catalog = migrated.subhalo_catalog_calc(
        M0=1.0e10,
        redshift=0.0,
        dz=0.5,
        zmax=1.0,
        N_ma=6,
        N_herm=3,
        logmamin=5.0,
        logmamax=7.0,
        N_hermNa=3,
    )

    for value in (*catalog.columns.values(), *catalog.weights.values()):
        assert np.all(np.isfinite(value))
    for value in catalog.weights.values():
        assert np.all(value >= 0.0)

    mvir_acc = migrated.Mvir_from_M200_fit(
        catalog.columns["m200_acc"],
        catalog.columns["z_acc"],
    )
    assert np.all(catalog.columns["m_bound"] <= mvir_acc * (1.0 + 1.0e-12))

    reconstructed_mass = (
        4.0
        * np.pi
        * catalog.columns["rho_s"]
        * catalog.columns["r_s"] ** 3
        * nfw_mass_function(catalog.columns["c_t"])
    )
    np.testing.assert_allclose(
        reconstructed_mass,
        catalog.columns["m_bound"],
        rtol=3.0e-13,
        atol=0.0,
    )


def test_catalog_preserves_target_redshift_host_mass_convention() -> None:
    """M0_at_redshift should retain the established inverse-history behavior."""

    parameters = {
        "M0": 1.0e10,
        "redshift": 1.0,
        "dz": 0.5,
        "zmax": 2.0,
        "N_ma": 6,
        "N_herm": 3,
        "logmamin": 5.0,
        "logmamax": 7.0,
        "N_hermNa": 3,
        "M0_at_redshift": True,
    }
    legacy = subhalo_properties().subhalo_properties_calc(**parameters)
    migrated = ItamaeSubhaloProperties().subhalo_properties_calc(**parameters)
    for index, (migrated_value, legacy_value) in enumerate(
        zip(migrated, legacy, strict=True)
    ):
        np.testing.assert_allclose(
            migrated_value,
            legacy_value,
            rtol=3.0e-3 if index == 7 else 5.0e-11,
            atol=2.0e-4 if index == 7 else 0.0,
        )


def test_opt_in_module_preserves_legacy_module_and_observable_results() -> None:
    """The public opt-in path must not alter the established legacy API."""

    assert sashimi_c.halo_model is halo_model
    assert sashimi_c.subhalo_properties is subhalo_properties
    assert sashimi_c.subhalo_observables is subhalo_observables
    assert sashimi_c_itamae.halo_model is ItamaeHaloModel
    assert sashimi_c_itamae.TidalStrippingSolver is ItamaeTidalStrippingSolver
    assert sashimi_c_itamae.subhalo_properties is ItamaeSubhaloProperties
    assert sashimi_c_itamae.subhalo_observables is ItamaeSubhaloObservables

    parameters = {
        "M0_per_Msun": 1.0e10,
        "redshift": 0.0,
        "dz": 0.5,
        "zmax": 1.0,
        "N_ma": 6,
        "N_herm": 3,
        "logmamin": 5.0,
        "logmamax": 7.0,
        "N_hermNa": 3,
    }
    legacy = subhalo_observables(**parameters)
    migrated = sashimi_c_itamae.subhalo_observables(**parameters)
    for name in (
        "ma200",
        "z_a",
        "rs_a",
        "rhos_a",
        "m0",
        "rs0",
        "rhos0",
        "ct0",
        "weight",
        "rmax",
        "Vmax",
        "rpeak",
        "Vpeak",
    ):
        np.testing.assert_allclose(
            getattr(migrated, name),
            getattr(legacy, name),
            rtol=3.0e-3 if name == "ct0" else 5.0e-11,
            atol=2.0e-4 if name == "ct0" else 0.0,
        )
    for evolved in (False, True):
        for migrated_value, legacy_value in zip(
            migrated.mass_function(evolved=evolved),
            legacy.mass_function(evolved=evolved),
            strict=True,
        ):
            np.testing.assert_allclose(
                migrated_value,
                legacy_value,
                rtol=5.0e-11,
                atol=0.0,
            )
        np.testing.assert_allclose(
            migrated.mass_fraction(evolved=evolved),
            legacy.mass_fraction(evolved=evolved),
            rtol=5.0e-11,
            atol=0.0,
        )


def test_migration_rejects_mixed_cosmology() -> None:
    """A partial migration must not combine incompatible cosmologies."""

    with pytest.raises(ValueError, match="requires OmegaM=0.315"):
        ItamaeHaloModel(cosmology_backend=NativeFlatLCDM(omega_m0=0.30, h=0.674))
    with pytest.raises(ValueError, match="requires h=0.674"):
        ItamaeHaloModel(cosmology_backend=NativeFlatLCDM(omega_m0=0.315, h=0.70))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"M0": 0.0}, "M0 must be positive"),
        ({"redshift": -0.1}, "redshift must be nonnegative"),
        ({"dz": 0.0}, "dz must be positive"),
        ({"zmax": 0.0}, "zmax must be greater than redshift"),
        ({"sigmalogc": -0.1}, "sigmalogc must be nonnegative"),
        ({"N_herm": 0}, "N_herm must be positive"),
        ({"logmamin": 8.0, "logmamax": 7.0}, "logmamin must be smaller"),
        ({"ct_th": -0.1}, "ct_th must be nonnegative"),
    ],
)
def test_catalog_rejects_unphysical_inputs(overrides, message) -> None:
    """Invalid physical domains should fail before numerical propagation."""

    parameters = {
        "M0": 1.0e10,
        "redshift": 0.0,
        "dz": 0.5,
        "zmax": 1.0,
        "N_ma": 4,
        "N_herm": 2,
        "logmamin": 5.0,
        "logmamax": 7.0,
        "N_hermNa": 2,
    }
    parameters.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        ItamaeSubhaloProperties().subhalo_catalog_calc(**parameters)
