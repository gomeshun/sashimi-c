"""Regression tests for the incremental SASHIMI-C ITAMAE migration."""

import json
from pathlib import Path

import numpy as np
import pytest

import itamae_catalog_migration
import itamae_migration
import sashimi_c
import sashimi_c_itamae
from itamae.cosmology import NativeFlatLCDM
from itamae.halo import nfw_mass_function
from itamae.numerics import gauss_hermite_lognormal
from itamae_catalog_migration import (
    ItamaeSubhaloProperties,
    ItamaeTidalStrippingSolver,
)
from itamae_migration import (
    ItamaeHaloModel,
    ItamaeSubhaloObservables,
    diagnose_stripping_approximation,
)
from sashimi_c import (
    TidalStrippingSolver,
    halo_model,
    subhalo_observables,
    subhalo_properties,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "sashimi_c_cdm_v1_2.json").read_text()
)


def test_itamae_cosmology_matches_legacy_background() -> None:
    """Legacy mode reproduces old constants; consistent mode uses ITAMAE's."""

    legacy = halo_model()
    migrated = ItamaeHaloModel(physics_mode="legacy")
    consistent = ItamaeHaloModel()
    backend = NativeFlatLCDM(omega_m0=0.315, h=0.674)
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
    np.testing.assert_allclose(
        consistent.Hubble(redshift), legacy.Hubble(redshift), rtol=2.0e-12, atol=0.0
    )
    np.testing.assert_allclose(
        consistent.growthD(redshift),
        legacy.growthD(redshift),
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        consistent.rhocrit(redshift),
        backend.rho_crit(redshift),
        rtol=2.0e-12,
        atol=0.0,
    )
    assert consistent.physics_mode == "consistent"
    assert migrated.physics_mode == "legacy"


def test_itamae_cosmology_preserves_halo_calculations() -> None:
    """Representative halo calculations should remain regression-equivalent."""

    legacy = halo_model()
    migrated = ItamaeHaloModel(physics_mode="legacy")
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
    """Legacy mode should reproduce every historical catalog field."""

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
    catalog = ItamaeSubhaloProperties(physics_mode="legacy").subhalo_catalog_calc(
        **parameters
    )

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
    np.testing.assert_allclose(catalog.columns["c_t"], legacy_ct, rtol=5.0e-14)
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
    assert catalog.metadata["model_identifier"] == "sashimi-c:cdm:legacy:v1.2"
    assert catalog.metadata["physics_mode"] == "legacy"


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
    migrated = ItamaeSubhaloProperties(physics_mode="legacy").subhalo_properties_calc(
        **parameters
    )
    for index, (migrated_value, legacy_value) in enumerate(
        zip(migrated, legacy, strict=True)
    ):
        np.testing.assert_allclose(
            migrated_value,
            legacy_value,
            rtol=5.0e-11,
            atol=0.0,
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
    migrated = sashimi_c_itamae.subhalo_observables(physics_mode="legacy", **parameters)
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
            rtol=5.0e-11,
            atol=0.0,
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
    assert migrated.catalog.metadata["physics_mode"] == "legacy"
    np.testing.assert_array_equal(
        migrated.catalog.columns["survive"],
        migrated.catalog.weights["weight_survival"].astype(bool),
    )


def test_migration_rejects_mixed_cosmology() -> None:
    """A partial migration must not combine incompatible cosmologies."""

    with pytest.raises(ValueError, match="requires OmegaM=0.315"):
        ItamaeHaloModel(cosmology_backend=NativeFlatLCDM(omega_m0=0.30, h=0.674))
    with pytest.raises(ValueError, match="requires h=0.674"):
        ItamaeHaloModel(cosmology_backend=NativeFlatLCDM(omega_m0=0.315, h=0.70))


def test_migration_rejects_unknown_physics_mode() -> None:
    """Mode selection is explicit and cannot silently fall back."""

    with pytest.raises(ValueError, match="physics_mode must be one of"):
        ItamaeHaloModel(physics_mode="hybrid")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"M0": 0.0}, "M0 must be positive"),
        ({"redshift": -0.1}, "redshift must be nonnegative"),
        ({"dz": 0.0}, "dz must be positive"),
        ({"zmax": 0.0}, "zmax must be greater than redshift"),
        ({"sigmalogc": -0.1}, "sigmalogc must be nonnegative"),
        ({"N_herm": 0}, "N_herm must be positive"),
        ({"Na_model": 4}, "Na_model must be 1, 2, or 3"),
        ({"logmamin": 8.0, "logmamax": 7.0}, "logmamin must be smaller"),
        ({"ct_th": -0.1}, "ct_th must be nonnegative"),
        ({"method": "unknown"}, "method must be one of"),
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


def _catalog_summary(catalog):
    """Return the frozen catalog diagnostics stored in the golden fixture."""
    legacy_weight = (
        catalog.weights["weight_base"] * catalog.weights["weight_concentration"]
    )
    return {
        "model_identifier": catalog.metadata["model_identifier"],
        "node_count": len(catalog),
        "survive_count": int(np.count_nonzero(catalog.columns["survive"])),
        "weight_sum": float(np.sum(legacy_weight)),
        "weight_final_sum": float(np.sum(catalog.weight_final)),
        "weighted_m200_acc": float(np.sum(legacy_weight * catalog.columns["m200_acc"])),
        "weighted_m_bound": float(np.sum(legacy_weight * catalog.columns["m_bound"])),
        "ct_min": float(np.min(catalog.columns["c_t"])),
        "ct_max": float(np.max(catalog.columns["c_t"])),
    }


def _observable_summary(observable):
    """Return frozen public-observable diagnostics."""
    mass_function_evolved = observable.mass_function(evolved=True)
    mass_function_unevolved = observable.mass_function(evolved=False)
    nsat_mpeak = observable.Nsat_Mpeak(1.0e8 * observable.Msun)
    nsat_vpeak = observable.Nsat_Vpeak(18.0 * observable.km / observable.s)
    return {
        "mass_fraction_evolved": float(observable.mass_fraction(evolved=True)),
        "mass_fraction_unevolved": float(observable.mass_fraction(evolved=False)),
        "boost_evolved": [
            float(value)
            for value in observable.annihilation_boost_factor(n=0, evolved=True)
        ],
        "boost_unevolved": [
            float(value)
            for value in observable.annihilation_boost_factor(n=0, evolved=False)
        ],
        "mass_function_evolved": [
            float(mass_function_evolved[0][0]),
            float(mass_function_evolved[0][-1]),
            float(np.sum(mass_function_evolved[1])),
            float(np.max(mass_function_evolved[1])),
        ],
        "mass_function_unevolved": [
            float(mass_function_unevolved[0][0]),
            float(mass_function_unevolved[0][-1]),
            float(np.sum(mass_function_unevolved[1])),
            float(np.max(mass_function_unevolved[1])),
        ],
        "Nsat_Mpeak": [
            float(nsat_mpeak[1][0]),
            float(np.max(nsat_mpeak[1])),
            float(nsat_mpeak[3][0]),
            float(np.max(nsat_mpeak[3])),
        ],
        "Nsat_Vpeak": [
            float(nsat_vpeak[1][0]),
            float(np.max(nsat_vpeak[1])),
            float(nsat_vpeak[3][0]),
            float(np.max(nsat_vpeak[3])),
        ],
    }


@pytest.mark.parametrize("physics_mode", ["consistent", "legacy"])
def test_full_catalog_and_observables_match_mode_specific_golden(physics_mode) -> None:
    """Exercise all catalog nodes and public CDM observables against goldens."""
    parameters = dict(GOLDEN["parameters"])
    model = ItamaeSubhaloProperties(physics_mode=physics_mode)
    catalog = model.subhalo_catalog_calc(**parameters)
    observable_parameters = {
        ("M0_per_Msun" if name == "M0" else name): value
        for name, value in parameters.items()
    }
    observable = ItamaeSubhaloObservables(
        physics_mode=physics_mode,
        **observable_parameters,
    )
    actual = {**_catalog_summary(catalog), **_observable_summary(observable)}
    expected = GOLDEN["modes"][physics_mode]

    assert actual["model_identifier"] == expected["model_identifier"]
    assert actual["node_count"] == expected["node_count"]
    assert actual["survive_count"] == expected["survive_count"]
    for name in expected.keys() - {
        "model_identifier",
        "node_count",
        "survive_count",
    }:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            rtol=5.0e-10,
            atol=0.0,
        )

    if physics_mode == "legacy":
        legacy_catalog = subhalo_properties().subhalo_properties_calc(**parameters)
        migrated_catalog = model.subhalo_properties_calc(**parameters)
        for migrated_value, legacy_value in zip(
            migrated_catalog,
            legacy_catalog,
            strict=True,
        ):
            if migrated_value.dtype == bool:
                np.testing.assert_array_equal(migrated_value, legacy_value)
            else:
                np.testing.assert_allclose(
                    migrated_value,
                    legacy_value,
                    rtol=5.0e-13,
                    atol=0.0,
                )

        legacy_observable = subhalo_observables(**observable_parameters)
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
                getattr(observable, name),
                getattr(legacy_observable, name),
                rtol=5.0e-13,
                atol=0.0,
            )


def test_catalog_metadata_records_mode_solver_weights_and_threshold() -> None:
    """Reproducibility metadata must expose every migration choice."""
    parameters = {
        "M0": 1.0e10,
        "redshift": 0.0,
        "dz": 0.5,
        "zmax": 1.5,
        "N_ma": 8,
        "N_herm": 3,
        "logmamin": 5.0,
        "logmamax": 8.0,
        "N_hermNa": 4,
        "ct_th": 2.0,
    }
    model = ItamaeSubhaloProperties()
    catalog = model.subhalo_catalog_calc(**parameters)
    metadata = catalog.metadata

    assert model.catalog is catalog
    assert metadata["physics_mode"] == "consistent"
    assert metadata["model_identifier"] == "sashimi-c:cdm:consistent:v1.2"
    assert metadata["stripping_method"] == "pert2_shanks"
    assert metadata["default_stripping_method"] == "pert2_shanks"
    assert metadata["shanks_small_correction_threshold"] == 0.02
    assert metadata["ct_threshold"] == 2.0
    assert metadata["default_ct_threshold"] == 0.0
    assert metadata["survival_rule"] == "c_t > ct_threshold"
    assert metadata["nfw_inversion"] == "itamae.brentq"
    assert metadata["backend_identifier"].startswith(
        "array=numpy;cosmology=native-flatlcdm:"
    )
    assert set(metadata["weight_semantics"]) == {
        "weight_base",
        "weight_concentration",
        "weight_survival",
    }
    survive = catalog.columns["c_t"] > parameters["ct_th"]
    np.testing.assert_array_equal(catalog.columns["survive"], survive)
    np.testing.assert_array_equal(
        catalog.weights["weight_survival"],
        survive.astype(float),
    )
    assert metadata["surviving_node_count"] == int(np.count_nonzero(survive))
    assert 0.0 <= metadata["surviving_weight_fraction"] <= 1.0


def test_shanks_vs_ode_diagnostic_does_not_change_defaults() -> None:
    """Expose the known approximation difference without selecting a new solver."""
    masses = np.logspace(6.0, 10.0, 9)
    diagnostic = diagnose_stripping_approximation(
        host_mass=1.0e12,
        mass_at_accretion=masses,
        accretion_redshift=1.0,
    )

    assert diagnostic.physics_mode == "consistent"
    assert diagnostic.summary()["comparison"] == "pert2_shanks-vs-odeint"
    assert diagnostic.summary()["sample_size"] == len(masses)
    np.testing.assert_allclose(
        diagnostic.max_relative_difference,
        0.0004768174547677608,
        rtol=5.0e-10,
        atol=0.0,
    )
    assert np.all(np.isfinite(diagnostic.relative_difference))
    assert not diagnostic.relative_difference.flags.writeable
    scalar_diagnostic = diagnose_stripping_approximation(
        host_mass=1.0e12,
        mass_at_accretion=1.0e8,
    )
    assert scalar_diagnostic.mass_at_accretion.shape == (1,)

    catalog = ItamaeSubhaloProperties().subhalo_catalog_calc(
        M0=1.0e10,
        dz=0.5,
        zmax=1.0,
        N_ma=4,
        N_herm=2,
        logmamin=5.0,
        logmamax=7.0,
        N_hermNa=2,
    )
    assert catalog.metadata["stripping_method"] == "pert2_shanks"
    assert catalog.metadata["ct_threshold"] == 0.0


def test_migration_imports_do_not_monkeypatch_legacy_globals() -> None:
    """Compatibility modules are side-effect-free aliases."""
    assert sashimi_c.TidalStrippingSolver is TidalStrippingSolver
    assert sashimi_c.subhalo_properties is subhalo_properties
    assert sashimi_c.subhalo_observables is subhalo_observables
    assert (
        itamae_catalog_migration.ItamaeTidalStrippingSolver
        is itamae_migration.ItamaeTidalStrippingSolver
    )
    assert (
        itamae_catalog_migration.ItamaeSubhaloProperties
        is itamae_migration.ItamaeSubhaloProperties
    )
