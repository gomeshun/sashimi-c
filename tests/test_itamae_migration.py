"""Standard CDM API, immutable references, observables and invariants."""

import json
from pathlib import Path

import numpy as np
import pytest
from itamae.cosmology import NativeFlatLCDM
from itamae.halo import nfw_mass_function
from itamae.numerics import gauss_hermite_lognormal
from itamae.provenance import CALCULATION_METADATA_KEYS

import sashimi_c
import sashimi_c_itamae_catalog
import sashimi_c_itamae_migration
from sashimi_c import (
    TidalStrippingSolver,
    subhalo_observables,
    subhalo_properties,
)
from sashimi_c_itamae_catalog import (
    ItamaeSubhaloProperties,
)
from sashimi_c_itamae_migration import (
    ItamaeHaloModel,
    ItamaeSubhaloObservables,
    diagnose_stripping_approximation,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "sashimi_c_cdm_v1_2.json").read_text()
)


def test_golden_fixture_provenance_is_complete() -> None:
    """The full C golden identifies its source and comparison policy."""
    provenance = GOLDEN["provenance"]
    assert provenance["fixture_schema"] == "sashimi-family:golden-provenance:v1"
    assert provenance["fixture_category"] == "full_small_catalog_golden"
    assert provenance["variant"] == "sashimi-c"
    assert len(provenance["generated_repository_revision"]) == 40
    assert len(provenance["itamae_source_revision"]) == 40
    assert set(provenance["physics_modes"]) == {"legacy", "consistent"}
    assert provenance["parameters_key"] == "parameters"
    assert provenance["comparison"]["rtol"] == 5e-10
    assert provenance["comparison"]["atol"] == 0.0
    assert provenance["constructor_parameters"]["physics_mode"] == [
        "consistent",
        "legacy",
    ]
    assert provenance["constructor_parameters"]["cosmology_backend"] == {
        "identifier": provenance["cosmology"]["backend_identifier"],
        "parameters": provenance["cosmology"]["parameters"],
    }


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


def test_migration_rejects_mixed_cosmology() -> None:
    """A partial migration must not combine incompatible cosmologies."""

    with pytest.raises(ValueError, match=r"requires OmegaM=0\.315"):
        ItamaeHaloModel(cosmology_backend=NativeFlatLCDM(omega_m0=0.30, h=0.674))
    with pytest.raises(ValueError, match=r"requires h=0\.674"):
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


def test_full_catalog_and_observables_match_preserved_consistent_golden() -> None:
    """Exercise all catalog nodes and public CDM observables against goldens."""
    parameters = dict(GOLDEN["parameters"])
    model = ItamaeSubhaloProperties()
    catalog = model.subhalo_catalog_calc(**parameters)
    observable_parameters = {
        ("M0_per_Msun" if name == "M0" else name): value
        for name, value in parameters.items()
    }
    observable = ItamaeSubhaloObservables(
        **observable_parameters,
    )
    actual = {**_catalog_summary(catalog), **_observable_summary(observable)}
    expected = GOLDEN["modes"]["consistent"]

    assert actual["model_identifier"] == sashimi_c.CALCULATION_SPECIFICATION
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


def test_catalog_metadata_records_mode_solver_weights_and_threshold(
    tmp_path: Path,
) -> None:
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
    assert set(CALCULATION_METADATA_KEYS) <= set(metadata)
    assert metadata["sashimi_variant"] == "sashimi-c"
    assert len(metadata["itamae_source_revision"]) == 40
    assert len(metadata["sashimi_source_revision"]) == 40
    assert metadata["sashimi_version"] == "1.2.0"
    assert metadata["catalog_schema_version"] == "1.0"
    assert metadata["canonical_unit_schema"] == "1.0"
    assert (
        metadata["cosmology_backend"]
        == GOLDEN["provenance"]["cosmology"]["backend_identifier"]
    )
    assert (
        metadata["cosmology_parameters"]
        == GOLDEN["provenance"]["cosmology"]["parameters"]
    )
    assert metadata["variance_identifier"] == "sashimi-c:analytic-cdm-fit:v1"
    assert metadata["power_identifier"] == "sashimi-c:cdm-linear-power:v1"
    assert metadata["solver_identifier"] == "sashimi-c:tidal-stripping:pert2_shanks:v1"
    assert metadata["cosmology_parameters"] == {
        "omega_m0": 0.315,
        "h": 0.674,
        "omega_lambda0": 0.685,
    }
    assert "physics_mode" not in metadata
    assert metadata["calculation_specification"] == sashimi_c.CALCULATION_SPECIFICATION
    assert metadata["model_identifier"] == sashimi_c.CALCULATION_SPECIFICATION
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
    np.testing.assert_allclose(
        catalog.weight_final,
        catalog.weights["weight_base"]
        * catalog.weights["weight_concentration"]
        * catalog.weights["weight_survival"],
        rtol=0.0,
        atol=0.0,
    )
    assert all(np.all(value >= 0.0) for value in catalog.weights.values())

    archive = tmp_path / "catalog.npz"
    catalog.to_npz(archive)
    restored = type(catalog).from_npz(archive)
    for name in catalog.columns:
        np.testing.assert_array_equal(restored.columns[name], catalog.columns[name])
    for name in catalog.weights:
        np.testing.assert_array_equal(restored.weights[name], catalog.weights[name])
    assert dict(restored.metadata) == dict(catalog.metadata)


def test_shanks_vs_ode_diagnostic_does_not_change_defaults() -> None:
    """Expose the known approximation difference without selecting a new solver."""
    masses = np.logspace(6.0, 10.0, 9)
    diagnostic = diagnose_stripping_approximation(
        host_mass=1.0e12,
        mass_at_accretion=masses,
        accretion_redshift=1.0,
    )

    assert diagnostic.calculation_specification == sashimi_c.CALCULATION_SPECIFICATION
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
        sashimi_c_itamae_catalog.ItamaeTidalStrippingSolver
        is sashimi_c_itamae_migration.ItamaeTidalStrippingSolver
    )
    assert (
        sashimi_c_itamae_catalog.ItamaeSubhaloProperties
        is sashimi_c_itamae_migration.ItamaeSubhaloProperties
    )
