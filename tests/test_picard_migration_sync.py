"""SYNC-C regressions for the optional Picard tidal-stripping path."""

import json
from pathlib import Path

import numpy as np

from sashimi_c import TidalStrippingSolver
from sashimi_c_itamae_migration import (
    ItamaeSubhaloObservables,
    ItamaeSubhaloProperties,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "sashimi_c_cdm_v1_2.json").read_text()
)

SMALL_CATALOG = {
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
}


def test_historical_reference_remains_explicit_and_shanks_based() -> None:
    """SYNC-C must not silently regenerate the historical migration fixture."""
    provenance = GOLDEN["provenance"]
    parameters = GOLDEN["parameters"]

    assert provenance["generated_repository_revision"] == (
        "9f6713b686805645da459e99522e2049e7dea793"
    )
    assert parameters["method"] == "pert2_shanks"
    assert parameters["ct_th"] == 0.0
    assert "legacy" in provenance["physics_modes"]


def test_public_default_remains_historical_pert2_shanks() -> None:
    """Mechanical Picard synchronization must not change the public solver default."""
    solver = TidalStrippingSolver(1.0e10, z_min=0.0, z_max=1.0, n_z_interp=32)
    masses = np.array([1.0e5, 1.0e6, 1.0e7])

    default = solver.subhalo_mass_stripped(masses, 1.0, 0.0)
    explicit = solver.subhalo_mass_stripped(masses, 1.0, 0.0, method="pert2_shanks")
    np.testing.assert_array_equal(default, explicit)


def test_picard_table_is_explicit_cached_and_invalidated_with_host_mass() -> None:
    solver = TidalStrippingSolver(1.0e10, z_min=0.0, z_max=1.0, n_z_interp=32)
    first = solver._get_picard_table(0.0)
    assert solver._get_picard_table(0.0) is first
    assert first.n_iterations == 3
    assert first.n_log_ratio == 128
    assert first.log10_ratio_min == -24.0

    solver.M0 = 2.0e10
    second = solver._get_picard_table(0.0)
    assert second is not first


def test_optional_picard_compares_at_full_catalog_level() -> None:
    """Picard uses identical accretion population and a bounded evolved-mass shift."""
    shanks_model = ItamaeSubhaloProperties()
    picard_model = ItamaeSubhaloProperties()
    shanks = shanks_model.subhalo_catalog_calc(**SMALL_CATALOG, method="pert2_shanks")
    picard = picard_model.subhalo_catalog_calc(**SMALL_CATALOG, method="picard_table")

    for name in ("m200_acc", "z_acc", "r_s_acc", "rho_s_acc"):
        np.testing.assert_allclose(
            picard.columns[name], shanks.columns[name], rtol=5.0e-11, atol=0.0
        )
    for name in ("weight_base", "weight_concentration"):
        np.testing.assert_allclose(
            picard.weights[name], shanks.weights[name], rtol=5.0e-11, atol=0.0
        )

    assert np.all(np.isfinite(picard.columns["m_bound"]))
    assert np.all(picard.columns["m_bound"] > 0.0)
    relative = np.abs(picard.columns["m_bound"] / shanks.columns["m_bound"] - 1.0)
    assert float(np.max(relative)) < 0.25

    metadata = picard.metadata
    assert "physics_mode" not in metadata
    assert metadata["stripping_method"] == "picard_table"
    assert metadata["default_stripping_method"] == "pert2_shanks"
    assert metadata["ct_threshold"] == 0.0
    assert metadata["picard_table_settings"] == {
        "n_iterations": 3,
        "n_z_acc": 96,
        "n_log_ratio": 128,
        "log10_ratio_min": -24.0,
        "log10_ratio_max": -0.5,
        "n_integration": 128,
    }


def test_optional_picard_compares_at_observable_level() -> None:
    """Record a bounded solver-only effect on evolved public observables."""
    observable_parameters = {
        ("M0_per_Msun" if key == "M0" else key): value
        for key, value in SMALL_CATALOG.items()
    }
    shanks = ItamaeSubhaloObservables(method="pert2_shanks", **observable_parameters)
    picard = ItamaeSubhaloObservables(method="picard_table", **observable_parameters)

    np.testing.assert_allclose(
        picard.mass_fraction(evolved=False),
        shanks.mass_fraction(evolved=False),
        rtol=5.0e-11,
        atol=0.0,
    )
    shanks_evolved = float(shanks.mass_fraction(evolved=True))
    picard_evolved = float(picard.mass_fraction(evolved=True))
    assert np.isfinite(shanks_evolved)
    assert np.isfinite(picard_evolved)
    assert abs(picard_evolved / shanks_evolved - 1.0) < 0.25
