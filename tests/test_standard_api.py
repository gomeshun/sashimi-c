"""Breaking standard API contract and independent corrected catalog reference."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import sashimi_c
from sashimi_c import (
    HaloModel,
    SubhaloObservables,
    SubhaloProperties,
    TidalStrippingSolver,
)
from sashimi_c_itamae_migration import ItamaeSubhaloProperties

SMALL = {
    "M0": 1e10,
    "dz": 0.5,
    "zmax": 1.0,
    "N_ma": 4,
    "N_herm": 2,
    "N_hermNa": 3,
    "logmamin": 6.0,
    "logmamax": 8.0,
}


def test_corrected_full_catalog_matches_independent_reference_b():
    path = Path(__file__).parent / "references/B-all.npz"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["role"] == "B"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["artifact_sha256"]
    parameters = metadata["calculation"]["parameters"]
    model = SubhaloProperties()
    actual = model.subhalo_properties_calc(**parameters)
    with np.load(path) as reference:
        for i, array in enumerate(actual):
            if array.dtype.kind == "b":
                np.testing.assert_array_equal(array, reference[f"tuple_{i}"])
            else:
                np.testing.assert_allclose(
                    array, reference[f"tuple_{i}"], rtol=5e-12, atol=0.0
                )
    assert (
        model.catalog.metadata["calculation_specification"]
        == sashimi_c.CALCULATION_SPECIFICATION
    )
    assert "physics_mode" not in model.catalog.metadata


@pytest.mark.parametrize("mode", ["legacy", "consistent", "hybrid"])
def test_removed_mode_cannot_select_a_hidden_calculation(mode):
    for cls, args in (
        (HaloModel, ()),
        (SubhaloProperties, ()),
        (TidalStrippingSolver, (1e10,)),
        (SubhaloObservables, (1e10,)),
    ):
        with pytest.raises(TypeError, match="physics_mode"):
            cls(*args, physics_mode=mode)
    with pytest.raises(TypeError, match="physics_mode"):
        SubhaloProperties().subhalo_catalog_calc(**SMALL, physics_mode=mode)


def test_standard_import_and_tuple_conversion_use_the_same_population():
    assert sashimi_c.subhalo_properties is SubhaloProperties is ItamaeSubhaloProperties
    model = SubhaloProperties()
    arrays = model.subhalo_properties_calc(**SMALL)
    catalog = model.catalog
    names = (
        "m200_acc",
        "z_acc",
        "r_s_acc",
        "rho_s_acc",
        "m_bound",
        "r_s",
        "rho_s",
        "c_t",
    )
    for name, values in zip(names, arrays[:8], strict=True):
        np.testing.assert_array_equal(catalog.columns[name], values)
    np.testing.assert_array_equal(
        arrays[8],
        catalog.weights["weight_base"] * catalog.weights["weight_concentration"],
    )
    np.testing.assert_array_equal(arrays[9], catalog.columns["survive"])
    assert all("Migration" not in cls.__name__ for cls in SubhaloProperties.__mro__)


def test_cosmology_units_and_growth_derivative_are_consistent():
    model = HaloModel()
    z = np.array([0.0, 0.5, 1.0, 3.0])
    np.testing.assert_allclose(
        model.rhocrit(z), 3 * model.Hubble(z) ** 2 / (8 * np.pi * model.G), rtol=5e-15
    )
    np.testing.assert_allclose(
        model.rhocrit(z), model.itamae_cosmology.rho_crit(z), rtol=2e-15
    )
    step = 1e-5
    derivative = (model.growthD(z + step) - model.growthD(z - step)) / (2 * step)
    np.testing.assert_allclose(model.dDdz(z), derivative, rtol=2e-9)
    assert model.growthD(0.0) == 1.0


def test_target_redshift_host_mass_keeps_the_recorded_mass_definition():
    parameters = {**SMALL, "redshift": 1.0, "zmax": 2.0}
    catalog = SubhaloProperties().subhalo_catalog_calc(
        **parameters, M0_at_redshift=True
    )
    corrected = {**parameters, "M0": catalog.metadata["host_mass_z0"]}
    at_zero = SubhaloProperties().subhalo_catalog_calc(**corrected)
    for name in catalog.columns:
        np.testing.assert_array_equal(catalog.columns[name], at_zero.columns[name])
    assert catalog.metadata["host_mass_input"] == parameters["M0"]
