"""Real relation replacement through initial and host stripping dependencies."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.optimize import brentq

from sashimi_c import CDM, HaloModel, TabulatedConcentration, TidalStrippingSolver


def analytic(mass, z):
    return 9.0 * (np.asarray(mass) / 1e8) ** (-0.08) * np.exp(-0.2 * np.asarray(z))


def table():
    mass = np.array([1e-6, 1e6, 1e10, 1e18])
    z = np.array([0.0, 1.0, 8.0])
    return TabulatedConcentration(mass, z, analytic(mass, z[:, None]))


def test_log_bilinear_interpolation_and_immutable_description():
    relation = table()
    mass, z = np.array([1e3, 3e7, 4e13]), np.array([0.0, 0.7, 5.0])
    np.testing.assert_allclose(
        relation.evaluate(mass, z), analytic(mass, z), rtol=2e-15
    )
    assert table().identifier == relation.identifier
    assert table() == relation
    assert hash(table()) == hash(relation)
    description = relation.describe()
    description["c200"][0][0] = 1
    assert relation.describe()["c200"][0][0] != 1
    with pytest.raises(FrozenInstanceError):
        relation.c200 = ()
    masses, redshifts, values = [1.0, 2.0], [0.0, 1.0], [[2.0, 3.0], [4.0, 5.0]]
    copied = TabulatedConcentration(masses, redshifts, values)
    values[0][0] = 999
    masses[0] = 999
    assert copied.c200[0][0] == 2 and copied.mass_msun[0] == 1
    np.testing.assert_allclose(copied.evaluate(1.0, 0.0), 2.0)


@pytest.mark.parametrize(
    "mass,z,c",
    [
        ([1, 1], [0, 1], [[1, 2], [1, 2]]),
        ([0, 1], [0, 1], [[1, 2], [1, 2]]),
        ([1, 2], [-1, 1], [[1, 2], [1, 2]]),
        ([1, 2], [0, 1], [[1, 0], [1, 2]]),
        ([1, 2], [0, 1], [[1, np.nan], [1, 2]]),
        ([1, 2], [0, 1], [[1, 2, 3], [1, 2, 3]]),
        ([1, 2], [0, 1], [[1, 2]]),
    ],
)
def test_invalid_table_rejected(mass, z, c):
    with pytest.raises((TypeError, ValueError)):
        TabulatedConcentration(mass, z, c)


def test_no_extrapolation_or_clipping():
    with pytest.raises(ValueError, match="host-test.*M200c"):
        table().evaluate(1e20, 0.0, component="host-test")
    with pytest.raises(ValueError, match="z=-0.1"):
        table().evaluate(1e10, -0.1)


def test_replacement_changes_nfw_mass_conversion_and_host_stripping():
    relation = table()
    physics = HaloModel(concentration_relation=relation)
    mass, z = 1e10, 0.5
    c = float(analytic(mass, z))
    np.testing.assert_allclose(physics.conc200(mass, z), c, rtol=1e-15)
    delta = physics.Delc(physics.OmegaM * (1 + z) ** 3 / physics.g(z) - 1)

    def f(x):
        return np.log1p(x) - x / (1 + x)

    cvir = brentq(lambda x: f(x) / x**3 - delta / 200 * f(c) / c**3, 0.01, 1000)
    expected_mvir = mass * f(cvir) / f(c)
    # Existing Hu-Kravtsov fit is approximate; retain its approximation, not refit it.
    np.testing.assert_allclose(
        physics.Mvir_from_M200_fit(mass, z), expected_mvir, rtol=0.005
    )
    replacement = TidalStrippingSolver(
        mass, z_min=0.2, z_max=1.2, concentration_relation=relation
    )
    original = TidalStrippingSolver(mass, z_min=0.2, z_max=1.2)
    host_m200 = physics.Mzzi(mass, z, 0.0)
    np.testing.assert_array_equal(
        replacement.Mzvir(z), physics.Mvir_from_M200_fit(host_m200, z)
    )
    assert replacement.Mzvir(z) != original.Mzvir(z)
    assert replacement.concentration_relation is relation
    modified_bound = replacement.subhalo_mass_stripped(np.array([1e6, 1e7]), 1.0, 0.2)
    original_bound = original.subhalo_mass_stripped(np.array([1e6, 1e7]), 1.0, 0.2)
    assert not np.allclose(modified_bound, original_bound, rtol=1e-6)


def test_population_uses_same_relation_for_subhalo_and_host():
    relation = table()
    model = CDM().configure(
        accretion={"mass_nodes": 3, "host_history_nodes": 3, "redshift_step": 0.5},
        concentration={"relation": relation, "scatter_dex": 0.0, "quadrature_nodes": 1},
    )
    catalog = model.population(
        host_mass_msun=1e10,
        redshift=0.2,
        accretion_mass_range_msun=(1e6, 1e8),
        accretion_redshift_range=(0.2, 1.2),
    )
    mass, z = catalog.columns["m200_acc"], catalog.columns["z_acc"]
    physics = HaloModel(concentration_relation=relation)
    radius_200 = (3 * mass / (4 * np.pi * 200 * physics.rhocrit0 * physics.g(z))) ** (
        1 / 3
    )
    np.testing.assert_allclose(
        radius_200 / catalog.columns["r_s_acc"], analytic(mass, z), rtol=2e-15
    )
    solver = TidalStrippingSolver(
        1e10, z_min=0.2, z_max=1.2, concentration_relation=relation
    )
    expected = np.concatenate(
        [
            solver.subhalo_mass_stripped(
                physics.Mvir_from_M200_fit(mass[z == za], za), za, 0.2
            )
            for za in np.unique(z)
        ]
    )
    np.testing.assert_array_equal(catalog.columns["m_bound"], expected)
    assert catalog.metadata["experimental_concentration"]
    assert (
        catalog.metadata["resolved_settings"]["concentration"]["relation"]["identifier"]
        == relation.identifier
    )
    assert (
        model.resolved_settings["concentration"]["prescription"]
        == "user-supplied-table"
    )
    assert "not refitted" in catalog.metadata["concentration_calibration_caveat"]


def test_domain_must_include_host_not_only_subhalo(monkeypatch):
    limited = TabulatedConcentration([1e6, 1e8], [0.0, 2.0], [[8.0, 8.0], [8.0, 8.0]])
    model = CDM().configure(
        accretion={"mass_nodes": 3, "redshift_step": 0.5},
        concentration={"relation": limited},
    )
    from sashimi_c import SubhaloProperties

    def cannot_execute(*args, **kwargs):
        raise AssertionError("domain failure should happen during preparation")

    monkeypatch.setattr(SubhaloProperties, "_calculate_population", cannot_execute)
    with pytest.raises(ValueError, match="host stripping preparation"):
        model.population(
            host_mass_msun=1e10,
            accretion_mass_range_msun=(1e6, 1e8),
            accretion_redshift_range=(0.0, 1.0),
        )


def test_stateful_custom_relation_is_not_accepted():
    with pytest.raises(TypeError, match="TabulatedConcentration"):
        CDM().configure(concentration={"relation": lambda mass, z: 10.0})


def test_table_provenance_roundtrips_and_identifies_changed_physics(tmp_path):
    from itamae.types import WeightedSubhaloCatalog

    model = CDM().configure(
        accretion={"mass_nodes": 3, "host_history_nodes": 3, "redshift_step": 0.5},
        concentration={"relation": table(), "quadrature_nodes": 1},
    )
    catalog = model.population(
        host_mass_msun=1e10,
        accretion_mass_range_msun=(1e6, 1e8),
        accretion_redshift_range=(0.0, 1.0),
    )
    path = tmp_path / "table.npz"
    catalog.to_npz(path)
    restored = WeightedSubhaloCatalog.from_npz(path)
    assert restored.metadata == catalog.metadata
    assert table().identifier in restored.metadata["model_identifier"]
    assert restored.metadata["concentration_identifier"] == table().identifier
    restored_table = restored.metadata["resolved_settings"]["concentration"]["relation"]
    relation = TabulatedConcentration(
        restored_table["mass_msun"], restored_table["redshift"], restored_table["c200"]
    )
    assert relation.identifier == table().identifier


def test_legacy_custom_relation_route_preserves_full_physical_provenance(tmp_path):
    from itamae.types import WeightedSubhaloCatalog

    from sashimi_c import SubhaloProperties

    catalog = SubhaloProperties(concentration_relation=table()).subhalo_catalog_calc(
        M0=1e10,
        N_ma=3,
        N_herm=1,
        N_hermNa=3,
        dz=0.5,
        zmax=1.0,
        logmamin=6.0,
        logmamax=8.0,
    )
    assert catalog.metadata["concentration_relation"] == table().describe()
    assert catalog.metadata["experimental_concentration"]
    assert "not refitted" in catalog.metadata["concentration_calibration_caveat"]
    path = tmp_path / "legacy-table.npz"
    catalog.to_npz(path)
    assert (
        WeightedSubhaloCatalog.from_npz(path).metadata["concentration_relation"]
        == table().describe()
    )


@pytest.mark.parametrize("target", [0.0, 0.2])
def test_table_odeint_respects_final_redshift_domain(target):
    relation = TabulatedConcentration(
        [1e-6, 1e18], [target, 8.0], [[10.0, 10.0], [10.0, 10.0]]
    )
    model = CDM().configure(
        accretion={"mass_nodes": 4, "host_history_nodes": 3, "redshift_step": 0.5},
        concentration={"relation": relation, "quadrature_nodes": 2},
        stripping={"solver": "odeint", "solver_options": {"rtol": 1e-8, "atol": 1e-10}},
    )
    catalog = model.population(
        host_mass_msun=1e10,
        redshift=target,
        accretion_mass_range_msun=(1e6, 1e8),
        accretion_redshift_range=(target, target + 1.0),
    )
    assert np.all(np.isfinite(catalog.columns["m_bound"]))
    assert (
        catalog.metadata["odeint_boundary_policy"]
        == "increasing-minus-redshift-tcrit-for-bounded-concentration"
    )
    assert catalog.metadata["odeint_tcrit"] == [-target]
    assert (
        catalog.metadata["resolved_settings"]["stripping"]["solver_options"]["rtol"]
        == 1e-8
    )
    solver = TidalStrippingSolver(
        1e10, z_min=target, z_max=target + 1.0, concentration_relation=relation
    )
    from scipy.integrate import solve_ivp

    physical = HaloModel(concentration_relation=relation)
    for accretion_z in np.unique(catalog.columns["z_acc"]):
        mask = catalog.columns["z_acc"] == accretion_z
        masses = catalog.columns["m200_acc"][mask]
        virial = physical.Mvir_from_M200_fit(masses, accretion_z)
        reference = solve_ivp(
            lambda z, m: solver.msolve(m, z),
            (accretion_z, target),
            virial,
            method="DOP853",
            rtol=1e-11,
            atol=1e-10,
        )
        assert reference.success
        np.testing.assert_allclose(
            catalog.columns["m_bound"][mask], reference.y[:, -1], rtol=3e-7
        )
    with pytest.raises(ValueError, match="tcrit is controlled"):
        solver.subhalo_mass_stripped(
            np.array([1e6]), target + 0.5, target, method="odeint", tcrit=[]
        )


def test_bounded_relation_ode_rejects_custom_jacobian_and_maps_h0():
    solver = TidalStrippingSolver(
        1e10, z_min=0.0, z_max=1.0, concentration_relation=table()
    )
    with pytest.raises(ValueError, match="Dfun is unsupported"):
        solver.subhalo_mass_stripped(
            np.array([1e6]), 0.5, 0.0, method="odeint", Dfun=lambda m, z: m
        )
    model = CDM().configure(
        accretion={"mass_nodes": 3, "host_history_nodes": 3, "redshift_step": 0.5},
        concentration={"relation": table(), "quadrature_nodes": 1},
        stripping={"solver": "odeint", "solver_options": {"h0": -1e-6}},
    )
    catalog = model.population(
        host_mass_msun=1e10,
        accretion_mass_range_msun=(1e6, 1e8),
        accretion_redshift_range=(0.0, 1.0),
    )
    assert catalog.metadata["odeint_effective_h0"] == 1e-6
    assert (
        catalog.metadata["resolved_settings"]["stripping"]["solver_options"]["h0"]
        == -1e-6
    )


def test_picard_reuse_tracks_immutable_concentration_relation_replacement():
    def relation(value):
        return TabulatedConcentration([1e-6, 1e18], [0.0, 8.0],
                                      [[value, value], [value, value]])

    host = TidalStrippingSolver(1e10, z_max=1.0, n_z_interp=16,
                               concentration_relation=relation(10.0))
    original = host._get_picard_table(0.0)
    before = original.mass(1e6, 0.5)
    assert host._get_picard_table(0.0) is original
    # Equal immutable payloads represent the same physical relation.
    host.concentration_relation = relation(10.0)
    assert host._get_picard_table(0.0) is original
    host.concentration_relation = relation(15.0)
    with pytest.raises(ValueError, match="stale"):
        original.mass(1e6, 0.5)
    replacement = host._get_picard_table(0.0)
    assert replacement is not original
    assert replacement.mass(1e6, 0.5) != before
    host.concentration_relation = None
    with pytest.raises(ValueError, match="stale"):
        replacement.mass(1e6, 0.5)
    standard = host._get_picard_table(0.0)
    assert host._get_picard_table(0.0) is standard
