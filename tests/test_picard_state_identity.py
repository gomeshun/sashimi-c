"""Narrow PR22 state-safety backport, independent of its solver changes."""

import numpy as np
import pytest

from sashimi_c import TidalStrippingSolver
from sashimi_c.picard_tidal_stripping import PicardTidalStrippingTable, physics_key


class AnalyticHost:
    _get_picard_table = TidalStrippingSolver._get_picard_table

    def __init__(self):
        self.M0 = 1e12
        self.z_min = 0.0
        self.z_max = 1.0
        self.rate = 0.5
        self.power = np.array([1.0])
        self._picard_tables = {}

    def Mzvir(self, z):
        return np.ones_like(np.asarray(z)) * self.M0

    def zetaMz(self, z):
        return np.zeros_like(np.asarray(z))

    def Phi(self, z):
        return np.ones_like(np.asarray(z)) * self.rate * self.power[0]


def test_unchanged_host_reuses_table_and_preserves_analytic_mass():
    host = AnalyticHost()
    table = host._get_picard_table(0.0)
    assert host._get_picard_table(0.0) is table
    np.testing.assert_allclose(table.mass(1e6, 0.5), 1e6 * np.exp(-0.25), rtol=3e-15)
    assert table.n_z_acc == 96 and table.n_log_ratio == 128
    assert table.n_integration == 128 and table.n_iterations == 3


@pytest.mark.parametrize("change", ["mass", "rate", "array", "callable"])
def test_mutation_rejects_retained_table_and_rebuilds_owned_cache(change):
    host = AnalyticHost()
    table = host._get_picard_table(0.0)
    if change == "mass":
        host.M0 *= 2
    elif change == "rate":
        host.rate = 0.7
    elif change == "array":
        host.power[0] = 2.0
    else:
        host.Phi = lambda z: np.ones_like(np.asarray(z)) * 0.9
    for method in (table.mass, table.delta_log_mass):
        with pytest.raises(ValueError, match="stale"):
            method(1e6, 0.5)
    new = host._get_picard_table(0.0)
    assert new is not table and len(host._picard_tables) == 1
    assert host._get_picard_table(0.0) is new
    np.testing.assert_allclose(new.mass(1e6, 0.5), 1e6 * np.exp(-0.5 * host.Phi(0.0)), rtol=4e-15)


def test_real_solver_mass_setter_and_legacy_table_rejection():
    host = TidalStrippingSolver(1e10, z_max=1.0, n_z_interp=16)
    old = host._get_picard_table(0.0)
    host.M0 = 2e10
    with pytest.raises(ValueError, match="stale"):
        old.mass(1e6, 0.5)
    new = host._get_picard_table(0.0)
    assert new is not old
    del new._physics_key  # An older serialized object cannot prove compatibility.
    with pytest.raises(ValueError, match="stale"):
        new.mass(1e6, 0.5)


def test_cache_fields_do_not_change_identity_and_provider_hook_covers_hidden_state():
    host = AnalyticHost()
    key = physics_key(host)
    host._eps_cache = np.arange(3)
    host._picard_tables[1] = object()
    assert physics_key(host) == key
    hidden = {"rate": 0.5}
    host.picard_physics_key = lambda: (hidden["rate"],)
    table = PicardTidalStrippingTable(host, n_z_acc=3, n_log_ratio=3)
    hidden["rate"] = 0.7
    with pytest.raises(ValueError, match="stale"):
        table.mass(1e6, 0.5)


def test_cosmology_replacement_and_mutable_scalar_state_are_detected():
    host = AnalyticHost()
    host.itamae_cosmology = AnalyticHost()
    key = physics_key(host)
    host.itamae_cosmology.rate = 0.7
    assert physics_key(host) != key
    key = physics_key(host)
    host.itamae_cosmology = AnalyticHost()
    assert physics_key(host) != key
