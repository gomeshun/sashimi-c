"""Narrow PR22 state-safety backport, independent of its solver changes."""

import numpy as np
import pytest

from sashimi_c import TidalStrippingSolver
from sashimi_c.picard_tidal_stripping import PicardTidalStrippingTable, physics_key


class AnalyticHost:
    """Supply an exponential tidal law with mutable scalar and array inputs for
    cache-invalidation checks.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """

    _get_picard_table = TidalStrippingSolver._get_picard_table

    def __init__(self):
        """Initialize a constant 1e12-Msun host and mutable tidal amplitude on z in [0,1].

        Notes
        -----
        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.
        """
        self.M0 = 1e12
        self.z_min = 0.0
        self.z_max = 1.0
        self.rate = 0.5
        self.power = np.array([1.0])
        self._picard_tables = {}

    def Mzvir(self, z):
        """Return a constant host mass with the requested redshift shape.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            M0 at every query epoch.

        Notes
        -----
        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.
        """
        return np.ones_like(np.asarray(z)) * self.M0

    def zetaMz(self, z):
        """Set the mass-ratio exponent to zero for an exact exponential stripping law.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            Zeros with the redshift shape.

        Notes
        -----
        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.
        """
        return np.zeros_like(np.asarray(z))

    def Phi(self, z):
        """Read both mutable scalar and array factors in the analytic tidal rate.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        numpy.ndarray
            rate*power[0], with the redshift shape.

        Notes
        -----
        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.
        """
        return np.ones_like(np.asarray(z)) * self.rate * self.power[0]


def test_unchanged_host_reuses_table_and_preserves_analytic_mass():
    """Recover exponential mass loss for an unchanged analytic host.

    Notes
    -----
    Pass criteria: The table gives 1e6*exp(-0.25) to rtol=3e-15 and reuses the configured
    default table.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    host = AnalyticHost()
    table = host._get_picard_table(0.0)
    assert host._get_picard_table(0.0) is table
    np.testing.assert_allclose(table.mass(1e6, 0.5), 1e6 * np.exp(-0.25), rtol=3e-15)
    assert table.n_z_acc == 48 and table.n_log_ratio == 32
    assert table.n_integration == 129 and table.n_iterations == 3


@pytest.mark.parametrize("change", ["mass", "rate", "array", "callable"])
def test_mutation_rejects_retained_table_and_rebuilds_owned_cache(change):
    """Invalidate tidal approximations after physical host inputs change.

    Parameters
    ----------
    change : object
        Controlled input supplied by the parametrization or fixture for the specific
        physical/reference case described below.

    Notes
    -----
    Pass criteria: Changing mass, rate, array content or callback rejects retained tables;
    rebuilding recovers the analytic mass with updated coefficients.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
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
    np.testing.assert_allclose(
        new.mass(1e6, 0.5), 1e6 * np.exp(-0.5 * host.Phi(0.0)), rtol=4e-15
    )


def test_real_solver_mass_setter_and_legacy_table_rejection():
    """Protect actual CDM host changes and tables lacking a verifiable state key.

    Notes
    -----
    Pass criteria: Changing M0 rejects the old table; a table with its state key removed is
    also rejected.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
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
    """Separate numerical cache contents from hidden physical provider state.

    Notes
    -----
    Pass criteria: Adding numerical caches leaves the key stable, while changing a custom
    provider key invalidates the retained table.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
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
    """Invalidate stripping tables when the cosmological provider or its parameters change.

    Notes
    -----
    Pass criteria: Both changing a backend scalar and replacing the backend change the
    physics key.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    host = AnalyticHost()
    host.itamae_cosmology = AnalyticHost()
    key = physics_key(host)
    host.itamae_cosmology.rate = 0.7
    assert physics_key(host) != key
    key = physics_key(host)
    host.itamae_cosmology = AnalyticHost()
    assert physics_key(host) != key
