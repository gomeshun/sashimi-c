"""Keep physical tidal values while sharing the numerical controller."""

import numpy as np
import pytest
from scipy.integrate import odeint

from sashimi_c import TidalStrippingSolver


@pytest.fixture
def solver():
    """Construct a short-interval CDM tidal solver for direct SciPy equivalence checks.

    Returns
    -------
    TidalStrippingSolver
        1e10-Msun host with z_max=1.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    return TidalStrippingSolver(1e10, z_max=1.0)


@pytest.mark.parametrize("controls", [{}, {"rtol": 1e-9, "atol": 1e-8, "hmax": 0.02}])
def test_tidal_solver_preserves_direct_scipy_values(solver, controls):
    """Preserve the variant's physical stripping equation when using the common ODE controller.

    Parameters
    ----------
    solver : object
        Variant tidal solver initialized by the local fixture with the stated host mass and
        redshift range.
    controls : dict
        Explicit ODE tolerances or step limits compared with the same direct SciPy
        integration.

    Notes
    -----
    Pass criteria: With default and explicit controls, the variant's returned history or
    endpoint is exactly equal to direct SciPy odeint on the same equation and grid.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    masses = np.array([1e6, 1e7])
    grid = np.linspace(0.8, 0.0, 100)
    expected = odeint(solver.msolve, masses, grid, **controls)[-1]
    actual = solver.subhalo_mass_stripped_odeint(masses, 0.8, 0.0, **controls)
    np.testing.assert_array_equal(actual, expected)


def test_zero_evolution_preserves_initial_masses(solver):
    """Keep the accreted mass unchanged when no cosmic evolution is requested.

    Parameters
    ----------
    solver : object
        Variant tidal solver initialized by the local fixture with the stated host mass and
        redshift range.

    Notes
    -----
    Pass criteria: Equal accretion and final redshifts return the input masses exactly.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    masses = np.array([1e6, 1e7])
    np.testing.assert_array_equal(
        solver.subhalo_mass_stripped_odeint(masses, 0.8, 0.8), masses
    )


def test_failed_integration_does_not_return_mass_values(solver):
    """Prevent failed physical stripping from being reported as a successful mass calculation.

    Parameters
    ----------
    solver : object
        Variant tidal solver initialized by the local fixture with the stated host mass and
        redshift range.

    Notes
    -----
    Pass criteria: Restricting odeint to mxstep=1 raises RuntimeError rather than returning
    masses.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    with pytest.raises(RuntimeError, match="odeint"):
        solver.subhalo_mass_stripped_odeint([1e6], 0.8, 0.0, mxstep=1)


def test_state_first_jacobian_adapter_preserves_stiff_solution(solver, monkeypatch):
    """Adapt the historical state-first Jacobian without changing stiff evolution.

    Parameters
    ----------
    solver : object
        Variant tidal solver initialized by the local fixture with the stated host mass and
        redshift range.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: For two decay rates 1 and 1000, the supplied Jacobian is actually called
    with correct argument shapes and the result equals direct odeint.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    rates = np.array([1.0, 1000.0])
    calls = []

    def rhs(state, time):
        """Supply a stiff diagonal decay equation through the historical state-first interface.

        Parameters
        ----------
        state : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        time : float or array_like
            Evolution coordinate or cosmic time, in the units of the supplied equation;
            canonical state times are in Gyr.

        Returns
        -------
        numpy.ndarray
            -rates*state for the controlled two-component system.

        Notes
        -----
        Adapt the historical state-first Jacobian without changing stiff evolution. Pass
        criteria: For two decay rates 1 and 1000, the supplied Jacobian is actually called with
        correct argument shapes and the result equals direct odeint. Failure occurs when the
        stated comparison or expected rejection is not met. This controlled calculation does not
        calibrate a subhalo population against simulations or observations.
        """
        return -rates * state

    def jacobian(state, time):
        """Check historical argument ordering and return the exact stiff decay Jacobian.

        Parameters
        ----------
        state : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        time : float or array_like
            Evolution coordinate or cosmic time, in the units of the supplied equation;
            canonical state times are in Gyr.

        Returns
        -------
        numpy.ndarray
            Negative diagonal rate matrix; the callback also records scalar evaluation times.

        Notes
        -----
        Adapt the historical state-first Jacobian without changing stiff evolution. Pass
        criteria: For two decay rates 1 and 1000, the supplied Jacobian is actually called with
        correct argument shapes and the result equals direct odeint. Failure occurs when the
        stated comparison or expected rejection is not met. This controlled calculation does not
        calibrate a subhalo population against simulations or observations.
        """
        assert np.shape(state) == (2,)
        assert np.ndim(time) == 0
        calls.append(time)
        return -np.diag(rates)

    monkeypatch.setattr(solver, "msolve", rhs)
    grid = np.linspace(0.0, 1.0, 100)
    expected = odeint(rhs, [1.0, 1.0], grid, Dfun=jacobian)
    calls.clear()
    actual = solver.subhalo_mass_stripped_odeint([1.0, 1.0], 0.0, 1.0, Dfun=jacobian)
    assert calls
    np.testing.assert_array_equal(actual, expected[-1])
