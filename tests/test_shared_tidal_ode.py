"""Keep physical tidal values while sharing the numerical controller."""

import numpy as np
import pytest
from scipy.integrate import odeint

from sashimi_c import TidalStrippingSolver


@pytest.fixture
def solver():
    return TidalStrippingSolver(1e10, z_max=1.0)


@pytest.mark.parametrize("controls", [{}, {"rtol": 1e-9, "atol": 1e-8, "hmax": 0.02}])
def test_tidal_solver_preserves_direct_scipy_values(solver, controls):
    masses = np.array([1e6, 1e7])
    grid = np.linspace(0.8, 0.0, 100)
    expected = odeint(solver.msolve, masses, grid, **controls)[-1]
    actual = solver.subhalo_mass_stripped_odeint(masses, 0.8, 0.0, **controls)
    np.testing.assert_array_equal(actual, expected)


def test_zero_evolution_preserves_initial_masses(solver):
    masses = np.array([1e6, 1e7])
    np.testing.assert_array_equal(
        solver.subhalo_mass_stripped_odeint(masses, 0.8, 0.8), masses
    )


def test_failed_integration_does_not_return_mass_values(solver):
    with pytest.raises(RuntimeError, match="odeint"):
        solver.subhalo_mass_stripped_odeint([1e6], 0.8, 0.0, mxstep=1)


def test_state_first_jacobian_adapter_preserves_stiff_solution(solver, monkeypatch):
    rates = np.array([1.0, 1000.0])
    calls = []

    def rhs(state, time):
        return -rates * state

    def jacobian(state, time):
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
