import numpy as np
import pytest
from scipy.integrate import solve_ivp

from picard_tidal_stripping import PicardTidalStrippingTable
from sashimi_c import TidalStrippingSolver


@pytest.fixture(scope="module")
def solver():
    return TidalStrippingSolver(M0=1.0e12, z_min=0.0, z_max=7.0, n_z_interp=64)


@pytest.fixture(scope="module")
def picard3(solver):
    return PicardTidalStrippingTable(
        solver,
        n_z_acc=96,
        n_log_ratio=96,
        log10_ratio_min=-18.0,
        log10_ratio_max=-0.5,
        n_integration=128,
        n_iterations=3,
    )


def reference_log_ode_mass(solver, ma, z_acc, z_final=0.0):
    """High-accuracy reference obtained by integrating ln(m), not m itself."""

    def rhs(z, y):
        return [
            solver.Phi(z)
            * np.exp(solver.zetaMz(z) * (y[0] - np.log(solver.Mzvir(z))))
        ]

    result = solve_ivp(
        rhs,
        (z_acc, z_final),
        [np.log(ma)],
        rtol=1.0e-11,
        atol=1.0e-12,
        method="DOP853",
    )
    assert result.success
    return np.exp(result.y[0, -1])


@pytest.mark.parametrize("z_acc", [0.5, 1.0, 3.0, 5.0, 7.0])
@pytest.mark.parametrize("mass_ratio", [1.0e-12, 1.0e-6, 1.0e-2])
def test_picard3_matches_log_ode_reference(solver, picard3, z_acc, mass_ratio):
    ma = mass_ratio * solver.Mzvir(z_acc)
    reference = reference_log_ode_mass(solver, ma, z_acc)
    approximate = picard3.mass(ma, z_acc)
    relative_error = abs(approximate / reference - 1.0)

    # The table is intended as a sub-per-mille approximation to the direct ODE.
    assert relative_error < 1.0e-3


def test_three_iterations_improve_challenging_high_redshift_case(solver):
    common = dict(
        solver=solver,
        n_z_acc=96,
        n_log_ratio=96,
        log10_ratio_min=-18.0,
        log10_ratio_max=-0.5,
        n_integration=128,
    )
    picard2 = PicardTidalStrippingTable(n_iterations=2, **common)
    picard3 = PicardTidalStrippingTable(n_iterations=3, **common)

    z_acc = 7.0
    ma = 1.0e-12 * solver.Mzvir(z_acc)
    reference = reference_log_ode_mass(solver, ma, z_acc)
    error2 = abs(picard2.mass(ma, z_acc) / reference - 1.0)
    error3 = abs(picard3.mass(ma, z_acc) / reference - 1.0)

    assert error3 < error2
    assert error3 < 1.0e-3


def test_vectorized_lookup_and_range_checks(solver, picard3):
    z_acc = 2.0
    ratios = np.array([1.0e-10, 1.0e-6, 1.0e-2])
    masses = ratios * solver.Mzvir(z_acc)

    result = picard3.mass(masses, z_acc)
    assert result.shape == masses.shape
    assert np.all(result > 0.0)
    assert np.all(result <= masses)

    with pytest.raises(ValueError):
        picard3.mass(1.0e-19 * solver.Mzvir(z_acc), z_acc)

    with pytest.raises(ValueError):
        picard3.mass(masses[0], 7.1)
