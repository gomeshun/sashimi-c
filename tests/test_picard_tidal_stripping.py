import numpy as np
import pytest
from scipy.integrate import solve_ivp

from picard_tidal_stripping import PicardTidalStrippingTable
from sashimi_c import TidalStrippingSolver


@pytest.fixture(scope="module")
def solver():
    """Prepare the calibrated 1e12-Msun CDM host for endpoint accuracy checks.

    Returns
    -------
    TidalStrippingSolver
        Host tidal law over z in [0,7], with retained epsilon-table resolution 64.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    return TidalStrippingSolver(M0=1.0e12, z_min=0.0, z_max=7.0, n_z_interp=64)


@pytest.fixture(scope="module")
def picard3(solver):
    """Build the explicit refined three-update endpoint table used in the sampled direct-ODE
    comparison.

    Parameters
    ----------
    solver : TidalStrippingSolver or object
        C-owned tidal coefficient provider exposing Phi(z), zetaMz(z), Mzvir(z) and its
        mutable physical state.

    Returns
    -------
    PicardTidalStrippingTable
        96-by-96 endpoint grid with 128 integration nodes and the stated mass-ratio support.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
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
    """High-accuracy reference obtained by integrating ln(m), not m itself.

    Parameters
    ----------
    solver : TidalStrippingSolver or object
        C-owned tidal coefficient provider exposing Phi(z), zetaMz(z), Mzvir(z) and its
        mutable physical state.
    ma : float or array_like
        Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
        queries broadcast over mass and redshift.
    z_acc : float or array_like
        Dimensionless accretion redshift. History integration requires one scalar accretion
        epoch.
    z_final : float or array_like, optional
        Target redshift no greater than the accretion redshift. Explicit tables use one
        scalar target; None selects solver.z_min.

    Returns
    -------
    float
        DOP853 endpoint mass after asserting integration success at tight log-mass
        tolerances.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """

    def rhs(z, y):
        """Evaluate the direct-reference derivative of the natural-log mass.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        y : array_like
            One-element natural-log mass vector integrated by SciPy.

        Returns
        -------
        list of float
            One d ln(m)/dz value under the same host and tidal coefficients.

        Notes
        -----
        This fixture/helper supports the controlled checks in this test module; its synthetic
        values are not a calibrated halo model.
        """
        return [
            solver.Phi(z) * np.exp(solver.zetaMz(z) * (y[0] - np.log(solver.Mzvir(z))))
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
    """Bound CDM stripping error against independent direct log-mass integration.

    Parameters
    ----------
    solver : object
        Variant tidal solver initialized by the local fixture with the stated host mass and
        redshift range.
    picard3 : PicardTidalStrippingTable
        Explicit refined three-iteration CDM table supplied by the local fixture.
    z_acc : float
        Dimensionless accretion redshift for the direct tidal-reference comparison.
    mass_ratio : float
        Accretion subhalo mass divided by host virial mass at accretion.

    Notes
    -----
    Pass criteria: Across five accretion redshifts and three mass ratios, the refined
    three-iteration table differs from DOP853 by less than 1e-3.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    ma = mass_ratio * solver.Mzvir(z_acc)
    reference = reference_log_ode_mass(solver, ma, z_acc)
    approximate = picard3.mass(ma, z_acc)
    relative_error = abs(approximate / reference - 1.0)

    assert relative_error < 1.0e-3


def test_three_iterations_improve_challenging_high_redshift_case(solver):
    """Check whether another Picard update improves a difficult CDM history.

    Parameters
    ----------
    solver : object
        Variant tidal solver initialized by the local fixture with the stated host mass and
        redshift range.

    Notes
    -----
    Pass criteria: At z_acc=7 and mass ratio 1e-12, three iterations improve on two and stay
    within 1e-3 of DOP853.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    common = {
        "solver": solver,
        "n_z_acc": 96,
        "n_log_ratio": 96,
        "log10_ratio_min": -18.0,
        "log10_ratio_max": -0.5,
        "n_integration": 128,
    }
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
    """Return physical stripped masses only inside an explicit CDM table's support.

    Parameters
    ----------
    solver : object
        Variant tidal solver initialized by the local fixture with the stated host mass and
        redshift range.
    picard3 : PicardTidalStrippingTable
        Explicit refined three-iteration CDM table supplied by the local fixture.

    Notes
    -----
    Pass criteria: Vectorized masses keep shape, remain positive and no larger than
    accretion mass; out-of-range mass ratios and redshifts raise ValueError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
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
