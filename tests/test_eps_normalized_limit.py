"""The normalized Yang kernel has a finite zero barrier-gap limit."""

import numpy as np
import pytest
from scipy.integrate import quad

from sashimi_c import SubhaloProperties


@pytest.mark.parametrize("gap", [0.0, 1e-12, 1e-6, 1e-3, 1.0, 10.0])
def test_normalized_yang_kernel_against_independent_definite_integral(gap):
    r"""Recover the finite normalized EPS limit when the collapse-barrier gap vanishes.

    Parameters
    ----------
    gap : float
        Collapse-barrier difference in the controlled EPS normalization case.

    Notes
    -----
    The independent definite integral is of the Yang et al. accretion kernel. Its tolerance
    tests algebra and quadrature near the limiting barrier, not the empirical merger rate.

    Pass criteria: For barrier gaps from zero to ten, the normalized Yang kernel agrees with
    an independent definite integral at the specified floating-point tolerance.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.

    References
    ----------

    .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
        https://arxiv.org/abs/1104.1757
    """
    model = SubhaloProperties()
    ds, dsmin = 8.0, 3.0
    x = gap / np.sqrt(2 * dsmin)
    normalization = quad(lambda t: np.exp(-((x * t) ** 2)), 0.0, 1.0, epsabs=1e-14)[0]
    expected = (
        np.sqrt(dsmin) / (2 * ds**1.5) * np.exp(-(gap**2) / (2 * ds)) / normalization
    )
    actual = model._normalized_yang_kernel(1.0, 1.0 + gap, 1.0, 1.0 + ds, 1.0 + dsmin)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)


def test_host_quadrature_refinement_includes_finite_zero_gap_nodes():
    """Keep accretion rates physical as host-history quadrature samples near-degenerate nodes.

    Notes
    -----
    Pass criteria: The refined host quadrature returns the expected mass-grid shape with
    finite nonnegative rates under strict floating-point handling.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model = SubhaloProperties()
    redshift = np.arange(0.25, 3.25, 0.25)
    m200 = model.Mzi(np.geomspace(1e6, 1e10, 16), redshift[:, None])
    mass = model.Mvir_from_M200_fit(m200, redshift[:, None])
    with np.errstate(all="raise"):
        accretion = model.Na_calc(mass, redshift, 1e12, N_herm=64)
    assert accretion.shape == mass.shape
    assert np.all(np.isfinite(accretion))
    assert np.all(accretion >= 0)


@pytest.mark.parametrize("prescription", [1, 2, 3])
def test_invalid_active_variance_is_not_replaced_by_zero(monkeypatch, prescription):
    """Reject a zero active variance interval rather than erasing its EPS contribution.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.
    prescription : int
        Selected EPS accretion prescription.

    Notes
    -----
    Pass criteria: Replacing the mass variance by a constant causes each supported active
    EPS prescription to raise ValueError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model = SubhaloProperties()
    monkeypatch.setattr(model, "s_func", lambda mass: np.ones_like(mass))
    with (
        np.errstate(divide="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="variance|EPS"),
    ):
        model.Na_calc(
            np.array([1e6, 1e7]),
            np.array([0.25, 0.5]),
            1e12,
            N_herm=3,
            Na_model=prescription,
        )


@pytest.mark.parametrize("prescription", [1, 2])
@pytest.mark.parametrize("order", [1, 4])
def test_zero_width_normalization_support_has_no_runtime_warning(prescription, order):
    """Handle genuine zero-support EPS normalization cells without undefined arithmetic.

    Parameters
    ----------
    prescription : int
        Selected EPS accretion prescription.
    order : int
        Quadrature or recursive-hierarchy order under test; malformed cases exercise
        validation.

    Notes
    -----
    Pass criteria: The tested prescriptions and host-quadrature orders return finite
    nonnegative rates with correct shape under strict floating-point handling.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model = SubhaloProperties()
    redshift = np.arange(0.25, 3.25, 0.25)
    mass = np.geomspace(1e6, 1e10, 16)
    with np.errstate(all="raise"):
        accretion = model.Na_calc(
            mass, redshift, 1e12, N_herm=order, Na_model=prescription
        )
    assert accretion.shape == (redshift.size, mass.size)
    assert np.all(np.isfinite(accretion))
    assert np.all(accretion >= 0)
