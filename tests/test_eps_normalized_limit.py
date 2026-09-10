"""The normalized Yang kernel has a finite zero barrier-gap limit."""

import numpy as np
import pytest
from scipy.integrate import quad

from sashimi_c import SubhaloProperties


@pytest.mark.parametrize("gap", [0., 1e-12, 1e-6, 1e-3, 1., 10.])
def test_normalized_yang_kernel_against_independent_definite_integral(gap):
    model = SubhaloProperties()
    ds, dsmin = 8., 3.
    x = gap / np.sqrt(2 * dsmin)
    normalization = quad(lambda t: np.exp(-(x*t)**2), 0., 1., epsabs=1e-14)[0]
    expected = np.sqrt(dsmin) / (2 * ds**1.5) * np.exp(-gap**2 / (2*ds)) / normalization
    actual = model._normalized_yang_kernel(1., 1.+gap, 1., 1.+ds, 1.+dsmin)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.)


def test_host_quadrature_refinement_includes_finite_zero_gap_nodes():
    model = SubhaloProperties()
    redshift = np.arange(.25, 3.25, .25)
    m200 = model.Mzi(np.geomspace(1e6, 1e10, 16), redshift[:, None])
    mass = model.Mvir_from_M200_fit(m200, redshift[:, None])
    with np.errstate(all="raise"):
        accretion = model.Na_calc(mass, redshift, 1e12, N_herm=64)
    assert accretion.shape == mass.shape
    assert np.all(np.isfinite(accretion))
    assert np.all(accretion >= 0)


@pytest.mark.parametrize("prescription", [1, 2, 3])
def test_invalid_active_variance_is_not_replaced_by_zero(monkeypatch, prescription):
    model = SubhaloProperties()
    monkeypatch.setattr(model, "s_func", lambda mass: np.ones_like(mass))
    with np.errstate(divide="ignore", invalid="ignore"), pytest.raises(ValueError, match="variance|EPS"):
        model.Na_calc(np.array([1e6, 1e7]), np.array([.25, .5]), 1e12,
                      N_herm=3, Na_model=prescription)


@pytest.mark.parametrize("prescription", [1, 2])
@pytest.mark.parametrize("order", [1, 4])
def test_zero_width_normalization_support_has_no_runtime_warning(prescription, order):
    model = SubhaloProperties()
    redshift = np.arange(.25, 3.25, .25)
    mass = np.geomspace(1e6, 1e10, 16)
    with np.errstate(all="raise"):
        accretion = model.Na_calc(mass, redshift, 1e12, N_herm=order, Na_model=prescription)
    assert accretion.shape == (redshift.size, mass.size)
    assert np.all(np.isfinite(accretion))
    assert np.all(accretion >= 0)
