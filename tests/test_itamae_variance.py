"""Regression tests for the SASHIMI-C variance protocol adapter."""

import numpy as np

from sashimi_c import halo_model
from sashimi_c_itamae_variance import make_variance_model


def test_c_variance_adapter_matches_legacy_model() -> None:
    """Expose the CDM variance fit through ITAMAE without changing its physical values.

    Notes
    -----
    Pass criteria: At the three mass/redshift pairs, sigma and dS/dM are exactly forwarded
    and variance equals sigma squared to rtol=2e-15.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """

    legacy = halo_model()
    variance = make_variance_model(legacy)
    mass = np.array([1.0e7, 1.0e9, 1.0e11]) * legacy.Msun
    redshift = np.array([0.0, 1.0, 3.0])

    np.testing.assert_allclose(
        variance.sigma(mass, redshift),
        legacy.sigmaMz(mass, redshift),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        variance.variance(mass, redshift),
        legacy.sigmaMz(mass, redshift) ** 2,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        variance.dvariance_dmass(mass, redshift),
        legacy.dsdm(mass, redshift),
        rtol=0.0,
        atol=0.0,
    )
    assert variance.identifier == "sashimi-c:analytic-cdm-fit:v1"
