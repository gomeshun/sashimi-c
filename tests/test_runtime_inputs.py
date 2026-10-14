"""Seeded realization and explicit scientific-file boundaries."""

import numpy as np
import pytest

from prompt_cusps import prompt_cusps
from sashimi_c import HaloModel, SubhaloObservables


@pytest.fixture(scope="module")
def observable():
    """Construct a small positive CDM population for threshold and random-generator boundary
    tests.

    Returns
    -------
    SubhaloObservables
        Small 1e10-Msun host population with masses between 1e6 and 1e8 Msun.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    return SubhaloObservables(
        1e10, dz=0.5, zmax=1.0, N_ma=4, N_herm=2, N_hermNa=3, logmamin=6.0, logmamax=8.0
    )


def test_seeded_mc_does_not_consume_global_random_state(observable):
    """Reproduce a subhalo realization without perturbing other stochastic calculations.

    Parameters
    ----------
    observable : object
        Small physical subhalo population used to test observable selection and Monte Carlo
        realization.

    Notes
    -----
    Pass criteria: Equal seeds yield identical one-dimensional output arrays above the
    selected mass cut and leave NumPy's global random state unchanged.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    np.random.seed(4321)
    before = np.random.get_state()
    a = observable.subhalo_catalog_MC(1e6, seed=20260910)
    b = observable.subhalo_catalog_MC(1e6, seed=20260910)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert all(x.ndim == 1 for x in a)
    assert np.all(a[4] > 1e6)


def test_mc_empty_population_and_rng_validation(observable):
    """Handle an empty selected population and unambiguous Monte Carlo inputs.

    Parameters
    ----------
    observable : object
        Small physical subhalo population used to test observable selection and Monte Carlo
        realization.

    Notes
    -----
    Pass criteria: A cut above all subhaloes returns empty arrays; simultaneous seed and
    generator or NaN threshold raises ValueError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    assert all(x.size == 0 for x in observable.subhalo_catalog_MC(1e20, seed=1))
    with pytest.raises(ValueError, match="either"):
        observable.subhalo_catalog_MC(1e6, rng=np.random.default_rng(1), seed=1)
    with pytest.raises(ValueError, match="threshold"):
        observable.subhalo_catalog_MC(float("nan"), seed=1)


def test_missing_prompt_spectra_raise_explicit_errors(tmp_path, monkeypatch):
    """Stop prompt-cusp calculations when the required physical spectra are unavailable.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: Missing Planck CAMB and prompt-cusp spectra raise explicit errors naming
    the required files.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match=r"Planck2018_CAMB_extrap\.dat"):
        HaloModel(prompt_cusps=True, data_dir=tmp_path)
    with pytest.raises(RuntimeError, match=r"powerspectrum31\.txt"):
        prompt_cusps(k_fs=1e6, data_dir=tmp_path).powerspectrum(np.array([1.0]))
