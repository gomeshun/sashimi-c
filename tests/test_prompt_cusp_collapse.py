"""Protect zero-collapse luminosities and the retained nonempty-sample contract."""

import warnings

import numpy as np
import pytest

from sashimi_c import SubhaloObservables
from sashimi_c.prompt_cusps import prompt_cusps


def _fixed_spectral_moments(self):
    """Supply controlled Gaussian-peak moments without an external power table.

    Parameters
    ----------
    self : prompt_cusps
        Cusp model whose spectrum is replaced only for the scoped test.

    Returns
    -------
    sigma : numpy.ndarray, shape (3,)
        Synthetic spectral moments in internal length**(-j), j=0, 1, 2.
    R_s : float
        Comoving correlation length sqrt(3)*sigma_1/sigma_2.
    gamma : float
        Dimensionless sigma_1**2/(sigma_0*sigma_2), equal to 0.1.

    Notes
    -----
    The moments (10, 1e6, 1e12) are numerical controls, not a published
    spectrum. They satisfy 0 < gamma < 1 so the real BBKS peak integration
    remains finite. This helper isolates collapse selection and luminosity
    handling; it does not test spectrum interpolation or cosmology.
    """
    sigma = np.array([10.0, 1e6, 1e12])
    return sigma, np.sqrt(3.0) * sigma[1] / sigma[2], 0.1


@pytest.fixture
def cusp_model(tmp_path, monkeypatch):
    """Construct a real cusp model with isolated caches and controlled moments.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-case directory for five-column peak caches; no user cache is read.
    monkeypatch : pytest.MonkeyPatch
        Restore the spectrum provider after each test, including observable
        calls that construct another cusp model.

    Returns
    -------
    prompt_cusps
        Actual model with a synthetic cutoff of 1e6 inverse internal length;
        the cutoff is unused by the fixed spectral-moment provider.

    Notes
    -----
    Only the spectral moments are replaced. Cache loading, collapse masks,
    BBKS integration, thermal/core quantities and luminosities execute the
    production code. Pass/fail conditions are specified in each test. This
    fixture is not an independent physical calibration.
    """
    monkeypatch.setattr(prompt_cusps, "sigma_function", _fixed_spectral_moments)
    return prompt_cusps(k_fs=1e6, data_dir=tmp_path)


@pytest.mark.parametrize("rejection", ["epoch", "shape", "failed-root"])
def test_no_collapsed_peaks_return_finite_scalar_zero(cusp_model, rejection):
    """Return zero luminosity when a nonempty peak sample has no collapsed cusps.

    Parameters
    ----------
    cusp_model : prompt_cusps
        Real model with controlled moments and an isolated peak-cache directory.
    rejection : {'epoch', 'shape', 'failed-root'}
        Reject both peaks through an early observation epoch, excessive
        ellipticity, or NaN collapse roots as produced by a failed root solve.

    Notes
    -----
    Two cached peaks have delta=10 and 20. At z=100 they fail the epoch
    threshold; at z=0 ellipticity e=1 fails the shape cut, or NaN f_ec fails
    the collapse comparison. No random draw or solver outcome is assumed.

    Pass requires a scalar f_coll=0 and a Python float J_cusp=0.0 with
    shape (), finite mean and zero population-weighted luminosity. Runtime
    warnings and invalid/divide/overflow floating-point operations fail the
    call. Exact zeros are an empty-population identity, not an accuracy
    tolerance. A nonempty input with no collapsed peaks is distinct from
    requesting zero Monte Carlo draws, whose fraction is undefined.

    This checks the return contract used by the boost consumer and prevents
    an undefined conditional mean from contaminating zero populations. It
    does not establish that a finite random sample proves the true collapse
    probability is zero, or calibrate the collapse prescription.
    """
    samples = np.array([[1.0, 1.0, 0.0, 0.0, 1.0], [2.0, 2.0, 0.0, 0.0, 1.0]])
    redshift = 100.0 if rejection == "epoch" else 0.0
    if rejection == "shape":
        samples[:, 2] = 1.0
    elif rejection == "failed-root":
        samples[:, 4] = np.nan
    cache_file = cusp_model.data_dir / "peaks.txt"
    np.savetxt(cache_file, samples)

    with (
        warnings.catch_warnings(),
        np.errstate(divide="raise", invalid="raise", over="raise"),
    ):
        warnings.simplefilter("error", RuntimeWarning)
        f_coll, luminosity = cusp_model.cusp_properties(
            z=redshift, mc_cache_file=cache_file
        )
        mean_luminosity = np.mean(luminosity)

    assert np.isscalar(f_coll) and np.shape(f_coll) == ()
    assert np.isfinite(f_coll) and f_coll == 0.0
    assert isinstance(luminosity, float) and np.shape(luminosity) == ()
    assert np.isfinite(luminosity) and luminosity == 0.0
    assert np.isfinite(mean_luminosity) and mean_luminosity == 0.0
    assert mean_luminosity * f_coll == 0.0


@pytest.mark.parametrize("collapsed_count", [1, 2])
@pytest.mark.parametrize("redshift", [0.0, 5.0])
@pytest.mark.parametrize("survival", [0.0, 1.0])
def test_nonempty_cusp_luminosities_preserve_baseline(
    cusp_model, collapsed_count, redshift, survival
):
    """Preserve individual cusp luminosities, selection order and array shape.

    Parameters
    ----------
    cusp_model : prompt_cusps
        Real model with controlled moments and an isolated peak cache.
    collapsed_count : {1, 2}
        Number of the three cached peaks that should collapse.
    redshift : {0.0, 5.0}
        Observation epoch; both epochs follow the selected peaks' collapse.
    survival : {0.0, 1.0}
        Population survival factor; it must not rescale an individual cusp.

    Notes
    -----
    The three peaks have (x, nu)=(1, 1), (2, 2), (3, 0.01), e=p=0 and
    f_ec=1, except that f_ec=20 rejects the second peak in the one-cusp
    case. The low-height third peak never collapses at either epoch.

    Expected luminosities were evaluated before the fix at owner PR head
    de061d456c4803d0682c661da5fdbada3dfcfac4 using these same controlled
    moments and caches. They are in internal mass**2/length**3. This is a
    historical regression reference, not an independent physical accuracy
    bound or calibration. The adopted luminosity/core prescription and its
    domain are documented with primary references in cusp_properties.

    Pass requires f_coll=collapsed_count/3, a floating one-dimensional
    array with exactly that length and the baseline values within relative
    tolerance 2e-13 (near-roundoff/platform allowance, zero absolute floor).
    Shape/value mismatches, scalar collapse of a one-element array,
    survival rescaling or numerical failures fail. Comparing two epochs
    also checks that already-formed cusp properties do not acquire an
    observation-time dependence. These cases do not test spectrum sampling
    or validate the prompt-cusp approximation against simulations.
    """
    samples = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [2.0, 2.0, 0.0, 0.0, 1.0],
            [3.0, 0.01, 0.0, 0.0, 1.0],
        ]
    )
    if collapsed_count == 1:
        samples[1, 4] = 20.0
    cache_file = cusp_model.data_dir / "peaks.txt"
    np.savetxt(cache_file, samples)

    f_coll, luminosity = cusp_model.cusp_properties(
        z=redshift, f_surv=survival, mc_cache_file=cache_file
    )

    assert np.isscalar(f_coll) and np.shape(f_coll) == ()
    assert f_coll == collapsed_count / 3.0
    assert isinstance(luminosity, np.ndarray)
    assert luminosity.shape == (collapsed_count,)
    assert np.issubdtype(luminosity.dtype, np.floating)
    assert np.all(np.isfinite(luminosity)) and np.all(luminosity > 0.0)
    baseline = np.array([815422340804.9146, 7699826193373.589])
    np.testing.assert_allclose(
        luminosity, baseline[:collapsed_count], rtol=2e-13, atol=0.0
    )


def test_no_collapsed_cusps_give_zero_boost_contributions(cusp_model):
    """Keep the observable finite and cusp boosts zero for an empty cusp sample.

    Parameters
    ----------
    cusp_model : prompt_cusps
        Supplies isolated data/cache paths and the scoped spectral provider.

    Notes
    -----
    Cache two peaks whose e=1, p=0 fail the shape cut, then call the actual
    n=0 prompt-cusp boost consumer on a small 1e10-Msun CDM host catalogue.
    Catalogue grid sizes and masses are synthetic numerical controls. The
    standard catalogue uses its analytic variance; only the cusp spectral
    moments are supplied by the fixture, avoiding an external power table.

    Pass requires the unchanged six-scalar return shape, all finite values,
    a positive smooth subhalo boost, and exactly zero dressed/naked cusp
    boosts. The total must equal 1-f_sh**2+B_sh to relative 1e-14, allowing
    rounding in that algebraic identity. Runtime warnings or invalid,
    divide-by-zero or overflow operations fail. An empty ndarray sentinel
    would cause np.mean to warn/produce NaN, so this protects real consumer
    behavior in addition to the direct return contract. It does not test
    recursive boost tables, spectrum integration or physical calibration.
    """
    cache_file = cusp_model.data_dir / "prompt_cusps" / "prompt_cusps_Monte_Carlo.txt"
    cache_file.parent.mkdir()
    np.savetxt(cache_file, [[1.0, 1.0, 1.0, 0.0, 1.0], [2.0, 2.0, 1.0, 0.0, 1.0]])
    observable = SubhaloObservables(
        1e10,
        dz=0.5,
        zmax=1.0,
        N_ma=4,
        N_herm=2,
        N_hermNa=3,
        logmamin=6.0,
        logmamax=8.0,
        data_dir=cusp_model.data_dir,
    )

    with (
        warnings.catch_warnings(),
        np.errstate(divide="raise", invalid="raise", over="raise"),
    ):
        warnings.simplefilter("error", RuntimeWarning)
        result = observable.annihilation_boost_factor_prompt_cusps(n=0)

    assert isinstance(result, tuple) and len(result) == 6
    assert all(np.shape(value) == () and np.isfinite(value) for value in result)
    subhalo_boost, dressed_boost, naked_boost, ratio, _, _ = result
    assert subhalo_boost > 0.0
    assert dressed_boost == 0.0 and naked_boost == 0.0
    np.testing.assert_allclose(
        ratio,
        1.0 - observable.mass_fraction() ** 2 + subhalo_boost,
        rtol=1e-14,
        atol=0.0,
    )
