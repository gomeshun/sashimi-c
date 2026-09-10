"""Seeded realization and explicit scientific-file boundaries."""

import numpy as np
import pytest

from prompt_cusps import prompt_cusps
from sashimi_c import HaloModel, SubhaloObservables


@pytest.fixture(scope="module")
def observable():
    return SubhaloObservables(
        1e10, dz=0.5, zmax=1.0, N_ma=4, N_herm=2, N_hermNa=3, logmamin=6.0, logmamax=8.0
    )


def test_seeded_mc_does_not_consume_global_random_state(observable):
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
    assert all(x.size == 0 for x in observable.subhalo_catalog_MC(1e20, seed=1))
    with pytest.raises(ValueError, match="either"):
        observable.subhalo_catalog_MC(1e6, rng=np.random.default_rng(1), seed=1)
    with pytest.raises(ValueError, match="threshold"):
        observable.subhalo_catalog_MC(float("nan"), seed=1)


def test_missing_prompt_spectra_raise_explicit_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match=r"Planck2018_CAMB_extrap\.dat"):
        HaloModel(prompt_cusps=True, data_dir=tmp_path)
    with pytest.raises(RuntimeError, match=r"powerspectrum31\.txt"):
        prompt_cusps(k_fs=1e6, data_dir=tmp_path).powerspectrum(np.array([1.0]))
