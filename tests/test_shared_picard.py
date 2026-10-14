"""The CDM production default and public controls use ITAMAE Picard."""

import numpy as np
import pytest
from itamae.evolution import PicardFallbackWarning, PicardMassLossSolver

from sashimi_c import SubhaloProperties, TidalStrippingSolver
from sashimi_c.picard_tidal_stripping import get_picard_solver


def test_public_default_covers_off_grid_masses_and_redshifts():
    solver = TidalStrippingSolver(1e12)
    for za in [0.03, 0.9, 3.17, 6.9]:
        masses = solver.Mzvir(za) * np.array([1e-20, 1e-12, 1e-6, 1e-2, 1.])
        default = solver.subhalo_mass_stripped(masses, za, 0.)
        direct = solver.subhalo_mass_stripped(masses, za, 0., method='dop853')
        np.testing.assert_allclose(default, direct, rtol=1e-3)
    assert isinstance(get_picard_solver(solver), PicardMassLossSolver)


def test_solver_options_are_applied_and_recorded_by_catalog():
    catalog = SubhaloProperties().subhalo_catalog_calc(
        M0=1e10, dz=.5, zmax=1., N_ma=4, N_herm=2, N_hermNa=2,
        logmamin=5., logmamax=7., n_iterations=4, n_integration=257)
    settings = catalog.metadata['picard_table_settings']
    assert settings['n_iterations'] == 4 and settings['n_integration'] == 257
    assert catalog.metadata['solver_options'] == {'n_iterations': 4, 'n_integration': 257}


def test_fallback_is_recorded_and_unknown_options_are_rejected():
    solver = TidalStrippingSolver(1e12)
    with pytest.warns(PicardFallbackWarning, match='convergence'):
        solver.subhalo_mass_stripped([1e6], 5., 0., n_iterations=1, convergence_tolerance=1e-15)
    assert get_picard_solver(solver).diagnostics()['fallback_mass_count'] == 1
    with pytest.raises(TypeError):
        solver.subhalo_mass_stripped([1e6], 5., 0., mxstep=1)
