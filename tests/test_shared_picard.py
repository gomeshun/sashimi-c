"""The CDM production default and public controls use ITAMAE Picard."""

import numpy as np
import pytest
from itamae.evolution import PicardFallbackWarning, PicardMassLossSolver

from sashimi_c import SubhaloProperties, TidalStrippingSolver
from sashimi_c.picard_tidal_stripping import get_picard_solver


def test_public_default_covers_off_grid_masses_and_redshifts():
    """Check the production CDM default away from interpolation knots.

    Notes
    -----
    Pass criteria: At z_acc=0.03,0.9,3.17,6.9 and ratios 1e-20..1, default masses agree with
    direct DOP853 to rtol=1e-3 and use the shared solver.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    solver = TidalStrippingSolver(1e12)
    for za in [0.03, 0.9, 3.17, 6.9]:
        masses = solver.Mzvir(za) * np.array([1e-20, 1e-12, 1e-6, 1e-2, 1.0])
        default = solver.subhalo_mass_stripped(masses, za, 0.0)
        direct = solver.subhalo_mass_stripped(masses, za, 0.0, method="dop853")
        np.testing.assert_allclose(default, direct, rtol=1e-3)
    assert isinstance(get_picard_solver(solver), PicardMassLossSolver)


def test_solver_options_are_applied_and_recorded_by_catalog():
    """Make a caller's refinement choices visible in the resulting CDM catalogue.

    Notes
    -----
    Pass criteria: Four iterations and 257 integration nodes appear in both actual table
    settings and recorded solver options.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    catalog = SubhaloProperties().subhalo_catalog_calc(
        M0=1e10,
        dz=0.5,
        zmax=1.0,
        N_ma=4,
        N_herm=2,
        N_hermNa=2,
        logmamin=5.0,
        logmamax=7.0,
        n_iterations=4,
        n_integration=257,
    )
    settings = catalog.metadata["picard_table_settings"]
    assert settings["n_iterations"] == 4 and settings["n_integration"] == 257
    assert catalog.metadata["solver_options"] == {
        "n_iterations": 4,
        "n_integration": 257,
    }


def test_fallback_is_recorded_and_unknown_options_are_rejected():
    """Expose a failed Picard convergence check and reject irrelevant solver controls.

    Notes
    -----
    Pass criteria: A deliberately strict one-iteration query warns and counts one fallback
    mass; an unsupported mxstep argument raises TypeError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    solver = TidalStrippingSolver(1e12)
    with pytest.warns(PicardFallbackWarning, match="convergence"):
        solver.subhalo_mass_stripped(
            [1e6], 5.0, 0.0, n_iterations=1, convergence_tolerance=1e-15
        )
    assert get_picard_solver(solver).diagnostics()["fallback_mass_count"] == 1
    with pytest.raises(TypeError):
        solver.subhalo_mass_stripped([1e6], 5.0, 0.0, mxstep=1)
