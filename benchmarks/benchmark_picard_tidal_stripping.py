"""Benchmark the precomputed Picard tidal-stripping table.

Run from the repository root:

    python benchmarks/benchmark_picard_tidal_stripping.py

The benchmark reports table-build time, vectorized lookup time, and sampled
accuracy against a high-accuracy ODE for ln(m). It also times the historical
``pert2_shanks`` method on the same catalog-shaped input.
"""

from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from picard_tidal_stripping import PicardTidalStrippingTable
from sashimi_c import TidalStrippingSolver


def reference_log_ode_mass(solver, ma, z_acc, z_final=0.0):
    """Integrate ln(m) directly with DOP853 for a sampled tidal endpoint comparison.

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
        Endpoint mass in the input unit using rtol=1e-11 and atol=1e-12 in log mass.

    Notes
    -----
    Comparisons use identical model inputs for the named numerical solvers. Timing and
    sampled mass differences are numerical diagnostics; they do not establish global
    convergence or empirical accuracy of the tidal-loss prescription.
    """

    def rhs(z, y):
        """Evaluate the one-state log-mass tidal derivative at the current redshift.

        Parameters
        ----------
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.
        y : array_like
            One-element natural-log mass state in the fixed internal mass unit.

        Returns
        -------
        list of float
            One derivative d ln(m)/dz, for the SciPy reference integration.

        Notes
        -----
        Comparisons use identical model inputs for the named numerical solvers. Timing and
        sampled mass differences are numerical diagnostics; they do not establish global
        convergence or empirical accuracy of the tidal-loss prescription.
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
    return np.exp(result.y[0, -1])


def main():
    """Measure explicit table construction/lookup costs and sampled endpoint differences
    against DOP853.

    Notes
    -----
    Comparisons use identical model inputs for the named numerical solvers. Timing and
    sampled mass differences are numerical diagnostics; they do not establish global
    convergence or empirical accuracy of the tidal-loss prescription.
    """
    solver = TidalStrippingSolver(M0=1.0e12, z_min=0.0, z_max=7.0, n_z_interp=64)

    start = time.perf_counter()
    table = PicardTidalStrippingTable(
        solver,
        n_z_acc=96,
        n_log_ratio=96,
        log10_ratio_min=-18.0,
        log10_ratio_max=-0.5,
        n_integration=128,
        n_iterations=3,
    )
    build_time = time.perf_counter() - start

    z_acc = np.arange(0.01, 7.0 + 0.01, 0.01)
    ratios = np.logspace(-18.0, -1.0, 500)
    za_grid, ratio_grid = np.meshgrid(z_acc, ratios, indexing="ij")
    masses = ratio_grid * solver.Mzvir(za_grid)

    start = time.perf_counter()
    picard_mass = table.mass(masses, za_grid)
    lookup_time = time.perf_counter() - start

    start = time.perf_counter()
    current_mass = np.empty_like(masses)
    for i, za in enumerate(z_acc):
        current_mass[i] = solver.subhalo_mass_stripped(
            masses[i], za, 0.0, method="pert2_shanks"
        )
    current_time = time.perf_counter() - start

    sample_z = [0.5, 1.0, 3.0, 5.0, 7.0]
    sample_ratio = [1.0e-12, 1.0e-6, 1.0e-2]
    errors = []
    current_errors = []
    for za in sample_z:
        for ratio in sample_ratio:
            ma = ratio * solver.Mzvir(za)
            reference = reference_log_ode_mass(solver, ma, za)
            errors.append(abs(table.mass(ma, za) / reference - 1.0))
            current = solver.subhalo_mass_stripped(ma, za, 0.0, method="pert2_shanks")
            current_errors.append(
                abs(float(np.asarray(current).squeeze()) / reference - 1.0)
            )

    print(f"Picard table build:        {build_time:.6f} s")
    print(f"Picard lookup, 350k pts:  {lookup_time:.6f} s")
    print(f"pert2_shanks, 350k pts:   {current_time:.6f} s")
    print(f"Picard sampled max error: {max(errors):.6e}")
    print(f"Shanks sampled max error: {max(current_errors):.6e}")
    print(f"Output shape: {picard_mass.shape}")


if __name__ == "__main__":
    main()
