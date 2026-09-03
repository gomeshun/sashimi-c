"""Precomputed Picard solver for the SASHIMI-C tidal-stripping ODE.

This module keeps the nonlinear factor

    exp[zeta(z) * ln(m / Mvir)]

unexpanded and solves the corresponding integral equation by a small number of
Picard iterations.  The final mass-loss factor is precomputed on a two-
dimensional grid in accretion redshift and log mass ratio, then evaluated with
bilinear interpolation.

The implementation is intentionally separate from ``sashimi_c.py`` so it can be
benchmarked and tested without changing the existing production default.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator


class PicardTidalStrippingTable:
    """Precompute tidal mass loss using Picard iteration.

    Parameters
    ----------
    solver : object
        A ``TidalStrippingSolver``-like object providing ``Phi(z)``,
        ``zetaMz(z)``, and ``Mzvir(z)``.
    z_final : float, optional
        Final redshift. Defaults to ``solver.z_min``.
    z_acc_max : float, optional
        Largest accretion redshift tabulated. Defaults to ``solver.z_max``.
    n_z_acc : int, optional
        Number of accretion-redshift grid points.
    n_log_ratio : int, optional
        Number of grid points in ``log10(ma / Mvir(za))``.
    log10_ratio_min, log10_ratio_max : float, optional
        Tabulated range in accretion mass ratio.
    n_integration : int, optional
        Number of redshift points used in each Picard integral.
    n_iterations : int, optional
        Number of Picard updates after the zeta=0 starting solution. The
        default is three; this noticeably improves the high-redshift accuracy
        compared with two iterations while retaining a very small build cost.
    """

    def __init__(
        self,
        solver,
        z_final=None,
        z_acc_max=None,
        n_z_acc=96,
        n_log_ratio=96,
        log10_ratio_min=-20.0,
        log10_ratio_max=-0.5,
        n_integration=128,
        n_iterations=3,
    ):
        self.solver = solver
        self.z_final = solver.z_min if z_final is None else float(z_final)
        self.z_acc_max = solver.z_max if z_acc_max is None else float(z_acc_max)
        self.n_z_acc = int(n_z_acc)
        self.n_log_ratio = int(n_log_ratio)
        self.log10_ratio_min = float(log10_ratio_min)
        self.log10_ratio_max = float(log10_ratio_max)
        self.n_integration = int(n_integration)
        self.n_iterations = int(n_iterations)

        if self.z_acc_max < self.z_final:
            raise ValueError("z_acc_max must be >= z_final")
        if self.n_z_acc < 2 or self.n_log_ratio < 2:
            raise ValueError("table grids must contain at least two points")
        if self.n_integration < 2:
            raise ValueError("n_integration must be at least two")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be positive")
        if self.log10_ratio_max <= self.log10_ratio_min:
            raise ValueError("log10_ratio_max must exceed log10_ratio_min")

        self.z_acc_grid = np.linspace(self.z_final, self.z_acc_max, self.n_z_acc)
        self.log_ratio_grid = np.linspace(
            self.log10_ratio_min, self.log10_ratio_max, self.n_log_ratio
        )
        self._ln_ratio_grid = np.log(10.0) * self.log_ratio_grid

        self.delta_ln_mass = self._build_table()
        self._interpolator = RegularGridInterpolator(
            (self.z_acc_grid, self._ln_ratio_grid),
            self.delta_ln_mass,
            method="linear",
            bounds_error=True,
        )

    def _build_table(self):
        table = np.empty((self.n_z_acc, self.n_log_ratio), dtype=float)

        for i, z_acc in enumerate(self.z_acc_grid):
            if np.isclose(z_acc, self.z_final):
                table[i] = 0.0
                continue

            z_path = np.linspace(z_acc, self.z_final, self.n_integration)
            phi = np.asarray(self.solver.Phi(z_path), dtype=float)
            zeta = np.asarray(self.solver.zetaMz(z_path), dtype=float)
            ln_mvir = np.log(np.asarray(self.solver.Mzvir(z_path), dtype=float))

            ln_mvir_acc = float(np.log(self.solver.Mzvir(z_acc)))
            ln_ma = ln_mvir_acc + self._ln_ratio_grid

            # zeta = 0 solution used as the Picard starting point.
            zeroth = cumulative_trapezoid(phi, x=z_path, initial=0.0)
            ln_m = ln_ma[:, None] + zeroth[None, :]

            for _ in range(self.n_iterations):
                rhs = phi[None, :] * np.exp(
                    zeta[None, :] * (ln_m - ln_mvir[None, :])
                )
                integral = cumulative_trapezoid(
                    rhs, x=z_path, axis=1, initial=0.0
                )
                ln_m = ln_ma[:, None] + integral

            table[i] = ln_m[:, -1] - ln_ma

        return table

    def _points(self, ma, z_acc):
        ma = np.asarray(ma, dtype=float)
        z_acc = np.asarray(z_acc, dtype=float)
        ma, z_acc = np.broadcast_arrays(ma, z_acc)

        if np.any(ma <= 0.0):
            raise ValueError("accretion masses must be positive")
        if np.any(z_acc < self.z_final) or np.any(z_acc > self.z_acc_max):
            raise ValueError("z_acc lies outside the precomputed redshift range")

        mvir_acc = np.asarray(self.solver.Mzvir(z_acc), dtype=float)
        ln_ratio = np.log(ma / mvir_acc)

        lo = self._ln_ratio_grid[0]
        hi = self._ln_ratio_grid[-1]
        if np.any(ln_ratio < lo) or np.any(ln_ratio > hi):
            raise ValueError(
                "ma/Mvir(z_acc) lies outside the precomputed mass-ratio range"
            )

        points = np.column_stack((z_acc.ravel(), ln_ratio.ravel()))
        return ma, points

    def delta_log_mass(self, ma, z_acc):
        """Return ``ln[m(z_final) / ma]`` for the requested subhalos."""
        ma, points = self._points(ma, z_acc)
        values = self._interpolator(points).reshape(ma.shape)
        return float(values) if values.ndim == 0 else values

    def mass(self, ma, z_acc):
        """Return the stripped subhalo mass at ``z_final``."""
        ma_array, points = self._points(ma, z_acc)
        delta = self._interpolator(points).reshape(ma_array.shape)
        values = ma_array * np.exp(delta)
        return float(values) if values.ndim == 0 else values
