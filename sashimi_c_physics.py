"""CDM-owned calibrated physics, tidal solvers and observable formulae.

The population controller and numerical primitives belong to ITAMAE. This
module contains no old catalog implementation or reproduction-mode switches.
All historical formula bodies are extracted from the reviewed C model; units
are initialized against the chosen, compatible cosmology before any work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from itamae.cosmology import NativeFlatLCDM
from itamae.protocols import CosmologyBackend
from scipy import optimize
from scipy.integrate import cumulative_trapezoid, odeint
from scipy.interpolate import griddata

from picard_tidal_stripping import PicardTidalStrippingTable

_CALIBRATED_OMEGA_M = 0.315
_CALIBRATED_H = 0.674


class CDMUnits:
    def __init__(self):
        self.Mpc = 1.0
        self.kpc = self.Mpc / 1000.0
        self.pc = self.kpc / 1000.0
        self.cm = self.pc / 3.086e18
        self.km = 1.0e5 * self.cm
        self.s = 1.0
        self.yr = 3.15576e7 * self.s
        self.Msun = 1.0
        self.gram = self.Msun / 1.988e33
        self.c = 2.9979e10 * self.cm / self.s
        self.G = 4.30091e-9 * self.Mpc * (self.km / self.s) ** 2 / self.Msun


class CDMPhysics(CDMUnits):
    """Planck-calibrated host history, concentration, EPS and structure relations."""

    def __init__(
        self,
        prompt_cusps=False,
        k_fs_Mpc=1.06e6,
        filter="Sharp-k",
        alpha=1.8,
        *,
        cosmology_backend=None,
    ):
        backend = cosmology_backend or NativeFlatLCDM(
            omega_m0=_CALIBRATED_OMEGA_M, h=_CALIBRATED_H
        )
        self._validate_migration_cosmology(backend)
        self.itamae_cosmology = backend
        CDMUnits.__init__(self)
        self.OmegaB = 0.049
        self.OmegaM = _CALIBRATED_OMEGA_M
        self.OmegaC = self.OmegaM - self.OmegaB
        self.OmegaL = 1.0 - self.OmegaM
        self.h = _CALIBRATED_H
        self._rho_crit_scale = 1.0
        self._synchronize_physics_constants()
        self.prompt_cusps = prompt_cusps
        self.k_fs = k_fs_Mpc * self.Mpc**-1
        self.filter = filter
        self.alpha = alpha
        if prompt_cusps:
            from prompt_cusps import build_ps_interpolators

            self.sigma_interp, self.dsdm_interp = build_ps_interpolators(
                self, self.k_fs, filter=filter, alpha=alpha
            )

    def _synchronize_physics_constants(self) -> None:
        """Align SASHIMI-C constants with the selected backend."""
        backend = self.itamae_cosmology
        self.H0 = float(np.asarray(backend.H(0.0))) * self.km / self.s / self.Mpc
        backend_rho0 = (
            float(np.asarray(backend.rho_crit(0.0))) * self.Msun / self.Mpc**3
        )
        self.G = 3.0 * self.H0**2 / (8.0 * np.pi * backend_rho0)
        self._rho_crit_scale = 1.0
        self.rhocrit0 = backend_rho0

    @staticmethod
    def _validate_migration_cosmology(backend: Any) -> None:
        """Require a complete backend matching the calibrated CDM physical model."""
        if not isinstance(backend, CosmologyBackend):
            raise TypeError(
                "cosmology_backend must implement the ITAMAE cosmology protocol."
            )

        omega_m0 = float(np.asarray(backend.omega_m(0.0)))
        h = float(np.asarray(backend.H(0.0))) / 100.0
        if not np.isclose(omega_m0, _CALIBRATED_OMEGA_M, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "SASHIMI-C calibrated physics requires "
                f"OmegaM={_CALIBRATED_OMEGA_M}; received {omega_m0}."
            )
        if not np.isclose(h, _CALIBRATED_H, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "SASHIMI-C calibrated physics requires "
                f"h={_CALIBRATED_H}; received {h}."
            )

    def Hubble(self, z: Any) -> np.ndarray:
        """Return the Hubble rate in the internal inverse-second unit."""
        return np.asarray(self.itamae_cosmology.H(z)) * self.km / self.s / self.Mpc

    def rhocrit(self, z: Any) -> np.ndarray:
        """Return critical density in the canonical solar masses per cubic megaparsec."""
        density = np.asarray(self.itamae_cosmology.rho_crit(z))
        return density * self.Msun / self.Mpc**3 * self._rho_crit_scale

    def growthD(self, z: Any) -> np.ndarray:
        """Return the normalized linear growth factor."""
        return np.asarray(self.itamae_cosmology.growth_factor(z))

    def g(self, z):
        return self.OmegaM * (1.0 + z) ** 3 + self.OmegaL

    def dDdz(self, z):
        def dOdz(z):
            return (
                -self.OmegaL
                * 3
                * self.OmegaM
                * (1 + z) ** 2
                * (self.OmegaL + self.OmegaM * (1 + z) ** 3.0) ** -2
            )

        Omega_Lz = self.OmegaL / (self.OmegaL + self.OmegaM * (1.0 + z) ** 3)
        Omega_Mz = 1 - Omega_Lz
        phiz = (
            Omega_Mz ** (4.0 / 7.0)
            - Omega_Lz
            + (1 + Omega_Mz / 2.0) * (1 + Omega_Lz / 70.0)
        )
        phi0 = (
            self.OmegaM ** (4.0 / 7.0)
            - self.OmegaL
            + (1 + self.OmegaM / 2.0) * (1 + self.OmegaL / 70.0)
        )
        dphidz = dOdz(z) * (
            -4.0 / 7.0 * Omega_Mz ** (-3.0 / 7.0)
            + (Omega_Mz - Omega_Lz) / 140.0
            + 1.0 / 70.0
            - 3.0 / 2.0
        )
        return (phi0 / self.OmegaM) * (
            -dOdz(z) / (phiz * (1 + z))
            - Omega_Mz * (dphidz * (1 + z) + phiz) / phiz**2 / (1 + z) ** 2
        )

    def sigmaMz(self, M, z):
        if not self.prompt_cusps:
            return self.sigma_Ludlow(M) * self.growthD(z)
        else:
            return self.sigma_interp(np.log10(M / self.Msun)) * self.growthD(z)

    def dsdm(self, M, z):
        if not self.prompt_cusps:
            return self.dsdm_Ludlow(M) * self.growthD(z) ** 2
        else:
            return self.dsdm_interp(np.log10(M / self.Msun)) * self.growthD(z) ** 2

    def sigma_Ludlow(self, M):
        """Ludlow et al. (2016)"""

        def xi(M):
            return (M / ((1.0e10 * self.Msun) / self.h)) ** -1

        return (
            22.26
            * xi(M) ** 0.292
            / (1.0 + 1.53 * xi(M) ** 0.275 + 3.36 * xi(M) ** 0.198)
        )

    def dsdm_Ludlow(self, M):
        """Ludlow et al. (2016)"""

        def xi(M):
            return (M / ((1.0e10 * self.Msun) / self.h)) ** -1

        dsdsigma = 2.0 * self.sigma_Ludlow(M)
        dxidm = -1.0e10 * self.Msun / self.h / M**2
        dsigmadxi = self.sigma_Ludlow(M) * (
            0.292 / xi(M)
            - (0.275 * 1.53 * xi(M) ** -0.725 + 0.198 * 3.36 * xi(M) ** -0.802)
            / (1.0 + 1.53 * xi(M) ** 0.275 + 3.36 * xi(M) ** 0.198)
        )
        return dsdsigma * dsigmadxi * dxidm

    def deltac_func(self, z):
        return 1.686 / self.growthD(z)

    def s_func(self, M):
        return self.sigmaMz(M, 0) ** 2

    def fc(self, x):
        return np.log(1 + x) - x * pow(1 + x, -1)

    def Delc(self, x):
        return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

    def conc200(self, M200, z):
        """Correa et al. (2015)"""
        alpha_cMz_1 = 1.7543 - 0.2766 * (1.0 + z) + 0.02039 * (1.0 + z) ** 2
        beta_cMz_1 = 0.2753 + 0.00351 * (1.0 + z) - 0.3038 * (1.0 + z) ** 0.0269
        gamma_cMz_1 = -0.01537 + 0.02102 * (1.0 + z) ** -0.1475
        c_Mz_1 = np.power(
            10.0,
            alpha_cMz_1
            + beta_cMz_1
            * np.log10(M200 / self.Msun)
            * (1 + gamma_cMz_1 * np.log10(M200 / self.Msun) ** 2),
        )
        alpha_cMz_2 = 1.3081 - 0.1078 * (1.0 + z) + 0.00398 * (1.0 + z) ** 2
        beta_cMz_2 = 0.0223 - 0.0944 * (1.0 + z) ** -0.3907
        c_Mz_2 = pow(10, alpha_cMz_2 + beta_cMz_2 * np.log10(M200 / self.Msun))
        return np.where(z <= 4.0, c_Mz_1, c_Mz_2)

    def Mvir_from_M200(self, M200, z):
        gz = self.g(z)
        c200 = self.conc200(M200, z)
        r200 = (3.0 * M200 / (4 * np.pi * 200 * self.rhocrit0 * gz)) ** (1.0 / 3.0)
        rs = r200 / c200
        fc200 = self.fc(c200)
        rhos = M200 / (4 * np.pi * rs**3 * fc200)
        Dc = self.Delc(self.OmegaM * (1.0 + z) ** 3 / self.g(z) - 1.0)
        rvir = optimize.fsolve(
            lambda r: (
                3.0 * (rs / r) ** 3 * self.fc(r / rs) * rhos - Dc * self.rhocrit0 * gz
            ),
            r200,
        )
        Mvir = 4 * np.pi * rs**3 * rhos * self.fc(rvir / rs)
        return Mvir

    def Mvir_from_M200_fit(self, M200, z):
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13e-3
        a4 = -3.52e-5
        Oz = self.OmegaM * (1.0 + z) ** 3 / self.g(z)

        def ffunc(x):
            return np.power(x, 3.0) * (np.log(1.0 + 1.0 / x) - 1.0 / (1.0 + x))

        def xfunc(f):
            p = a2 + a3 * np.log(f) + a4 * np.power(np.log(f), 2.0)
            return (
                np.power(a1 * np.power(f, 2.0 * p) + (3.0 / 4.0) ** 2, -0.5) + 2.0 * f
            )

        return (
            self.Delc(Oz - 1)
            / 200.0
            * M200
            * np.power(
                self.conc200(M200, z)
                * xfunc(self.Delc(Oz - 1) / 200.0 * ffunc(1.0 / self.conc200(M200, z))),
                -3.0,
            )
        )

    def Mzi(self, M0, z):
        a = 1.686 * np.sqrt(2.0 / np.pi) * self.dDdz(0) + 1.0
        zf = (
            -0.0064 * np.log10(M0 / self.Msun) ** 2
            + 0.0237 * np.log10(M0 / self.Msun)
            + 1.8837
        )
        q = 4.137 / zf**0.9476
        fM0 = (self.sigmaMz(M0 / q, 0) ** 2 - self.sigmaMz(M0, 0) ** 2) ** -0.5
        return M0 * np.power(1.0 + z, a * fM0) * np.exp(-fM0 * z)

    def Mzzi(self, M0, z, zi):
        Mzi0 = self.Mzi(M0, zi)
        zf = (
            -0.0064 * np.log10(M0 / self.Msun) ** 2
            + 0.0237 * np.log10(M0 / self.Msun)
            + 1.8837
        )
        q = 4.137 / zf**0.9476
        fMzi = (self.sigmaMz(Mzi0 / q, zi) ** 2 - self.sigmaMz(Mzi0, zi) ** 2) ** -0.5
        alpha = fMzi * (
            1.686 * np.sqrt(2.0 / np.pi) / self.growthD(zi) ** 2 * self.dDdz(zi) + 1.0
        )
        beta = -fMzi
        return Mzi0 * np.power(1.0 + z - zi, alpha) * np.exp(beta * (z - zi))

    def dMdz(self, M0, z, zi):
        Mzi0 = self.Mzi(M0, zi)
        zf = (
            -0.0064 * np.log10(M0 / self.Msun) ** 2
            + 0.0237 * np.log10(M0 / self.Msun)
            + 1.8837
        )
        q = 4.137 / zf**0.9476
        fMzi = (self.sigmaMz(Mzi0 / q, zi) ** 2 - self.sigmaMz(Mzi0, zi) ** 2) ** -0.5
        alpha = fMzi * (
            1.686 * np.sqrt(2.0 / np.pi) / self.growthD(zi) ** 2 * self.dDdz(zi) + 1
        )
        beta = -fMzi
        Mzzidef = Mzi0 * (1.0 + z - zi) ** alpha * np.exp(beta * (z - zi))
        Mzzivir = self.Mvir_from_M200_fit(Mzzidef, z)
        return (beta + alpha / (1.0 + z - zi)) * Mzzivir


class CDMTidalKernels(CDMPhysics):
    """C-owned stripping laws, coefficient tables and solver choices."""

    def __init__(
        self, M0, z_min=0.0, z_max=7.0, n_z_interp=64, *, cosmology_backend=None
    ):
        """Initial function of the class.

        -----
        Input
        -----
        M0: Mass of the host halo defined as M_{200} (200 times critial density) at *z = 0*.
        (Optional) z_min:          Minimum redshift to end the calculation of evolution to. (default: 0.)
        (Optional) z_max:          Maximum redshift to start the calculation of evolution from. (default: 7.)
        (Optional) n_z_interp:     Number of redshifts to calculate epsilon functions. (default: 64)
        """
        CDMPhysics.__init__(self, cosmology_backend=cosmology_backend)
        self.z_min = z_min
        self.z_max = z_max
        self.n_z_interp = n_z_interp
        self._picard_tables = {}
        self.M0 = M0

    @property
    def M0(self):
        return self._M0

    @M0.setter
    def M0(self, value):
        self._M0 = value
        if hasattr(self, "_picard_tables"):
            self._picard_tables.clear()
        self.reset_interpolation(
            z_max=self.z_max, z_min=self.z_min, n_z=self.n_z_interp
        )

    def reset_interpolation(self, z_max, z_min, n_z):
        """Reset interpolation for epsilon functions.

        This function is called when the mass of the host
        halo is changed.

        -----
        Input
        -----
        za_max: float
            Maximum redshift to start the calculation of evolution from.
        z_min: float
            Minimum redshift to end the calculation of evolution to.
        n_z: int
            Number of redshifts to calculate epsilon functions.
        """
        _z, _eps_0 = self._eps_0(z_max, z_min, n_z)
        _, _eps_10, _eps_11 = self._eps_1(z_max, z_min, n_z)
        _, _eps_20, _eps_21, _eps_22 = self._eps_2(z_max, z_min, n_z)
        _, _eps_30, _eps_31, _eps_32, _eps_33 = self._eps_3(z_max, z_min, n_z)
        # get the interpolation functions as indefinite integrals
        self._eps_0_interp = lambda z: np.interp(z, _z[::-1], _eps_0[::-1])
        self._eps_10_interp = lambda z: np.interp(z, _z[::-1], _eps_10[::-1])
        self._eps_11_interp = lambda z: np.interp(z, _z[::-1], _eps_11[::-1])
        self._eps_20_interp = lambda z: np.interp(z, _z[::-1], _eps_20[::-1])
        self._eps_21_interp = lambda z: np.interp(z, _z[::-1], _eps_21[::-1])
        self._eps_22_interp = lambda z: np.interp(z, _z[::-1], _eps_22[::-1])
        self._eps_30_interp = lambda z: np.interp(z, _z[::-1], _eps_30[::-1])
        self._eps_31_interp = lambda z: np.interp(z, _z[::-1], _eps_31[::-1])
        self._eps_32_interp = lambda z: np.interp(z, _z[::-1], _eps_32[::-1])
        self._eps_33_interp = lambda z: np.interp(z, _z[::-1], _eps_33[::-1])
        # define the epsilon functions as definite integrals from za to z
        self.eps_0 = lambda _za, _z: self._eps_0_interp(_z) - self._eps_0_interp(_za)
        self.eps_10 = lambda _za, _z: self._eps_10_interp(_z) - self._eps_10_interp(_za)
        self.eps_11 = lambda _za, _z: self._eps_11_interp(_z) - self._eps_11_interp(_za)
        self.eps_20 = lambda _za, _z: self._eps_20_interp(_z) - self._eps_20_interp(_za)
        self.eps_21 = lambda _za, _z: self._eps_21_interp(_z) - self._eps_21_interp(_za)
        self.eps_22 = lambda _za, _z: self._eps_22_interp(_z) - self._eps_22_interp(_za)
        self.eps_30 = lambda _za, _z: self._eps_30_interp(_z) - self._eps_30_interp(_za)
        self.eps_31 = lambda _za, _z: self._eps_31_interp(_z) - self._eps_31_interp(_za)
        self.eps_32 = lambda _za, _z: self._eps_32_interp(_z) - self._eps_32_interp(_za)
        self.eps_33 = lambda _za, _z: self._eps_33_interp(_z) - self._eps_33_interp(_za)

    def Mzvir(self, z):
        Mz200 = self.Mzzi(self.M0, z, 0.0)
        Mvir = self.Mvir_from_M200_fit(Mz200, z)
        return Mvir

    def AMz(self, z):
        log10a = (-0.0003 * np.log10(self.Mzvir(z) / self.Msun) + 0.02) * z + (
            0.011 * np.log10(self.Mzvir(z) / self.Msun) - 0.354
        )
        return 10.0**log10a

    def zetaMz(self, z):
        return (0.00012 * np.log10(self.Mzvir(z) / self.Msun) - 0.0033) * z + (
            -0.0011 * np.log10(self.Mzvir(z) / self.Msun) + 0.026
        )

    def tdynz(self, z):
        Oz_z = self.OmegaM * (1.0 + z) ** 3 / self.g(z)
        return (
            1.628
            / self.h
            * (self.Delc(Oz_z - 1.0) / 178.0) ** -0.5
            / (self.Hubble(z) / self.H0)
            * 1.0e9
            * self.yr
        )

    def _get_picard_table(self, z_final):
        """Return a cached x3 Picard table for the requested final redshift."""
        key = float(z_final)
        table = self._picard_tables.get(key)
        if table is None:
            table = PicardTidalStrippingTable(self, z_final=key)
            self._picard_tables[key] = table
        return table

    def subhalo_mass_stripped_picard_table(self, ma, za, z):
        """Calculate tidal mass loss with the precomputed Picard table."""
        return self._get_picard_table(z).mass(ma, za)

    def msolve(self, m, z):
        return (
            self.AMz(z)
            * (m / self.tdynz(z))
            * (m / self.Mzvir(z)) ** self.zetaMz(z)
            / (self.Hubble(z) * (1 + z))
        )

    def subhalo_mass_stripped_odeint(self, ma, za, z0, **kwargs):
        zcalc = np.linspace(za, z0, 100)
        sol = odeint(self.msolve, ma, zcalc, **kwargs)
        return sol[-1]

    def Phi(self, z):
        """subhalo stripping factor assuming zetaMz(z) = 0.
        The stripping rate dm/dt is given by
          dm/dt(z) = m(z) * Phi(z) * (m(z)/Mzvir(z))**zetaMz(z)
        """
        return self.AMz(z) / self.tdynz(z) / self.Hubble(z) / (1 + z)

    def _eps_0(self, za, z, n_z=64):
        """calculate epsilon0.

        Returns
        -------
        _z : array
            redshift array
        eps0 : array
            epsilon0 array.
        """
        _z = np.linspace(za, z, n_z)
        Phi_z = self.Phi(_z)
        return _z, cumulative_trapezoid(Phi_z, x=_z, initial=0)

    def _eps_1(self, za, z, n_z=64):
        """calculate the first order correction.

        The first order correction epsilon_1 is given by the following equation:
            epsilon_1 = epsilon_10 + epsilon_11 * ln_ma

        Returns
        -------
        _z : array
            redshift array
        eps10 : array
            epsilon10 array.
        eps11 : array
            epsilon11 array.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_10 = Phi_z * (eps_0 - ln_Mvir_z) * zeta_z
        integrand_11 = Phi_z * zeta_z
        integral_10 = cumulative_trapezoid(integrand_10, x=_z, initial=0)
        integral_11 = cumulative_trapezoid(integrand_11, x=_z, initial=0)
        return _z, integral_10, integral_11

    def _eps_2(self, za, z, n_z=64):
        """calculate the second order correction.

        The second order correction epsilon_2 is given by the following equation:
            epsilon_2 = epsilon_20 + epsilon_21 * ln_ma + epsilon_22 * ln_ma^2

        Returns
        -------
        _z : array
            redshift array
        eps20 : array
            epsilon20 array.
        eps21 : array
            epsilon21 array.
        eps22 : array
            epsilon22 array.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        _, eps_10, eps_11 = self._eps_1(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_20 = (
            Phi_z * zeta_z**2 * (eps_0 - ln_Mvir_z) ** 2 / 2 + Phi_z * zeta_z * eps_10
        )
        integrand_21 = Phi_z * zeta_z**2 * (eps_0 - ln_Mvir_z) + Phi_z * zeta_z * eps_11
        integrand_22 = Phi_z * zeta_z**2 / 2
        integral_20 = cumulative_trapezoid(integrand_20, x=_z, initial=0)
        integral_21 = cumulative_trapezoid(integrand_21, x=_z, initial=0)
        integral_22 = cumulative_trapezoid(integrand_22, x=_z, initial=0)
        return _z, integral_20, integral_21, integral_22

    def _eps_3(self, za, z, n_z=64):
        """calculate the third order correction.

        The third order correction epsilon_3 is given by the following equation:
            epsilon_3 = epsilon_30 + epsilon_31 * ln_ma + epsilon_32 * ln_ma^2 + epsilon_33 * ln_ma^3

        Returns
        -------
        _z : array
            redshift array
        eps30 : array
            epsilon30 array.
        eps31 : array
            epsilon31 array.
        eps32 : array
            epsilon32 array.
        eps33 : array
            epsilon33 array.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        _, eps_10, eps_11 = self._eps_1(za, z, n_z)
        _, eps_20, eps_21, eps_22 = self._eps_2(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_30 = (
            Phi_z * (eps_0 - ln_Mvir_z) ** 3 * zeta_z**3 / 6.0
            + Phi_z * (eps_0 - ln_Mvir_z) * eps_10 * (zeta_z**2)
            + Phi_z * eps_20 * zeta_z
        )
        integrand_31 = (
            Phi_z * (eps_0 - ln_Mvir_z) ** 2 * (zeta_z**3) / 2.0
            + Phi_z * eps_10 * (zeta_z**2)
            + Phi_z * eps_21 * zeta_z
            + Phi_z * (eps_0 - ln_Mvir_z) * eps_11 * (zeta_z**2)
        )
        integrand_32 = (
            Phi_z * (eps_0 - ln_Mvir_z) * (zeta_z**3) / 2.0
            + Phi_z * eps_11 * (zeta_z**2)
            + Phi_z * eps_22 * zeta_z
        )
        integrand_33 = Phi_z * (zeta_z**3) / 6.0
        integral_30 = cumulative_trapezoid(integrand_30, x=_z, initial=0)
        integral_31 = cumulative_trapezoid(integrand_31, x=_z, initial=0)
        integral_32 = cumulative_trapezoid(integrand_32, x=_z, initial=0)
        integral_33 = cumulative_trapezoid(integrand_33, x=_z, initial=0)
        return _z, integral_30, integral_31, integral_32, integral_33

    def subhalo_mass_stripped_pert0(self, ma, za, z):
        """Calculate subhalo mass stripping using zeroth-order perturbation."""
        eps_0 = self.eps_0(za, z)
        return ma * np.exp(eps_0)

    def subhalo_mass_stripped_pert1(self, ma, za, z):
        """Calculate subhalo mass stripping using first-order perturbation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        ln_ma = np.log(ma)
        eps = eps_0 + eps_10 + ln_ma * eps_11
        return ma * np.exp(eps)

    def subhalo_mass_stripped_pert2(self, ma, za, z):
        """Calculate subhalo mass stripping using second-order perturbation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        ln_ma = np.log(ma)
        eps = (
            eps_0
            + eps_10
            + ln_ma * eps_11
            + eps_20
            + ln_ma * eps_21
            + ln_ma**2 * eps_22
        )
        return ma * np.exp(eps)

    def subhalo_mass_stripped_pert3(self, ma, za, z):
        """Calculate subhalo mass stripping using third-order perturbation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        eps_30 = self.eps_30(za, z)
        eps_31 = self.eps_31(za, z)
        eps_32 = self.eps_32(za, z)
        eps_33 = self.eps_33(za, z)
        ln_ma = np.log(ma)
        eps = (
            eps_0
            + eps_10
            + ln_ma * eps_11
            + eps_20
            + ln_ma * eps_21
            + ln_ma**2 * eps_22
            + eps_30
            + ln_ma * eps_31
            + ln_ma**2 * eps_32
            + ln_ma**3 * eps_33
        )
        return ma * np.exp(eps)

    def subhalo_mass_stripped(self, ma, za, z, method="pert2_shanks", **kwargs):
        """A wrapper function to calculate subhalo mass stripping.

        Parameters
        ----------
        ma : float
            initial subhalo mass.
        za : float
            initial redshift.
        z : float
            final redshift.
        method : str, optional
            method to calculate the subhalo mass stripping.
            - "picard_table" : use the precomputed third-order Picard table.
            - "odeint" : use odeint to solve the differential equation.
            - "pert0" : use perturbative method with zeroth-order correction.
            - "pert1" : use perturbative method with first-order correction.
            - "pert2" : use perturbative method with second-order correction.
            - "pert2_shanks" : use perturbative method with second-order correction and Shanks transformation.
            - "pert3" : use perturbative method with third-order correction.
        kwargs : dict, optional
            additional arguments for the odeint function.

        Returns
        -------
        zcalc : array
            redshift array.
        mcalc : array
            subhalo mass array.
        """
        match method:
            case "picard_table":
                return self.subhalo_mass_stripped_picard_table(ma, za, z)
            case "odeint":
                return self.subhalo_mass_stripped_odeint(ma, za, z, **kwargs)
            case "pert0":
                return self.subhalo_mass_stripped_pert0(ma, za, z)
            case "pert1":
                return self.subhalo_mass_stripped_pert1(ma, za, z)
            case "pert2":
                return self.subhalo_mass_stripped_pert2(ma, za, z)
            case "pert2_shanks":
                return self.subhalo_mass_stripped_pert2_shanks(ma, za, z)
            case "pert3":
                return self.subhalo_mass_stripped_pert3(ma, za, z)
            case _:
                raise ValueError(f"Invalid method: {method}")


class CDMObservableKernels:
    """Observable reductions over a previously computed named population."""

    def mass_function(self, evolved=True):
        """
        Subhalo mass function

        -----
        Input
        -----
        (Optional) evolved: If True (False), this function calculates evolved (unevolved) mass function.
                            Here 'evolved' means that subhalos experiences tidal mass loss, whereas
                            'unevolved' means that mass loss is ignored.

        ------
        Output
        ------
        m:       Mass of subhalo in units of [Msun].
        dNdlnm:  Subhalo mass function dN/dln(m).

        """

        if evolved:
            N, lnm_edges = np.histogram(np.log(self.m0), weights=self.weight, bins=100)
        else:
            N, lnm_edges = np.histogram(
                np.log(self.ma200), weights=self.weight, bins=100
            )

        lnm = (lnm_edges[1:] + lnm_edges[:-1]) / 2.0
        dlnm = lnm_edges[1:] - lnm_edges[:-1]

        m = np.exp(lnm)
        dNdlnm = N / dlnm

        return m / self.Msun, dNdlnm

    def Nsat_Mpeak(self, Mpeak_th):
        """
        Calculate expected number of satellites for a given host halo. Satellites are assumed
        to be formed in a subhalo whose peak mass (equivalent to the mass at accretion) is above
        a given threshold value, Mpeak_th.

        -----
        Input
        -----
        Mpeak_th:   Threshold value for m_{peak} (= mass at accretion) above which a satellite
                    galaxy is (assumed to be) formed. E.g., Mpeak_th = 1.e8*Msun.

        ------
        Output
        ------
        m:          Subhalo masses within tidal radius in units of [Msun].
        Nccum_m:    Complementary cumulative number of subhalos N(>m)
                    with the condition m_{peak}>Mpeak_th.
        Vmax:       Vmax of a subhalo in units of [km/s].
        Nccum_Vmax: Complementary cumulative number of subhalos N(>Vmax)
                    with the condition m_{peak}>Mpeak_th.

        """

        N, lnm_edges = np.histogram(
            np.log(self.m0[self.ma200 > Mpeak_th]),
            weights=self.weight[self.ma200 > Mpeak_th],
            bins=100,
        )
        lnm = (lnm_edges[1:] + lnm_edges[:-1]) / 2.0
        m = np.exp(lnm)
        Ncum = np.cumsum(N)
        Nccum_m = Ncum[-1] - Ncum

        N, lnVmax_edges = np.histogram(
            np.log(self.Vmax[self.ma200 > Mpeak_th]),
            weights=self.weight[self.ma200 > Mpeak_th],
            bins=100,
        )
        lnVmax = (lnVmax_edges[1:] + lnVmax_edges[:-1]) / 2.0
        Vmax = np.exp(lnVmax)
        Ncum = np.cumsum(N)
        Nccum_Vmax = Ncum[-1] - Ncum

        return m / self.Msun, Nccum_m, Vmax / (self.km / self.s), Nccum_Vmax

    def Nsat_Vpeak(self, Vpeak_th):
        """
        Calculate expected number of satellites for a given host halo. Satellites are assumed
        to be formed in a subhalo whose Vpeak (equivalent to the Vmax at accretion) is above
        a given threshold value, Vpeak_th.

        -----
        Input
        -----
        Vpeak_th:   Threshold value for V_{peak} (= V_{max} at accretion) above which a satellite
                    galaxy is (assumed to be) formed. E.g., Vpeak_th = 18*km/s.

        ------
        Output
        ------
        m:          Subhalo masses within tidal radius in units of [Msun].
        Nccum_m:    Complementary cumulative number of subhalos N(>m)
                    with the condition V_{peak}>Vpeak_th.
        Vmax:       Vmax of a subhalo in units of [km/s].
        Nccum_Vmax: Complementary cumulative number of subhalos N(>Vmax)
                    with the condition V_{peak}>Vpeak_th.

        """

        N, lnm_edges = np.histogram(
            np.log(self.m0[self.Vpeak > Vpeak_th]),
            weights=self.weight[self.Vpeak > Vpeak_th],
            bins=100,
        )
        lnm = (lnm_edges[1:] + lnm_edges[:-1]) / 2.0
        m = np.exp(lnm)
        Ncum = np.cumsum(N)
        Nccum_m = Ncum[-1] - Ncum

        N, lnVmax_edges = np.histogram(
            np.log(self.Vmax[self.Vpeak > Vpeak_th]),
            weights=self.weight[self.Vpeak > Vpeak_th],
            bins=100,
        )
        lnVmax = (lnVmax_edges[1:] + lnVmax_edges[:-1]) / 2.0
        Vmax = np.exp(lnVmax)
        Ncum = np.cumsum(N)
        Nccum_Vmax = Ncum[-1] - Ncum

        return m / self.Msun, Nccum_m, Vmax / (self.km / self.s), Nccum_Vmax

    def mass_fraction(self, evolved=True):
        """
        Subhalo mass fraction: sum_i m_i / M_host

        -----
        Input
        -----
        (Optional) evolved: If True (False), this function calculates evolved (unevolved) mass function.
                            Here 'evolved' means that subhalos experiences tidal mass loss, whereas
                            'unevolved' means that mass loss is ignored.

        ------
        Output
        ------
        fsh: Fractional mass of the host in the form of subhalos.

        """

        Mhost = self.Mzi(self.M0, self.redshift)
        if evolved:
            fsh = np.sum(self.m0 * self.weight) / Mhost
        else:
            fsh = np.sum(self.ma200 * self.weight) / Mhost

        return fsh

    def annihilation_boost_factor(self, n=0, evolved=True):
        """
        Annihilation boost factor B_{sh}. Note that the effect of sub-subhalos and higher order
        structure is not included.

        -----
        Input
        -----
        (Optional) n:       The effects up to sub^{n}-subhalos will be included.
                            If n=0 (default), no sub-subhalos and beyond is considered.
                            For other values of n, the function requires pre-computed boost factors
                            B_sh from the previous (n-1)th iteration and subhalo mass fraction f_sh.
                            These are stored under 'data/boost/' directory. If the directory does not
                            exist, excecuting 'boost_iteraction.py' will generate the  necessary files
                            and store them in the directory, up to n = 3.
        (Optional) evolved: If True (False), this function calculates evolved (unevolved) mass function.
                            Here 'evolved' means that subhalos experiences tidal mass loss, whereas
                            'unevolved' means that mass loss is ignored.

        ------
        Output
        ------
        Bsh:              Annihilation boost factor B_{sh}. See Eq. (37) of Ando et al.
                          arXiv:1903.11427 for definition.
        luminosity_ratio: Ratio of the total luminosity (subhalos+host) and the luminosity due
                          to host only in the absence of subhalos:
                          L_{total}/L_{host,0} = 1-f_{sh}^2+B_{sh}

        """

        fsh = self.mass_fraction(evolved)
        if n == 0:
            fssh = 0.0
            Bssh = 0.0
        else:
            list_Bssh = np.loadtxt("data/boost/Bsh_%s.txt" % (n - 1))
            list_fssh = np.loadtxt("data/boost/fsh.txt")
            list_za = np.loadtxt("data/boost/za.txt")
            list_ma = np.loadtxt("data/boost/ma.txt")

            list_log_ma_flat = np.log10(list_ma.flatten())
            list_za_flat = (list_za.reshape(-1, 1) * np.ones_like(list_ma[0])).flatten()
            list_log_fssh_flat = np.log10(list_fssh.flatten())
            list_log_Bssh_flat = np.log10((list_Bssh + 1.0e-30).flatten())

            log_Bssh = griddata(
                (list_log_ma_flat, list_za_flat),
                list_log_Bssh_flat,
                (np.log10(self.ma200), self.z_a),
                method="linear",
            )
            log_fssh = griddata(
                (list_log_ma_flat, list_za_flat),
                list_log_fssh_flat,
                (np.log10(self.ma200), self.z_a),
                method="linear",
            )
            log_Bssh[~np.isfinite(log_Bssh)] = -np.inf
            log_fssh[~np.isfinite(log_fssh)] = -np.inf
            Bssh = 10.0**log_Bssh
            fssh = 10.0**log_fssh

            mavir = self.Mvir_from_M200_fit(self.ma200, self.z_a)
            Oz = self.OmegaM * (1.0 + self.z_a) ** 3 / self.g(self.z_a)
            ravir = np.cbrt(
                3.0
                * mavir
                / (4.0 * np.pi * self.rhocrit(self.z_a) * self.Delc(Oz - 1.0))
            )
            cavir = ravir / self.rs_a

            Bssh = (
                Bssh
                * self.rs0**3
                * (np.arcsinh(self.ct0) - self.ct0 / np.sqrt(1.0 + self.ct0**2))
            )
            Bssh = Bssh / (
                self.rs_a**3 * (np.arcsinh(cavir) - cavir / np.sqrt(1.0 + cavir**2))
            )
            Bssh = Bssh / (
                self.rhos0**2 * self.rs0**3 * (1.0 - 1.0 / (1.0 + self.ct0) ** 3)
            )
            Bssh = Bssh * (
                self.rhos_a**2 * self.rs_a**3 * (1.0 - 1.0 / (1.0 + cavir) ** 3)
            )
            fssh = (
                fssh
                * self.rs0**3
                * (np.arcsinh(self.ct0) - self.ct0 / np.sqrt(1.0 + self.ct0**2))
            )
            fssh = fssh / (
                self.rs_a**3 * (np.arcsinh(cavir) - cavir / np.sqrt(1.0 + cavir**2))
            )
            fssh = fssh / (self.rhos0 * self.rs0**3 * self.fc(self.ct0))
            fssh = fssh * (self.rhos_a * self.rs_a**3 * self.fc(cavir))

        if evolved:
            Lsh = np.sum(
                (1.0 - fssh**2 + Bssh)
                * self.rhos0**2
                * self.rs0**3
                * (1.0 - 1.0 / (1.0 + self.ct0) ** 3)
                * self.weight
            )
        else:
            r200 = (
                3.0 * self.ma200 / (4.0 * np.pi * self.rhocrit(self.redshift) * 200.0)
            ) ** (1.0 / 3.0)
            c200 = r200 / self.rs_a
            Lsh = np.sum(
                self.rhos_a**2
                * self.rs_a**3
                * (1.0 - 1.0 / (1.0 + c200) ** 3)
                * self.weight
            )

        Mhost = self.Mzi(self.M0, self.redshift)
        r200_host = (
            3.0 * Mhost / (4.0 * np.pi * self.rhocrit(self.redshift) * 200.0)
        ) ** (1.0 / 3.0)
        c200_host = self.conc200(Mhost, self.redshift)
        rs_host = r200_host / c200_host
        rhos_host = Mhost / (4.0 * np.pi * rs_host**3 * self.fc(c200_host))
        Lhost0 = rhos_host**2 * rs_host**3 * (1.0 - 1.0 / (1.0 + c200_host) ** 3)

        Bsh = Lsh / Lhost0
        luminosity_ratio = 1.0 - fsh**2 + Bsh

        return Bsh, luminosity_ratio

    def annihilation_boost_factor_prompt_cusps(
        self, n=0, f_surv=1.0, f_surv_stripped=1.0
    ):
        """
        Annihilation boost factor B_{sh}. Note that the effect of sub-subhalos and higher order
        structure is not included.

        -----
        Input
        -----
        (Optional) n:       The effects up to sub^{n}-subhalos will be included.
                            If n=0 (default), no sub-subhalos and beyond is considered.
                            For other values of n, the function requires pre-computed boost factors
                            B_sh from the previous (n-1)th iteration and subhalo mass fraction f_sh.
                            These are stored under 'data/boost/' directory. If the directory does not
                            exist, excecuting 'boost_iteraction.py' will generate the  necessary files
                            and store them in the directory, up to n = 3.
        (Optional) evolved: If True (False), this function calculates evolved (unevolved) mass function.
                            Here 'evolved' means that subhalos experiences tidal mass loss, whereas
                            'unevolved' means that mass loss is ignored.

        ------
        Output
        ------
        Bsh:              Annihilation boost factor B_{sh}. See Eq. (37) of Ando et al.
                          arXiv:1903.11427 for definition.
        luminosity_ratio: Ratio of the total luminosity (subhalos+host) and the luminosity due
                          to host only in the absence of subhalos:
                          L_{total}/L_{host,0} = 1-f_{sh}^2+B_{sh}

        """

        from prompt_cusps import prompt_cusps as _prompt_cusps

        prc = _prompt_cusps(k_fs=self.k_fs)
        f_coll, J_cusps = prc.cusp_properties(f_surv=1.0, z=self.redshift)
        J_cusps_mean = np.mean(J_cusps)

        fsh = self.mass_fraction()
        if n == 0:
            fssh = 0.0
            Bssh = 0.0
            Ncusp_dressed = 0.0
            Ncusp_naked = 0.0
        else:
            list_Bssh = np.loadtxt(
                "data/prompt_cusps/boost/Bsh_%s_%.1f_%.1f.txt"
                % ((n - 1), f_surv, f_surv_stripped)
            )
            list_Ncusp_dressed = np.loadtxt(
                "data/prompt_cusps/boost/Ncusp_dressed_%s_%.1f_%.1f.txt"
                % ((n - 1), f_surv, f_surv_stripped)
            )
            list_Ncusp_naked = np.loadtxt(
                "data/prompt_cusps/boost/Ncusp_naked_%s_%.1f_%.1f.txt"
                % ((n - 1), f_surv, f_surv_stripped)
            )
            list_fssh = np.loadtxt("data/prompt_cusps/boost/fsh.txt")
            list_za = np.loadtxt("data/prompt_cusps/boost/za.txt")
            list_ma = np.loadtxt("data/prompt_cusps/boost/ma.txt")

            list_log_ma_flat = np.log10(list_ma.flatten())
            list_za_flat = (list_za.reshape(-1, 1) * np.ones_like(list_ma[0])).flatten()
            list_log_fssh_flat = np.log10(list_fssh.flatten())
            list_log_Bssh_flat = np.log10((list_Bssh + 1.0e-30).flatten())
            list_log_Ncusp_dressed_flat = np.log10((list_Ncusp_dressed).flatten())
            list_log_Ncusp_naked_flat = np.log10((list_Ncusp_naked).flatten())

            log_Bssh = griddata(
                (list_log_ma_flat, list_za_flat),
                list_log_Bssh_flat,
                (np.log10(self.ma200), self.z_a),
                method="linear",
            )
            log_Ncusp_dressed = griddata(
                (list_log_ma_flat, list_za_flat),
                list_log_Ncusp_dressed_flat,
                (np.log10(self.ma200), self.z_a),
                method="linear",
            )
            log_Ncusp_naked = griddata(
                (list_log_ma_flat, list_za_flat),
                list_log_Ncusp_naked_flat,
                (np.log10(self.ma200), self.z_a),
                method="linear",
            )
            log_fssh = griddata(
                (list_log_ma_flat, list_za_flat),
                list_log_fssh_flat,
                (np.log10(self.ma200), self.z_a),
                method="linear",
            )

            log_Bssh[~np.isfinite(log_Bssh)] = -np.inf
            log_Ncusp_dressed[~np.isfinite(log_Ncusp_dressed)] = -np.inf
            log_Ncusp_naked[~np.isfinite(log_Ncusp_naked)] = -np.inf
            log_fssh[~np.isfinite(log_fssh)] = -np.inf
            Bssh = 10.0**log_Bssh
            Ncusp_dressed0 = 10.0**log_Ncusp_dressed
            Ncusp_naked0 = 10.0**log_Ncusp_naked
            fssh = 10.0**log_fssh

            mavir = self.Mvir_from_M200_fit(self.ma200, self.z_a)
            Oz = self.OmegaM * (1.0 + self.z_a) ** 3 / self.g(self.z_a)
            ravir = np.cbrt(
                3.0
                * mavir
                / (4.0 * np.pi * self.rhocrit(self.z_a) * self.Delc(Oz - 1.0))
            )
            cavir = ravir / self.rs_a

            fssh = (
                fssh
                * self.rs0**3
                * (np.arcsinh(self.ct0) - self.ct0 / np.sqrt(1.0 + self.ct0**2))
            )
            fssh = fssh / (
                self.rs_a**3 * (np.arcsinh(cavir) - cavir / np.sqrt(1.0 + cavir**2))
            )
            fssh = fssh / (self.rhos0 * self.rs0**3 * self.fc(self.ct0))
            fssh = fssh * (self.rhos_a * self.rs_a**3 * self.fc(cavir))

            Bssh = (
                Bssh
                * self.rs0**3
                * (np.arcsinh(self.ct0) - self.ct0 / np.sqrt(1.0 + self.ct0**2))
            )
            Bssh = Bssh / (
                self.rs_a**3 * (np.arcsinh(cavir) - cavir / np.sqrt(1.0 + cavir**2))
            )
            Bssh = Bssh / (
                self.rhos0**2 * self.rs0**3 * (1.0 - 1.0 / (1.0 + self.ct0) ** 3)
            )
            Bssh = Bssh * (
                self.rhos_a**2 * self.rs_a**3 * (1.0 - 1.0 / (1.0 + cavir) ** 3)
            )

            Ncusp_dressed = (
                Ncusp_dressed0
                * self.rs0**3
                * (np.arcsinh(self.ct0) - self.ct0 / np.sqrt(1.0 + self.ct0**2))
                / (self.rs_a**3 * (np.arcsinh(cavir) - cavir / np.sqrt(1.0 + cavir**2)))
            )
            Ncusp_naked = Ncusp_naked0 + f_surv_stripped * (
                Ncusp_dressed0 - Ncusp_dressed
            )

        Ncusp_dressed = f_surv * np.sum(self.weight) + np.sum(
            Ncusp_dressed * self.weight
        )
        Ncusp_naked = np.sum(Ncusp_naked * self.weight)

        Lsh = np.sum(
            (1.0 - fssh**2 + Bssh)
            * 4.0
            * np.pi
            / 3.0
            * self.rhos0**2
            * self.rs0**3
            * (1.0 - 1.0 / (1.0 + self.ct0) ** 3)
            * self.weight
        )
        Lcusp_dressed = J_cusps_mean * Ncusp_dressed
        Lcusp_naked = J_cusps_mean * Ncusp_naked

        Mhost = self.Mzi(self.M0, self.redshift)
        r200_host = (
            3.0 * Mhost / (4.0 * np.pi * self.rhocrit(self.redshift) * 200.0)
        ) ** (1.0 / 3.0)
        c200_host = self.conc200(Mhost, self.redshift)
        rs_host = r200_host / c200_host
        rhos_host = Mhost / (4.0 * np.pi * rs_host**3 * self.fc(c200_host))
        Lhost0 = (
            4.0
            * np.pi
            / 3.0
            * rhos_host**2
            * rs_host**3
            * (1.0 - 1.0 / (1.0 + c200_host) ** 3)
        )

        Bsh = Lsh / Lhost0
        Bcusp_dressed = Lcusp_dressed / Lhost0
        Bcusp_naked = Lcusp_naked / Lhost0
        luminosity_ratio = 1.0 - fsh**2 + Bsh + Bcusp_dressed + Bcusp_naked

        return (
            Bsh,
            Bcusp_dressed,
            Bcusp_naked,
            luminosity_ratio,
            Ncusp_dressed,
            Ncusp_naked,
        )

    def subhalo_catalog_MC(self, mth):
        """
        This function returns a subhalo catalog generated with the Monte Carlo simulations.

        -----
        Input
        -----
        mth:  Threshold of subhalo mass above which the catalog is generated.

        ------
        Output
        ------
        ma200_MC:    Mass m_{200} at accretion. [Msun]
        z_a_MC:      Redshift at accretion.
        rs_a_MC:     Scale radius r_s at accretion. [kpc]
        rhos_a_MC:   Characteristic density \rho_s at accretion. [Msun/pc^3]
        m0_MC:       Mass up to tidal truncation radius at a given redshift. [Msun]
        rs0_MC:      Scale radius r_s at a given redshift. [kpc]
        rhos0_MC:    Characteristic density \rho_s at a given redshift. [Msun/pc^3]
        ct0_MC:      Tidal truncation radius in units of r_s at a given redshift.

        """

        condition = self.m0 > mth
        ma200 = self.ma200[condition]
        z_a = self.z_a[condition]
        rs_a = self.rs_a[condition]
        rhos_a = self.rhos_a[condition]
        m0 = self.m0[condition]
        rs0 = self.rs0[condition]
        rhos0 = self.rhos0[condition]
        ct0 = self.ct0[condition]
        weight = self.weight[condition]

        mu_sh = np.sum(weight)
        prob = weight / mu_sh
        subhalo_id = np.arange(len(prob))
        N_sh = np.random.poisson(mu_sh)
        id_MC = np.random.choice(subhalo_id, size=N_sh, p=prob)
        ma200_MC = ma200[id_MC]
        z_a_MC = z_a[id_MC]
        rs_a_MC = rs_a[id_MC]
        rhos_a_MC = rhos_a[id_MC]
        m0_MC = m0[id_MC]
        rs0_MC = rs0[id_MC]
        rhos0_MC = rhos0[id_MC]
        ct0_MC = ct0[id_MC]

        return (
            ma200_MC / self.Msun,
            z_a_MC,
            rs_a_MC / self.kpc,
            rhos_a_MC / (self.Msun / self.pc**3),
            m0_MC / self.Msun,
            rs0_MC / self.kpc,
            rhos0_MC / (self.Msun / self.pc**3),
            ct0_MC,
        )
