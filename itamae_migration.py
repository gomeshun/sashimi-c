"""Incremental ITAMAE migration layer for SASHIMI-C.

The legacy public classes remain unchanged. This module provides parallel
ITAMAE-backed classes whose background cosmology, Gauss-Hermite quadrature,
Shanks acceleration, NFW mass inversion, and weighted catalog representation
come from ITAMAE. Keeping both paths available makes numerical differences
explicit and testable before the legacy implementations are removed.
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from scipy import integrate, special
from scipy.interpolate import interp1d

from itamae.cosmology import NativeFlatLCDM
from itamae.evolution import shanks_transform
from itamae.halo import invert_nfw_mass_function
from itamae.numerics import gauss_hermite_lognormal
from itamae.types import CATALOG_SCHEMA_VERSION, WeightedSubhaloCatalog
from sashimi_c import TidalStrippingSolver, halo_model, subhalo_properties

_Base = TypeVar("_Base", bound=type)


class ItamaeMigrationMixin:
    """Supply shared ITAMAE mechanisms to a legacy SASHIMI-C class.

    Parameters
    ----------
    *args
        Positional arguments forwarded to the legacy class.
    cosmology_backend : object, optional
        ITAMAE-compatible cosmology backend. When omitted, a native flat-LCDM
        backend is configured from the legacy ``OmegaM`` and ``h`` values.
    **kwargs
        Keyword arguments forwarded to the legacy class.

    Notes
    -----
    SASHIMI-C uses Mpc, solar mass, and seconds as implicit floating-point base
    units. ITAMAE returns Hubble rates in km s^-1 Mpc^-1 and densities in
    Msun Mpc^-3, so conversions are explicit at this adapter boundary.

    Historical SASHIMI constants are rounded differently from ITAMAE constants.
    The adapter preserves the legacy critical-density normalization at redshift
    zero while delegating its redshift dependence to ITAMAE. This prevents a
    constants-only change from contaminating migration regression tests.
    """

    def __init__(self, *args: Any, cosmology_backend: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        backend = cosmology_backend or NativeFlatLCDM(omega_m0=self.OmegaM, h=self.h)
        self.itamae_cosmology = backend

        if hasattr(backend, "omega_m0"):
            self.OmegaM = float(backend.omega_m0)
            self.OmegaL = 1.0 - self.OmegaM
        if hasattr(backend, "h"):
            self.h = float(backend.h)

        self.H0 = float(np.asarray(backend.H(0.0))) * self.km / self.s / self.Mpc
        backend_rho0 = (
            float(np.asarray(backend.rho_crit(0.0))) * self.Msun / self.Mpc**3
        )
        legacy_rho0 = 3.0 * self.H0**2 / (8.0 * np.pi * self.G)
        self._rho_crit_scale = legacy_rho0 / backend_rho0
        self.rhocrit0 = legacy_rho0

    def Hubble(self, z: Any) -> np.ndarray:
        """Return the Hubble rate in the legacy inverse-second unit."""
        return np.asarray(self.itamae_cosmology.H(z)) * self.km / self.s / self.Mpc

    def rhocrit(self, z: Any) -> np.ndarray:
        """Return critical density in the legacy SASHIMI-C convention."""
        density = np.asarray(self.itamae_cosmology.rho_crit(z))
        return density * self.Msun / self.Mpc**3 * self._rho_crit_scale

    def growthD(self, z: Any) -> np.ndarray:
        """Return the normalized linear growth factor."""
        return np.asarray(self.itamae_cosmology.growth_factor(z))

    def Na_calc(self, ma, zacc, Mhost, z0=0.0, N_herm=200, Nrand=1000, Na_model=3):
        """Evaluate the Yang et al. accretion rate with ITAMAE quadrature.

        Parameters
        ----------
        ma : numpy.ndarray
            Accretion-mass grid. A two-dimensional input must have redshift as
            its leading axis, matching the legacy implementation.
        zacc : numpy.ndarray
            Accretion-redshift grid.
        Mhost : float
            Host mass at ``z0`` in legacy solar-mass units.
        z0 : float, optional
            Host reference redshift.
        N_herm : int, optional
            Gauss-Hermite order for host-history scatter.
        Nrand : int, optional
            Resolution of the auxiliary redshift inversion grid.
        Na_model : {1, 2, 3}, optional
            Yang et al. normalization variant.

        Returns
        -------
        numpy.ndarray
            Differential accretion abundance with shape ``(len(zacc), len(ma))``.
        """

        zacc_2d = np.asarray(zacc).reshape(-1, 1)
        M200_0 = self.Mzzi(Mhost, zacc_2d, z0)
        sigmalogM200 = 0.12 - 0.15 * np.log10(M200_0 / Mhost)
        M200, host_weight = gauss_hermite_lognormal(
            M200_0, sigmalogM200, order=N_herm
        )

        mmax = np.minimum(M200, Mhost / 2.0)
        Mmax = np.minimum(M200_0 + mmax, Mhost)

        if Na_model == 3:
            zlist = zacc_2d * np.linspace(1.0, 0.0, Nrand)
            iMmax = np.argmin(np.abs(self.Mzzi(Mhost, zlist, z0) - Mmax), axis=-1)
            z_Max = zlist[np.arange(len(zlist)), iMmax]
            z_Max_3d = z_Max.reshape(N_herm, len(zlist), 1)
            delcM = self.deltac_func(z_Max_3d)
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(Mmax)
            sa = self.s_func(ma)
            xmax = (delca - delcM) ** 2 / (2.0 * (self.s_func(mmax) - sM))
            normB = special.gamma(0.5) * special.gammainc(0.5, xmax) / np.sqrt(np.pi)
            Phi = (
                self.Ffunc_Yang(delcM, delca, sM, sa)
                / normB
                * np.heaviside(mmax - ma, 0)
            )
        elif Na_model == 1:
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(M200)
            sa = self.s_func(ma)
            xmin = self.s_func(mmax) - self.s_func(M200)
            normB = (
                1.0
                / np.sqrt(2.0 * np.pi)
                * delca
                * 2.0
                / xmin**0.5
                * special.hyp2f1(0.5, 0.0, 1.5, -sM / xmin)
            )
            Phi = self.Ffunc(delca, sM, sa) / normB * np.heaviside(mmax - ma, 0)
        elif Na_model == 2:
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(M200)
            sa = self.s_func(ma)
            xmin = self.s_func(mmax) - self.s_func(M200)
            normB = (
                1.0
                / np.sqrt(2.0 * np.pi)
                * delca
                * 0.57
                * (delca / np.sqrt(sM)) ** -0.01
                * (2.0 / (1.0 - 0.38))
                * sM ** (-0.38 / 2.0)
                * xmin ** (0.5 * (0.38 - 1.0))
                * special.hyp2f1(
                    0.5 * (1.0 - 0.38),
                    -0.38 / 2.0,
                    0.5 * (3.0 - 0.38),
                    -sM / xmin,
                )
            )
            Phi = (
                self.Ffunc(delca, sM, sa)
                * self.Gfunc(delca, sM, sa)
                / normB
                * np.heaviside(mmax - ma, 0)
            )
        else:
            raise ValueError("Na_model must be 1, 2, or 3.")

        F2 = np.sum(np.nan_to_num(Phi) * host_weight, axis=0)
        return F2 * self.dsdm(ma, 0.0) * self.dMdz(Mhost, zacc_2d, z0) * (1.0 + zacc_2d)

    def subhalo_mass_stripped_pert2_shanks(self, ma, za, z):
        """Evaluate second-order stripping with ITAMAE Shanks acceleration.

        The SASHIMI-specific two-percent stability criterion is retained around
        ITAMAE's generic sequence transformation.
        """

        eps_0 = self.eps_0(za, z)
        ln_ma = np.log(ma)
        eps_1 = self.eps_10(za, z) + ln_ma * self.eps_11(za, z)
        eps_2 = (
            self.eps_20(za, z)
            + ln_ma * self.eps_21(za, z)
            + ln_ma**2 * self.eps_22(za, z)
        )
        partial_0 = eps_0
        partial_1 = eps_0 + eps_1
        partial_2 = partial_1 + eps_2
        accelerated = shanks_transform(partial_0, partial_1, partial_2)
        with np.errstate(divide="ignore", invalid="ignore"):
            small_correction = np.abs((eps_1 + eps_2) / eps_0) < 0.02
        eps = np.where(small_correction, partial_2, accelerated)
        return ma * np.exp(eps)

    def subhalo_catalog_calc(
        self,
        M0,
        redshift=0.0,
        dz=0.01,
        zmax=7.0,
        N_ma=500,
        sigmalogc=0.128,
        N_herm=5,
        logmamin=-6,
        logmamax=None,
        N_hermNa=200,
        Na_model=3,
        ct_th=0.0,
        profile_change=True,
        M0_at_redshift=False,
        method="pert2_shanks",
        **kwargs,
    ) -> WeightedSubhaloCatalog:
        """Generate an ITAMAE weighted catalog while preserving SASHIMI physics.

        Returns
        -------
        itamae.types.WeightedSubhaloCatalog
            Catalog with separate population, concentration, and survival
            weights. The product of population and concentration weights equals
            the historical tuple ``weight``; survival remains an independent
            factor for diagnostics and reweighting.
        """

        if M0_at_redshift:
            Mz = M0
            M0_list = np.logspace(0.0, 5.0, 1500) * Mz
            fint = interp1d(
                self.Mzi(M0_list, redshift),
                M0_list,
                bounds_error=False,
                fill_value="extrapolate",
            )
            M0 = float(fint(Mz))

        self.M0 = M0
        self.redshift = redshift
        zdist = np.arange(redshift + dz, zmax + dz, dz)
        if logmamax is None:
            logmamax = np.log10(0.1 * M0 / self.Msun)
        ma200_grid = np.logspace(logmamin, logmamax, N_ma) * self.Msun

        shape = (len(zdist), N_herm, len(ma200_grid))
        rs_acc = np.zeros(shape)
        rhos_acc = np.zeros(shape)
        rs_z0 = np.zeros(shape)
        rhos_z0 = np.zeros(shape)
        ct_z0 = np.zeros(shape)
        survive = np.zeros(shape, dtype=bool)
        m0_matrix = np.zeros(shape)
        concentration_weight = np.zeros(shape)

        solver = ItamaeTidalStrippingSolver(
            M0=M0,
            z_min=redshift,
            z_max=zmax,
            n_z_interp=64,
            cosmology_backend=self.itamae_cosmology,
        )

        for iz, z_acc_value in enumerate(zdist):
            ma = self.Mvir_from_M200_fit(ma200_grid, z_acc_value)
            Oz = self.OmegaM * (1.0 + z_acc_value) ** 3 / self.g(z_acc_value)
            m0 = solver.subhalo_mass_stripped(
                ma, z_acc_value, redshift, method=method, **kwargs
            )
            c200sub = self.conc200(ma200_grid, z_acc_value)
            rvirsub = (
                3.0
                * ma
                / (
                    4.0
                    * np.pi
                    * self.rhocrit0
                    * self.g(z_acc_value)
                    * self.Delc(Oz - 1.0)
                )
            ) ** (1.0 / 3.0)
            r200sub = (
                3.0
                * ma200_grid
                / (4.0 * np.pi * self.rhocrit0 * self.g(z_acc_value) * 200.0)
            ) ** (1.0 / 3.0)
            c_mz = c200sub * rvirsub / r200sub
            c_sub, concentration_weight[iz] = gauss_hermite_lognormal(
                c_mz, sigmalogc, order=N_herm
            )
            rs_acc[iz] = rvirsub / c_sub
            rhos_acc[iz] = ma / (
                4.0 * np.pi * rs_acc[iz] ** 3 * self.fc(c_sub)
            )

            if profile_change:
                rmax_acc = rs_acc[iz] * 2.163
                Vmax_acc = (
                    np.sqrt(rhos_acc[iz] * 4.0 * np.pi * self.G / 4.625)
                    * rs_acc[iz]
                )
                Vmax_z0 = Vmax_acc * (
                    2.0**0.4 * (m0 / ma) ** 0.3 * (1.0 + m0 / ma) ** -0.4
                )
                rmax_z0 = rmax_acc * (
                    2.0**-0.3 * (m0 / ma) ** 0.4 * (1.0 + m0 / ma) ** 0.3
                )
                rs_z0[iz] = rmax_z0 / 2.163
                rhos_z0[iz] = (
                    4.625 / (4.0 * np.pi * self.G)
                ) * (Vmax_z0 / rs_z0[iz]) ** 2
            else:
                rs_z0[iz] = rs_acc[iz]
                rhos_z0[iz] = rhos_acc[iz]

            enclosed_fraction = m0 / (
                4.0 * np.pi * rhos_z0[iz] * rs_z0[iz] ** 3
            )
            ct_z0[iz] = invert_nfw_mass_function(enclosed_fraction)
            survive[iz] = ct_z0[iz] > ct_th
            m0_matrix[iz] = m0 * np.ones((N_herm, 1))

        Na = self.Na_calc(
            ma200_grid,
            zdist,
            M0,
            z0=0.0,
            N_herm=N_hermNa,
            Nrand=1000,
            Na_model=Na_model,
        )
        Na_total = integrate.simpson(
            integrate.simpson(Na, x=np.log(ma200_grid)), x=np.log(1.0 + zdist)
        )
        population_2d = Na / (1.0 + zdist.reshape(-1, 1))
        population_2d = population_2d / np.sum(population_2d) * Na_total
        population_weight = np.broadcast_to(
            population_2d[:, None, :], shape
        ).copy()

        z_acc = np.broadcast_to(zdist[:, None, None], shape)
        ma200 = np.broadcast_to(ma200_grid[None, None, :], shape)
        metadata = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "model_identifier": "sashimi-c:cdm:itamae-migration:v1",
            "backend_identifier": (
                "array=numpy;cosmology="
                f"{self.itamae_cosmology.identifier};units=legacy-sashimi-c"
            ),
            "source_identifier": "sashimi-c:itamae-migration",
            "migration_backend": getattr(
                self.itamae_cosmology, "identifier", type(self.itamae_cosmology).__name__
            ),
            "target_redshift": float(redshift),
            "legacy_weight_excludes_survival": True,
            "nfw_inversion": "itamae.brentq",
        }
        return WeightedSubhaloCatalog(
            columns={
                "m200_acc": ma200.reshape(-1),
                "z_acc": z_acc.reshape(-1),
                "r_s_acc": rs_acc.reshape(-1),
                "rho_s_acc": rhos_acc.reshape(-1),
                "m_bound": m0_matrix.reshape(-1),
                "r_s": rs_z0.reshape(-1),
                "rho_s": rhos_z0.reshape(-1),
                "c_t": ct_z0.reshape(-1),
                "survive": survive.reshape(-1),
            },
            weights={
                "weight_base": population_weight.reshape(-1),
                "weight_concentration": concentration_weight.reshape(-1),
                "weight_survival": survive.astype(float).reshape(-1),
            },
            metadata=metadata,
        )

    def subhalo_properties_calc(self, *args: Any, **kwargs: Any):
        """Return the historical tuple from the ITAMAE catalog calculation."""
        catalog = self.subhalo_catalog_calc(*args, **kwargs)
        legacy_weight = (
            np.asarray(catalog.weights["weight_base"])
            * np.asarray(catalog.weights["weight_concentration"])
        )
        columns = catalog.columns
        return (
            columns["m200_acc"],
            columns["z_acc"],
            columns["r_s_acc"],
            columns["rho_s_acc"],
            columns["m_bound"],
            columns["r_s"],
            columns["rho_s"],
            columns["c_t"],
            legacy_weight,
            columns["survive"],
        )


def migrate_class(base_class: _Base) -> _Base:
    """Construct an ITAMAE-backed subclass without modifying legacy defaults."""
    return type(
        f"Itamae{base_class.__name__[0].upper()}{base_class.__name__[1:]}",
        (ItamaeMigrationMixin, base_class),
        {"__module__": __name__},
    )


ItamaeHaloModel = migrate_class(halo_model)
ItamaeTidalStrippingSolver = migrate_class(TidalStrippingSolver)
ItamaeSubhaloProperties = migrate_class(subhalo_properties)

__all__ = [
    "ItamaeHaloModel",
    "ItamaeMigrationMixin",
    "ItamaeSubhaloProperties",
    "ItamaeTidalStrippingSolver",
    "migrate_class",
]
