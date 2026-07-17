"""Incremental ITAMAE migration layer for SASHIMI-C.

The legacy public classes remain unchanged. This module provides parallel
ITAMAE-backed classes whose background cosmology, Gauss-Hermite quadrature,
Shanks acceleration, NFW mass inversion, and weighted catalog representation
come from ITAMAE. Keeping both paths available makes numerical differences
explicit and testable before the legacy implementations are removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, TypeVar

import numpy as np
from scipy import integrate, special
from scipy.interpolate import interp1d

from itamae import __version__ as ITAMAE_VERSION
from itamae.cosmology import NativeFlatLCDM
from itamae.evolution import shanks_transform
from itamae.halo import invert_nfw_mass_function
from itamae.numerics import gauss_hermite_lognormal
from itamae.protocols import CosmologyBackend
from itamae.types import (
    CATALOG_SCHEMA_VERSION,
    CatalogMetadata,
    WeightedSubhaloCatalog,
)
from sashimi_c import (
    TidalStrippingSolver,
    halo_model,
    subhalo_observables,
    subhalo_properties,
)

_Base = TypeVar("_Base", bound=type)
_LEGACY_OMEGA_M = 0.315
_LEGACY_H = 0.674
_PHYSICS_MODES = ("consistent", "legacy")
_STRIPPING_METHODS = (
    "odeint",
    "pert0",
    "pert1",
    "pert2",
    "pert2_shanks",
    "pert3",
)
_DEFAULT_STRIPPING_METHOD = "pert2_shanks"
_DEFAULT_CT_THRESHOLD = 0.0
_SHANKS_SMALL_CORRECTION_THRESHOLD = 0.02


@dataclass(frozen=True, slots=True)
class StrippingDiagnostics:
    """Record the SASHIMI-C Shanks approximation against direct ODE integration.

    The diagnostic is intentionally separate from catalog generation. Running
    a direct ODE solve for every catalog node would change the established
    performance characteristics of the default ``pert2_shanks`` calculation.

    Attributes
    ----------
    mass_at_accretion
        Input subhalo masses in the legacy SASHIMI-C mass unit.
    mass_pert2_shanks
        Bound masses from the established second-order Shanks approximation.
    mass_odeint
        Bound masses from direct legacy ODE integration.
    relative_difference
        Absolute fractional difference relative to the ODE result.
    host_mass
        Host mass at redshift zero.
    accretion_redshift, target_redshift
        Evolution interval used by both solvers.
    physics_mode
        ITAMAE migration physics convention.
    backend_identifier
        Stable cosmology backend identifier.
    """

    mass_at_accretion: np.ndarray
    mass_pert2_shanks: np.ndarray
    mass_odeint: np.ndarray
    relative_difference: np.ndarray
    host_mass: float
    accretion_redshift: float
    target_redshift: float
    physics_mode: str
    backend_identifier: str

    def __post_init__(self) -> None:
        """Freeze aligned diagnostic arrays."""
        arrays = {}
        for name in (
            "mass_at_accretion",
            "mass_pert2_shanks",
            "mass_odeint",
            "relative_difference",
        ):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            value.setflags(write=False)
            arrays[name] = value
        shapes = {value.shape for value in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(f"Diagnostic arrays must share one shape; got {shapes}.")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def max_relative_difference(self) -> float:
        """Return the largest Shanks-versus-ODE fractional difference."""
        return float(np.max(self.relative_difference, initial=0.0))

    def summary(self) -> Mapping[str, Any]:
        """Return immutable JSON-compatible diagnostic provenance."""
        return MappingProxyType(
            {
                "comparison": "pert2_shanks-vs-odeint",
                "host_mass": self.host_mass,
                "accretion_redshift": self.accretion_redshift,
                "target_redshift": self.target_redshift,
                "physics_mode": self.physics_mode,
                "backend_identifier": self.backend_identifier,
                "sample_size": int(self.mass_at_accretion.size),
                "max_relative_difference": self.max_relative_difference,
            }
        )


class ItamaeMigrationMixin:
    """Supply shared ITAMAE mechanisms to a legacy SASHIMI-C class.

    Parameters
    ----------
    *args
        Positional arguments forwarded to the legacy class.
    cosmology_backend : itamae.protocols.CosmologyBackend, optional
        ITAMAE-compatible cosmology backend. When omitted, a native flat-LCDM
        backend is configured from the legacy ``OmegaM`` and ``h`` values.
    physics_mode : {"consistent", "legacy"}, optional
        ``"consistent"`` (default) uses one gravitational-constant convention
        for the ITAMAE critical density and SASHIMI-C structure calculations.
        ``"legacy"`` retains the historical rounded SASHIMI-C constant and
        critical-density normalization for numerical reproduction.
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

    During this result-preserving migration phase, the selected backend must
    match the legacy SASHIMI-C values of ``OmegaM`` and ``h``. Allowing a
    different background before all host-history and halo-definition formulae
    use that backend would silently mix two cosmologies.
    """

    def __init__(
        self,
        *args: Any,
        cosmology_backend: Any | None = None,
        physics_mode: str = "consistent",
        **kwargs: Any,
    ) -> None:
        if physics_mode not in _PHYSICS_MODES:
            raise ValueError(
                f"physics_mode must be one of {_PHYSICS_MODES}; received {physics_mode!r}."
            )
        backend = cosmology_backend or NativeFlatLCDM(
            omega_m0=_LEGACY_OMEGA_M,
            h=_LEGACY_H,
        )
        self._validate_migration_cosmology(backend)

        # Some legacy constructors build interpolation tables or a full catalog
        # before returning. Install the backend first so overridden cosmology
        # methods are valid throughout that initialization.
        self.itamae_cosmology = backend
        self.physics_mode = physics_mode
        self._rho_crit_scale = 1.0
        super().__init__(*args, **kwargs)
        self._synchronize_physics_constants()

    def _synchronize_physics_constants(self) -> None:
        """Align SASHIMI-C constants with the selected migration convention."""
        backend = self.itamae_cosmology
        self.H0 = float(np.asarray(backend.H(0.0))) * self.km / self.s / self.Mpc
        backend_rho0 = (
            float(np.asarray(backend.rho_crit(0.0))) * self.Msun / self.Mpc**3
        )
        if self.physics_mode == "consistent":
            # Derive G from the selected backend so halo radii, densities, and
            # circular velocities use the same constant convention.
            self.G = 3.0 * self.H0**2 / (8.0 * np.pi * backend_rho0)
            self._rho_crit_scale = 1.0
            self.rhocrit0 = backend_rho0
        else:
            legacy_rho0 = 3.0 * self.H0**2 / (8.0 * np.pi * self.G)
            self._rho_crit_scale = legacy_rho0 / backend_rho0
            self.rhocrit0 = legacy_rho0

    @staticmethod
    def _validate_migration_cosmology(backend: Any) -> None:
        """Require a complete backend matching the legacy physical model."""
        if not isinstance(backend, CosmologyBackend):
            raise TypeError(
                "cosmology_backend must implement the ITAMAE cosmology protocol."
            )

        omega_m0 = float(np.asarray(backend.omega_m(0.0)))
        h = float(np.asarray(backend.H(0.0))) / 100.0
        if not np.isclose(omega_m0, _LEGACY_OMEGA_M, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "The result-preserving SASHIMI-C migration currently requires "
                f"OmegaM={_LEGACY_OMEGA_M}; received {omega_m0}."
            )
        if not np.isclose(h, _LEGACY_H, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "The result-preserving SASHIMI-C migration currently requires "
                f"h={_LEGACY_H}; received {h}."
            )

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
        M200, host_weight = gauss_hermite_lognormal(M200_0, sigmalogM200, order=N_herm)

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
        # ``subhalo_observables`` performs its legacy base initialization via
        # an explicit class call. Re-synchronize here so its first catalog is
        # already generated with the selected convention, not only subsequent
        # calls after the constructor returns.
        self._synchronize_physics_constants()
        self._validate_catalog_inputs(
            M0=M0,
            redshift=redshift,
            dz=dz,
            zmax=zmax,
            N_ma=N_ma,
            sigmalogc=sigmalogc,
            N_herm=N_herm,
            logmamin=logmamin,
            logmamax=logmamax,
            N_hermNa=N_hermNa,
            Na_model=Na_model,
            ct_th=ct_th,
            method=method,
        )

        requested_host_mass = float(M0)
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
        if float(logmamin) >= float(logmamax):
            raise ValueError("logmamin must be smaller than logmamax.")
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
            physics_mode=self.physics_mode,
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
            rhos_acc[iz] = ma / (4.0 * np.pi * rs_acc[iz] ** 3 * self.fc(c_sub))

            if profile_change:
                rmax_acc = rs_acc[iz] * 2.163
                Vmax_acc = (
                    np.sqrt(rhos_acc[iz] * 4.0 * np.pi * self.G / 4.625) * rs_acc[iz]
                )
                Vmax_z0 = Vmax_acc * (
                    2.0**0.4 * (m0 / ma) ** 0.3 * (1.0 + m0 / ma) ** -0.4
                )
                rmax_z0 = rmax_acc * (
                    2.0**-0.3 * (m0 / ma) ** 0.4 * (1.0 + m0 / ma) ** 0.3
                )
                rs_z0[iz] = rmax_z0 / 2.163
                rhos_z0[iz] = (4.625 / (4.0 * np.pi * self.G)) * (
                    Vmax_z0 / rs_z0[iz]
                ) ** 2
            else:
                rs_z0[iz] = rs_acc[iz]
                rhos_z0[iz] = rhos_acc[iz]

            enclosed_fraction = m0 / (4.0 * np.pi * rhos_z0[iz] * rs_z0[iz] ** 3)
            if self.physics_mode == "consistent":
                ct_z0[iz] = invert_nfw_mass_function(enclosed_fraction)
            else:
                # Reproduce the historical finite-grid interpolation exactly.
                # The consistent mode uses ITAMAE's bracketed inverse instead.
                legacy_c_t = np.linspace(0.0, 100.0, 1000)
                inverse = interp1d(
                    self.fc(legacy_c_t),
                    legacy_c_t,
                    fill_value="extrapolate",
                )
                ct_z0[iz] = inverse(enclosed_fraction)
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
        population_weight = np.broadcast_to(population_2d[:, None, :], shape).copy()

        z_acc = np.broadcast_to(zdist[:, None, None], shape)
        ma200 = np.broadcast_to(ma200_grid[None, None, :], shape)
        legacy_weight = population_weight * concentration_weight
        total_legacy_weight = float(np.sum(legacy_weight))
        surviving_weight_fraction = (
            float(np.sum(legacy_weight * survive) / total_legacy_weight)
            if total_legacy_weight > 0.0
            else 0.0
        )
        backend_identifier = (
            "array=numpy;cosmology="
            f"{self.itamae_cosmology.identifier};units=legacy-sashimi-c-floats"
        )
        metadata = CatalogMetadata(
            model_identifier=f"sashimi-c:cdm:{self.physics_mode}:v1.2",
            backend_identifier=backend_identifier,
            source_identifier=f"sashimi-c:itamae-adapter:{self.physics_mode}:v1",
            schema_version=CATALOG_SCHEMA_VERSION,
            extra={
                "itamae_version": ITAMAE_VERSION,
                "physics_mode": self.physics_mode,
                "cosmology_backend": self.itamae_cosmology.identifier,
                "critical_density_convention": (
                    "itamae-backend"
                    if self.physics_mode == "consistent"
                    else "legacy-sashimi-c-rounded-G"
                ),
                "host_mass_input": requested_host_mass,
                "host_mass_z0": float(M0),
                "host_mass_input_at_target_redshift": bool(M0_at_redshift),
                "target_redshift": float(redshift),
                "zmax": float(zmax),
                "dz": float(dz),
                "n_mass": int(N_ma),
                "n_concentration": int(N_herm),
                "n_host_history": int(N_hermNa),
                "log10_mass_min": float(logmamin),
                "log10_mass_max": float(logmamax),
                "sigma_log10_concentration": float(sigmalogc),
                "accretion_model": int(Na_model),
                "profile_change": bool(profile_change),
                "stripping_method": method,
                "default_stripping_method": _DEFAULT_STRIPPING_METHOD,
                "shanks_small_correction_threshold": (
                    _SHANKS_SMALL_CORRECTION_THRESHOLD
                ),
                "ct_threshold": float(ct_th),
                "default_ct_threshold": _DEFAULT_CT_THRESHOLD,
                "survival_rule": "c_t > ct_threshold",
                "surviving_node_count": int(np.count_nonzero(survive)),
                "node_count": int(survive.size),
                "surviving_weight_fraction": surviving_weight_fraction,
                "legacy_weight_excludes_survival": True,
                "weight_semantics": {
                    "weight_base": "population measure",
                    "weight_concentration": "lognormal concentration quadrature",
                    "weight_survival": "binary c_t threshold",
                },
                "nfw_inversion": (
                    "itamae.brentq"
                    if self.physics_mode == "consistent"
                    else "legacy-linear-grid-0-100-1000"
                ),
            },
        )
        catalog = WeightedSubhaloCatalog(
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
        self.catalog = catalog
        return catalog

    def _validate_catalog_inputs(
        self,
        *,
        M0: Any,
        redshift: Any,
        dz: Any,
        zmax: Any,
        N_ma: Any,
        sigmalogc: Any,
        N_herm: Any,
        logmamin: Any,
        logmamax: Any,
        N_hermNa: Any,
        Na_model: Any,
        ct_th: Any,
        method: Any,
    ) -> None:
        """Reject inputs outside the physical and numerical catalog domain."""
        scalar_values = {
            "M0": M0,
            "redshift": redshift,
            "dz": dz,
            "zmax": zmax,
            "sigmalogc": sigmalogc,
            "logmamin": logmamin,
            "ct_th": ct_th,
        }
        if logmamax is not None:
            scalar_values["logmamax"] = logmamax
        for name, value in scalar_values.items():
            array = np.asarray(value)
            if array.ndim != 0 or not np.isfinite(float(array)):
                raise ValueError(f"{name} must be a finite scalar.")

        if float(M0) <= 0.0:
            raise ValueError("M0 must be positive.")
        if float(redshift) < 0.0:
            raise ValueError("redshift must be nonnegative.")
        if float(dz) <= 0.0:
            raise ValueError("dz must be positive.")
        if float(zmax) <= float(redshift):
            raise ValueError("zmax must be greater than redshift.")
        if float(sigmalogc) < 0.0:
            raise ValueError("sigmalogc must be nonnegative.")
        if float(ct_th) < 0.0:
            raise ValueError("ct_th must be nonnegative.")
        if method not in _STRIPPING_METHODS:
            raise ValueError(
                f"method must be one of {_STRIPPING_METHODS}; received {method!r}."
            )

        for name, value in {
            "N_ma": N_ma,
            "N_herm": N_herm,
            "N_hermNa": N_hermNa,
            "Na_model": Na_model,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if int(Na_model) not in (1, 2, 3):
            raise ValueError("Na_model must be 1, 2, or 3.")

        if logmamax is not None and float(logmamin) >= float(logmamax):
            raise ValueError("logmamin must be smaller than logmamax.")

    def subhalo_properties_calc(self, *args: Any, **kwargs: Any):
        """Return the historical tuple from the ITAMAE catalog calculation."""
        catalog = self.subhalo_catalog_calc(*args, **kwargs)
        self.catalog = catalog
        legacy_weight = np.asarray(catalog.weights["weight_base"]) * np.asarray(
            catalog.weights["weight_concentration"]
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
ItamaeSubhaloObservables = migrate_class(subhalo_observables)


def diagnose_stripping_approximation(
    host_mass: float,
    mass_at_accretion: Any,
    *,
    accretion_redshift: float = 1.0,
    target_redshift: float = 0.0,
    n_z_interp: int = 64,
    cosmology_backend: Any | None = None,
    physics_mode: str = "consistent",
    odeint_options: Mapping[str, Any] | None = None,
) -> StrippingDiagnostics:
    """Compare the default Shanks approximation with direct ODE integration.

    Parameters
    ----------
    host_mass
        Host mass at redshift zero in the legacy SASHIMI-C mass unit.
    mass_at_accretion
        Positive scalar or array of subhalo accretion masses.
    accretion_redshift, target_redshift
        Start and end redshifts. Accretion redshift must be larger.
    n_z_interp
        Interpolation resolution for the perturbative solver.
    cosmology_backend
        Optional canonical SASHIMI-C-compatible ITAMAE cosmology backend.
    physics_mode
        ITAMAE migration physical convention.
    odeint_options
        Optional keyword arguments forwarded only to SciPy ``odeint``.

    Returns
    -------
    StrippingDiagnostics
        Aligned solver outputs and their fractional difference.

    Notes
    -----
    This function does not select or modify the catalog stripping method. It is
    an explicit validation tool; catalog generation remains
    ``method="pert2_shanks"`` by default.
    """
    masses = np.atleast_1d(np.asarray(mass_at_accretion, dtype=float))
    if masses.ndim > 1 or masses.size == 0:
        raise ValueError("mass_at_accretion must be a non-empty scalar or 1D array.")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("mass_at_accretion must contain finite positive values.")
    host_mass_array = np.asarray(host_mass)
    if host_mass_array.ndim != 0:
        raise ValueError("host_mass must be a finite positive scalar.")
    host_mass_value = float(host_mass_array)
    if not np.isfinite(host_mass_value) or host_mass_value <= 0.0:
        raise ValueError("host_mass must be finite and positive.")
    accretion_redshift_array = np.asarray(accretion_redshift)
    target_redshift_array = np.asarray(target_redshift)
    if accretion_redshift_array.ndim != 0 or target_redshift_array.ndim != 0:
        raise ValueError("redshifts must be finite scalars.")
    accretion_redshift_value = float(accretion_redshift_array)
    target_redshift_value = float(target_redshift_array)
    if not np.isfinite(accretion_redshift_value) or not np.isfinite(
        target_redshift_value
    ):
        raise ValueError("redshifts must be finite.")
    if accretion_redshift_value <= target_redshift_value:
        raise ValueError("accretion_redshift must be greater than target_redshift.")
    if isinstance(n_z_interp, bool) or not isinstance(n_z_interp, (int, np.integer)):
        raise TypeError("n_z_interp must be an integer.")
    if n_z_interp < 2:
        raise ValueError("n_z_interp must be at least two.")

    solver = ItamaeTidalStrippingSolver(
        host_mass_value,
        z_min=target_redshift_value,
        z_max=accretion_redshift_value,
        n_z_interp=n_z_interp,
        cosmology_backend=cosmology_backend,
        physics_mode=physics_mode,
    )
    shanks_mass = solver.subhalo_mass_stripped(
        masses,
        accretion_redshift_value,
        target_redshift_value,
        method=_DEFAULT_STRIPPING_METHOD,
    )
    ode_mass = solver.subhalo_mass_stripped(
        masses,
        accretion_redshift_value,
        target_redshift_value,
        method="odeint",
        **({} if odeint_options is None else dict(odeint_options)),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_difference = np.abs(shanks_mass - ode_mass) / np.abs(ode_mass)

    return StrippingDiagnostics(
        mass_at_accretion=masses,
        mass_pert2_shanks=shanks_mass,
        mass_odeint=ode_mass,
        relative_difference=relative_difference,
        host_mass=host_mass_value,
        accretion_redshift=accretion_redshift_value,
        target_redshift=target_redshift_value,
        physics_mode=physics_mode,
        backend_identifier=solver.itamae_cosmology.identifier,
    )


__all__ = [
    "ItamaeHaloModel",
    "ItamaeMigrationMixin",
    "ItamaeSubhaloObservables",
    "ItamaeSubhaloProperties",
    "ItamaeTidalStrippingSolver",
    "StrippingDiagnostics",
    "diagnose_stripping_approximation",
    "migrate_class",
]
