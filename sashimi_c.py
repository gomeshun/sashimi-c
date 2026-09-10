"""Standard SASHIMI-C API using ITAMAE population execution.

The validated CDM calculation is the only product path. Historical version
reproduction is provided by the independent sashimi-family A/B workflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from itamae.evolution import shanks_transform
from itamae.execution import PopulationComponents
from itamae.numerics import gauss_hermite_lognormal
from itamae.provenance import build_calculation_metadata
from itamae.types import (
    WeightedSubhaloCatalog,
)
from scipy import integrate, special
from scipy.interpolate import interp1d

from sashimi_c_itamae_components import (
    CDMAccretionSlices,
    CDMCatalogColumns,
    NFWInitialStructure,
    TidalProfileEvolution,
    TruncationThresholdSurvival,
)
from sashimi_c_physics import (
    CDMObservableKernels,
    CDMPhysics,
    CDMTidalKernels,
    CDMUnits,
)

CALCULATION_SPECIFICATION = "sashimi-c:cdm:2026-09-10:v1"
_STRIPPING_METHODS = (
    "picard_table",
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
    calculation_specification
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
    calculation_specification: str
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
                "calculation_specification": self.calculation_specification,
                "backend_identifier": self.backend_identifier,
                "sample_size": int(self.mass_at_accretion.size),
                "max_relative_difference": self.max_relative_difference,
            }
        )


class TidalStrippingSolver(CDMTidalKernels):
    """C stripping solvers with the common Shanks primitive."""

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


class SubhaloProperties(CDMPhysics):
    """Generate named, weighted CDM catalogs through ITAMAE."""

    def Ffunc(self, dela, s1, s2):
        """Returns Eq. (12) of Yang et al. (2011)"""
        return 1 / np.sqrt(2.0 * np.pi) * dela / (s2 - s1) ** 1.5

    def Gfunc(self, dela, s1, s2):
        G0 = 0.57
        gamma1 = 0.38
        gamma2 = -0.01
        sig1 = np.sqrt(s1)
        sig2 = np.sqrt(s2)
        return G0 * pow(sig2 / sig1, gamma1) * pow(dela / sig1, gamma2)

    def Ffunc_Yang(self, delc1, delc2, s1, s2):
        """Returns Eq. (14) of Yang et al. (2011)"""
        return (
            1.0
            / np.sqrt(2.0 * np.pi)
            * (delc2 - delc1)
            / (s2 - s1) ** 1.5
            * np.exp(-((delc2 - delc1) ** 2) / (2.0 * (s2 - s1)))
        )

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
        if "physics_mode" in kwargs:
            raise TypeError(
                "physics_mode was removed; use the independent A/B reference workflow."
            )
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
        solver = TidalStrippingSolver(
            M0=M0,
            z_min=redshift,
            z_max=zmax,
            n_z_interp=64,
            cosmology_backend=self.itamae_cosmology,
        )
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
        slice_builder = CDMAccretionSlices(
            model=self,
            ma200_grid=ma200_grid,
            zdist=zdist,
            population_2d=population_2d,
            sigmalogc=sigmalogc,
            N_herm=N_herm,
        )
        slices = [slice_builder.build(index) for index in range(len(zdist))]
        execution = PopulationComponents(
            initializer=NFWInitialStructure(model=self, N_herm=N_herm),
            evolver=TidalProfileEvolution(
                model=self,
                solver=solver,
                redshift=redshift,
                method=method,
                N_herm=N_herm,
                profile_change=profile_change,
                kwargs=kwargs,
            ),
            survival=TruncationThresholdSurvival(ct_threshold=ct_th),
            columns=CDMCatalogColumns(),
        ).execute(
            [batch for batch, _ in slices], contexts=[context for _, context in slices]
        )
        survive = execution.survival["default"]
        quadrature_weight = (
            execution.weight_factors["weight_base"]
            * execution.weight_factors["weight_concentration"]
        )
        total_quadrature_weight = float(np.sum(quadrature_weight))
        surviving_weight_fraction = (
            float(np.sum(quadrature_weight * survive) / total_quadrature_weight)
            if total_quadrature_weight > 0.0
            else 0.0
        )
        backend_identifier = f"array=numpy;cosmology={self.itamae_cosmology.identifier};units=canonical-Msun-Mpc-km-s"
        picard_table_settings = None
        if method == "picard_table":
            table = solver._get_picard_table(redshift)
            picard_table_settings = {
                "n_iterations": int(table.n_iterations),
                "n_z_acc": int(table.n_z_acc),
                "n_log_ratio": int(table.n_log_ratio),
                "log10_ratio_min": float(table.log10_ratio_min),
                "log10_ratio_max": float(table.log10_ratio_max),
                "n_integration": int(table.n_integration),
            }
        metadata = build_calculation_metadata(
            variant="sashimi-c",
            distribution_name="sashimi-c",
            module_file=__file__,
            calculation_specification=CALCULATION_SPECIFICATION,
            model_identifier=CALCULATION_SPECIFICATION,
            backend_identifier=backend_identifier,
            source_identifier="sashimi-c:standard-api:v1",
            variance_identifier="sashimi-c:analytic-cdm-fit:v1",
            power_identifier="sashimi-c:cdm-linear-power:v1",
            solver_identifier=f"sashimi-c:tidal-stripping:{method}:v1",
            extra={
                "cosmology_backend": self.itamae_cosmology.identifier,
                "cosmology_parameters": {
                    "omega_m0": float(np.asarray(self.itamae_cosmology.omega_m(0.0))),
                    "h": float(np.asarray(self.itamae_cosmology.H(0.0))) / 100.0,
                    "omega_lambda0": 1.0
                    - float(np.asarray(self.itamae_cosmology.omega_m(0.0))),
                },
                "canonical_units": {
                    "mass": "Msun",
                    "length": "Mpc",
                    "velocity": "km / s",
                    "density": "Msun / Mpc3",
                },
                "critical_density_convention": "itamae-backend",
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
                "picard_table_settings": picard_table_settings,
                "shanks_small_correction_threshold": _SHANKS_SMALL_CORRECTION_THRESHOLD,
                "ct_threshold": float(ct_th),
                "default_ct_threshold": _DEFAULT_CT_THRESHOLD,
                "survival_rule": "c_t > ct_threshold",
                "surviving_node_count": int(np.count_nonzero(survive)),
                "node_count": int(survive.size),
                "surviving_weight_fraction": surviving_weight_fraction,
                "tuple_weight_excludes_survival": True,
                "weight_semantics": {
                    "weight_base": "population measure",
                    "weight_concentration": "lognormal concentration quadrature",
                    "weight_survival": "binary c_t threshold",
                },
                "nfw_inversion": "itamae.brentq",
            },
        )
        catalog = execution.to_catalog(metadata)
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
        quadrature_weight = np.asarray(catalog.weights["weight_base"]) * np.asarray(
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
            quadrature_weight,
            columns["survive"],
        )


class SubhaloObservables(SubhaloProperties, CDMObservableKernels):
    """CDM catalog plus existing mass, satellite and annihilation observables."""

    def __init__(
        self,
        M0_per_Msun,
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
        prompt_cusps=False,
        k_fs_Mpc=1.06e6,
        filter="Sharp-k",
        alpha=1.8,
        method="pert2_shanks",
        cosmology_backend=None,
        data_dir=None,
        **kwargs,
    ):
        """
        This class computes various subhalo observables in a host halo.

        -----
        Input
        -----
        M0_per_Msun:               Mass of the host halo defined as M_{200} (200 times critial density)
                                   at z = 0, in units of solar mass, Msun. Note that this is *not* the
                                   host mass at the given redshift! It can be obtained via Mzi(M0,redshift).
                                   If you want to give this parameter as the mass at the given redshift,
                                   then turn 'M0_at_redshift' parameter on (see below).

        (Optional) redshift:       Redshift of interest. (default: 0)
        (Optional) dz:             Grid of redshift of halo accretion. (default 0.1)
        (Optional) zmax:           Maximum redshift to start the calculation of evolution from. (default: 7.)
        (Optional) N_ma:           Number of logarithmic grid of subhalo mass at accretion defined as M_{200}.
                                   (default: 500)
        (Optional) sigmalogc:      rms scatter of concentration parameter defined for log_{10}(c).
                                   (default: 0.128)
        (Optional) N_herm:         Number of grid in Gauss-Hermite quadrature for integral over concentration.
                                   (default: 5)
        (Optional) logmamin:       Minimum value of subhalo mass at accretion defined as log_{10}(m_{min}/Msun).
                                   (default: -6)
        (Optional) logmamax:       Maximum value of subhalo mass at accretion defined as log_{10}(m_{max}/Msun).
                                   If None, m_{max}=0.1*M0. (default: None)
        (Optional) N_hermNa:       Number of grid in Gauss-Hermite quadrature for integral over host evoluation,
                                   used in Na_calc. (default: 200)
        (Optional) Na_model:       Model number of EPS defined in Yang et al. (2011). (default: 3)
        (Optional) ct_th:          Threshold value for c_t(=r_t/r_s) parameter, below which a subhalo is assumed to
                                   be completely disrupted. Suggested values: 0.77 (disruption) or 0.0
                                   (no disruption; default).
        (Optional) profile_change: Whether we implement the evolution of subhalo density profile through tidal
                                   mass loss. (default: True)
        (Optional) M0_at_redshift: If True, M0 is regarded as the mass at a given redshift, instead of z=0.
        (Optional) method:         Method to calculate the subhalo mass stripping. (default: "pert2_shanks")
                                   - "odeint" : use odeint to solve the differential equation.
                                   - "pert0" : use perturbative method with zeroth-order correction.
                                   - "pert1" : use perturbative method with first-order correction.
                                   - "pert2" : use perturbative method with second-order correction.
                                   - "pert2_shanks" : use perturbative method with second-order correction
                                     and Shanks transformation.
                                   - "pert3" : use perturbative method with third-order correction.
        (Optional) kwargs:         Additional arguments for the odeint function.



        When called with these input parameters, this class initially creates a list of subhalos, characterized
        by the following output parameters.

        ------
        Output
        ------

        ma200:    Mass m_{200} at accretion.
        z_a:      Redshift at accretion.
        rs_a:     Scale radius r_s at accretion.
        rhos_a:   Characteristic density \rho_s at accretion.
        m0:       Mass up to tidal truncation radius at a given redshift.
        rs0:      Scale radius r_s at a given redshift.
        rhos0:    Characteristic density \rho_s at a given redshift.
        ct0:      Tidal truncation radius in units of r_s at a given redshift.
        weight:   Effective number of subhalos that are characterized by the same set of the parameters above.
        survive:  If that subhalo survive against tidal disruption or not.

        Vmax:     Maximum circular velocity at a given redshift.
        rmax:     Radius at which the orbital speed reaches Vmax.
        Vpeak:    Maximum circular velocity at accretion.
        rpeak:    Radius at which the orbital speed reaches Vpeak.

        """

        SubhaloProperties.__init__(
            self,
            cosmology_backend=cosmology_backend,
            prompt_cusps=prompt_cusps,
            k_fs_Mpc=k_fs_Mpc,
            filter=filter,
            alpha=alpha,
            data_dir=data_dir,
        )
        ma200, z_a, rs_a, rhos_a, m0, rs0, rhos0, ct0, weight, survive = (
            self.subhalo_properties_calc(
                M0_per_Msun * self.Msun,
                redshift,
                dz,
                zmax,
                N_ma,
                sigmalogc,
                N_herm,
                logmamin,
                logmamax,
                N_hermNa,
                Na_model,
                ct_th,
                profile_change,
                M0_at_redshift,
                method,
                **kwargs,
            )
        )
        self.ma200 = ma200[survive]
        self.z_a = z_a[survive]
        self.rs_a = rs_a[survive]
        self.rhos_a = rhos_a[survive]
        self.m0 = m0[survive]
        self.rs0 = rs0[survive]
        self.rhos0 = rhos0[survive]
        self.ct0 = ct0[survive]
        self.weight = weight[survive]
        self.rmax = 2.163 * self.rs0
        self.Vmax = np.sqrt(4.0 * np.pi * self.G * self.rhos0 / 4.625) * self.rs0
        self.rpeak = 2.163 * self.rs_a
        self.Vpeak = np.sqrt(4.0 * np.pi * self.G * self.rhos_a / 4.625) * self.rs_a

        self.redshift = redshift


def diagnose_stripping_approximation(
    host_mass: float,
    mass_at_accretion: Any,
    *,
    accretion_redshift: float = 1.0,
    target_redshift: float = 0.0,
    n_z_interp: int = 64,
    cosmology_backend: Any | None = None,
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

    solver = TidalStrippingSolver(
        host_mass_value,
        z_min=target_redshift_value,
        z_max=accretion_redshift_value,
        n_z_interp=n_z_interp,
        cosmology_backend=cosmology_backend,
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
        calculation_specification=CALCULATION_SPECIFICATION,
        backend_identifier=solver.itamae_cosmology.identifier,
    )


HaloModel = CDMPhysics
halo_model = HaloModel
subhalo_properties = SubhaloProperties
subhalo_observables = SubhaloObservables
cosmology = CDMPhysics
units_and_constants = CDMUnits

__all__ = [
    "CALCULATION_SPECIFICATION",
    "HaloModel",
    "StrippingDiagnostics",
    "SubhaloObservables",
    "SubhaloProperties",
    "TidalStrippingSolver",
    "cosmology",
    "diagnose_stripping_approximation",
    "halo_model",
    "subhalo_observables",
    "subhalo_properties",
    "units_and_constants",
]
