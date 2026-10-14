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

from ._itamae_components import (
    CDMAccretionSlices,
    CDMCatalogColumns,
    NFWInitialStructure,
    TidalProfileEvolution,
    TruncationThresholdSurvival,
)
from ._physics import (
    CDMObservableKernels,
    CDMPhysics,
    CDMTidalKernels,
    CDMUnits,
)

CALCULATION_SPECIFICATION = "sashimi-c:cdm:2026-09-10:v1"
_STRIPPING_METHODS = (
    "picard_table",
    "dop853",
    "odeint",
    "pert0",
    "pert1",
    "pert2",
    "pert2_shanks",
    "pert3",
)
_DEFAULT_STRIPPING_METHOD = "picard_table"
_DEFAULT_CT_THRESHOLD = 0.0
_SHANKS_SMALL_CORRECTION_THRESHOLD = 0.02


@dataclass(frozen=True, slots=True)
class StrippingDiagnostics:
    """Record the SASHIMI-C Shanks approximation against direct ODE integration.

    The diagnostic is intentionally separate from catalog generation. Running
    a direct ODE solve for every catalog node would change the established
    performance characteristics of the historical ``pert2_shanks`` calculation.

    Parameters
    ----------
    mass_at_accretion : numpy.ndarray
        Aligned subhalo masses at accretion or after the named stripping calculation, in the
        internal mass unit.
    mass_pert2_shanks : numpy.ndarray
        Aligned subhalo masses at accretion or after the named stripping calculation, in the
        internal mass unit.
    mass_odeint : numpy.ndarray
        Aligned subhalo masses at accretion or after the named stripping calculation, in the
        internal mass unit.
    relative_difference : numpy.ndarray
        Absolute fractional Shanks-versus-ODE mass difference, normalized by the direct ODE
        mass.
    host_mass : float
        Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
        explicit epoch flags are described by the caller.
    accretion_redshift : float
        Scalar dimensionless start or final epoch for the stripping comparison; accretion
        must precede the target in cosmic time.
    target_redshift : float
        Scalar dimensionless start or final epoch for the stripping comparison; accretion
        must precede the target in cosmic time.
    calculation_specification : str
        Versioned identifier of the current physical and numerical calculation choices.
    backend_identifier : str
        Identity of the cosmology, unit and array backends used for this calculation.

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

    Notes
    -----
    The CDM population combines EPS accretion, lognormal concentration scatter, the selected
    tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
    concentration and survival as separate factors; changing a numerical solver does not by
    itself recalibrate any fitted physical relation.
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
        """Freeze aligned diagnostic arrays.

        Notes
        -----
        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.
        """
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
        """Return the largest Shanks-versus-ODE fractional difference.

        Returns
        -------
        float
            Maximum absolute fractional mass discrepancy relative to the direct ODE output.

        Notes
        -----
        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.
        """
        return float(np.max(self.relative_difference, initial=0.0))

    def summary(self) -> Mapping[str, Any]:
        """Return immutable JSON-compatible diagnostic provenance.

        Returns
        -------
        Mapping[str, object]
            Immutable comparison settings and maximum discrepancy; not a physical-model accuracy
            certificate.

        Notes
        -----
        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.
        """
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
    """C stripping solvers with the common Shanks primitive.

    Notes
    -----
    The CDM population combines EPS accretion, lognormal concentration scatter, the selected
    tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
    concentration and survival as separate factors; changing a numerical solver does not by
    itself recalibrate any fitted physical relation.
    """

    def subhalo_mass_stripped_pert2_shanks(self, ma, za, z):
        """Evaluate second-order stripping with ITAMAE Shanks acceleration.

        The SASHIMI-specific two-percent stability criterion is retained around
        ITAMAE's generic sequence transformation.

        Parameters
        ----------
        ma : float or array_like
            Positive subhalo mass at accretion, in the same units as reference_mass; endpoint
            queries broadcast over mass and redshift.
        za : float or array_like
            Dimensionless accretion redshift. History integration requires one scalar accretion
            epoch.
        z : float or array_like
            Dimensionless redshift. The cosmological background requires z > -1; population
            nodes use nonnegative redshifts.

        Returns
        -------
        float or numpy.ndarray
            Final bound masses in the ma unit; this endpoint interface does not return a
            redshift/mass tuple.

        Notes
        -----
        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.
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
    """Generate named, weighted CDM catalogs through ITAMAE.

    Notes
    -----
    The CDM population combines EPS accretion, lognormal concentration scatter, the selected
    tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
    concentration and survival as separate factors; changing a numerical solver does not by
    itself recalibrate any fitted physical relation.
    """

    def Ffunc(self, dela, s1, s2):
        r"""Evaluate the Yang et al. EPS kernel before support normalization.

        Parameters
        ----------
        dela : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        s1 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.
        s2 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.

        Returns
        -------
        array_like
            delta_a/[sqrt(2*pi)*(s2-s1)**(3/2)], on positive variance support.

        Notes
        -----
        The EPS accretion kernel and, where used, host-history scatter 0.12-0.15*log10(Ma/M0)
        follow Yang et al., Eqs. (12)-(14), (21). Finite-gap normalization and zero-barrier
        limits are numerical treatments of that kernel, not new empirical fits.

        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.

        References
        ----------

        .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
            https://arxiv.org/abs/1104.1757
        """
        return 1 / np.sqrt(2.0 * np.pi) * dela / (s2 - s1) ** 1.5

    def Gfunc(self, dela, s1, s2):
        r"""Evaluate the retained empirical multiplicative correction to the EPS kernel.

        Parameters
        ----------
        dela : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        s1 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.
        s2 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.

        Returns
        -------
        array_like
            0.57*(sqrt(s2/s1))**0.38*(delta_a/sqrt(s1))**(-0.01), dimensionless.

        Notes
        -----
        The empirical correction G0=0.57, gamma1=0.38, gamma2=-0.01 is from Parkinson et al.,
        Section 3.1, and is adopted in Yang et al., Eq. (13).

        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.

        References
        ----------

        .. [cite-parkinson] Parkinson, Cole & Helly (2008), Generating Dark Matter Halo Merger Trees, Section 3.1.
            https://arxiv.org/abs/0708.1382

        .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
            https://arxiv.org/abs/1104.1757
        """
        G0 = 0.57
        gamma1 = 0.38
        gamma2 = -0.01
        sig1 = np.sqrt(s1)
        sig2 = np.sqrt(s2)
        return G0 * pow(sig2 / sig1, gamma1) * pow(dela / sig1, gamma2)

    def Ffunc_Yang(self, delc1, delc2, s1, s2):
        r"""Evaluate the conditional EPS kernel with its collapse-barrier difference.

        Parameters
        ----------
        delc1 : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        delc2 : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        s1 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.
        s2 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.

        Returns
        -------
        array_like
            Gaussian first-crossing density in variance, before support normalization.

        Notes
        -----
        The EPS accretion kernel and, where used, host-history scatter 0.12-0.15*log10(Ma/M0)
        follow Yang et al., Eqs. (12)-(14), (21). Finite-gap normalization and zero-barrier
        limits are numerical treatments of that kernel, not new empirical fits.

        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.

        References
        ----------

        .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
            https://arxiv.org/abs/1104.1757
        """
        return (
            1.0
            / np.sqrt(2.0 * np.pi)
            * (delc2 - delc1)
            / (s2 - s1) ** 1.5
            * np.exp(-((delc2 - delc1) ** 2) / (2.0 * (s2 - s1)))
        )

    def _normalized_yang_kernel(self, delc1, delc2, s1, s2, smin):
        r"""Yang Eq. (14) divided by its mass-support integral, including gap=0.

        For x=delta/sqrt(2*(smin-s1)), erf(x)/x tends to 2/sqrt(pi).
        The small-x series evaluates that ratio before the vanishing factors
        are divided. This is the same normalized kernel, with no weight cut.

        Parameters
        ----------
        delc1 : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        delc2 : float or array_like
            Dimensionless linear-collapse barrier at the indicated epoch, extrapolated in the
            same growth convention.
        s1 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.
        s2 : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.
        smin : float or array_like
            Dimensionless linear variance S=sigma**2 at the indicated host/subhalo/support mass.
            Active EPS terms require positive variance gaps.

        Returns
        -------
        numpy.ndarray
            Conditional variance kernel divided by its finite-support integral, including the
            zero-barrier-gap limit.

        Notes
        -----
        The EPS accretion kernel and, where used, host-history scatter 0.12-0.15*log10(Ma/M0)
        follow Yang et al., Eqs. (12)-(14), (21). Finite-gap normalization and zero-barrier
        limits are numerical treatments of that kernel, not new empirical fits.

        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.

        References
        ----------

        .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
            https://arxiv.org/abs/1104.1757
        """
        d1, d2, a, b, minimum = np.broadcast_arrays(delc1, delc2, s1, s2, smin)
        gap, ds, dsmin = d2 - d1, b - a, minimum - a
        if np.any(gap < 0) or np.any(ds <= 0) or np.any(dsmin <= 0):
            raise ValueError(
                "Normalized Yang kernel requires nonnegative barrier gap and positive variance gaps."
            )
        x2 = gap**2 / (2.0 * dsmin)
        small = x2 < 1e-8
        result = np.empty_like(x2, dtype=float)
        x = x2[small]
        integral = 1.0 - x / 3.0 + x**2 / 10.0 - x**3 / 42.0 + x**4 / 216.0
        result[small] = (
            np.sqrt(dsmin[small])
            / (2.0 * ds[small] ** 1.5)
            * np.exp(-(gap[small] ** 2) / (2.0 * ds[small]))
            / integral
        )
        regular = ~small
        norm = special.gamma(0.5) * special.gammainc(0.5, x2[regular]) / np.sqrt(np.pi)
        result[regular] = (
            self.Ffunc_Yang(d1[regular], d2[regular], a[regular], b[regular]) / norm
        )
        return result

    def Na_calc(self, ma, zacc, Mhost, z0=0.0, N_herm=200, Nrand=1000, Na_model=3):
        r"""Evaluate the Yang et al. accretion rate with ITAMAE quadrature.

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
            Expected accretion rate per d ln(ma) d ln(1+zacc), after host-scatter averaging;
            first axis is redshift.

        Notes
        -----
        The EPS accretion kernel and, where used, host-history scatter 0.12-0.15*log10(Ma/M0)
        follow Yang et al., Eqs. (12)-(14), (21). Finite-gap normalization and zero-barrier
        limits are numerical treatments of that kernel, not new empirical fits.

        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.

        References
        ----------

        .. [cite-yang] Yang et al. (2011), An Analytical Model for the Accretion of Dark Matter Subhalos, Eqs. (12)-(14), (21).
            https://arxiv.org/abs/1104.1757
        """

        zacc_2d = np.asarray(zacc).reshape(-1, 1)
        M200_0 = self.Mzzi(Mhost, zacc_2d, z0)
        sigmalogM200 = 0.12 - 0.15 * np.log10(M200_0 / Mhost)
        M200, host_weight = gauss_hermite_lognormal(M200_0, sigmalogM200, order=N_herm)

        mmax = np.minimum(M200, Mhost / 2.0)
        Mmax = np.minimum(M200_0 + mmax, Mhost)

        if Na_model == 3:
            zlist = zacc_2d * np.linspace(1, 0, Nrand)
            iMmax = np.argmin(np.abs(self.Mzzi(Mhost, zlist, z0) - Mmax), axis=-1)
            z_Max = zlist[np.arange(len(zlist)), iMmax]
            z_Max_3d = z_Max.reshape(N_herm, len(zlist), 1)
            delcM = self.deltac_func(z_Max_3d)
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(Mmax)
            sa = self.s_func(ma)
            # those reside in the exponential part of Eq. (14)
            d1, d2, s1, s2, smin, allowed = np.broadcast_arrays(
                delcM, delca, sM, sa, self.s_func(mmax), mmax > ma
            )
            Phi = np.zeros(s2.shape)
            Phi[allowed] = self._normalized_yang_kernel(
                d1[allowed], d2[allowed], s1[allowed], s2[allowed], smin[allowed]
            )
        elif Na_model == 1:
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(M200)
            sa = self.s_func(ma)
            xmin = self.s_func(mmax) - self.s_func(M200)
            d_norm, s_norm, x_norm = np.broadcast_arrays(delca, sM, xmin)
            if np.any(x_norm < 0):
                raise ValueError(
                    "EPS normalization requires a nonnegative variance support gap."
                )
            positive = x_norm > 0
            normB = np.full(x_norm.shape, np.inf)
            d_pos, s_pos, x_pos = d_norm[positive], s_norm[positive], x_norm[positive]
            normB[positive] = (
                1.0
                / np.sqrt(2 * np.pi)
                * d_pos
                * 2.0
                / x_pos**0.5
                * special.hyp2f1(0.5, 0.0, 1.5, -s_pos / x_pos)
            )
            d, s1, s2, norm, allowed = np.broadcast_arrays(
                delca, sM, sa, normB, mmax > ma
            )
            if np.any((s2 - s1)[allowed] <= 0):
                raise ValueError(
                    "EPS accretion requires positive variance gaps in its active mass domain."
                )
            Phi = np.zeros(s2.shape)
            Phi[allowed] = (
                self.Ffunc(d[allowed], s1[allowed], s2[allowed]) / norm[allowed]
            )
        elif Na_model == 2:
            delca = self.deltac_func(zacc_2d)
            sM = self.s_func(M200)
            sa = self.s_func(ma)
            xmin = self.s_func(mmax) - self.s_func(M200)
            d_norm, s_norm, x_norm = np.broadcast_arrays(delca, sM, xmin)
            if np.any(x_norm < 0):
                raise ValueError(
                    "EPS normalization requires a nonnegative variance support gap."
                )
            positive = x_norm > 0
            normB = np.full(x_norm.shape, np.inf)
            d_pos, s_pos, x_pos = d_norm[positive], s_norm[positive], x_norm[positive]
            normB[positive] = (
                1.0
                / np.sqrt(2.0 * np.pi)
                * d_pos
                * 0.57
                * (d_pos / np.sqrt(s_pos)) ** -0.01
                * (2.0 / (1.0 - 0.38))
                * s_pos ** (-0.38 / 2.0)
                * x_pos ** (0.5 * (0.38 - 1.0))
                * special.hyp2f1(
                    0.5 * (1 - 0.38), -0.38 / 2.0, 0.5 * (3.0 - 0.38), -s_pos / x_pos
                )
            )
            d, s1, s2, norm, allowed = np.broadcast_arrays(
                delca, sM, sa, normB, mmax > ma
            )
            if np.any((s2 - s1)[allowed] <= 0):
                raise ValueError(
                    "EPS accretion requires positive variance gaps in its active mass domain."
                )
            Phi = np.zeros(s2.shape)
            Phi[allowed] = (
                self.Ffunc(d[allowed], s1[allowed], s2[allowed])
                * self.Gfunc(d[allowed], s1[allowed], s2[allowed])
                / norm[allowed]
            )
        else:
            raise ValueError("Na_model must be 1, 2, or 3.")
        if not np.all(np.isfinite(Phi)):
            raise ValueError(
                "Non-finite EPS accretion kernel inside its active mass domain."
            )
        F2 = np.sum(Phi * host_weight, axis=0)
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
        method="picard_table",
        **kwargs,
    ) -> WeightedSubhaloCatalog:
        r"""Generate an ITAMAE weighted catalog while preserving SASHIMI physics.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        redshift : float
            Nonnegative target redshift of the subhalo population.
        dz : float
            Positive nominal spacing of the accretion-redshift quadrature grid.
        zmax : float
            Largest requested accretion redshift, above the target epoch; the actual grid
            convention belongs to this variant.
        N_ma : int
            Number of logarithmically spaced accretion-mass nodes.
        sigmalogc : float
            Nonnegative standard deviation of log10 concentration, in dex.
        N_herm : int
            Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
            concentration in catalogue stages or host history in Na_calc.
        logmamin : float, optional
            Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
            host-relative mass limit.
        logmamax : float, optional
            Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
            host-relative mass limit.
        N_hermNa : int
            Positive Gauss-Hermite order for the host-history scatter in the EPS accretion rate.
        Na_model : {1, 2, 3}
            Yang EPS accretion/normalization prescription. This is a physical model selection,
            not a numerical tolerance.
        ct_th : float
            Dimensionless tidal truncation threshold r_t/r_s; the specified strict survival rule
            is applied once through its mask.
        profile_change : bool
            Apply the variant tidal response of scale radius and density in addition to
            bound-mass loss.
        M0_at_redshift : bool
            Interpret the input host mass at the target redshift and invert the prescribed
            history to obtain its z=0 reference value.
        method : str
            Explicit tidal integration method: picard_table, dop853, odeint or a retained
            perturbative approximation.
        **kwargs : dict
            Additional keyword options forwarded to the called routine; reserved scientific
            controls remain owned by that routine.

        Returns
        -------
        itamae.types.WeightedSubhaloCatalog
            Catalog with separate population, concentration, and survival
            weights. The product of population and concentration weights equals
            the historical tuple ``weight``; survival remains an independent
            factor for diagnostics and reweighting.

        Notes
        -----
        The retained default scatter sigma(log10(c)) = 0.128 dex is the Ishiyama et al. Eq. (7),
        Fig. 10 value for z=0 halos of 1e9 to 1e10 solar masses; Hiroshima et al. adopt the
        rounded value 0.13 dex. Using one value across masses and redshifts is a model
        assumption, not a quadrature error or a new calibration. The literature threshold rt/rs
        > 0.77 follows the Hayashi et al. disruption prescription adopted by Hiroshima et al. A
        caller-supplied threshold takes precedence in variants that expose it (C and SI
        currently default to 0). This is a model selection rule; an exact-boundary test does not
        calibrate disruption physics.

        Catalogue masses and lengths are Msun and Mpc in this C implementation, densities
        Msun/Mpc^3. All accretion nodes are retained; weight_final includes the survival mask.
        The legacy tuple exposes the same nodes with a separate mask. The z grid uses the
        retained arange(target+dz,zmax+dz,dz) convention, so a non-aligned zmax can be exceeded
        by less than one nominal step.

        References
        ----------

        .. [cite-scatter] Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.
            https://arxiv.org/abs/1101.2020

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
            https://arxiv.org/abs/astro-ph/0203004
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
        picard_diagnostics = None
        picard_table_settings = None
        if method == "picard_table":
            from .picard_tidal_stripping import get_picard_solver

            picard_diagnostics = get_picard_solver(solver).diagnostics()
            if picard_diagnostics["tables"]:
                picard_table_settings = picard_diagnostics["tables"][0]
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
            solver_identifier=f"sashimi-c:tidal-stripping:{method}:itamae:v2",
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
                "picard_diagnostics": picard_diagnostics,
                "solver_options": dict(kwargs),
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
        """Reject inputs outside the physical and numerical catalog domain.

        Parameters
        ----------
        M0 : float
            Reference host M200 in the variant internal mass unit. M0 normally refers to z=0;
            explicit epoch flags are described by the caller.
        redshift : float
            Nonnegative target redshift of the subhalo population.
        dz : float
            Positive nominal spacing of the accretion-redshift quadrature grid.
        zmax : float
            Largest requested accretion redshift, above the target epoch; the actual grid
            convention belongs to this variant.
        N_ma : int
            Number of logarithmically spaced accretion-mass nodes.
        sigmalogc : float
            Nonnegative standard deviation of log10 concentration, in dex.
        N_herm : int
            Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
            concentration in catalogue stages or host history in Na_calc.
        logmamin : float, optional
            Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
            host-relative mass limit.
        logmamax : float, optional
            Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
            host-relative mass limit.
        N_hermNa : int
            Positive Gauss-Hermite order for the host-history scatter in the EPS accretion rate.
        Na_model : {1, 2, 3}
            Yang EPS accretion/normalization prescription. This is a physical model selection,
            not a numerical tolerance.
        ct_th : float
            Dimensionless tidal truncation threshold r_t/r_s; the specified strict survival rule
            is applied once through its mask.
        method : str
            Explicit tidal integration method: picard_table, dop853, odeint or a retained
            perturbative approximation.

        Notes
        -----
        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.
        """
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
        r"""Return the historical tuple from the ITAMAE catalog calculation.

        Parameters
        ----------
        *args : tuple
            Additional positional parameters forwarded unchanged to the supplied numerical or
            physical callback.
        **kwargs : dict
            Additional keyword options forwarded to the called routine; reserved scientific
            controls remain owned by that routine.

        Returns
        -------
        tuple of numpy.ndarray
            Historical ten arrays: M200_acc, z_acc, r_s_acc, rho_s_acc, m_bound, r_s, rho_s,
            c_t, generation weight and survival mask. The weight excludes survival.

        Notes
        -----
        The retained default scatter sigma(log10(c)) = 0.128 dex is the Ishiyama et al. Eq. (7),
        Fig. 10 value for z=0 halos of 1e9 to 1e10 solar masses; Hiroshima et al. adopt the
        rounded value 0.13 dex. Using one value across masses and redshifts is a model
        assumption, not a quadrature error or a new calibration. The literature threshold rt/rs
        > 0.77 follows the Hayashi et al. disruption prescription adopted by Hiroshima et al. A
        caller-supplied threshold takes precedence in variants that expose it (C and SI
        currently default to 0). This is a model selection rule; an exact-boundary test does not
        calibrate disruption physics.

        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.

        References
        ----------

        .. [cite-scatter] Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.
            https://arxiv.org/abs/1101.2020

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
            https://arxiv.org/abs/astro-ph/0203004
        """
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
    """CDM catalog plus existing mass, satellite and annihilation observables.

    Parameters
    ----------
    M0_per_Msun : float
        Host M200 numerical value in solar masses, normally at z=0. M0_at_redshift=True
        instead assigns it to the target epoch.
    redshift : float
        Nonnegative target redshift of the subhalo population.
    dz : float
        Positive nominal spacing of the accretion-redshift quadrature grid.
    zmax : float
        Largest requested accretion redshift, above the target epoch; the actual grid
        convention belongs to this variant.
    N_ma : int
        Number of logarithmically spaced accretion-mass nodes.
    sigmalogc : float
        Nonnegative standard deviation of log10 concentration, in dex.
    N_herm : int
        Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
        concentration in catalogue stages or host history in Na_calc.
    logmamin : float, optional
        Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
        host-relative mass limit.
    logmamax : float, optional
        Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
        host-relative mass limit.
    N_hermNa : int
        Positive Gauss-Hermite order for the host-history scatter in the EPS accretion rate.
    Na_model : {1, 2, 3}
        Yang EPS accretion/normalization prescription. This is a physical model selection,
        not a numerical tolerance.
    ct_th : float
        Dimensionless tidal truncation threshold r_t/r_s; the specified strict survival rule
        is applied once through its mask.
    profile_change : bool
        Apply the variant tidal response of scale radius and density in addition to
        bound-mass loss.
    M0_at_redshift : bool
        Interpret the input host mass at the target redshift and invert the prescribed
        history to obtain its z=0 reference value.
    prompt_cusps : bool
        Enable the variant prompt-cusp prescription and its explicitly supplied spectrum
        inputs.
    k_fs_Mpc : float
        Free-streaming wavenumber numerical value in Mpc^-1 for the prompt-cusp spectrum
        prescription.
    filter : str
        Named physical smoothing/filter prescription used in the prompt-cusp calculation.
    alpha : float
        Dimensionless shape parameter of the selected prompt-cusp filter/transfer
        prescription.
    method : str
        Explicit tidal integration method: picard_table, dop853, odeint or a retained
        perturbative approximation.
    cosmology_backend : CosmologyBackend, optional
        ITAMAE background provider. The variant checks compatibility with the cosmology of
        its fitted halo relations.
    data_dir : pathlib.Path or str, optional
        Explicit root of the required scientific input tables or generated products; no
        substitute spectrum is selected implicitly.
    **kwargs : dict
        Additional keyword options forwarded to the called routine; reserved scientific
        controls remain owned by that routine.

    Notes
    -----
    The CDM population combines EPS accretion, lognormal concentration scatter, the selected
    tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
    concentration and survival as separate factors; changing a numerical solver does not by
    itself recalibrate any fitted physical relation.
    """

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
        method="picard_table",
        cosmology_backend=None,
        data_dir=None,
        **kwargs,
    ):
        r"""Compute a CDM population and retain surviving nodes for the standard observable
        reductions.

        Parameters
        ----------
        M0_per_Msun : float
            Host M200 numerical value in solar masses, normally at z=0. M0_at_redshift=True
            instead assigns it to the target epoch.
        redshift : float
            Nonnegative target redshift of the subhalo population.
        dz : float
            Positive nominal spacing of the accretion-redshift quadrature grid.
        zmax : float
            Largest requested accretion redshift, above the target epoch; the actual grid
            convention belongs to this variant.
        N_ma : int
            Number of logarithmically spaced accretion-mass nodes.
        sigmalogc : float
            Nonnegative standard deviation of log10 concentration, in dex.
        N_herm : int
            Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
            concentration in catalogue stages or host history in Na_calc.
        logmamin : float, optional
            Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
            host-relative mass limit.
        logmamax : float, optional
            Lower or upper log10(M200_acc/Msun) bound. A None upper bound uses the model
            host-relative mass limit.
        N_hermNa : int
            Positive Gauss-Hermite order for the host-history scatter in the EPS accretion rate.
        Na_model : {1, 2, 3}
            Yang EPS accretion/normalization prescription. This is a physical model selection,
            not a numerical tolerance.
        ct_th : float
            Dimensionless tidal truncation threshold r_t/r_s; the specified strict survival rule
            is applied once through its mask.
        profile_change : bool
            Apply the variant tidal response of scale radius and density in addition to
            bound-mass loss.
        M0_at_redshift : bool
            Interpret the input host mass at the target redshift and invert the prescribed
            history to obtain its z=0 reference value.
        prompt_cusps : bool
            Enable the variant prompt-cusp prescription and its explicitly supplied spectrum
            inputs.
        k_fs_Mpc : float
            Free-streaming wavenumber numerical value in Mpc^-1 for the prompt-cusp spectrum
            prescription.
        filter : str
            Named physical smoothing/filter prescription used in the prompt-cusp calculation.
        alpha : float
            Dimensionless shape parameter of the selected prompt-cusp filter/transfer
            prescription.
        method : str
            Explicit tidal integration method: picard_table, dop853, odeint or a retained
            perturbative approximation.
        cosmology_backend : CosmologyBackend, optional
            ITAMAE background provider. The variant checks compatibility with the cosmology of
            its fitted halo relations.
        data_dir : pathlib.Path or str, optional
            Explicit root of the required scientific input tables or generated products; no
            substitute spectrum is selected implicitly.
        **kwargs : dict
            Additional keyword options forwarded to the called routine; reserved scientific
            controls remain owned by that routine.

        Notes
        -----
        The NFW profile is from Navarro et al. The rounded factors rmax/rs = 2.163 (2.1626 where
        explicitly used) and 4.625 are NFW maximum-circular-speed identities, as written in
        Hiroshima et al., Eqs. (5)-(7), rather than independent empirical fits.

        The retained default scatter sigma(log10(c)) = 0.128 dex is the Ishiyama et al. Eq. (7),
        Fig. 10 value for z=0 halos of 1e9 to 1e10 solar masses; Hiroshima et al. adopt the
        rounded value 0.13 dex. Using one value across masses and redshifts is a model
        assumption, not a quadrature error or a new calibration. The literature threshold rt/rs
        > 0.77 follows the Hayashi et al. disruption prescription adopted by Hiroshima et al. A
        caller-supplied threshold takes precedence in variants that expose it (C and SI
        currently default to 0). This is a model selection rule; an exact-boundary test does not
        calibrate disruption physics.

        If prompt cusps are enabled, the benchmark cutoff k_fs=1.06e6 Mpc**(-1) and sharp-k
        multiplier alpha=1.8 follow the prompt-cusp model of Ando et al.; these are separate
        from the CDM variance fit used by default.

        The CDM population combines EPS accretion, lognormal concentration scatter, the selected
        tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
        concentration and survival as separate factors; changing a numerical solver does not by
        itself recalibrate any fitted physical relation.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691

        .. [cite-scatter] Ishiyama et al. (2013), The Cosmogrid Simulation: Statistical Properties of Small Dark Matter Halos, Eq. (7), Fig. 10.
            https://arxiv.org/abs/1101.2020

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
            https://arxiv.org/abs/astro-ph/0203004

        .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
            https://arxiv.org/abs/2601.19863
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
    """Compare the explicitly selected Shanks approximation with the same tidal law integrated
    by odeint.

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
    ``method="picard_table"`` by default; this diagnostic explicitly compares Shanks and odeint.

    The CDM population combines EPS accretion, lognormal concentration scatter, the selected
    tidal solver and the strict c_t survival cut. Named catalogues retain abundance,
    concentration and survival as separate factors; changing a numerical solver does not by
    itself recalibrate any fitted physical relation.
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
        method="pert2_shanks",
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
