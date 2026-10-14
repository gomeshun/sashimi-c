"""CDM prescriptions composed through ITAMAE's population executor.

These stages retain SASHIMI-C's historical internal floating units. The adapter
converts the finished catalog at its explicit canonical-unit boundary. Model
objects and stripping solvers are supplied by C, never selected by ITAMAE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from itamae.halo import invert_nfw_mass_function
from itamae.measure import build_accretion_batch
from itamae.numerics import gauss_hermite_lognormal
from itamae.protocols.execution import PopulationState
from itamae.types import AccretionBatch


@dataclass(frozen=True, slots=True)
class CDMAccretionSlices:
    """Prepare C-owned virial conversion and concentration quadrature slices.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    ma200_grid : numpy.ndarray
        One-dimensional accretion M200 grid in the variant internal mass unit.
    zdist : numpy.ndarray
        One-dimensional dimensionless accretion-redshift grid, paired with the population
        grid first axis.
    population_2d : numpy.ndarray
        Expected accretion abundance per redshift/mass node, shape (n_redshift,n_mass),
        before concentration scatter.
    sigmalogc : float
        Nonnegative standard deviation of log10 concentration, in dex.
    N_herm : int
        Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
        concentration in catalogue stages or host history in Na_calc.

    Notes
    -----
    Each accretion slice is flattened in concentration-major, then mass order. Stages retain
    C internal Msun/Mpc/second units and explicit independent weights; the model owns virial
    conversion, NFW structure, tidal response and disruption.
    """

    model: Any
    ma200_grid: np.ndarray
    zdist: np.ndarray
    population_2d: np.ndarray
    sigmalogc: float
    N_herm: int

    def build(self, index: int):
        """Construct one EPS accretion slice with concentration quadrature and explicit virial
        conversion.

        Parameters
        ----------
        index : int
            Index of the selected accretion-redshift slice in zdist and population_2d.

        Returns
        -------
        tuple
            Flattened AccretionBatch and context holding the unflattened virial mass/radius
            arrays at this redshift.

        Notes
        -----
        Each accretion slice is flattened in concentration-major, then mass order. Stages retain
        C internal Msun/Mpc/second units and explicit independent weights; the model owns virial
        conversion, NFW structure, tidal response and disruption.
        """
        z_acc_value = self.zdist[index]
        ma = self.model.Mvir_from_M200_fit(self.ma200_grid, z_acc_value)
        Oz = self.model.OmegaM * (1.0 + z_acc_value) ** 3 / self.model.g(z_acc_value)
        c200sub = self.model.conc200(self.ma200_grid, z_acc_value)
        rvirsub = (
            3.0
            * ma
            / (
                4.0
                * np.pi
                * self.model.rhocrit0
                * self.model.g(z_acc_value)
                * self.model.Delc(Oz - 1.0)
            )
        ) ** (1.0 / 3.0)
        r200sub = (
            3.0
            * self.ma200_grid
            / (4.0 * np.pi * self.model.rhocrit0 * self.model.g(z_acc_value) * 200.0)
        ) ** (1.0 / 3.0)
        c_mz = c200sub * rvirsub / r200sub
        c_sub, concentration_weight = gauss_hermite_lognormal(
            c_mz, self.sigmalogc, order=self.N_herm
        )
        batch = build_accretion_batch(
            self.ma200_grid,
            z_acc_value,
            c_sub,
            self.population_2d[index],
            concentration_weight,
            mvir_acc=ma,
            metadata={
                "model": "sashimi-c",
                "calculation_specification": "sashimi-c:cdm:2026-09-10:v1",
            },
        )
        context = {
            "redshift_index": index,
            "mvir_acc": ma,
            "z_acc": z_acc_value,
            "rvir_acc": rvirsub,
            "r200_acc": r200sub,
        }
        return batch, context


@dataclass(frozen=True, slots=True)
class NFWInitialStructure:
    """Create the unstripped NFW structure at each accretion node.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    N_herm : int
        Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
        concentration in catalogue stages or host history in Na_calc.

    Notes
    -----
    Each accretion slice is flattened in concentration-major, then mass order. Stages retain
    C internal Msun/Mpc/second units and explicit independent weights; the model owns virial
    conversion, NFW structure, tidal response and disruption.
    """

    model: Any
    N_herm: int

    def initialize(self, batch, context):
        r"""Normalize the accreted NFW profile to the virial mass for every concentration node.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        dict
            Flattened r_s_acc and rho_s_acc arrays in Mpc and Msun/Mpc^3.

        Notes
        -----
        The NFW profile is from Navarro et al. The rounded factors rmax/rs = 2.163 (2.1626 where
        explicitly used) and 4.625 are NFW maximum-circular-speed identities, as written in
        Hiroshima et al., Eqs. (5)-(7), rather than independent empirical fits.

        Each accretion slice is flattened in concentration-major, then mass order. Stages retain
        C internal Msun/Mpc/second units and explicit independent weights; the model owns virial
        conversion, NFW structure, tidal response and disruption.

        References
        ----------

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691
        """
        c_sub = batch.concentration_acc.reshape(self.N_herm, -1)
        rs_acc = context["rvir_acc"] / c_sub
        rhos_acc = context["mvir_acc"] / (
            4.0 * np.pi * rs_acc**3 * self.model.fc(c_sub)
        )
        return {
            "r_s_acc": rs_acc.reshape(-1),
            "rho_s_acc": rhos_acc.reshape(-1),
        }


@dataclass(frozen=True, slots=True)
class TidalProfileEvolution:
    """Apply the supplied C mass-loss solver and C profile-response formulae.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    solver : TidalStrippingSolver or object
        C-owned tidal coefficient provider exposing Phi(z), zetaMz(z), Mzvir(z) and its
        mutable physical state.
    redshift : float
        Nonnegative target redshift of the subhalo population.
    method : str
        Explicit tidal integration method: picard_table, dop853, odeint or a retained
        perturbative approximation.
    N_herm : int
        Positive Gauss-Hermite quadrature order for the scatter integrated by this routine:
        concentration in catalogue stages or host history in Na_calc.
    profile_change : bool
        Apply the variant tidal response of scale radius and density in addition to
        bound-mass loss.
    kwargs : dict
        Additional keyword options forwarded to the called routine; reserved scientific
        controls remain owned by that routine.

    Notes
    -----
    Each accretion slice is flattened in concentration-major, then mass order. Stages retain
    C internal Msun/Mpc/second units and explicit independent weights; the model owns virial
    conversion, NFW structure, tidal response and disruption.
    """

    model: Any
    solver: Any
    redshift: float
    method: str
    N_herm: int
    profile_change: bool
    kwargs: dict[str, Any]

    def evolve(self, batch, initial, context):
        r"""Strip the accreted mass and apply the C NFW structural response when selected.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        initial : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        dict
            Flattened bound mass, scale radius/density and tidal concentration; the NFW inverse
            enforces the enclosed bound mass.

        Notes
        -----
        The factors 2**0.4*mu**0.3/(1+mu)**0.4 and 2**(-0.3)*mu**0.4*(1+mu)**0.3 are the cuspy
        tidal tracks of Penarrubia et al., reproduced in Hiroshima et al., Eqs. (8)-(9), with mu
        the retained bound-mass fraction. The NFW profile is from Navarro et al. The rounded
        factors rmax/rs = 2.163 (2.1626 where explicitly used) and 4.625 are NFW maximum-
        circular-speed identities, as written in Hiroshima et al., Eqs. (5)-(7), rather than
        independent empirical fits.

        The NFW profile is from Navarro et al. The rounded factors rmax/rs = 2.163 (2.1626 where
        explicitly used) and 4.625 are NFW maximum-circular-speed identities, as written in
        Hiroshima et al., Eqs. (5)-(7), rather than independent empirical fits.

        Each accretion slice is flattened in concentration-major, then mass order. Stages retain
        C internal Msun/Mpc/second units and explicit independent weights; the model owns virial
        conversion, NFW structure, tidal response and disruption.

        References
        ----------

        .. [cite-tracks] Penarrubia et al. (2010), The impact of dark matter cusps and cores on the satellite galaxy population around spiral galaxies, Eq. (8), Fig. 6.
            https://arxiv.org/abs/1002.3376

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691

        .. [cite-nfw] Navarro, Frenk & White (1997), A Universal Density Profile from Hierarchical Clustering.
            https://arxiv.org/abs/astro-ph/9611107
        """
        n_mass = context["mvir_acc"].size
        m0 = self.solver.subhalo_mass_stripped(
            context["mvir_acc"],
            context["z_acc"],
            self.redshift,
            method=self.method,
            **self.kwargs,
        )
        m0 = np.broadcast_to(m0, (self.N_herm, n_mass))
        ma = context["mvir_acc"]
        rs_acc = initial["r_s_acc"].reshape(self.N_herm, n_mass)
        rhos_acc = initial["rho_s_acc"].reshape(self.N_herm, n_mass)

        if self.profile_change:
            rmax_acc = rs_acc * 2.163
            Vmax_acc = np.sqrt(rhos_acc * 4.0 * np.pi * self.model.G / 4.625) * rs_acc
            Vmax_z0 = Vmax_acc * (2.0**0.4 * (m0 / ma) ** 0.3 * (1.0 + m0 / ma) ** -0.4)
            rmax_z0 = rmax_acc * (2.0**-0.3 * (m0 / ma) ** 0.4 * (1.0 + m0 / ma) ** 0.3)
            rs_z0 = rmax_z0 / 2.163
            rhos_z0 = (4.625 / (4.0 * np.pi * self.model.G)) * (Vmax_z0 / rs_z0) ** 2
        else:
            rs_z0 = rs_acc
            rhos_z0 = rhos_acc

        enclosed_fraction = m0 / (4.0 * np.pi * rhos_z0 * rs_z0**3)
        c_t = invert_nfw_mass_function(enclosed_fraction)
        return {
            "m_bound": m0.reshape(-1),
            "r_s": rs_z0.reshape(-1),
            "rho_s": rhos_z0.reshape(-1),
            "c_t": c_t.reshape(-1),
        }


@dataclass(frozen=True, slots=True)
class CDMCatalogColumns:
    """Name accreted/evolved columns without modifying units or node order.

    Notes
    -----
    Each accretion slice is flattened in concentration-major, then mass order. Stages retain
    C internal Msun/Mpc/second units and explicit independent weights; the model owns virial
    conversion, NFW structure, tidal response and disruption.
    """

    def build(self, batch, initial, evolved, survival_masks, context):
        """Name aligned accretion and evolved fields without multiplying their statistical weights.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        initial : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        evolved : bool
            Use present bound masses/profiles when true or accretion masses/profiles when false,
            under the observable population selection.
        survival_masks : Mapping[str, numpy.ndarray]
            Named boolean survival decisions aligned with the physical state; kept separate from
            population and concentration weights.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        dict
            Physical columns and the default survival mask in original node order.

        Notes
        -----
        Each accretion slice is flattened in concentration-major, then mass order. Stages retain
        C internal Msun/Mpc/second units and explicit independent weights; the model owns virial
        conversion, NFW structure, tidal response and disruption.
        """
        return {
            "m200_acc": batch.m200_acc,
            "z_acc": batch.z_acc,
            "r_s_acc": initial["r_s_acc"],
            "rho_s_acc": initial["rho_s_acc"],
            "m_bound": evolved["m_bound"],
            "r_s": evolved["r_s"],
            "rho_s": evolved["rho_s"],
            "c_t": evolved["c_t"],
            "survive": survival_masks["default"],
        }


@dataclass(frozen=True, slots=True)
class TruncationThresholdSurvival:
    r"""Apply the historical SASHIMI-C truncation-concentration survival cut.

    The strict ``c_t > ct_threshold`` comparison is the existing SASHIMI-C
    catalog rule. Keeping it in the variant package makes the scientific
    disruption prescription explicit while ITAMAE remains responsible only
    for transporting and validating the resulting survival mask.

    Parameters
    ----------
    ct_threshold : float
        Dimensionless tidal truncation threshold r_t/r_s; the specified strict survival rule
        is applied once through its mask.

    Notes
    -----
    The literature threshold rt/rs > 0.77 follows the Hayashi et al. disruption prescription
    adopted by Hiroshima et al. A caller-supplied threshold takes precedence in variants
    that expose it (C and SI currently default to 0). This is a model selection rule; an
    exact-boundary test does not calibrate disruption physics.

    Each accretion slice is flattened in concentration-major, then mass order. Stages retain
    C internal Msun/Mpc/second units and explicit independent weights; the model owns virial
    conversion, NFW structure, tidal response and disruption.

    References
    ----------

    .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
        https://arxiv.org/abs/astro-ph/0203004

    .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
        https://arxiv.org/abs/1803.07691
    """

    ct_threshold: float

    def select(
        self,
        batch: AccretionBatch,
        initial: PopulationState,
        evolved: PopulationState,
        context: Any,
    ) -> np.ndarray:
        r"""Return the historical boolean survival mask for one batch.

        Parameters
        ----------
        batch : AccretionBatch
            Aligned accretion nodes with separate population, concentration and optional
            host/orbit factors.
        initial : Mapping or array_like
            Per-node physical state for the indicated evolution stage; its fields and units are
            owned by the supplied physical model.
        evolved : bool
            Use present bound masses/profiles when true or accretion masses/profiles when false,
            under the observable population selection.
        context : object
            Variant-owned state for this batch, for example host history, epoch or numerical
            settings; it is not interpreted by the generic executor.

        Returns
        -------
        numpy.ndarray of bool
            True exactly when evolved c_t exceeds ct_threshold.

        Notes
        -----
        The literature threshold rt/rs > 0.77 follows the Hayashi et al. disruption prescription
        adopted by Hiroshima et al. A caller-supplied threshold takes precedence in variants
        that expose it (C and SI currently default to 0). This is a model selection rule; an
        exact-boundary test does not calibrate disruption physics.

        Each accretion slice is flattened in concentration-major, then mass order. Stages retain
        C internal Msun/Mpc/second units and explicit independent weights; the model owns virial
        conversion, NFW structure, tidal response and disruption.

        References
        ----------

        .. [cite-survival] Hayashi et al. (2003), The Structural Evolution of Substructure.
            https://arxiv.org/abs/astro-ph/0203004

        .. [cite-structure] Hiroshima, Ando & Ishiyama (2018), Modeling evolution of dark matter substructure and annihilation boost, Section II, Eqs. (5)-(9).
            https://arxiv.org/abs/1803.07691
        """
        del batch, initial, context
        return evolved["c_t"] > self.ct_threshold


__all__ = [
    "CDMAccretionSlices",
    "CDMCatalogColumns",
    "NFWInitialStructure",
    "TidalProfileEvolution",
    "TruncationThresholdSurvival",
]
