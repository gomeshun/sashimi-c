"""CDM prescriptions composed through ITAMAE's population executor.

These stages retain SASHIMI-C's historical internal floating units. The adapter
converts the finished catalog at its explicit canonical-unit boundary. Model
objects and stripping solvers are supplied by C, never selected by ITAMAE.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from itamae.halo import invert_nfw_mass_function
from itamae.measure import build_accretion_batch
from itamae.numerics import gauss_hermite_lognormal
from itamae.protocols.execution import PopulationState
from itamae.types import AccretionBatch


@dataclass(frozen=True, slots=True)
class CDMSliceContext(Mapping):
    """C's concentration-major, mass-minor layout for one redshift slice.

    This is a variant-specific context, not a family-wide mass-grid contract.
    The mapping access preserves existing component consumers.
    """

    redshift_index: int
    mvir_acc: np.ndarray
    z_acc: float
    rvir_acc: np.ndarray
    r200_acc: np.ndarray
    concentration_nodes: int

    def __post_init__(self):
        shape = None
        for name in ("mvir_acc", "rvir_acc", "r200_acc"):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if value.ndim != 1 or not value.size or not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ValueError(f"{name} requires a finite positive mass-axis vector.")
            if shape is not None and value.shape != shape:
                raise ValueError("CDM slice context mass-axis arrays must align.")
            shape = value.shape
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.concentration_nodes < 1:
            raise ValueError("concentration_nodes must be positive.")

    def __iter__(self):
        return iter(("redshift_index", "mvir_acc", "z_acc", "rvir_acc", "r200_acc"))

    def __len__(self):
        return 5

    def __getitem__(self, key):
        if key not in tuple(self):
            raise KeyError(key)
        return getattr(self, key)

    def validate_batch(self, batch, concentration_nodes):
        if concentration_nodes != self.concentration_nodes:
            raise ValueError("Component concentration layout differs from its prepared slice.")
        expected = self.concentration_nodes * self.mvir_acc.size
        if batch.m200_acc.shape != (expected,):
            raise ValueError("CDM batch does not match its prepared population layout.")
        if not np.all(batch.z_acc == self.z_acc):
            raise ValueError("CDM batch and context accretion redshifts differ.")
        if not np.array_equal(batch.mvir_acc, np.tile(self.mvir_acc, self.concentration_nodes)):
            raise ValueError("CDM batch and context virial-mass layouts differ.")


@dataclass(frozen=True, slots=True)
class CDMAccretionSlices:
    """Prepare C-owned virial conversion and concentration quadrature slices."""

    model: Any
    ma200_grid: np.ndarray
    zdist: np.ndarray
    population_2d: np.ndarray
    sigmalogc: float
    N_herm: int

    def build(self, index: int):
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
        context = CDMSliceContext(
            redshift_index=index, mvir_acc=ma, z_acc=float(z_acc_value),
            rvir_acc=rvirsub, r200_acc=r200sub, concentration_nodes=self.N_herm,
        )
        context.validate_batch(batch, self.N_herm)
        return batch, context


@dataclass(frozen=True, slots=True)
class NFWInitialStructure:
    """Create the unstripped NFW structure at each accretion node."""

    model: Any
    N_herm: int

    def initialize(self, batch, context):
        if isinstance(context, CDMSliceContext):
            context.validate_batch(batch, self.N_herm)
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
    """Apply the supplied C mass-loss solver and C profile-response formulae."""

    model: Any
    solver: Any
    redshift: float
    method: str
    N_herm: int
    profile_change: bool
    kwargs: dict[str, Any]

    def evolve(self, batch, initial, context):
        if isinstance(context, CDMSliceContext):
            context.validate_batch(batch, self.N_herm)
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
    """Name accreted/evolved columns without modifying units or node order."""

    def build(self, batch, initial, evolved, survival_masks, context):
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
    """Apply the historical SASHIMI-C truncation-concentration survival cut.

    The strict ``c_t > ct_threshold`` comparison is the existing SASHIMI-C
    catalog rule. Keeping it in the variant package makes the scientific
    disruption prescription explicit while ITAMAE remains responsible only
    for transporting and validating the resulting survival mask.
    """

    ct_threshold: float

    def select(
        self,
        batch: AccretionBatch,
        initial: PopulationState,
        evolved: PopulationState,
        context: Any,
    ) -> np.ndarray:
        """Return the historical boolean survival mask for one batch."""
        del batch, initial, context
        return evolved["c_t"] > self.ct_threshold


__all__ = [
    "CDMAccretionSlices",
    "CDMCatalogColumns",
    "CDMSliceContext",
    "NFWInitialStructure",
    "TidalProfileEvolution",
    "TruncationThresholdSurvival",
]
