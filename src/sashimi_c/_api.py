"""Immutable CDM specifications and explicit per-population preparation."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from ._settings import finite_scalar, freeze, resolve, thaw


@dataclass(frozen=True, slots=True)
class _CDMPreparation:
    """C-owned resolved quadrature and provenance for one execution."""

    redshift_nodes: np.ndarray
    interpolation_nodes: int
    metadata: Mapping[str, Any]


def _range(value, name, *, allow_none_upper=False):
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a pair of bounds.")
    try:
        lower, upper = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain exactly two bounds.") from error
    lower = finite_scalar(lower, f"{name}[0]", strict=allow_none_upper)
    if upper is None and allow_none_upper:
        return lower, None
    upper = finite_scalar(upper, f"{name}[1]", strict=allow_none_upper)
    if upper <= lower:
        raise ValueError(
            f"{name} requires an upper bound greater than its lower bound."
        )
    return lower, upper


def _host_mass_at_zero(physics, mass, epoch):
    """Apply the historical inversion once, without unbounded extrapolation."""
    if epoch == 0.0:
        return mass
    candidates = np.logspace(0.0, 5.0, 1500) * mass
    at_epoch = physics.Mzi(candidates, epoch)
    if not np.all(np.isfinite(at_epoch)) or not np.all(np.diff(at_epoch) > 0):
        raise ValueError(
            "Host-history inversion is not finite and monotonic in its supported bracket."
        )
    if not at_epoch[0] <= mass <= at_epoch[-1]:
        raise ValueError(
            "Host mass lies outside the supported host-history inversion bracket."
        )
    # Same interpolation as M0_at_redshift, but explicitly bounded for the native API.
    return float(interp1d(at_epoch, candidates, bounds_error=True)(mass))


@dataclass(frozen=True, slots=True)
class CDM:
    """A reusable immutable specification of the calibrated SASHIMI-C model.

    ``configure`` returns a new specification. ``population`` creates a fresh
    physics/solver context and returns an ITAMAE ``WeightedSubhaloCatalog``.
    Physical prescriptions and their accepted settings are owned by SASHIMI-C.
    """

    _settings: Mapping[str, Any] = field(
        default_factory=resolve, init=False, repr=False
    )

    @property
    def resolved_settings(self) -> Mapping[str, Any]:
        """Read-only, recursively immutable physical and numerical settings."""
        result = thaw(self._settings)
        relation = self._settings["concentration"]["relation"]
        if relation is not None:
            result["concentration"]["relation"] = relation.describe()
            result["concentration"]["prescription"] = "user-supplied-table"
        return freeze(result)

    def configure(
        self, *, accretion=None, concentration=None, stripping=None, disruption=None
    ):
        """Return a new CDM with validated partial component overrides.

        ``solver_options={}`` clears existing solver options; a nonempty
        mapping partially updates them. No caller-owned mutable object is kept.
        """
        configured = CDM()
        object.__setattr__(
            configured,
            "_settings",
            resolve(
                self._settings,
                accretion=accretion,
                concentration=concentration,
                stripping=stripping,
                disruption=disruption,
            ),
        )
        return configured

    def population(
        self,
        *,
        host_mass_msun,
        host_mass_definition="200c",
        host_mass_redshift=0.0,
        redshift=0.0,
        accretion_mass_range_msun=(1e-6, None),
        accretion_mass_definition="200c",
        accretion_redshift_range=None,
    ):
        """Calculate a named, weighted catalog with explicit physical inputs.

        Masses are in Msun and both mass definitions must be ``'200c'``.
        The host mass is specified at ``host_mass_redshift`` independently of
        the target ``redshift``. An omitted upper mass bound is 0.1 times the
        resolved host mass at zero. Explicit redshift support is ``(lo, hi]``;
        quadrature nodes never exceed it. Omitted support retains the historical
        ``arange(redshift + step, 7 + step, step)`` convention, including any
        off-grid final node. At least two redshift and mass nodes are required.
        """
        if host_mass_definition != "200c" or accretion_mass_definition != "200c":
            raise ValueError(
                "The CDM native API currently supports only '200c' masses."
            )
        mass = finite_scalar(host_mass_msun, "host_mass_msun", strict=True)
        epoch = finite_scalar(host_mass_redshift, "host_mass_redshift")
        target = finite_scalar(redshift, "redshift")
        mass_lo, mass_hi = _range(
            accretion_mass_range_msun,
            "accretion_mass_range_msun",
            allow_none_upper=True,
        )
        explicit_range = accretion_redshift_range is not None
        z_lo, z_hi = _range(
            accretion_redshift_range if explicit_range else (target, 7.0),
            "accretion_redshift_range",
        )
        if z_lo < target:
            raise ValueError(
                "Accretion redshift lower bound must be at least the target redshift."
            )
        a, c, s, d = (
            self._settings[key]
            for key in ("accretion", "concentration", "stripping", "disruption")
        )
        step = a["redshift_step"]
        nodes = np.arange(z_lo + step, z_hi + step, step)
        if explicit_range:
            # Correct floating-point endpoint drift only; never extend support.
            tolerance = 8 * np.finfo(float).eps * max(1.0, abs(z_hi))
            nodes = nodes[nodes <= z_hi + tolerance]
            nodes = np.minimum(nodes, z_hi)
        if nodes.size < 2:
            raise ValueError(
                "Accretion support and redshift_step must yield at least two redshift nodes."
            )
        from . import SubhaloProperties

        physics = SubhaloProperties(concentration_relation=c["relation"])
        mass_zero = _host_mass_at_zero(physics, mass, epoch)
        if mass_hi is None:
            mass_hi = 0.1 * mass_zero
        if mass_lo >= mass_hi:
            raise ValueError(
                "The resolved accretion mass upper bound must exceed its lower bound."
            )
        if mass_lo >= 0.5 * mass_zero:
            raise ValueError(
                "The accretion mass lower bound must be below half the resolved "
                "host mass at z=0 to contain supported accretion masses."
            )
        settings = thaw(self._settings)
        relation = c["relation"]
        if relation is not None:
            # Validate both initial-grid queries and host queries at interpolation
            # nodes before creating a stripping solver or computing accretion weights.
            relation.evaluate(
                np.array([mass_lo, mass_hi]),
                np.array([[nodes[0]], [nodes[-1]]]),
                component="subhalo preparation",
            )
            host_z = np.linspace(
                target, max(z_hi, float(nodes[-1])), s["interpolation_nodes"]
            )
            relation.evaluate(
                physics.Mzzi(mass_zero, host_z, 0.0),
                host_z,
                component="host stripping preparation",
            )
            settings["concentration"]["relation"] = relation.describe()
            settings["concentration"]["prescription"] = "user-supplied-table"

        metadata = {
            "native_api": "sashimi-c:cdm-configure:v1",
            "resolved_settings": settings,
            "host_mass_msun": mass,
            "host_mass_input": mass,
            "host_mass_input_at_target_redshift": epoch == target and epoch != 0.0,
            "host_mass_definition": host_mass_definition,
            "host_mass_redshift": epoch,
            "host_mass_z0": mass_zero,
            "host_mass_inversion": {
                "method": "identity" if epoch == 0 else "linear-interpolation",
                "nodes": 1500,
                "log10_mass_ratio_bracket": [0.0, 5.0],
            },
            "accretion_mass_definition": accretion_mass_definition,
            "accretion_mass_range_msun": [mass_lo, mass_hi],
            "accretion_redshift_range": [z_lo, z_hi],
            "accretion_redshift_range_policy": "explicit-bounded"
            if explicit_range
            else "legacy-default-grid",
            "accretion_redshift_nodes": nodes.tolist(),
            "accretion_sampled_redshift_range": [float(nodes[0]), float(nodes[-1])],
            "accretion_auxiliary_redshift_nodes": 1000,
            "odeint_output_nodes": 100,
            "odeint_tolerance_base_policy": "scipy-odeint-default"
            if s["solver"] == "odeint"
            else None,
        }
        nodes.setflags(write=False)
        prepared = _CDMPreparation(nodes, s["interpolation_nodes"], metadata)
        return physics._calculate_population(
            mass_zero,
            redshift=target,
            dz=step,
            zmax=z_hi,
            N_ma=a["mass_nodes"],
            sigmalogc=c["scatter_dex"],
            N_herm=c["quadrature_nodes"],
            logmamin=np.log10(mass_lo),
            logmamax=np.log10(mass_hi),
            N_hermNa=a["host_history_nodes"],
            Na_model=a["model"],
            ct_th=d["ct_threshold"],
            profile_change=s["profile_change"],
            M0_at_redshift=False,
            method=s["solver"],
            preparation=prepared,
            **dict(s["solver_options"]),
        )
