"""Explicit C-owned concentration relations with immutable, bounded inputs."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ConcentrationRelation(Protocol):
    """Minimal C relation contract, not a family-wide plugin registry.

    Evaluate median c200 (dimensionless) for M200c in Msun and redshift z.
    Implementations describe their exact data, domain and interpolation meaning.
    The public CDM specification currently accepts only TabulatedConcentration.
    """

    def evaluate(self, mass_msun, redshift, *, component: str = "concentration"): ...

    def describe(self) -> dict: ...


@dataclass(frozen=True, slots=True)
class TabulatedConcentration:
    """An immutable experimental median-c200 relation supplied by the caller.

    ``c200`` has shape ``(len(redshift), len(mass_msun))``. Both axes must
    contain at least two strictly increasing finite points. Mass and c200 must
    be positive; redshifts are nonnegative. Interpolation is bilinear in natural
    log(c200), log10(M200c/Msun), and linear redshift z. No extrapolation or
    clipping is performed. The domain must cover host and subhalo queries.

    Supplying a table does not recalibrate CDM host history, variance, tidal
    prescriptions, or validate the table as a physical model.
    """

    mass_msun: tuple[float, ...]
    redshift: tuple[float, ...]
    c200: tuple[tuple[float, ...], ...]
    _interpolator: RegularGridInterpolator = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self):
        arrays = []
        for name in ("mass_msun", "redshift", "c200"):
            raw = np.asarray(getattr(self, name))
            if raw.dtype.kind not in "iuf":
                raise TypeError(f"{name} must contain real numeric values.")
            value = np.asarray(raw, dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite.")
            arrays.append(value)
        mass, z, concentration = arrays
        for name, axis in (("mass_msun", mass), ("redshift", z)):
            if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0):
                raise ValueError(
                    f"{name} must be a strictly increasing axis with at least two nodes."
                )
        if np.any(mass <= 0) or np.any(z < 0) or np.any(concentration <= 0):
            raise ValueError(
                "mass_msun and c200 must be positive; redshift must be nonnegative."
            )
        if concentration.shape != (z.size, mass.size):
            raise ValueError("c200 must have shape (redshift nodes, mass nodes).")
        object.__setattr__(self, "mass_msun", tuple(map(float, mass)))
        object.__setattr__(self, "redshift", tuple(map(float, z)))
        object.__setattr__(
            self, "c200", tuple(tuple(map(float, row)) for row in concentration)
        )
        object.__setattr__(
            self,
            "_interpolator",
            RegularGridInterpolator(
                (np.asarray(self.redshift), np.log10(self.mass_msun)),
                np.log(self.c200),
                bounds_error=True,
            ),
        )

    def _description(self):
        return {
            "prescription": "user-supplied-table",
            "mass_definition": "200c",
            "mass_unit": "Msun",
            "concentration_unit": "dimensionless",
            "axes_order": ["redshift", "mass_msun"],
            "interpolation": "bilinear-ln(c200)-in-linear-z-and-log10(M200c/Msun)",
            "extrapolation": "forbidden",
            "mass_msun": list(self.mass_msun),
            "redshift": list(self.redshift),
            "c200": [list(row) for row in self.c200],
            "mass_domain_msun": [self.mass_msun[0], self.mass_msun[-1]],
            "redshift_domain": [self.redshift[0], self.redshift[-1]],
        }

    @property
    def identifier(self):
        encoded = json.dumps(
            self._description(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return (
            "sashimi-c:concentration-table:sha256:"
            + hashlib.sha256(encoded).hexdigest()
        )

    def describe(self):
        return {
            **self._description(),
            "identifier": self.identifier,
            "experimental": True,
            "calibration": "user-supplied; not externally validated",
        }

    def evaluate(self, mass_msun, redshift, *, component="concentration"):
        raw_m, raw_z = np.asarray(mass_msun), np.asarray(redshift)
        if raw_m.dtype.kind not in "iuf" or raw_z.dtype.kind not in "iuf":
            raise TypeError(
                f"{component}: concentration coordinates must be real numeric values."
            )
        mass, z = np.broadcast_arrays(
            np.asarray(raw_m, dtype=float), np.asarray(raw_z, dtype=float)
        )
        valid = (
            np.isfinite(mass)
            & np.isfinite(z)
            & (mass >= self.mass_msun[0])
            & (mass <= self.mass_msun[-1])
            & (z >= self.redshift[0])
            & (z <= self.redshift[-1])
        )
        if not np.all(valid):
            index = np.flatnonzero(~valid)[0]
            raise ValueError(
                f"{component}: concentration table does not cover M200c={mass.flat[index]} Msun, z={z.flat[index]}; "
                f"required domain is mass {self.mass_msun[0], self.mass_msun[-1]} Msun, z {self.redshift[0], self.redshift[-1]}."
            )
        points = np.column_stack((z.ravel(), np.log10(mass).ravel()))
        return np.exp(self._interpolator(points)).reshape(mass.shape)
