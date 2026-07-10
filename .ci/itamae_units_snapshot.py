"""Pinned ITAMAE unit backends for cross-repository migration CI.

Source commit: 3d1eaa01082c24d951da32a12c01f9904a792565

This temporary fixture mirrors the tested ITAMAE unit interfaces while ITAMAE
is private and unpublished. Astropy is imported lazily so repositories testing
only the native backend do not acquire an unnecessary runtime dependency.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NativeUnits:
    """Interpret numeric inputs in canonical ITAMAE units."""

    identifier: str = "native-v1"

    def to_internal(self, value, physical_type: str):
        """Convert numeric input to a floating NumPy array."""
        del physical_type
        return np.asarray(value, dtype=float)

    def from_internal(self, value, unit=None):
        """Return canonical numeric values without attaching units."""
        del unit
        return np.asarray(value, dtype=float)

    def validate(self, value, physical_type: str) -> None:
        """Validate finite numeric input."""
        del physical_type
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("Unit input contains non-finite values.")


@dataclass(frozen=True, slots=True)
class AstropyUnits:
    """Validate Astropy quantities and convert them to canonical ITAMAE units."""

    identifier: str = "astropy-v1"

    @staticmethod
    def _units():
        import astropy.units as u

        return {
            "dimensionless": u.dimensionless_unscaled,
            "mass": u.Msun,
            "length": u.Mpc,
            "velocity": u.km / u.s,
            "time": u.Gyr,
            "density": u.Msun / u.Mpc**3,
            "cross_section_per_mass": u.cm**2 / u.g,
        }

    def to_internal(self, value, physical_type: str):
        """Convert a Quantity to a floating array in canonical units."""
        import astropy.units as u

        units = self._units()
        if physical_type not in units:
            raise KeyError(f"Unknown physical type: {physical_type}")
        return np.asarray(u.Quantity(value).to_value(units[physical_type]), dtype=float)

    def from_internal(self, value, unit):
        """Attach an Astropy unit to canonical floating values."""
        import astropy.units as u

        return np.asarray(value, dtype=float) * u.Unit(unit)

    def validate(self, value, physical_type: str) -> None:
        """Raise when a value is dimensionally incompatible."""
        self.to_internal(value, physical_type)


__all__ = ["AstropyUnits", "NativeUnits"]
