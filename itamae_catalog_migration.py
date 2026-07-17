"""Compatibility imports for the original catalog-migration module.

Backend bootstrap now happens in :mod:`itamae_migration` before every legacy
constructor. Keeping this module as a side-effect-free re-export preserves the
existing migration import path without mutating module globals.
"""

from itamae_migration import ItamaeSubhaloProperties, ItamaeTidalStrippingSolver

__all__ = ["ItamaeSubhaloProperties", "ItamaeTidalStrippingSolver"]
