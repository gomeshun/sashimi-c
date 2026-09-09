"""Catalog-focused imports for the SASHIMI-C ITAMAE migration.

Backend bootstrap happens in :mod:`sashimi_c_itamae_migration` before every
legacy constructor. This variant-scoped module provides side-effect-free
catalog imports without introducing generic top-level names that collide with
other SASHIMI distributions.
"""

from sashimi_c_itamae_migration import (
    ItamaeSubhaloProperties,
    ItamaeTidalStrippingSolver,
)

__all__ = ["ItamaeSubhaloProperties", "ItamaeTidalStrippingSolver"]
