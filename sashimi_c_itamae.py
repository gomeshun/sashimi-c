"""Opt-in ITAMAE-backed public API for SASHIMI-C.

Importing this module leaves :mod:`sashimi_c` unchanged. The lower-case aliases
mirror the established SASHIMI-C class names and method signatures while using
ITAMAE for migrated numerical mechanisms and structured catalog output.
"""

from itamae_migration import (
    ItamaeHaloModel,
    ItamaeSubhaloObservables,
    ItamaeSubhaloProperties,
    ItamaeTidalStrippingSolver,
    StrippingDiagnostics,
    diagnose_stripping_approximation,
)

halo_model = ItamaeHaloModel
TidalStrippingSolver = ItamaeTidalStrippingSolver
subhalo_properties = ItamaeSubhaloProperties
subhalo_observables = ItamaeSubhaloObservables

__all__ = [
    "TidalStrippingSolver",
    "StrippingDiagnostics",
    "diagnose_stripping_approximation",
    "halo_model",
    "subhalo_observables",
    "subhalo_properties",
]
