"""Transitional import aliases for the standard ITAMAE-backed SASHIMI-C API."""

from ._itamae_migration import (
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
    "StrippingDiagnostics",
    "TidalStrippingSolver",
    "diagnose_stripping_approximation",
    "halo_model",
    "subhalo_observables",
    "subhalo_properties",
]
