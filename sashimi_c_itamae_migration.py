"""Transitional import aliases for the standard SASHIMI-C implementation.

No alternate calculation is selected by importing this module. physics_mode
and dynamic migration of old classes have been removed.
"""

from sashimi_c import (
    HaloModel as ItamaeHaloModel,
)
from sashimi_c import (
    StrippingDiagnostics,
    diagnose_stripping_approximation,
)
from sashimi_c import (
    SubhaloObservables as ItamaeSubhaloObservables,
)
from sashimi_c import (
    SubhaloProperties as ItamaeSubhaloProperties,
)
from sashimi_c import (
    TidalStrippingSolver as ItamaeTidalStrippingSolver,
)

__all__ = [
    "ItamaeHaloModel",
    "ItamaeSubhaloObservables",
    "ItamaeSubhaloProperties",
    "ItamaeTidalStrippingSolver",
    "StrippingDiagnostics",
    "diagnose_stripping_approximation",
]
