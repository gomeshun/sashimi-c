"""Variant-owned execution components for the SASHIMI-C ITAMAE adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from itamae.protocols.execution import PopulationState
from itamae.types import AccretionBatch


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


__all__ = ["TruncationThresholdSurvival"]
