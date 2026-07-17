"""Variance adapter for the SASHIMI-C ITAMAE migration.

The analytic CDM fit remains implemented and calibrated in ``sashimi_c.py``.
This module only exposes it through ITAMAE's common variance interface so EPS
and catalog machinery no longer need to know the concrete implementation.
"""

from __future__ import annotations

from typing import Any

from itamae.variance import CallableVarianceModel
from sashimi_c_itamae_migration import ItamaeHaloModel


def make_variance_model(model: Any | None = None) -> CallableVarianceModel:
    """Wrap the SASHIMI-C analytic variance implementation.

    Parameters
    ----------
    model : object, optional
        SASHIMI-C halo model. A migrated ITAMAE-backed halo model is constructed
        when omitted.

    Returns
    -------
    itamae.variance.CallableVarianceModel
        Variance model returning ``sigmaMz`` and ``dsdm`` from SASHIMI-C.
    """

    model = model or ItamaeHaloModel()
    return CallableVarianceModel(
        identifier="sashimi-c:analytic-cdm-fit:v1",
        sigma_function=lambda mass, z: model.sigmaMz(mass, z),
        derivative_function=lambda mass, z: model.dsdm(mass, z),
    )


__all__ = ["make_variance_model"]
