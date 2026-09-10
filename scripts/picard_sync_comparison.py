"""Generate durable SYNC-C comparison diagnostics for the Picard solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sashimi_c import subhalo_properties
from sashimi_c_itamae_migration import ItamaeSubhaloObservables, ItamaeSubhaloProperties

HISTORICAL_REFERENCE = "9f6713b686805645da459e99522e2049e7dea793"
PICARD_MAIN_REFERENCE = "e09571be7a1daaef343e97887e34449faf21db7b"

PARAMETERS = {
    "M0": 1.0e10,
    "redshift": 0.0,
    "dz": 0.5,
    "zmax": 1.0,
    "N_ma": 6,
    "sigmalogc": 0.128,
    "N_herm": 3,
    "logmamin": 5.0,
    "logmamax": 7.0,
    "N_hermNa": 3,
    "Na_model": 3,
    "ct_th": 0.0,
    "profile_change": True,
}


def max_relative(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(np.max(np.abs(left / right - 1.0)))


def build_report():
    public = subhalo_properties().subhalo_properties_calc(
        **PARAMETERS, method="pert2_shanks"
    )
    migrated_shanks_model = ItamaeSubhaloProperties()
    migrated_picard_model = ItamaeSubhaloProperties()
    shanks = migrated_shanks_model.subhalo_catalog_calc(
        **PARAMETERS, method="pert2_shanks"
    )
    picard = migrated_picard_model.subhalo_catalog_calc(
        **PARAMETERS, method="picard_table"
    )

    public_m_bound = public[4]
    shanks_m_bound = shanks.columns["m_bound"]
    picard_m_bound = picard.columns["m_bound"]

    observable_parameters = {
        ("M0_per_Msun" if key == "M0" else key): value
        for key, value in PARAMETERS.items()
    }
    shanks_obs = ItamaeSubhaloObservables(
        method="pert2_shanks", **observable_parameters
    )
    picard_obs = ItamaeSubhaloObservables(
        method="picard_table", **observable_parameters
    )

    report = {
        "schema": "sashimi-c:solver-comparison:v2",
        "historical_reference_not_executed_here": {
            "repository_revision": HISTORICAL_REFERENCE,
            "physics_mode": "legacy",
            "stripping_method": "pert2_shanks",
            "ct_threshold": 0.0,
        },
        "picard_source": {
            "main_revision": PICARD_MAIN_REFERENCE,
            "default_adopted": False,
            "selection": "explicit method=picard_table only",
        },
        "parameters": PARAMETERS,
        "product_metadata": dict(shanks.metadata),
        "picard_table_settings": dict(picard.metadata["picard_table_settings"]),
        "comparisons": {
            "tuple_vs_named_shanks": {
                "max_relative_m_bound_difference": max_relative(
                    shanks_m_bound, public_m_bound
                ),
            },
            "picard_vs_shanks": {
                "max_relative_m_bound_difference": max_relative(
                    picard_m_bound, shanks_m_bound
                ),
                "mass_fraction_evolved_shanks": float(
                    shanks_obs.mass_fraction(evolved=True)
                ),
                "mass_fraction_evolved_picard": float(
                    picard_obs.mass_fraction(evolved=True)
                ),
                "relative_mass_fraction_evolved_difference": abs(
                    float(picard_obs.mass_fraction(evolved=True))
                    / float(shanks_obs.mass_fraction(evolved=True))
                    - 1.0
                ),
                "mass_fraction_unevolved_shanks": float(
                    shanks_obs.mass_fraction(evolved=False)
                ),
                "mass_fraction_unevolved_picard": float(
                    picard_obs.mass_fraction(evolved=False)
                ),
            },
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
