"""Validated, explicitly located tables for the CDM annihilation observables."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

from sashimi_c_data import data_directory, require_input

MODEL_KEYS = (
    "cosmology_backend",
    "variance_identifier",
    "profile_change",
    "stripping_method",
    "ct_threshold",
    "log10_mass_min",
    "sigma_log10_concentration",
    "accretion_model",
)


def validate_order(order):
    """Reject invalid iteration counts before accessing tables or computing cusps."""
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("Boost order n must be a nonnegative integer.")


def model_contract(model, **extra):
    """Select the physical assumptions that must agree with the supplied tables."""
    contract = {key: model.catalog.metadata[key] for key in MODEL_KEYS}
    contract["prompt_cusps"] = bool(model.prompt_cusps)
    if model.prompt_cusps or extra:
        contract.update(
            k_fs_Mpc=float(model.k_fs * model.Mpc),
            filter=model.filter,
            alpha=float(model.alpha),
        )
        contract["input_sha256"] = {
            name: hashlib.sha256(
                require_input(model.data_dir / name).read_bytes()
            ).hexdigest()
            for name in ("Planck2018_CAMB_extrap.dat", "powerspectrum31.txt")
        }
    return {**contract, **extra}


def interpolate_boost_tables(
    model, directory, fields, *, extra_contract=None, allow_incomplete_tables=False
):
    """Preserve historical positive-table interpolation and reject undefined points.

    ``fields`` maps each output to its filename and the historical additive
    offset before log10. The historical zero-fill requires explicit opt-in and
    exposes its invalid points; it is not a calibrated boundary model.
    """
    if not isinstance(allow_incomplete_tables, (bool, np.bool_)):
        raise TypeError("allow_incomplete_tables must be an explicit boolean.")
    directory = directory.resolve()
    model.last_boost_provenance = {"directory": str(directory), "valid": False}
    manifest_path = require_input(directory / "manifest.json")
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if not isinstance(manifest, dict):
        raise TypeError("Boost manifest must be an object.")
    expected = {
        "schema_version": 1,
        "variant": "sashimi-c",
        "mass_unit": "Msun",
        "calculation_specification": model.catalog.metadata[
            "calculation_specification"
        ],
        "model_contract": model_contract(model, **(extra_contract or {})),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Boost manifest {key} does not match the requested model."
            )
    for key in ("source_revision", "itamae_source_revision"):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(
            "[0-9a-f]{40}", manifest[key]
        ):
            raise ValueError(f"Boost manifest {key} must record a full source SHA.")
    if not isinstance(manifest.get("generation_parameters"), dict):
        raise TypeError("Boost manifest must record generation_parameters.")
    hashes = manifest.get("files")
    if not isinstance(hashes, dict):
        raise TypeError("Boost manifest must record file hashes.")
    invalid_cells = manifest.get("generation_diagnostics", {}).get(
        "invalid_cell_count", 0
    )
    if invalid_cells and not allow_incomplete_tables:
        raise ValueError(
            "Boost input contains invalid generation cells; explicit allow_incomplete_tables=True is required for the historical approximation."
        )

    def read(filename):
        raw = require_input(directory / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != hashes.get(filename):
            raise ValueError(f"Boost table hash mismatch: {filename}.")
        values = np.loadtxt(io.BytesIO(raw))
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Boost table must contain finite values: {filename}.")
        return values

    redshift = read("za.txt")
    mass = read("ma.txt")
    if (
        redshift.ndim != 1
        or redshift.size < 2
        or np.any(redshift < 0)
        or np.any(np.diff(redshift) <= 0)
    ):
        raise ValueError(
            "Boost redshift grid must be nonnegative and strictly increasing."
        )
    if (
        mass.ndim != 2
        or mass.shape[0] != redshift.size
        or mass.shape[1] < 2
        or np.any(mass <= 0)
        or np.any(np.diff(mass, axis=1) <= 0)
    ):
        raise ValueError("Boost mass grid must have aligned, positive increasing rows.")
    coordinates = (
        np.log10(mass.flatten()),
        (redshift[:, None] * np.ones_like(mass[0])).flatten(),
    )
    queries = (np.log10(model.ma200), model.z_a)
    result = {}
    invalid_union = np.zeros(np.shape(model.ma200), dtype=bool)
    field_diagnostics = {}
    for name, (filename, offset) in fields.items():
        values = read(filename)
        if values.shape != mass.shape or np.any(values < 0):
            raise ValueError(
                f"Boost field {filename} must be nonnegative and aligned with mass."
            )
        if np.any(values + offset <= 0) and not allow_incomplete_tables:
            raise ValueError(
                f"Boost log-interpolation requires positive values in {filename}."
            )
        log_values = np.full(values.shape, -np.inf)
        np.log10(values + offset, out=log_values, where=values + offset > 0)
        interpolated = griddata(
            coordinates, log_values.flatten(), queries, method="linear"
        )
        invalid = ~np.isfinite(interpolated)
        if np.any(invalid) and not allow_incomplete_tables:
            raise ValueError(
                f"{int(np.count_nonzero(invalid))} boost queries lie outside the valid table domain "
                f"for {filename}. Expand/recompute the table; no zero substitution is applied."
            )
        invalid_union |= invalid
        field_diagnostics[name] = {
            "invalid_query_count": int(np.count_nonzero(invalid)),
            "zero_table_cells": int(np.count_nonzero(values == 0)),
        }
        interpolated[invalid] = -np.inf
        result[name] = 10.0**interpolated
    invalid_count = int(np.count_nonzero(invalid_union))
    model.last_boost_provenance = {
        "directory": str(directory),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "manifest": manifest,
        "fields": {name: filename for name, (filename, _) in fields.items()},
        "interpolation": "linear-griddata-in-log10-mass-and-log10-field",
        "valid": invalid_count == 0 and invalid_cells == 0,
        "allow_incomplete_tables": bool(allow_incomplete_tables),
        "approximation": "historical-undefined-log-interpolation-to-zero"
        if allow_incomplete_tables
        else None,
        "invalid_query_count": invalid_count,
        "query_count": int(invalid_union.size),
        "invalid_query_mask_sha256": hashlib.sha256(
            invalid_union.astype(np.uint8).tobytes()
        ).hexdigest(),
        "field_diagnostics": field_diagnostics,
    }
    if invalid_count or invalid_cells:
        warnings.warn(
            f"Using the explicitly requested historical boost approximation: "
            f"{invalid_count}/{invalid_union.size} query points had undefined interpolation "
            f"and were set to zero; the input records {invalid_cells} invalid generation cells. "
            "Inspect last_boost_provenance. This is not a calibrated low-mass boundary.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def generate_boost_tables(configuration, *, data_dir=None, prompt_cusps=False):
    """Generate tables from an explicit grid without overwriting existing results.

    Host masses are M200 at each table redshift in Msun. This uses the existing
    observable formulas and requires every requested population to be nonempty
    and every previous-order interpolation to be defined. Unsupported low-mass
    boundaries fail; they are not replaced with a new physical prescription.
    """
    from sashimi_c import SubhaloObservables

    order = configuration.get("max_order", 0)
    validate_order(order)
    allow_incomplete = configuration.get("allow_incomplete_tables", False)
    if not isinstance(allow_incomplete, bool):
        raise TypeError("allow_incomplete_tables must be an explicit boolean.")
    redshifts = np.asarray(configuration["redshift_grid"])
    masses = np.asarray(configuration["mass_grid_Msun"])
    if redshifts.dtype.kind not in "iuf" or masses.dtype.kind not in "iuf":
        raise ValueError("Boost grid coordinates must be real numeric values.")
    redshifts, masses = redshifts.astype(float), masses.astype(float)
    if (
        redshifts.ndim != 1
        or redshifts.size < 2
        or not np.all(np.isfinite(redshifts))
        or np.any(redshifts < 0)
        or np.any(np.diff(redshifts) <= 0)
    ):
        raise ValueError("redshift_grid must be finite, nonnegative and increasing.")
    if (
        masses.ndim != 2
        or masses.shape[0] != redshifts.size
        or masses.shape[1] < 2
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0)
        or np.any(np.diff(masses, axis=1) <= 0)
    ):
        raise ValueError(
            "mass_grid_Msun must contain aligned positive increasing rows."
        )
    parameters = dict(configuration["calculation_parameters"])
    reserved = {"M0_per_Msun", "redshift", "M0_at_redshift", "data_dir", "prompt_cusps"}
    if reserved.intersection(parameters):
        raise ValueError(
            f"Calculation parameters must not override {sorted(reserved)}."
        )
    root = data_directory(data_dir)
    subdirectory = Path("prompt_cusps/boost" if prompt_cusps else "boost")
    target = root / subdirectory
    if target.exists():
        raise FileExistsError(
            f"Boost output already exists: {target}. Choose a new data directory."
        )
    root.mkdir(parents=True, exist_ok=True)
    survival = {
        key: float(configuration.get(key, 1.0)) for key in ("f_surv", "f_surv_stripped")
    }
    with tempfile.TemporaryDirectory(prefix="boost-build-", dir=root) as temporary:
        staging_root = Path(temporary)
        staging = staging_root / subdirectory
        staging.mkdir(parents=True)
        # The prompt-cusp spectra remain in the configured input root. Only
        # generated previous-order tables are read from the staging directory.
        arrays = {"za.txt": redshifts, "ma.txt": masses}
        manifest = None
        incomplete_lookups = []
        degenerate_cells = []
        for current in range(order + 1):
            fields = {name: np.empty_like(masses) for name in ("fsh", "Bsh")}
            if prompt_cusps:
                fields.update(
                    {
                        name: np.empty_like(masses)
                        for name in (
                            "Bcusp_dressed",
                            "Bcusp_naked",
                            "Ncusp_dressed",
                            "Ncusp_naked",
                        )
                    }
                )
            for i, redshift in enumerate(redshifts):
                for j, mass in enumerate(masses[i]):
                    if (
                        allow_incomplete
                        and not prompt_cusps
                        and redshift == 0.0
                        and parameters.get("logmamax") is None
                        and np.log10(0.1 * mass)
                        == float(parameters.get("logmamin", -6.0))
                    ):
                        # Frozen A gives exactly fsh=Bsh=0 for this zero-width
                        # grid. Reproduce that known old numerical result only
                        # under the same explicit approximation opt-in.
                        for values in fields.values():
                            values[i, j] = 0.0
                        degenerate_cells.append(
                            {
                                "order": current,
                                "redshift_index": i,
                                "mass_index": j,
                                "reason": "historical-zero-width-log-mass-grid",
                            }
                        )
                        continue
                    model = SubhaloObservables(
                        mass,
                        redshift=float(redshift),
                        M0_at_redshift=True,
                        data_dir=root if prompt_cusps else staging_root,
                        prompt_cusps=prompt_cusps,
                        **parameters,
                    )
                    # Separate scientific inputs from staged generated tables.
                    model._boost_table_directory = staging
                    contract = model_contract(
                        model, **(survival if prompt_cusps else {})
                    )
                    if manifest is None:
                        manifest = {
                            "schema_version": 1,
                            "variant": "sashimi-c",
                            "mass_unit": "Msun",
                            "calculation_specification": model.catalog.metadata[
                                "calculation_specification"
                            ],
                            "source_revision": model.catalog.metadata[
                                "sashimi_source_revision"
                            ],
                            "itamae_source_revision": model.catalog.metadata[
                                "itamae_source_revision"
                            ],
                            "model_contract": contract,
                            "generation_parameters": configuration,
                            "numerical_example_metadata": dict(model.catalog.metadata),
                            "generator_sha256": hashlib.sha256(
                                Path(__file__).read_bytes()
                            ).hexdigest(),
                            "files": {},
                        }
                    elif manifest["model_contract"] != contract:
                        raise ValueError("The table grid changed the model contract.")
                    fields["fsh"][i, j] = model.mass_fraction()
                    if prompt_cusps:
                        values = model.annihilation_boost_factor_prompt_cusps(
                            n=current,
                            allow_incomplete_tables=allow_incomplete,
                            **survival,
                        )
                        for name, value in zip(
                            (
                                "Bsh",
                                "Bcusp_dressed",
                                "Bcusp_naked",
                                "luminosity_ratio",
                                "Ncusp_dressed",
                                "Ncusp_naked",
                            ),
                            values,
                            strict=True,
                        ):
                            if name in fields:
                                fields[name][i, j] = value
                    else:
                        fields["Bsh"][i, j] = model.annihilation_boost_factor(
                            n=current, allow_incomplete_tables=allow_incomplete
                        )[0]
                    if current > 0 and not model.last_boost_provenance["valid"]:
                        incomplete_lookups.append(
                            {
                                "order": current,
                                "redshift_index": i,
                                "mass_index": j,
                                "invalid_query_count": model.last_boost_provenance[
                                    "invalid_query_count"
                                ],
                                "query_count": model.last_boost_provenance[
                                    "query_count"
                                ],
                                "input_manifest_sha256": model.last_boost_provenance[
                                    "manifest_sha256"
                                ],
                                "invalid_query_mask_sha256": model.last_boost_provenance[
                                    "invalid_query_mask_sha256"
                                ],
                            }
                        )
            for name, values in fields.items():
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(
                        f"Generated boost field {name} contains invalid values."
                    )
                if name == "fsh":
                    filename = "fsh.txt"
                    if filename in arrays and not np.array_equal(
                        arrays[filename], values
                    ):
                        raise ValueError("Mass fraction changed between boost orders.")
                elif prompt_cusps:
                    filename = f"{name}_{current}_{survival['f_surv']:.1f}_{survival['f_surv_stripped']:.1f}.txt"
                else:
                    filename = f"{name}_{current}.txt"
                arrays[filename] = values
            for filename, values in arrays.items():
                np.savetxt(staging / filename, values)
            if manifest is None:
                raise ValueError(
                    "Boost generation needs at least one nondegenerate population."
                )
            manifest["files"] = {
                filename: hashlib.sha256((staging / filename).read_bytes()).hexdigest()
                for filename in arrays
            }
            manifest["generation_diagnostics"] = {
                "allow_incomplete_tables": allow_incomplete,
                "invalid_cell_count": len(incomplete_lookups) + len(degenerate_cells),
                "incomplete_lookups": incomplete_lookups,
                "degenerate_zero_measure_cells": degenerate_cells,
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), target)
    if incomplete_lookups or degenerate_cells:
        warnings.warn(
            f"Generated approximate boost tables with {len(incomplete_lookups)} incomplete lookups "
            f"and {len(degenerate_cells)} historical zero-width cells; see generation_diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )
    return target


def generation_cli(*, prompt_cusps=False):
    """Run explicit JSON-configured table generation from an installed package."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SASHIMI-C boost tables with provenance."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON with explicit mass/redshift grids and calculation parameters",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Scientific data root; otherwise SASHIMI_C_DATA_DIR/cache",
    )
    args = parser.parse_args()
    output = generate_boost_tables(
        json.loads(args.config.read_text()),
        data_dir=args.data_dir,
        prompt_cusps=prompt_cusps,
    )
    print(output)
