"""Evaluate a frozen reference in its own process without importing the product.

The controller supplies an exported source directory and explicit calculation
parameters. This worker deliberately has no dependency on ITAMAE or a current
SASHIMI checkout. Numerical tuples preserve their original ordering and units.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    config = json.loads(args.config.read_text())
    if config["role"] not in ("A", "B"):
        raise ValueError(
            "The isolated reference worker accepts only A or B references."
        )
    expected = config.get("environment", {})
    if expected.get("python") and expected["python"] != sys.version.split()[0]:
        raise ValueError(
            "Reference Python version does not match its frozen environment."
        )
    for name, version in expected.get("dependencies", {}).items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Reference dependency version mismatch: {name}")
    verified_inputs = []
    for item in config.get("input_files", []):
        path = (source / item["path"]).resolve()
        if not path.is_relative_to(source):
            raise ValueError(
                "Reference input paths must stay within the exported source."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Reference input digest mismatch: {item['path']}")
        verified_inputs.append({**item, "verified_sha256": digest})
    args.output = args.output.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as observed:
        module = importlib.import_module(config["module"])
        if Path(module.__file__).resolve().parent != source:
            raise RuntimeError(
                "The reference module was imported from an unexpected location."
            )
        # Old modules install broad warning filters. Record their actual arithmetic
        # without changing any expressions or suppressing their numerical results.
        warnings.simplefilter("always", RuntimeWarning)
        model = getattr(module, config["class"])(**config.get("constructor", {}))
        arrays = getattr(model, config["method"])(**config["parameters"])
        payload = {
            f"tuple_{index}": np.asarray(value) for index, value in enumerate(arrays)
        }
        for probe in config.get("probes", []):
            value = getattr(model, probe["method"])(
                *[
                    np.asarray(arg) if isinstance(arg, list) else arg
                    for arg in probe.get("args", [])
                ],
                **probe.get("kwargs", {}),
            )
            payload["probe_" + probe["name"]] = np.asarray(value)
    if "itamae" in sys.modules:
        raise RuntimeError(
            "An independent reference must not import the product's ITAMAE core."
        )
    packages = {}
    for name in sorted(
        set(("numpy", "scipy", "colossus", "numexpr", "matplotlib", "tqdm"))
        | set(config.get("environment", {}).get("dependencies", {}))
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {
        "schema": "sashimi-family:independent-reference:v1",
        "role": config["role"],
        "repository": config["repository"],
        "source_revision": config["source_revision"],
        "module_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "input_files": verified_inputs,
        "patches": config.get("patches", []),
        "calculation": config,
        "python": sys.version,
        "platform": sys.platform,
        "dependencies": packages,
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "warnings": sorted({str(w.message) for w in observed}),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "nonfinite": int(np.sum(~np.isfinite(value))),
            }
            for name, value in payload.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    metadata["artifact_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "role": config["role"],
                "repository": config["repository"],
                "output": str(args.output),
                "seconds": metadata["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
