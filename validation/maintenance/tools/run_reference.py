"""Export a pinned reference, verify its inputs/patches, and run it in isolation.

The supplied interpreter owns the historical dependency environment. This
controller imports neither current products nor historical modules. It exports
the exact Git object, never the changing working tree. B is made solely by
applying hash-verified, reviewable patches to A in a temporary export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    revision = config["source_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("References require a complete immutable Git revision.")
    if config["role"] not in ("A", "B"):
        raise ValueError("Reference role must be A or B.")
    if config["role"] == "A" and config.get("patches"):
        raise ValueError("Frozen reference A must not have source patches.")
    repository = args.repository.resolve()
    resolved = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", revision + "^{commit}"], text=True
    ).strip()
    if resolved != revision:
        raise ValueError(
            "The requested object is not the exact commit recorded in the config."
        )
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sashimi-reference-") as temporary:
        temporary = Path(temporary)
        archive = temporary / "source.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                "--output=" + str(archive),
                revision,
            ],
            check=True,
        )
        source = temporary / "source"
        source.mkdir()
        with tarfile.open(archive) as exported:
            exported.extractall(source, filter="data")
        export_hash = sha256(archive)
        for patch in config.get("patches", []):
            path = (config_path.parent / patch["path"]).resolve()
            if sha256(path) != patch["sha256"]:
                raise ValueError(f"Patch digest mismatch: {patch['id']}")
            subprocess.run(
                ["git", "apply", "--check", str(path)], cwd=source, check=True
            )
            subprocess.run(["git", "apply", str(path)], cwd=source, check=True)
        subprocess.run(
            [
                str(args.python.absolute()),
                "-I",
                str(Path(__file__).with_name("reference_worker.py")),
                "--source",
                str(source),
                "--config",
                str(config_path),
                "--output",
                str(args.output),
            ],
            cwd=temporary,
            check=True,
        )
        sidecar = args.output.with_suffix(".json")
        metadata = json.loads(sidecar.read_text())
        metadata["source_export_sha256"] = export_hash
        metadata["config_sha256"] = sha256(config_path)
        metadata["controller_sha256"] = sha256(Path(__file__))
        metadata["independent_process"] = True
        sidecar.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
