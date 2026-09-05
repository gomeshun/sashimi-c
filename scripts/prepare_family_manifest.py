"""Derive a candidate or promoted manifest from a validated family checkout."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

PACKAGES = ("itamae", "sashimi-c", "sashimi-si", "sashimi-w", "sashimi-f")


def prepare(family_root, output, candidate, candidate_ref, mode="candidate"):
    """Preserve the base set and override only the explicitly tested candidate."""
    family_root, output = Path(family_root).resolve(), Path(output).resolve()
    source = family_root / "compatibility.toml"
    if output == source:
        raise ValueError("The effective manifest must not overwrite the canonical manifest")
    if candidate not in PACKAGES or mode not in ("candidate", "promoted"):
        raise ValueError("Unknown candidate package or validation mode")
    if re.fullmatch(r"[0-9a-f]{40}", candidate_ref) is None:
        raise ValueError("The candidate revision must be a full lowercase commit SHA")

    # Check the untouched base against its own gitlinks, never against an override.
    subprocess.run(
        [sys.executable, str(family_root / "scripts/check_compatibility.py")],
        check=True,
        stdout=sys.stderr,
    )
    base_ref = subprocess.check_output(
        ["git", "-C", str(family_root), "rev-parse", "HEAD"], text=True
    ).strip()
    base = tomllib.loads(source.read_text())
    effective = copy.deepcopy(base)
    if mode == "candidate":
        effective[candidate]["ref"] = candidate_ref

    # Schema 1 has scalar-valued tables. Verify the serialization round trip so
    # future unsupported fields fail explicitly instead of being discarded.
    rendered = "\n\n".join(
        f"[{json.dumps(name)}]\n"
        + "\n".join(f"{json.dumps(key)} = {json.dumps(value)}" for key, value in entry.items())
        for name, entry in effective.items()
    ) + "\n"
    if tomllib.loads(rendered) != effective:
        raise ValueError("Effective manifest serialization changed its contents")
    output.write_text(rendered)
    output.with_suffix(".json").write_text(json.dumps({
        "family_base_ref": base_ref,
        "validation_mode": mode,
        "candidate_package": candidate,
        "workflow_source_ref": candidate_ref,
        "effective_revisions": {name: effective[name]["ref"] for name in PACKAGES},
    }, indent=2) + "\n")
    return base_ref, effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=PACKAGES, required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--mode", choices=("candidate", "promoted"), default="candidate")
    args = parser.parse_args()
    base_ref, manifest = prepare(
        args.family_root, args.output, args.candidate, args.candidate_ref, args.mode
    )
    print(f"family_base_ref={base_ref}")
    print(f"validation_mode={args.mode}")
    for name in PACKAGES:
        key = name.replace("-", "_")
        print(f"{key}_repo={manifest[name]['repo']}")
        print(f"{key}_ref={manifest[name]['ref']}")


if __name__ == "__main__":
    main()
