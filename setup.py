"""Embed the SASHIMI-C source revision in build artifacts."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

_SOURCE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_MODULE = "_sashimi_c_build_provenance"
_TARGET = f"{_MODULE}.py"


def _valid_revision(value: str | None) -> str | None:
    """Return a full lowercase source SHA or ``None``."""
    if value is not None:
        value = value.strip()
    return value if value and _SOURCE_REVISION_PATTERN.fullmatch(value) else None


def _existing_revision(root: Path) -> str | None:
    """Preserve a revision carried by an exported source archive."""
    path = root / _TARGET
    if not path.is_file():
        return None
    match = re.search(
        r"SOURCE_REVISION\s*=\s*['\"]([0-9a-f]{40})['\"]",
        path.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def _git_checkout_revision(root: Path) -> str | None:
    """Read Git only when the source root is the repository root."""
    try:
        root_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        if root_result.returncode != 0:
            return None
        if Path(root_result.stdout.strip()).resolve() != root.resolve():
            return None
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return _valid_revision(result.stdout if result.returncode == 0 else None)


def _resolve_revision(root: Path) -> str:
    """Preserve archive identity and reject conflicting build overrides."""
    supplied = os.environ.get("SASHIMI_C_SOURCE_REVISION")
    explicit = _valid_revision(supplied)
    if supplied is not None and explicit is None:
        raise RuntimeError(
            "SASHIMI_C_SOURCE_REVISION must contain a full 40-character lowercase SHA."
        )
    revision = _existing_revision(root)
    if revision is None:
        revision = _git_checkout_revision(root)
    if revision is not None and explicit is not None and revision != explicit:
        raise RuntimeError(
            "Explicit source revision conflicts with the source archive or checkout."
        )
    revision = revision or explicit
    if revision is None:
        raise RuntimeError(
            "SASHIMI-C artifact builds require an exact source revision from "
            "the source archive, SASHIMI_C_SOURCE_REVISION, or its own Git checkout."
        )
    return revision


class _ProvenanceBuildPy(_build_py):
    """Write provenance into the isolated build directory."""

    _provenance_output: Path | None = None

    def run(self) -> None:
        """Build normal modules, then add the generated provenance module."""
        original_modules = self.py_modules
        self.py_modules = [module for module in original_modules if module != _MODULE]
        try:
            super().run()
        finally:
            self.py_modules = original_modules
        self._provenance_output = Path(self.build_lib) / _TARGET
        self._provenance_output.write_text(
            f"SOURCE_REVISION = {_resolve_revision(Path(__file__).parent)!r}\n",
            encoding="utf-8",
        )

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        """Include the generated module in setuptools' install file list."""
        outputs = super().get_outputs(include_bytecode=include_bytecode)
        if self._provenance_output is not None:
            output = str(self._provenance_output)
            if output not in outputs:
                outputs.append(output)
        return outputs


class _ProvenanceSdist(_sdist):
    """Add provenance to the isolated source-distribution tree."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        """Copy normal files and add a durable provenance module."""
        super().make_release_tree(base_dir, files)
        target = Path(base_dir) / _TARGET
        if not target.is_file():
            target.write_text(
                f"SOURCE_REVISION = {_resolve_revision(Path(__file__).parent)!r}\n",
                encoding="utf-8",
            )


setup(cmdclass={"build_py": _ProvenanceBuildPy, "sdist": _ProvenanceSdist})
