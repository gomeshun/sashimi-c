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


def _resolve_revision(root: Path) -> str:
    """Resolve the exact revision used to create the artifact."""
    revision = _valid_revision(os.environ.get("SASHIMI_C_SOURCE_REVISION"))
    if revision is None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            result = None
        revision = _valid_revision(result.stdout if result and result.returncode == 0 else None)
    if revision is None:
        revision = _existing_revision(root)
    if revision is None:
        raise RuntimeError(
            "SASHIMI-C artifact builds require an exact source revision from "
            "the source archive, SASHIMI_C_SOURCE_REVISION, or Git."
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
