"""Explicit locations for user-supplied scientific tables and generated data."""

from __future__ import annotations

import os
from pathlib import Path


def data_directory(path=None) -> Path:
    """Return the configured data root without creating or reading files.

    An explicit path wins, followed by SASHIMI_C_DATA_DIR, then the user's cache
    directory. Scientific input files are never downloaded or substituted.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the scientific data archive; no pickle evaluation is used by the NPZ
        readers.

    Returns
    -------
    pathlib.Path
        Resolved input/output root, without creating files.

    Notes
    -----
    Data location is explicit so changing the process working directory cannot silently
    select different power spectra or recursive boost tables. Missing scientific inputs fail
    with their required paths.
    """
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.environ.get("SASHIMI_C_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (cache / "sashimi-c").expanduser().resolve()


def require_input(path) -> Path:
    """Fail with the exact required scientific input instead of a cwd lookup.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the scientific data archive; no pickle evaluation is used by the NPZ
        readers.

    Returns
    -------
    pathlib.Path
        Resolved existing input file; missing data raise RuntimeError with the required
        location.

    Notes
    -----
    Data location is explicit so changing the process working directory cannot silently
    select different power spectra or recursive boost tables. Missing scientific inputs fail
    with their required paths.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(
            f"Required SASHIMI-C input file is missing: {path}. "
            "Provide the documented spectrum/table through data_dir or "
            "SASHIMI_C_DATA_DIR. It is not bundled with this release; "
            "no alternative spectrum is substituted."
        )
    return path
