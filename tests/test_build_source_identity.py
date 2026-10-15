"""An archive keeps its source identity regardless of its build directory."""

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("source_build_hook", ROOT / "setup.py")
hook_module = importlib.util.module_from_spec(spec)
with patch("setuptools.setup"):
    spec.loader.exec_module(hook_module)
TARGET = hook_module._TARGET
ENV = (
    "ITAMAE"
    if TARGET.startswith("itamae/")
    else TARGET.removeprefix("_").removesuffix("_build_provenance.py").upper()
) + "_SOURCE_REVISION"
ARCHIVE_SHA = "1234567890abcdef1234567890abcdef12345678"


def resolve(root):
    """Resolve scientific source attribution from the controlled package root.

    Parameters
    ----------
    root : pathlib.Path
        Package checkout or isolated test repository whose own source identity is under
        examination.

    Returns
    -------
    str
        Exact package revision, or the failure expected by an invalid-attribution test.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    return hook_module._resolve_revision(root)


def git(root, *args):
    """Run a Git query in the isolated attribution fixture.

    Parameters
    ----------
    root : pathlib.Path
        Package checkout or isolated test repository whose own source identity is under
        examination.
    *args : tuple
        Additional positional parameters forwarded unchanged to the supplied numerical or
        physical callback.

    Returns
    -------
    str
        Stripped command output used to identify or prepare the controlled checkout.

    Notes
    -----
    This is packaging/reproducibility support. Tests must recover the package revision and
    reject missing or conflicting attribution; a parent repository or environment must not
    silently relabel a scientific calculation.
    """
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    """Create a parent checkout to detect accidental attribution to unrelated source.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated test directory holding temporary scientific data or source-attribution
        artifacts.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the provider or environment under test, restored after the
        case.

    Returns
    -------
    pathlib.Path
        Temporary repository containing a known unrelated commit.

    Notes
    -----
    This is packaging/reproducibility support. Tests must recover the package revision and
    reject missing or conflicting attribution; a parent repository or environment must not
    silently relabel a scientific calculation.
    """
    monkeypatch.delenv(ENV, raising=False)
    git(tmp_path, "init", "--quiet")
    (tmp_path / "unrelated.txt").write_text("parent source")
    git(tmp_path, "add", "unrelated.txt")
    git(
        tmp_path,
        "-c",
        "user.name=Build test",
        "-c",
        "user.email=build@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "Parent source",
    )
    return tmp_path


def archive(root):
    """Write a source-archive fixture with an embedded known package revision.

    Parameters
    ----------
    root : pathlib.Path
        Package checkout or isolated test repository whose own source identity is under
        examination.

    Returns
    -------
    pathlib.Path
        Archive root carrying the expected original source identity.

    Notes
    -----
    This is packaging/reproducibility support. Tests must recover the package revision and
    reject missing or conflicting attribution; a parent repository or environment must not
    silently relabel a scientific calculation.
    """
    root.mkdir(parents=True)
    descriptor = root / TARGET
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    descriptor.write_text(f"SOURCE_REVISION = {ARCHIVE_SHA!r}\n")
    return root


def test_archive_in_unrelated_repository_keeps_embedded_revision(repository):
    """Keep a published catalogue attributable to the code inside its source archive.

    Parameters
    ----------
    repository : pathlib.Path
        Temporary Git checkout used to distinguish a package's own scientific source from an
        unrelated enclosing repository.

    Notes
    -----
    Pass criteria: An exported package inside an unrelated checkout resolves to its embedded
    source SHA rather than the enclosing repository SHA.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: These checks establish source attribution, not the numerical or physical
    correctness of the packaged model.
    """
    source = archive(repository / "unpacked")
    assert resolve(source) == ARCHIVE_SHA


def test_unidentified_export_rejects_enclosing_repository(repository):
    """Reject an export whose scientific source revision cannot be established.

    Parameters
    ----------
    repository : pathlib.Path
        Temporary Git checkout used to distinguish a package's own scientific source from an
        unrelated enclosing repository.

    Notes
    -----
    Pass criteria: A source export lacking embedded identity raises RuntimeError instead of
    using the surrounding repository's SHA.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: These checks establish source attribution, not the numerical or physical
    correctness of the packaged model.
    """
    source = repository / "export"
    source.mkdir()
    with pytest.raises(RuntimeError, match="source revision"):
        resolve(source)


def test_exact_source_checkout_uses_own_revision(repository):
    """Identify the actual source checkout used to produce a catalogue.

    Parameters
    ----------
    repository : pathlib.Path
        Temporary Git checkout used to distinguish a package's own scientific source from an
        unrelated enclosing repository.

    Notes
    -----
    Pass criteria: A package at its own Git root resolves to that checkout's exact revision.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: These checks establish source attribution, not the numerical or physical
    correctness of the packaged model.
    """
    assert resolve(repository) == git(repository, "rev-parse", "HEAD")


def test_archive_rejects_conflicting_explicit_revision(repository, monkeypatch):
    """Prevent a build override from relabelling a preserved scientific source archive.

    Parameters
    ----------
    repository : pathlib.Path
        Temporary Git checkout used to distinguish a package's own scientific source from an
        unrelated enclosing repository.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: A supplied revision that conflicts with the archive's embedded SHA raises
    RuntimeError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: These checks establish source attribution, not the numerical or physical
    correctness of the packaged model.
    """
    source = archive(repository / "unpacked")
    monkeypatch.setenv(ENV, "f" * 40)
    with pytest.raises(RuntimeError, match="conflict"):
        resolve(source)


def test_exact_checkout_rejects_conflicting_explicit_revision(repository, monkeypatch):
    """Prevent an environment override from relabelling the checked-out calculation.

    Parameters
    ----------
    repository : pathlib.Path
        Temporary Git checkout used to distinguish a package's own scientific source from an
        unrelated enclosing repository.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: A supplied revision different from the checkout's SHA raises
    RuntimeError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: These checks establish source attribution, not the numerical or physical
    correctness of the packaged model.
    """
    monkeypatch.setenv(ENV, "f" * 40)
    with pytest.raises(RuntimeError, match="conflict"):
        resolve(repository)


def test_malformed_explicit_revision_does_not_silently_fall_back(
    repository, monkeypatch
):
    """Reject an unusable source identifier before a scientific package is built.

    Parameters
    ----------
    repository : pathlib.Path
        Temporary Git checkout used to distinguish a package's own scientific source from an
        unrelated enclosing repository.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: Malformed explicit revision values raise RuntimeError even when another
    source identity is available.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: These checks establish source attribution, not the numerical or physical
    correctness of the packaged model.
    """
    monkeypatch.setenv(ENV, "short")
    with pytest.raises(RuntimeError, match="40"):
        resolve(repository)


def test_matching_explicit_archive_revision_is_allowed(repository, monkeypatch):
    """Permit an explicitly confirmed archive identity without changing its provenance.

    Parameters
    ----------
    repository : pathlib.Path
        Temporary Git checkout used to distinguish a package's own scientific source from an
        unrelated enclosing repository.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: An explicit SHA equal to the embedded archive SHA is accepted and
    recorded unchanged.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: These checks establish source attribution, not the numerical or physical
    correctness of the packaged model.
    """
    source = archive(repository / "unpacked")
    monkeypatch.setenv(ENV, ARCHIVE_SHA)
    assert resolve(source) == ARCHIVE_SHA
