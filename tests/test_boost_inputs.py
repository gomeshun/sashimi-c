"""Configured higher-order boost tables and explicit failure boundaries."""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import griddata

from sashimi_c import CALCULATION_SPECIFICATION, SubhaloObservables

REFERENCE = json.loads(
    (Path(__file__).parent / "references/boost-positive-inputs.json").read_text()
)


def frozen_boost(model, directory, monkeypatch, *, n):
    """Evaluate the hash-checked pre-loader boost formula on identical model arrays and table
    data.

    Parameters
    ----------
    model : object
        Variant halo/population object supplying its own background, units, structure and
        accretion prescriptions.
    directory : pathlib.Path or str, optional
        Explicit root of the required scientific input tables or generated products; no
        substitute spectrum is selected implicitly.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the provider or environment under test, restored after the
        case.
    n : int
        Nonnegative substructure hierarchy order. Zero uses the direct subhalo contribution;
        higher orders require previous-order tables.

    Returns
    -------
    tuple
        Historical boost and luminosity ratio used only as a regression reference.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    path = Path(__file__).parent / "references/boost-pre-loader.txt"
    source = path.read_text()
    provenance = json.loads(
        path.with_name("boost-pre-loader-provenance.json").read_text()
    )
    assert hashlib.sha256(source.encode()).hexdigest() == provenance["method_sha256"]
    namespace = {"np": np, "griddata": griddata}
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102 - pinned, hash-checked local reference
    with monkeypatch.context() as context, warnings.catch_warnings(record=True):
        context.chdir(directory.parent.parent)
        warnings.simplefilter("always")
        return namespace["annihilation_boost_factor"](model, n=n)


def prepare(tmp_path, monkeypatch):
    """Create controlled boost tables and matching physical/source provenance for loader tests.

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
    tuple
        Model, isolated table directory and manifest used to vary one input condition at a
        time.

    Notes
    -----
    This fixture/helper supports the controlled checks in this test module; its synthetic
    values are not a calibrated halo model.
    """
    root = tmp_path / "data"
    directory = root / "boost"
    directory.mkdir(parents=True)
    for filename, values in REFERENCE["arrays"].items():
        np.savetxt(directory / filename, values)
    monkeypatch.chdir(tmp_path)
    model = SubhaloObservables(**REFERENCE["parameters"], data_dir=root)
    manifest = {
        "schema_version": 1,
        "variant": "sashimi-c",
        "calculation_specification": CALCULATION_SPECIFICATION,
        "mass_unit": "Msun",
        "source_revision": REFERENCE["source_revision"],
        "itamae_source_revision": "e5c77cdf2832104752d641dd553902fc507885ba",
        "model_contract": {
            "prompt_cusps": False,
            **{
                name: model.catalog.metadata[name]
                for name in (
                    "cosmology_backend",
                    "variance_identifier",
                    "profile_change",
                    "stripping_method",
                    "ct_threshold",
                    "log10_mass_min",
                    "sigma_log10_concentration",
                    "accretion_model",
                )
            },
        },
        "generation_parameters": REFERENCE["parameters"],
        "files": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in REFERENCE["arrays"]
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return model, directory, manifest


def test_positive_table_result_is_unchanged_outside_source(tmp_path, monkeypatch):
    """Preserve annihilation boosts when valid higher-order tables are loaded from an explicit
    location.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: At orders zero and one the result equals the hash-checked historical
    method; changing the working directory does not alter the tables or their recorded
    manifest.

    Failure criteria: any mismatch with the stated value, shape or identity requirements
    fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model, directory, manifest = prepare(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    np.testing.assert_array_equal(
        model.annihilation_boost_factor(n=1),
        frozen_boost(model, directory, monkeypatch, n=1),
    )
    np.testing.assert_array_equal(
        model.annihilation_boost_factor(),
        frozen_boost(model, directory, monkeypatch, n=0),
    )
    assert model.last_boost_provenance["manifest"] == manifest
    assert model.last_boost_provenance["directory"] == str(directory)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mass_unit", "Msun/h"),
        ("calculation_specification", "old"),
        ("variant", "sashimi-f"),
    ],
)
def test_wrong_physical_table_contract_fails(tmp_path, monkeypatch, field, value):
    """Prevent reuse of boost tables with a different mass unit, variant or prescription.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.
    field : str
        Physical field or provenance field deliberately varied in this test.
    value : object
        Parametrized physical value or malformed input, as specified by this case.

    Notes
    -----
    Pass criteria: Changing the declared mass unit, calculation specification or variant
    raises ValueError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model, directory, manifest = prepare(tmp_path, monkeypatch)
    manifest[field] = value
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        model.annihilation_boost_factor(n=1)


def test_table_corruption_fails(tmp_path, monkeypatch):
    """Reject a boost table modified after its provenance was recorded.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: Changing fsh while retaining the old manifest digest raises ValueError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model, directory, _ = prepare(tmp_path, monkeypatch)
    np.savetxt(directory / "fsh.txt", np.full((2, 2), 0.3))
    with pytest.raises(ValueError, match="hash"):
        model.annihilation_boost_factor(n=1)


def test_out_of_domain_is_not_silently_replaced_with_zero(tmp_path, monkeypatch):
    """Avoid treating an unsupported boost-table query as zero substructure luminosity.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: Moving queried accretion masses beyond the table domain raises
    ValueError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model, _, _ = prepare(tmp_path, monkeypatch)
    model.ma200 = model.ma200 * 1.0e10
    with pytest.raises(ValueError, match="outside"):
        model.annihilation_boost_factor(n=1)


def test_zero_log_table_is_not_silently_replaced_with_zero(tmp_path, monkeypatch):
    """Reject logarithmic interpolation of a zero table entry.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.

    Notes
    -----
    Pass criteria: A hash-consistent but zero fsh table raises a positivity error.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model, directory, manifest = prepare(tmp_path, monkeypatch)
    np.savetxt(directory / "fsh.txt", np.zeros((2, 2)))
    manifest["files"]["fsh.txt"] = hashlib.sha256(
        (directory / "fsh.txt").read_bytes()
    ).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="positive"):
        model.annihilation_boost_factor(n=1)


@pytest.mark.parametrize("order", [-1, 0.5, True])
def test_invalid_order_fails_before_file_access(tmp_path, monkeypatch, order):
    """Require a defined nonnegative order for the recursive boost hierarchy.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.
    order : int
        Quadrature or recursive-hierarchy order under test; malformed cases exercise
        validation.

    Notes
    -----
    Pass criteria: Negative, fractional or Boolean recursion order raises ValueError.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model, _, _ = prepare(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="nonnegative integer"):
        model.annihilation_boost_factor(n=order)


def test_generator_records_reproducible_inputs_and_refuses_overwrite(tmp_path):
    """Preserve the physical inputs used to generate reusable boost tables.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.

    Notes
    -----
    Pass criteria: The saved manifest records the full configuration and matching file
    hashes; a repeated write raises FileExistsError and leaves the manifest unchanged.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    from sashimi_c_boost import generate_boost_tables

    configuration = {
        "max_order": 0,
        "redshift_grid": [0.0, 0.25],
        "mass_grid_Msun": [[1.0e10, 1.0e11], [1.0e10, 1.0e11]],
        "calculation_parameters": {
            "dz": 0.25,
            "zmax": 1.0,
            "N_ma": 4,
            "N_herm": 2,
            "N_hermNa": 3,
            "logmamin": 6.0,
            "logmamax": 8.0,
        },
    }
    path = generate_boost_tables(configuration, data_dir=tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["generation_parameters"] == configuration
    assert manifest["model_contract"]["prompt_cusps"] is False
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
    before = (path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        generate_boost_tables(configuration, data_dir=tmp_path)
    assert (path / "manifest.json").read_bytes() == before


def test_failed_generator_leaves_no_partial_table(tmp_path):
    """Do not leave a failed boost calculation looking like a valid reusable dataset.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.

    Notes
    -----
    Pass criteria: An invalid lower mass boundary raises ValueError and leaves the
    destination empty.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    from sashimi_c_boost import generate_boost_tables

    configuration = {
        "max_order": 0,
        "redshift_grid": [0.0, 0.25],
        "mass_grid_Msun": [[1.0e-5, 1.0], [1.0e-5, 1.0]],
        "calculation_parameters": {
            "dz": 0.5,
            "zmax": 1.0,
            "N_ma": 4,
            "N_herm": 2,
            "N_hermNa": 3,
        },
    }
    with pytest.raises(ValueError, match="logmamin"):
        generate_boost_tables(configuration, data_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("case", ["zero-fsh", "outside-domain"])
def test_explicit_approximation_retains_frozen_results_and_reports_invalid_points(
    tmp_path, monkeypatch, case
):
    """Expose the limitations when explicitly replaying the historical incomplete-table
    approximation.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacement of the particular physical provider or environment under test;
        restored after the case.
    case : object
        Controlled input supplied by the parametrization or fixture for the specific
        physical/reference case described below.

    Notes
    -----
    Pass criteria: For zero or out-of-domain tables, opt-in execution warns, matches the
    frozen method, and records invalid query count and mask identity.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    model, directory, manifest = prepare(tmp_path, monkeypatch)
    if case == "zero-fsh":
        np.savetxt(directory / "fsh.txt", np.zeros((2, 2)))
        manifest["files"]["fsh.txt"] = hashlib.sha256(
            (directory / "fsh.txt").read_bytes()
        ).hexdigest()
        (directory / "manifest.json").write_text(json.dumps(manifest))
    else:
        model.ma200 *= 1.0e10
    with pytest.warns(RuntimeWarning, match="historical boost approximation"):
        result = model.annihilation_boost_factor(n=1, allow_incomplete_tables=True)
    np.testing.assert_array_equal(
        result, frozen_boost(model, directory, monkeypatch, n=1)
    )
    diagnostics = model.last_boost_provenance
    assert diagnostics["valid"] is False
    assert diagnostics["allow_incomplete_tables"] is True
    assert (
        diagnostics["invalid_query_count"]
        == diagnostics["query_count"]
        == REFERENCE["incomplete_cases"][case]["query_count"]
    )
    assert len(diagnostics["invalid_query_mask_sha256"]) == 64


def test_generator_can_explicitly_preserve_incomplete_hierarchy(tmp_path):
    """Keep recursive boost-table incompleteness visible in saved generation diagnostics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.

    Notes
    -----
    Pass criteria: Opt-in order-one generation warns, records four incomplete cells/lookups,
    and reproduces the historical equality of Bsh_0 and Bsh_1.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    from sashimi_c_boost import generate_boost_tables

    configuration = {
        "max_order": 1,
        "allow_incomplete_tables": True,
        "redshift_grid": [0.0, 0.25],
        "mass_grid_Msun": [[1.0e10, 1.0e11], [1.0e10, 1.0e11]],
        "calculation_parameters": {
            "dz": 0.25,
            "zmax": 1.0,
            "N_ma": 4,
            "N_herm": 2,
            "N_hermNa": 3,
            "logmamin": 6.0,
            "logmamax": 8.0,
        },
    }
    with pytest.warns(RuntimeWarning, match="approxima"):
        path = generate_boost_tables(configuration, data_dir=tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["generation_diagnostics"]["invalid_cell_count"] == 4
    assert len(manifest["generation_diagnostics"]["incomplete_lookups"]) == 4
    np.testing.assert_array_equal(
        np.loadtxt(path / "Bsh_0.txt"), np.loadtxt(path / "Bsh_1.txt")
    )


def test_generator_preserves_only_the_documented_zero_width_boundary(tmp_path):
    """Limit the historical zero-contribution exception to a zero-measure mass interval.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test catalogue, cache or provenance files.

    Notes
    -----
    Pass criteria: The single documented zero-width cell is zero in fsh and Bsh_0, warns,
    and is recorded as one degenerate cell.

    Failure criteria: a stated numerical/identity mismatch, a missing required exception or
    warning, or an unexpected calculation failure fails the test.

    Scope: This controlled calculation does not calibrate a subhalo population against
    simulations or observations.
    """
    from sashimi_c_boost import generate_boost_tables

    configuration = {
        "max_order": 0,
        "allow_incomplete_tables": True,
        "redshift_grid": [0.0, 0.25],
        "mass_grid_Msun": [[1.0e-5, 1.0], [1.0e-5, 1.0]],
        "calculation_parameters": {
            "dz": 0.5,
            "zmax": 1.0,
            "N_ma": 4,
            "N_herm": 2,
            "N_hermNa": 3,
        },
    }
    with pytest.warns(RuntimeWarning, match="1 historical zero-width cells"):
        path = generate_boost_tables(configuration, data_dir=tmp_path)
    assert (
        np.loadtxt(path / "fsh.txt")[0, 0]
        == np.loadtxt(path / "Bsh_0.txt")[0, 0]
        == 0.0
    )
    diagnostics = json.loads((path / "manifest.json").read_text())[
        "generation_diagnostics"
    ]
    assert diagnostics["invalid_cell_count"] == 1
    assert (
        diagnostics["degenerate_zero_measure_cells"][0]["reason"]
        == "historical-zero-width-log-mass-grid"
    )
