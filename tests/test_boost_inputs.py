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
    """Run the exact pre-loader method on identical arrays and runtime math."""
    path = Path(__file__).parent / "references/boost-pre-loader.txt"
    source = path.read_text()
    provenance = json.loads(path.with_name("boost-pre-loader-provenance.json").read_text())
    assert hashlib.sha256(source.encode()).hexdigest() == provenance["method_sha256"]
    namespace = {"np": np, "griddata": griddata}
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102 - pinned, hash-checked local reference
    with monkeypatch.context() as context, warnings.catch_warnings(record=True):
        context.chdir(directory.parent.parent)
        warnings.simplefilter("always")
        return namespace["annihilation_boost_factor"](model, n=n)


def prepare(tmp_path, monkeypatch):
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
    model, directory, manifest = prepare(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    np.testing.assert_array_equal(
        model.annihilation_boost_factor(n=1), frozen_boost(model, directory, monkeypatch, n=1)
    )
    np.testing.assert_array_equal(model.annihilation_boost_factor(), frozen_boost(model, directory, monkeypatch, n=0))
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
    model, directory, manifest = prepare(tmp_path, monkeypatch)
    manifest[field] = value
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        model.annihilation_boost_factor(n=1)


def test_table_corruption_fails(tmp_path, monkeypatch):
    model, directory, _ = prepare(tmp_path, monkeypatch)
    np.savetxt(directory / "fsh.txt", np.full((2, 2), 0.3))
    with pytest.raises(ValueError, match="hash"):
        model.annihilation_boost_factor(n=1)


def test_out_of_domain_is_not_silently_replaced_with_zero(tmp_path, monkeypatch):
    model, _, _ = prepare(tmp_path, monkeypatch)
    model.ma200 = model.ma200 * 1.0e10
    with pytest.raises(ValueError, match="outside"):
        model.annihilation_boost_factor(n=1)


def test_zero_log_table_is_not_silently_replaced_with_zero(tmp_path, monkeypatch):
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
    model, _, _ = prepare(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="nonnegative integer"):
        model.annihilation_boost_factor(n=order)


def test_generator_records_reproducible_inputs_and_refuses_overwrite(tmp_path):
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
    np.testing.assert_array_equal(result, frozen_boost(model, directory, monkeypatch, n=1))
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
