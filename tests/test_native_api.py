"""Native CDM specification against independently saved migration outputs."""

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from sashimi_c import CDM, SubhaloProperties
from sashimi_c._api import _host_mass_at_zero
from sashimi_c._itamae_components import CDMAccretionSlices, NFWInitialStructure

REFERENCE = Path(__file__).parent / "references/native-api-baseline"


def small():
    return CDM().configure(
        accretion={"mass_nodes": 4, "host_history_nodes": 3, "redshift_step": 0.5},
        concentration={"quadrature_nodes": 2},
    )


def inputs():
    return {
        "host_mass_msun": 1e10,
        "accretion_mass_range_msun": (1e6, 1e8),
        "accretion_redshift_range": (0.0, 1.0),
    }


@pytest.mark.parametrize("name", [p.stem for p in REFERENCE.glob("*.json")])
def test_native_and_legacy_match_independent_baseline(name):
    provenance = json.loads((REFERENCE / f"{name}.json").read_text())
    path = REFERENCE / f"{name}.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["sha256"]
    assert provenance["variant"] == "0aa33a2a0935b3f0c4bf62072b36186ecc9f1047"
    assert provenance["core"] == "23d01e8758a88b061b87de9e488c38ec89fd8e4f"
    p = provenance["parameters"]
    native = (
        small()
        .configure(
            stripping={
                "solver": p.get("method", "pert2_shanks"),
                "profile_change": p.get("profile_change", True),
                "solver_options": {k: p[k] for k in ("rtol", "atol") if k in p},
            },
            disruption={"ct_threshold": p.get("ct_th", 0.0)},
        )
        .population(
            **{
                **inputs(),
                "redshift": p.get("redshift", 0.0),
                "host_mass_redshift": p.get("redshift", 0.0)
                if p.get("M0_at_redshift")
                else 0.0,
                "accretion_redshift_range": (p.get("redshift", 0.0), p["zmax"]),
            }
        )
    )
    legacy = SubhaloProperties().subhalo_catalog_calc(**p)
    with np.load(path) as baseline:
        for catalog in (native, legacy):
            for column_name, array in {**catalog.columns, **catalog.weights}.items():
                if array.dtype.kind == "b":
                    np.testing.assert_array_equal(array, baseline[column_name])
                else:
                    np.testing.assert_allclose(
                        array, baseline[column_name], rtol=5e-12, atol=0.0
                    )
    for key in native.columns:
        np.testing.assert_array_equal(native.columns[key], legacy.columns[key])
    for key in native.weights:
        np.testing.assert_array_equal(native.weights[key], legacy.weights[key])


def test_omitted_range_preserves_historical_grid_even_off_grid():
    model = small().configure(accretion={"redshift_step": 2.0})
    native = model.population(
        **{k: v for k, v in inputs().items() if k != "accretion_redshift_range"}
    )
    old = SubhaloProperties().subhalo_catalog_calc(
        M0=1e10, N_ma=4, N_herm=2, N_hermNa=3, dz=2.0, logmamin=6.0, logmamax=8.0
    )
    for key in native.columns:
        np.testing.assert_array_equal(native.columns[key], old.columns[key])
    assert native.metadata["accretion_redshift_nodes"] == [2.0, 4.0, 6.0, 8.0]
    assert native.metadata["accretion_redshift_range_policy"] == "legacy-default-grid"


def test_explicit_range_never_expands_physical_support():
    native = small().population(
        **{**inputs(), "redshift": 0.1, "accretion_redshift_range": (0.2, 1.3)}
    )
    np.testing.assert_array_equal(np.unique(native.columns["z_acc"]), [0.7, 1.2])
    assert np.max(native.columns["z_acc"]) <= 1.3
    assert native.metadata["accretion_redshift_range"] == [0.2, 1.3]
    assert native.metadata["accretion_redshift_range_policy"] == "explicit-bounded"


def test_configure_is_detached_immutable_and_partial():
    options = {"rtol": 1e-8}
    supplied = {"solver": "odeint", "solver_options": options}
    original = small()
    changed = original.configure(stripping=supplied)
    options["rtol"] = 99
    supplied["solver"] = "bad"
    assert changed.resolved_settings["stripping"]["solver_options"]["rtol"] == 1e-8
    assert original.resolved_settings["stripping"]["solver"] == "pert2_shanks"
    assert (
        changed.resolved_settings["accretion"]
        == original.resolved_settings["accretion"]
    )
    with pytest.raises(TypeError):
        changed.resolved_settings["stripping"]["solver_options"]["rtol"] = 2
    with pytest.raises(FrozenInstanceError):
        changed._settings = {}
    next_model = changed.configure(stripping={"solver_options": {"atol": 1e-10}})
    assert dict(next_model.resolved_settings["stripping"]["solver_options"]) == {
        "rtol": 1e-8,
        "atol": 1e-10,
    }
    with pytest.raises(ValueError, match="odeint"):
        next_model.configure(stripping={"solver": "pert0"})
    assert not next_model.configure(
        stripping={"solver": "pert0", "solver_options": {}}
    ).resolved_settings["stripping"]["solver_options"]


def test_repeated_runs_and_models_do_not_reuse_mutable_state():
    model = small()
    first = model.population(**inputs())
    saved = {k: v.copy() for k, v in first.columns.items()}
    model.configure(concentration={"scatter_dex": 0.25}).population(
        **{**inputs(), "host_mass_msun": 2e10}
    )
    model.population(
        **{**inputs(), "redshift": 0.5, "accretion_redshift_range": (0.5, 1.5)}
    )
    again = model.population(**inputs())
    for key, expected in saved.items():
        np.testing.assert_array_equal(first.columns[key], expected)
        np.testing.assert_array_equal(again.columns[key], expected)
    assert not hasattr(model, "catalog") and not hasattr(model, "M0")


@pytest.mark.parametrize(
    "override",
    [
        {"accretion": {"oops": 1}},
        {"concentration": {"prescription": "unknown"}},
        {"stripping": {"solver": "unknown"}},
        {"stripping": {"solver_options": {"rtol": 1e-8}}},
        {"stripping": {"solver": "odeint", "solver_options": {"Dfun": lambda x: x}}},
        {"stripping": {"solver": "odeint", "solver_options": {"tfirst": True}}},
        {"stripping": {"solver": "odeint", "solver_options": {"rtol": 0}}},
        {"stripping": {"solver": "odeint", "solver_options": {"mxordn": 13}}},
        {"accretion": {"mass_nodes": 1}},
        {"concentration": {"scatter_dex": float("nan")}},
        {"stripping": {"profile_change": "False"}},
        {"accretion": {"model": True}},
    ],
)
def test_invalid_settings_fail_during_configuration(override):
    with pytest.raises((TypeError, ValueError)):
        CDM().configure(**override)


@pytest.mark.parametrize(
    "override",
    [
        {"host_mass_msun": 0},
        {"host_mass_msun": True},
        {"host_mass_redshift": -1},
        {"host_mass_redshift": [1.0]},
        {"host_mass_definition": "vir"},
        {"accretion_mass_definition": "vir"},
        {"accretion_mass_range_msun": (1, 0)},
        {"redshift": 1, "accretion_redshift_range": (0, 2)},
        {"accretion_redshift_range": (0, 0.1)},
        {"accretion_redshift_range": (1, 0)},
    ],
)
def test_invalid_population_inputs_fail(override):
    with pytest.raises((TypeError, ValueError)):
        small().population(**{**inputs(), **override})


@pytest.mark.parametrize("accretion_model", [1, 2, 3])
@pytest.mark.parametrize("host_epoch", [0.0, 0.7])
@pytest.mark.parametrize("lower_fraction", [0.5, 0.6])
def test_mass_range_without_host_support_fails_before_execution(
    monkeypatch, accretion_model, host_epoch, lower_fraction
):
    mass_zero = _host_mass_at_zero(SubhaloProperties(), 1e10, host_epoch)

    def cannot_execute(*args, **kwargs):
        raise AssertionError("Unsupported mass range must fail before execution.")

    monkeypatch.setattr(SubhaloProperties, "_calculate_population", cannot_execute)
    with pytest.raises(ValueError, match="lower bound.*half.*host mass at z=0"):
        small().configure(accretion={"model": accretion_model}).population(
            **{
                **inputs(),
                "host_mass_redshift": host_epoch,
                "accretion_mass_range_msun": (lower_fraction * mass_zero, 0.8 * mass_zero),
            }
        )


@pytest.mark.parametrize("accretion_model", [1, 2, 3])
def test_mass_grid_without_sampled_support_fails_before_normalization(accretion_model):
    model = small().configure(accretion={"model": accretion_model, "host_history_nodes": 1})
    # Below the global half-host ceiling, but above all sampled host masses at z=6.5,7.
    with (
        np.errstate(divide="raise", invalid="raise"),
        pytest.raises(ValueError, match="Accretion grid.*positive finite normalization"),
    ):
        model.population(
            **{
                **inputs(),
                "accretion_mass_range_msun": (4e9, 4.5e9),
                "accretion_redshift_range": (6.0, 7.0),
            }
        )


def test_host_reference_epoch_is_independent_of_target():
    at_epoch = small().population(
        **{
            **inputs(),
            "host_mass_redshift": 0.7,
            "redshift": 0.2,
            "accretion_redshift_range": (0.2, 1.2),
        }
    )
    at_zero = small().population(
        **{
            **inputs(),
            "host_mass_msun": at_epoch.metadata["host_mass_z0"],
            "redshift": 0.2,
            "accretion_redshift_range": (0.2, 1.2),
        }
    )
    for name in at_epoch.columns:
        np.testing.assert_array_equal(at_epoch.columns[name], at_zero.columns[name])
    assert at_epoch.metadata["host_mass_redshift"] == 0.7


def test_host_inversion_zero_is_identity_and_extrapolation_is_rejected():
    class NoPhysics:
        def Mzi(self, masses, z):
            raise AssertionError("zero-epoch mass should bypass inversion")

    assert _host_mass_at_zero(NoPhysics(), 3.0, 0.0) == 3.0

    class OutsideBracket:
        def Mzi(self, masses, z):
            return masses * 1e-10

    with pytest.raises(ValueError, match="bracket"):
        _host_mass_at_zero(OutsideBracket(), 3.0, 1.0)


def test_units_weight_factors_and_executed_provenance():
    catalog = (
        small().configure(disruption={"ct_threshold": 0.77}).population(**inputs())
    )
    assert catalog.metadata["canonical_units"]["mass"] == "Msun"
    assert np.min(catalog.columns["m200_acc"]) == 1e6
    assert np.max(catalog.columns["m200_acc"]) == 1e8
    np.testing.assert_array_equal(
        catalog.weights["weight_survival"], catalog.columns["survive"]
    )
    assert catalog.metadata["tuple_weight_excludes_survival"]
    assert catalog.metadata["resolved_settings"]["disruption"]["ct_threshold"] == 0.77
    assert catalog.metadata["accretion_auxiliary_redshift_nodes"] == 1000
    assert catalog.metadata["host_mass_inversion"]["nodes"] == 1500
    assert catalog.metadata["solver_identifier"].endswith(":pert2_shanks:v1")


def test_disruption_threshold_changes_survival_and_final_weights():
    original = small().population(**inputs())
    threshold = 10.0
    selected = (
        small().configure(disruption={"ct_threshold": threshold}).population(**inputs())
    )
    expected = original.columns["c_t"] > threshold
    assert np.all(original.columns["survive"])
    assert np.any(expected) and not np.all(expected)
    np.testing.assert_array_equal(selected.columns["survive"], expected)
    np.testing.assert_array_equal(selected.weights["weight_survival"], expected)
    for key in ("weight_base", "weight_concentration"):
        np.testing.assert_array_equal(selected.weights[key], original.weights[key])
    np.testing.assert_array_equal(
        selected.weight_final, original.weight_final * expected
    )
    assert selected.weight_final.sum() < original.weight_final.sum()


def test_typed_cdm_context_rejects_mismatched_concentration_layout():
    model = SubhaloProperties()
    batch, context = CDMAccretionSlices(
        model, np.array([1e6, 1e7]), np.array([0.5]), np.ones((1, 2)), 0.128, 2
    ).build(0)
    with pytest.raises(ValueError, match="layout"):
        NFWInitialStructure(model, 3).initialize(batch, context)
    with pytest.raises(ValueError):
        context.mvir_acc[0] = 1


@pytest.mark.parametrize("model", [1, 2])
def test_existing_accretion_laws_are_explicit_and_match_legacy(model):
    native = small().configure(accretion={"model": model}).population(**inputs())
    old = SubhaloProperties().subhalo_catalog_calc(
        M0=1e10,
        N_ma=4,
        N_herm=2,
        N_hermNa=3,
        dz=0.5,
        zmax=1.0,
        logmamin=6.0,
        logmamax=8.0,
        Na_model=model,
    )
    for key in native.columns:
        np.testing.assert_array_equal(native.columns[key], old.columns[key])
    for key in native.weights:
        np.testing.assert_array_equal(native.weights[key], old.weights[key])
    assert native.metadata["resolved_settings"]["accretion"]["model"] == model


def test_native_effective_provenance_roundtrips(tmp_path):
    from itamae.types import WeightedSubhaloCatalog

    catalog = small().population(**inputs())
    path = tmp_path / "native.npz"
    catalog.to_npz(path)
    restored = WeightedSubhaloCatalog.from_npz(path)
    assert restored.metadata == catalog.metadata
    for name in catalog.weights:
        np.testing.assert_array_equal(restored.weights[name], catalog.weights[name])


@pytest.mark.parametrize(
    "options", [{"h0": 0.01}, {"mxordn": 0}, {"mxords": 0}, {"hmin": 0.2, "hmax": 0.1}]
)
def test_incompatible_ode_controls_fail_before_execution(options):
    with pytest.raises(ValueError):
        CDM().configure(stripping={"solver": "odeint", "solver_options": options})
