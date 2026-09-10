import numpy as np
from itamae.execution import PopulationComponents

from sashimi_c_itamae import subhalo_properties
from sashimi_c_itamae_components import (
    CDMAccretionSlices,
    CDMCatalogColumns,
    NFWInitialStructure,
    TidalProfileEvolution,
    TruncationThresholdSurvival,
)


def test_truncation_threshold_survival_preserves_strict_legacy_cut():
    component = TruncationThresholdSurvival(ct_threshold=0.1)
    c_t = np.array([0.0, 0.1, np.nextafter(0.1, np.inf), 0.2, np.nan])

    mask = component.select(
        batch=None,
        initial={},
        evolved={"c_t": c_t},
        context=None,
    )

    np.testing.assert_array_equal(mask, [False, False, True, True, False])
    assert mask.dtype == np.bool_


def test_cdm_stages_preserve_node_identity_and_initial_enclosed_mass():
    model = subhalo_properties()
    mass = np.geomspace(1.0e6, 1.0e8, 3)
    batch, context = CDMAccretionSlices(
        model=model,
        ma200_grid=mass,
        zdist=np.array([0.5]),
        population_2d=np.array([[1.0, 2.0, 3.0]]),
        sigmalogc=0.128,
        N_herm=2,
    ).build(0)
    initializer = NFWInitialStructure(model=model, N_herm=2)
    initial = initializer.initialize(batch, context)
    enclosed = (
        4.0
        * np.pi
        * initial["rho_s_acc"]
        * initial["r_s_acc"] ** 3
        * (
            np.log1p(batch.concentration_acc)
            - batch.concentration_acc / (1.0 + batch.concentration_acc)
        )
    )
    np.testing.assert_allclose(enclosed, np.tile(context["mvir_acc"], 2), rtol=2e-14)
    assert "m_bound" not in initial

    class HalfMassSolver:
        def subhalo_mass_stripped(self, mass, za, redshift, *, method):
            assert za == 0.5 and redshift == 0.0 and method == "test-half"
            return 0.5 * mass

    result = PopulationComponents(
        initializer=initializer,
        evolver=TidalProfileEvolution(
            model=model,
            solver=HalfMassSolver(),
            redshift=0.0,
            method="test-half",
            N_herm=2,
            profile_change=False,
            kwargs={},
        ),
        survival=TruncationThresholdSurvival(0.0),
        columns=CDMCatalogColumns(),
    ).execute([batch], contexts=[context])
    np.testing.assert_array_equal(result.columns["m200_acc"], np.tile(mass, 2))
    np.testing.assert_array_equal(result.columns["r_s"], initial["r_s_acc"])
    np.testing.assert_array_equal(result.columns["rho_s"], initial["rho_s_acc"])
    np.testing.assert_array_equal(
        result.columns["m_bound"], np.tile(context["mvir_acc"] / 2.0, 2)
    )
    recovered = (
        4.0
        * np.pi
        * result.columns["rho_s"]
        * result.columns["r_s"] ** 3
        * (
            np.log1p(result.columns["c_t"])
            - result.columns["c_t"] / (1.0 + result.columns["c_t"])
        )
    )
    np.testing.assert_allclose(recovered, result.columns["m_bound"], rtol=2e-12)
    assert np.all(result.survival["default"])
