import numpy as np

from sashimi_c_itamae_components import TruncationThresholdSurvival


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
