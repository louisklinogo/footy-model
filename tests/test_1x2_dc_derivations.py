import numpy as np

from src.modeling.layer2_markets.market_outcome_calibrator import (
    _binary_prob_for_market,
    _binary_target_for_market,
)


def test_1x2_dc_binary_targets_are_derived_from_three_class_outcome():
    y_class = np.array([0, 1, 2, 0, 2, 1], dtype=int)

    assert np.array_equal(_binary_target_for_market("1x2_h", y_class), np.array([1, 0, 0, 1, 0, 0]))
    assert np.array_equal(_binary_target_for_market("1x2_d", y_class), np.array([0, 1, 0, 0, 0, 1]))
    assert np.array_equal(_binary_target_for_market("1x2_a", y_class), np.array([0, 0, 1, 0, 1, 0]))
    assert np.array_equal(_binary_target_for_market("dc_1x", y_class), np.array([1, 1, 0, 1, 0, 1]))
    assert np.array_equal(_binary_target_for_market("dc_x2", y_class), np.array([0, 1, 1, 0, 1, 1]))
    assert np.array_equal(_binary_target_for_market("dc_12", y_class), np.array([1, 0, 1, 1, 1, 0]))


def test_1x2_dc_binary_probabilities_are_derived_from_class_probabilities():
    p_home = np.array([0.52, 0.25], dtype=float)
    p_draw = np.array([0.18, 0.40], dtype=float)
    p_away = np.array([0.30, 0.35], dtype=float)

    assert np.allclose(_binary_prob_for_market("1x2_h", p_home, p_draw, p_away), p_home)
    assert np.allclose(_binary_prob_for_market("1x2_d", p_home, p_draw, p_away), p_draw)
    assert np.allclose(_binary_prob_for_market("1x2_a", p_home, p_draw, p_away), p_away)
    assert np.allclose(_binary_prob_for_market("dc_1x", p_home, p_draw, p_away), p_home + p_draw)
    assert np.allclose(_binary_prob_for_market("dc_x2", p_home, p_draw, p_away), p_draw + p_away)
    assert np.allclose(_binary_prob_for_market("dc_12", p_home, p_draw, p_away), p_home + p_away)
