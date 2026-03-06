import numpy as np

from src.modeling.layer2_markets.market_outcome_calibrator import (
    _mean_binary_brier_for_markets,
)


def test_mean_binary_brier_for_markets_prefers_better_multiclass_probabilities():
    y_class = np.array([0, 1, 2, 0, 2, 1], dtype=int)
    markets = {"1x2_h", "1x2_d", "1x2_a", "dc_1x", "dc_x2", "dc_12"}

    p_home_good = np.array([0.80, 0.10, 0.10, 0.75, 0.15, 0.10], dtype=float)
    p_draw_good = np.array([0.10, 0.80, 0.10, 0.10, 0.10, 0.75], dtype=float)
    p_away_good = np.array([0.10, 0.10, 0.80, 0.15, 0.75, 0.15], dtype=float)
    good_score = _mean_binary_brier_for_markets(
        y_class=y_class,
        p_home=p_home_good,
        p_draw=p_draw_good,
        p_away=p_away_good,
        markets=markets,
    )

    p_home_bad = np.array([0.34, 0.34, 0.34, 0.34, 0.34, 0.34], dtype=float)
    p_draw_bad = np.array([0.33, 0.33, 0.33, 0.33, 0.33, 0.33], dtype=float)
    p_away_bad = np.array([0.33, 0.33, 0.33, 0.33, 0.33, 0.33], dtype=float)
    bad_score = _mean_binary_brier_for_markets(
        y_class=y_class,
        p_home=p_home_bad,
        p_draw=p_draw_bad,
        p_away=p_away_bad,
        markets=markets,
    )

    assert np.isfinite(good_score)
    assert np.isfinite(bad_score)
    assert float(good_score) < float(bad_score)
