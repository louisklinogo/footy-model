from __future__ import annotations

import numpy as np
import pytest

from src.modeling.v2.families.scoreline.derive_markets import (
    assert_probability_identities,
    derive_and_validate,
    derive_markets_from_score_matrix,
)


def test_derive_scoreline_market_identities_hold() -> None:
    mat = np.array(
        [
            [0.08, 0.07, 0.03, 0.01],
            [0.11, 0.12, 0.07, 0.03],
            [0.09, 0.11, 0.09, 0.05],
            [0.04, 0.05, 0.03, 0.03],
        ],
        dtype=float,
    )
    probs = derive_and_validate(mat)
    assert abs(probs["1x2_h"] + probs["1x2_d"] + probs["1x2_a"] - 1.0) < 1e-9
    assert abs(probs["dc_12"] - (probs["1x2_h"] + probs["1x2_a"])) < 1e-9
    assert abs(
        probs["ms_h_1_0_2_0_3_0"]
        + probs["ms_a_0_1_0_2_0_3"]
        + probs["ms_h_4_0_5_0_6_0"]
        + probs["ms_a_0_4_0_5_0_6"]
        + probs["ms_h_2_1_3_1_4_1"]
        + probs["ms_h_1_2_1_3_1_4"]
        + probs["ms_h_3_2_4_2_5_1"]
        + probs["ms_a_2_3_2_4_1_5"]
        + probs["ms_other_homewin"]
        + probs["ms_other_awaywin"]
        + probs["ms_draw"]
        - 1.0
    ) < 1e-9


def test_derive_scoreline_market_ranges() -> None:
    mat = np.full((5, 5), 1.0, dtype=float)
    probs = derive_markets_from_score_matrix(mat)
    for value in probs.values():
        assert 0.0 <= value <= 1.0
    assert "mg_1_2" in probs and "hmg_1_2" in probs and "ms_draw" in probs
    assert probs["mg_1_2"] <= probs["mg_1_3"] + 1e-9
    assert probs["mg_1_3"] <= probs["mg_1_4"] + 1e-9
    assert probs["hmg_1_2"] <= probs["hmg_1_3"] + 1e-9
    assert probs["amg_1_2"] <= probs["amg_1_3"] + 1e-9


def test_derive_scoreline_negative_probs_raise() -> None:
    mat = np.array([[0.4, -0.1], [0.2, 0.5]], dtype=float)
    with pytest.raises(ValueError):
        derive_markets_from_score_matrix(mat)


def test_identity_validator_rejects_broken_payload() -> None:
    bad = {
        "1x2_h": 0.2,
        "1x2_d": 0.2,
        "1x2_a": 0.2,
        "dc_1x": 0.4,
        "dc_x2": 0.4,
        "dc_12": 0.3,
        "o15": 0.7,
        "u35": 0.6,
        "ah_h05": 0.2,
        "ah_a05": 0.2,
        "ah_h15": 0.1,
        "ah_a15": 0.1,
        "eh_h1": 0.1,
        "eh_a1": 0.1,
    }
    with pytest.raises(ValueError):
        assert_probability_identities(bad)
