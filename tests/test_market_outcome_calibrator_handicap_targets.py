from __future__ import annotations

import pandas as pd

from src.modeling.layer2_markets.market_outcome_calibrator import _add_handicap_target_columns


def test_add_handicap_target_columns_includes_canonical_contract_targets() -> None:
    frame = pd.DataFrame(
        {
            "home_goals": [2, 1, 0],
            "away_goals": [1, 1, 2],
        }
    )

    out = _add_handicap_target_columns(frame.copy())

    assert out["target_eh3_0_1_home"].tolist() == [0, 0, 0]
    assert out["target_eh3_0_1_draw"].tolist() == [1, 0, 0]
    assert out["target_eh3_0_1_away"].tolist() == [0, 1, 1]
    assert out["target_eh3_1_0_home"].tolist() == [1, 1, 0]
    assert out["target_eh3_1_0_draw"].tolist() == [0, 0, 0]
    assert out["target_eh3_1_0_away"].tolist() == [0, 0, 1]
    assert out["target_ah2_away_p05"].tolist() == [0, 1, 1]
    assert out["target_ah2_home_p15"].tolist() == [1, 1, 0]