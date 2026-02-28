from __future__ import annotations

import pytest

from src.modeling.layer2_situational.rule_layer import (
    apply_rule_adjustment,
    rule_adjustment_for_side,
)


BASE_SAFETY_CONFIG = {
    "safety_enabled_for_disabled_leagues": True,
    "safety_max_down_pct": 0.06,
    "safety_max_up_pct": 0.02,
    "safety_upcoming_tier_2_pct": -0.015,
    "safety_upcoming_tier_3_pct": -0.03,
    "safety_key_absent_pct": -0.04,
    "safety_congestion_threshold_games_14d": 3,
    "safety_congestion_pct": -0.015,
    "safety_rest_disadvantage_threshold_days": -2.5,
    "safety_rest_disadvantage_pct": -0.015,
    "safety_top4_pressure_pct": -0.012,
    "safety_table_adjacent_position_gap_max": 2,
    "safety_table_adjacent_points_gap_max": 6,
    "safety_table_adjacent_pct": -0.01,
    "safety_table_adjacent_only_if_weaker": True,
    "safety_require_odds_confirmation": True,
    "safety_allow_without_odds": False,
    "safety_odds_min_abs_gap": 0.0,
}


def test_safety_scope_applies_capped_adjustment_when_odds_confirm() -> None:
    row = {
        "home_upcoming_tier": 3,
        "home_key_absent": 1,
        "home_recent": 3,
        "rest_delta": -3.0,
        "home_playing_top4": 1,
        "position_gap": -1,
        "points_gap": -2,
        "odds_model_gap_home": -0.08,
    }

    out = rule_adjustment_for_side(
        row=row,
        side="home",
        config=BASE_SAFETY_CONFIG,
        scope="disabled_league_safety",
    )

    assert out["scope"] == "disabled_league_safety"
    assert out["mode"] == "safety"
    assert set(out["rules"]) == {
        "upcoming_tier_3",
        "key_absent",
        "congestion",
        "rest_disadvantage",
        "top4_pressure",
        "table_adjacent_volatility",
    }
    assert out["odds_confirmed"] is True
    assert out["odds_conflict"] is False
    assert out["odds_gate_blocked"] is False
    assert out["pct_raw"] == pytest.approx(-0.122)
    assert out["pct_capped_pre_gate"] == pytest.approx(-0.06)
    assert out["pct_capped"] == pytest.approx(-0.06)
    assert out["applied"] is True


def test_safety_scope_blocks_when_odds_conflict() -> None:
    row = {
        "home_upcoming_tier": 3,
        "home_key_absent": 1,
        "home_recent": 3,
        "rest_delta": -3.0,
        "home_playing_top4": 1,
        "position_gap": -1,
        "points_gap": -2,
        "odds_model_gap_home": 0.03,
    }

    out = rule_adjustment_for_side(
        row=row,
        side="home",
        config=BASE_SAFETY_CONFIG,
        scope="disabled_league_safety",
    )

    assert out["pct_capped_pre_gate"] == pytest.approx(-0.06)
    assert out["pct_capped"] == pytest.approx(0.0)
    assert out["applied"] is False
    assert out["odds_conflict"] is True
    assert out["odds_gate_blocked"] is True


def test_apply_rule_adjustment_safety_updates_lambda() -> None:
    row = {
        "home_key_absent": 1,
        "odds_model_gap_home": -0.02,
    }
    cfg = dict(BASE_SAFETY_CONFIG)
    cfg["safety_upcoming_tier_2_pct"] = 0.0
    cfg["safety_upcoming_tier_3_pct"] = 0.0
    cfg["safety_congestion_pct"] = 0.0
    cfg["safety_rest_disadvantage_pct"] = 0.0
    cfg["safety_top4_pressure_pct"] = 0.0
    cfg["safety_table_adjacent_pct"] = 0.0

    out = apply_rule_adjustment(
        base_lambda=1.5,
        row=row,
        side="home",
        config=cfg,
        scope="disabled_league_safety",
    )

    assert out["pct_capped"] == pytest.approx(-0.04)
    assert out["lambda_after"] == pytest.approx(1.44)
