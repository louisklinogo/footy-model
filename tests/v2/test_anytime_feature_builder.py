from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import patch

from src.modeling.v2.families.anytime.features import build_anytime_features


def test_build_anytime_features_resolves_adj_lambda_fallbacks() -> None:
    frame = pd.DataFrame(
        [
            {
                "adj_lambda_home_final": None,
                "adj_lambda_away_final": None,
                "lambda_home_l1": 1.6,
                "lambda_away_l1": 1.1,
                "home_rolling_xg": 1.4,
                "away_rolling_xg": 1.0,
            },
            {
                "adj_lambda_home_final": 1.8,
                "adj_lambda_away_final": 0.9,
                "lambda_home_l1": 1.5,
                "lambda_away_l1": 1.2,
                "home_rolling_xg": 1.3,
                "away_rolling_xg": 1.1,
            },
        ]
    )
    out = build_anytime_features(frame)

    assert out.loc[0, "adj_lambda_home_resolved"] == 1.6
    assert out.loc[0, "adj_lambda_away_resolved"] == 1.1
    assert out.loc[0, "adj_lambda_home_resolved_is_fallback"] == 1.0
    assert out.loc[0, "adj_lambda_away_resolved_is_fallback"] == 1.0

    assert out.loc[1, "adj_lambda_home_resolved"] == 1.8
    assert out.loc[1, "adj_lambda_away_resolved"] == 0.9
    assert out.loc[1, "adj_lambda_home_resolved_is_fallback"] == 0.0
    assert out.loc[1, "adj_lambda_away_resolved_is_fallback"] == 0.0


def test_build_anytime_features_adds_missing_indicators() -> None:
    frame = pd.DataFrame([{"home_rolling_xg": 1.2, "away_rolling_xg": 1.1}])
    out = build_anytime_features(frame)
    assert out.loc[0, "home_rolling_xg_p1_is_missing"] == 1.0
    assert out.loc[0, "away_rolling_xg_h2_delta_is_missing"] == 1.0


def test_build_anytime_features_adds_phase_and_lead_rate_derivatives() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_rolling_xg": 1.8,
                "away_rolling_xg": 1.2,
                "home_rolling_xg_p1": 0.9,
                "away_rolling_xg_p1": 0.3,
                "home_rolling_xg_h2_delta": 0.4,
                "away_rolling_xg_h2_delta": 0.2,
                "home_rolling_sot_p1": 2.0,
                "away_rolling_sot_p1": 1.0,
                "home_rolling_sot_h2_delta": 1.5,
                "away_rolling_sot_h2_delta": 0.4,
                "home_rolling_lead_rate_1up": 0.42,
                "away_rolling_lead_rate_1up": 0.24,
                "home_rolling_lead_rate_1up_against": 0.18,
                "away_rolling_lead_rate_1up_against": 0.31,
                "home_rolling_lead_rate_2up": 0.17,
                "away_rolling_lead_rate_2up": 0.08,
                "home_rolling_lead_rate_2up_against": 0.06,
                "away_rolling_lead_rate_2up_against": 0.12,
            }
        ]
    )
    out = build_anytime_features(frame)

    assert out.loc[0, "home_xg_phase1_share"] == 0.5
    assert out.loc[0, "away_xg_phase1_share"] == 0.25
    assert out.loc[0, "home_xg_phase2_est"] == 1.3
    assert out.loc[0, "away_xg_phase2_est"] == 0.5
    assert out.loc[0, "lead_rate_1up_gap"] == pytest.approx(0.18)
    assert out.loc[0, "lead_rate_2up_gap"] == pytest.approx(0.09)
    assert out.loc[0, "lead_rate_1up_net"] == pytest.approx(0.11)
    assert out.loc[0, "lead_rate_2up_net"] == pytest.approx(0.05)


def test_build_anytime_features_adds_db_reuse_player_context_derivatives() -> None:
    frame = pd.DataFrame(
        [
            {
                "fixture_id": 1001,
                "home_rolling_xg": 1.8,
                "away_rolling_xg": 1.2,
                "lambda_home_l1": 1.5,
                "lambda_away_l1": 1.0,
            }
        ]
    )
    enriched = frame.assign(
        home_missing_market_value_total=40_000_000,
        away_missing_market_value_total=15_000_000,
        home_missing_market_value_attack=22_000_000,
        away_missing_market_value_attack=6_000_000,
        home_missing_market_value_share=0.24,
        away_missing_market_value_share=0.10,
        home_top1_attack_xg_share=0.41,
        away_top1_attack_xg_share=0.28,
        home_top2_attack_xg_share=0.72,
        away_top2_attack_xg_share=0.55,
        home_top2_attack_xga_share=0.77,
        away_top2_attack_xga_share=0.61,
        home_top2_attack_xg_sum=0.84,
        away_top2_attack_xg_sum=0.53,
        home_bench_attack_form_xga_sum=0.30,
        away_bench_attack_form_xga_sum=0.18,
        home_bench_attack_market_value=9_000_000,
        away_bench_attack_market_value=4_000_000,
        home_listed_player_count=16,
        away_listed_player_count=15,
        home_starter_known_count=11,
        away_starter_known_count=10,
        home_bench_known_count=5,
        away_bench_known_count=5,
        home_missing_known_count=2,
        away_missing_known_count=1,
        home_availability_refresh_hours_dbreuse=3.0,
        away_availability_refresh_hours_dbreuse=5.0,
    )

    with patch(
        "src.modeling.v2.families.anytime.features.add_db_reuse_context_features",
        return_value=enriched,
    ):
        out = build_anytime_features(frame)

    assert out.loc[0, "missing_market_value_total_gap"] == 25_000_000
    assert out.loc[0, "missing_market_value_attack_gap"] == 16_000_000
    assert out.loc[0, "missing_market_value_share_gap"] == pytest.approx(0.14)
    assert out.loc[0, "top1_attack_xg_share_gap"] == pytest.approx(0.13)
    assert out.loc[0, "top2_attack_xg_share_gap"] == pytest.approx(0.17)
    assert out.loc[0, "top2_attack_xga_share_gap"] == pytest.approx(0.16)
    assert out.loc[0, "top2_attack_xg_sum_gap"] == pytest.approx(0.31)
    assert out.loc[0, "bench_attack_form_xga_gap"] == pytest.approx(0.12)
    assert out.loc[0, "bench_attack_market_value_gap"] == 5_000_000
    assert out.loc[0, "lineup_completeness_home"] == pytest.approx(16 / 11)
    assert out.loc[0, "lineup_completeness_gap"] == pytest.approx((16 / 11) - (15 / 11))
    assert out.loc[0, "lineup_known_share_home"] == pytest.approx(1.0)
    assert out.loc[0, "lineup_known_share_gap"] == pytest.approx(1.0 - 1.0)
