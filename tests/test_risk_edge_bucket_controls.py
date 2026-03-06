from __future__ import annotations

from datetime import datetime, timezone

from src.modeling.evaluation.assess_prediction_risk import (
    CalibrationStats,
    EdgeBucketPerformance,
    RollingPerformance,
    assess_row,
)


def test_assess_row_edge_bucket_precision_gate_forces_pass() -> None:
    now = datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc)
    row = {
        "prediction_id": 101,
        "fixture_id": 202,
        "market_code": "o15",
        "model_name": "market_outcome_gbm",
        "model_version": "fixtures_first_prematch_v1",
        "p_model": 0.56,
        "metadata_json": {
            "features_missing_count": 0,
            "fallback_used": False,
            "home_sample_size": 12,
            "away_sample_size": 12,
        },
        "league_code": "ENG",
        "match_datetime_utc": now.isoformat(),
        "odds_rows": [
            {
                "provider": "sofascore",
                "snapshot_type": "latest_pre_match",
                "snapshot_time_utc": now.isoformat(),
                "market_code": "ou",
                "line_num": 1.5,
                "odds_json": {"market_code": "ou", "line_num": 1.5},
                "prices_latest": {"over": 1.9, "under": 1.9},
            }
        ],
    }
    policy = {
        "min_edge_watch": 0.01,
        "min_edge_bet_small": 0.02,
        "min_edge_bet": 0.04,
        "min_precision_rolling": 0.4,
        "precision_min_samples": 80,
        "loss_streak_pause": 3,
        "hard_precision_gate": False,
        "hard_loss_streak_pause": False,
        "tradable_markets": ["o15"],
        "calibration_min_samples": 20,
        "calibration_poor_brier": 0.22,
        "calibration_very_poor_brier": 0.24,
        "kelly_scale": 0.25,
        "max_stake_fraction": 0.02,
        "max_small_stake_fraction": 0.01,
        "edge_bucket_min_samples": 20,
        "edge_bucket_min_precision": 0.54,
        "edge_bucket_loss_streak_pause": 3,
        "hard_edge_bucket_precision_gate": True,
        "hard_edge_bucket_loss_streak_pause": True,
        "penalties": {
            "missing_odds": 12.0,
            "fallback_used": 18.0,
            "feature_missing_unit": 0.75,
            "stale_odds_6h": 2.0,
            "stale_odds_24h": 4.0,
            "low_sample_lt8": 10.0,
            "low_sample_lt5": 18.0,
            "calibration_low_sample": 8.0,
            "calibration_poor": 10.0,
            "calibration_very_poor": 16.0,
            "precision_gate_fail": 25.0,
            "loss_streak_pause": 30.0,
            "edge_bucket_precision_gate_fail": 20.0,
            "edge_bucket_loss_streak_pause": 24.0,
        },
    }

    out = assess_row(
        row,
        policy=policy,
        tradable_summary={},
        calibration_by_league_market={("ENG", "o15"): CalibrationStats(n=100, mean_brier=0.18)},
        calibration_by_market={"o15": CalibrationStats(n=100, mean_brier=0.18)},
        performance_by_league_market={
            ("ENG", "o15"): RollingPerformance(n=120, precision=0.62, loss_streak=0)
        },
        performance_by_market={"o15": RollingPerformance(n=120, precision=0.62, loss_streak=0)},
        edge_bucket_by_market={
            ("o15", "2-4%"): EdgeBucketPerformance(
                n=90, precision=0.40, loss_streak=1, mean_roi_unit=-0.08
            )
        },
        edge_bucket_global={},
    )

    assert out["action"] == "pass"
    assert "edge_bucket_precision_gate_fail" in out["risk_flags_json"]
    assert out["metadata_json"]["edge_bucket"] == "2-4%"


def test_assess_row_edge_bucket_market_override_is_applied() -> None:
    now = datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc)
    row = {
        "prediction_id": 111,
        "fixture_id": 222,
        "market_code": "o15",
        "model_name": "market_outcome_gbm",
        "model_version": "fixtures_first_prematch_v1",
        "p_model": 0.56,
        "metadata_json": {
            "features_missing_count": 0,
            "fallback_used": False,
            "home_sample_size": 12,
            "away_sample_size": 12,
        },
        "league_code": "ENG",
        "match_datetime_utc": now.isoformat(),
        "odds_rows": [
            {
                "provider": "sofascore",
                "snapshot_type": "latest_pre_match",
                "snapshot_time_utc": now.isoformat(),
                "market_code": "ou",
                "line_num": 1.5,
                "odds_json": {"market_code": "ou", "line_num": 1.5},
                "prices_latest": {"over": 1.9, "under": 1.9},
            }
        ],
    }
    policy = {
        "min_edge_watch": 0.01,
        "min_edge_bet_small": 0.02,
        "min_edge_bet": 0.04,
        "min_precision_rolling": 0.4,
        "precision_min_samples": 80,
        "loss_streak_pause": 3,
        "hard_precision_gate": False,
        "hard_loss_streak_pause": False,
        "tradable_markets": ["o15"],
        "calibration_min_samples": 20,
        "calibration_poor_brier": 0.22,
        "calibration_very_poor_brier": 0.24,
        "kelly_scale": 0.25,
        "max_stake_fraction": 0.02,
        "max_small_stake_fraction": 0.01,
        "edge_bucket_min_samples": 20,
        "edge_bucket_min_precision": 0.54,
        "edge_bucket_loss_streak_pause": 3,
        "hard_edge_bucket_precision_gate": True,
        "hard_edge_bucket_loss_streak_pause": True,
        "edge_bucket_overrides": {
            "o15": {
                "2-4%": {
                    "min_samples": 20,
                    "min_precision": 0.72,
                    "loss_streak_pause": 2,
                    "hard_precision_gate": True,
                    "hard_loss_streak_pause": True,
                }
            }
        },
        "penalties": {
            "missing_odds": 12.0,
            "fallback_used": 18.0,
            "feature_missing_unit": 0.75,
            "stale_odds_6h": 2.0,
            "stale_odds_24h": 4.0,
            "low_sample_lt8": 10.0,
            "low_sample_lt5": 18.0,
            "calibration_low_sample": 8.0,
            "calibration_poor": 10.0,
            "calibration_very_poor": 16.0,
            "precision_gate_fail": 25.0,
            "loss_streak_pause": 30.0,
            "edge_bucket_precision_gate_fail": 20.0,
            "edge_bucket_loss_streak_pause": 24.0,
        },
    }

    out = assess_row(
        row,
        policy=policy,
        tradable_summary={},
        calibration_by_league_market={("ENG", "o15"): CalibrationStats(n=100, mean_brier=0.18)},
        calibration_by_market={"o15": CalibrationStats(n=100, mean_brier=0.18)},
        performance_by_league_market={
            ("ENG", "o15"): RollingPerformance(n=120, precision=0.62, loss_streak=0)
        },
        performance_by_market={"o15": RollingPerformance(n=120, precision=0.62, loss_streak=0)},
        edge_bucket_by_market={
            ("o15", "2-4%"): EdgeBucketPerformance(
                n=90, precision=0.70, loss_streak=1, mean_roi_unit=-0.02
            )
        },
        edge_bucket_global={},
    )

    assert out["action"] == "pass"
    assert "edge_bucket_precision_gate_fail" in out["risk_flags_json"]
    gate = out["metadata_json"]["edge_bucket_gate"]
    assert float(gate["min_precision"]) == 0.72
