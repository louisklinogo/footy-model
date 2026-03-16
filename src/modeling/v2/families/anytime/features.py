from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling.v2.db_reuse_features import add_db_reuse_context_features


def _ensure_numeric_column(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        frame[col] = np.nan
    return pd.to_numeric(frame[col], errors="coerce")


def build_anytime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_db_reuse_context_features(df.copy(), include_anytime_player_context=True)

    home_adj = _ensure_numeric_column(out, "adj_lambda_home_final")
    away_adj = _ensure_numeric_column(out, "adj_lambda_away_final")
    home_l1 = _ensure_numeric_column(out, "lambda_home_l1")
    away_l1 = _ensure_numeric_column(out, "lambda_away_l1")
    home_xg = _ensure_numeric_column(out, "home_rolling_xg")
    away_xg = _ensure_numeric_column(out, "away_rolling_xg")

    out["adj_lambda_home_final_is_missing"] = home_adj.isna().astype(float)
    out["adj_lambda_away_final_is_missing"] = away_adj.isna().astype(float)

    home_resolved = home_adj.where(home_adj > 0.0, np.nan)
    away_resolved = away_adj.where(away_adj > 0.0, np.nan)
    home_fallback = home_resolved.isna()
    away_fallback = away_resolved.isna()

    home_resolved = home_resolved.fillna(home_l1.where(home_l1 > 0.0, np.nan))
    away_resolved = away_resolved.fillna(away_l1.where(away_l1 > 0.0, np.nan))
    home_resolved = home_resolved.fillna(home_xg.where(home_xg > 0.0, np.nan))
    away_resolved = away_resolved.fillna(away_xg.where(away_xg > 0.0, np.nan))

    # Conservative global fallback for rare rows with no lambda/xG anchors.
    out["adj_lambda_home_resolved"] = home_resolved.fillna(1.25).astype(float)
    out["adj_lambda_away_resolved"] = away_resolved.fillna(1.05).astype(float)
    out["adj_lambda_home_resolved_is_fallback"] = home_fallback.astype(float)
    out["adj_lambda_away_resolved_is_fallback"] = away_fallback.astype(float)

    for col in (
        "home_rolling_xg_p1",
        "away_rolling_xg_p1",
        "home_rolling_xg_p1_against",
        "away_rolling_xg_p1_against",
        "home_rolling_xg_h2_delta",
        "away_rolling_xg_h2_delta",
        "home_rolling_sot_p1",
        "away_rolling_sot_p1",
        "home_rolling_sot_p1_against",
        "away_rolling_sot_p1_against",
        "home_rolling_sot_h2_delta",
        "away_rolling_sot_h2_delta",
        "home_rolling_sot_h2_delta_against",
        "away_rolling_sot_h2_delta_against",
        "home_rolling_lead_rate_1up",
        "away_rolling_lead_rate_1up",
        "home_rolling_lead_rate_1up_against",
        "away_rolling_lead_rate_1up_against",
        "home_rolling_lead_rate_2up",
        "away_rolling_lead_rate_2up",
        "home_rolling_lead_rate_2up_against",
        "away_rolling_lead_rate_2up_against",
    ):
        s = _ensure_numeric_column(out, col)
        out[f"{col}_is_missing"] = s.isna().astype(float)

    out["home_xg_phase1_share"] = np.where(
        home_xg > 0.0,
        _ensure_numeric_column(out, "home_rolling_xg_p1") / home_xg,
        np.nan,
    )
    out["away_xg_phase1_share"] = np.where(
        away_xg > 0.0,
        _ensure_numeric_column(out, "away_rolling_xg_p1") / away_xg,
        np.nan,
    )
    out["home_xg_phase2_est"] = (
        _ensure_numeric_column(out, "home_rolling_xg_p1")
        + _ensure_numeric_column(out, "home_rolling_xg_h2_delta")
    )
    out["away_xg_phase2_est"] = (
        _ensure_numeric_column(out, "away_rolling_xg_p1")
        + _ensure_numeric_column(out, "away_rolling_xg_h2_delta")
    )
    out["home_sot_phase2_est"] = (
        _ensure_numeric_column(out, "home_rolling_sot_p1")
        + _ensure_numeric_column(out, "home_rolling_sot_h2_delta")
    )
    out["away_sot_phase2_est"] = (
        _ensure_numeric_column(out, "away_rolling_sot_p1")
        + _ensure_numeric_column(out, "away_rolling_sot_h2_delta")
    )
    out["phase1_share_gap"] = (
        pd.to_numeric(out["home_xg_phase1_share"], errors="coerce")
        - pd.to_numeric(out["away_xg_phase1_share"], errors="coerce")
    )
    out["lead_rate_1up_gap"] = (
        _ensure_numeric_column(out, "home_rolling_lead_rate_1up")
        - _ensure_numeric_column(out, "away_rolling_lead_rate_1up")
    )
    out["lead_rate_2up_gap"] = (
        _ensure_numeric_column(out, "home_rolling_lead_rate_2up")
        - _ensure_numeric_column(out, "away_rolling_lead_rate_2up")
    )
    out["lead_rate_1up_net"] = (
        _ensure_numeric_column(out, "home_rolling_lead_rate_1up")
        - _ensure_numeric_column(out, "away_rolling_lead_rate_1up_against")
    )
    out["lead_rate_2up_net"] = (
        _ensure_numeric_column(out, "home_rolling_lead_rate_2up")
        - _ensure_numeric_column(out, "away_rolling_lead_rate_2up_against")
    )
    out["xg_p1_delta"] = (
        _ensure_numeric_column(out, "home_rolling_xg_p1")
        - _ensure_numeric_column(out, "away_rolling_xg_p1")
    )
    out["sot_p1_delta"] = (
        _ensure_numeric_column(out, "home_rolling_sot_p1")
        - _ensure_numeric_column(out, "away_rolling_sot_p1")
    )
    out["h2_surge_delta"] = (
        _ensure_numeric_column(out, "home_rolling_xg_h2_delta")
        - _ensure_numeric_column(out, "away_rolling_xg_h2_delta")
    )

    # DB-reuse wave 1 player / lineup context.
    for col in (
        "home_missing_market_value_total",
        "away_missing_market_value_total",
        "home_missing_market_value_attack",
        "away_missing_market_value_attack",
        "home_missing_market_value_share",
        "away_missing_market_value_share",
        "home_top1_attack_xg_share",
        "away_top1_attack_xg_share",
        "home_top2_attack_xg_share",
        "away_top2_attack_xg_share",
        "home_top2_attack_xga_share",
        "away_top2_attack_xga_share",
        "home_top2_attack_xg_sum",
        "away_top2_attack_xg_sum",
        "home_bench_attack_form_xga_sum",
        "away_bench_attack_form_xga_sum",
        "home_bench_attack_market_value",
        "away_bench_attack_market_value",
        "home_availability_refresh_hours_dbreuse",
        "away_availability_refresh_hours_dbreuse",
        "home_listed_player_count",
        "away_listed_player_count",
        "home_starter_known_count",
        "away_starter_known_count",
        "home_bench_known_count",
        "away_bench_known_count",
        "home_missing_known_count",
        "away_missing_known_count",
    ):
        s = _ensure_numeric_column(out, col)
        out[f"{col}_is_missing"] = s.isna().astype(float)

    out["missing_market_value_total_gap"] = (
        _ensure_numeric_column(out, "home_missing_market_value_total")
        - _ensure_numeric_column(out, "away_missing_market_value_total")
    )
    out["missing_market_value_attack_gap"] = (
        _ensure_numeric_column(out, "home_missing_market_value_attack")
        - _ensure_numeric_column(out, "away_missing_market_value_attack")
    )
    out["missing_market_value_share_gap"] = (
        _ensure_numeric_column(out, "home_missing_market_value_share")
        - _ensure_numeric_column(out, "away_missing_market_value_share")
    )
    out["top1_attack_xg_share_gap"] = (
        _ensure_numeric_column(out, "home_top1_attack_xg_share")
        - _ensure_numeric_column(out, "away_top1_attack_xg_share")
    )
    out["top2_attack_xg_share_gap"] = (
        _ensure_numeric_column(out, "home_top2_attack_xg_share")
        - _ensure_numeric_column(out, "away_top2_attack_xg_share")
    )
    out["top2_attack_xga_share_gap"] = (
        _ensure_numeric_column(out, "home_top2_attack_xga_share")
        - _ensure_numeric_column(out, "away_top2_attack_xga_share")
    )
    out["top2_attack_xg_sum_gap"] = (
        _ensure_numeric_column(out, "home_top2_attack_xg_sum")
        - _ensure_numeric_column(out, "away_top2_attack_xg_sum")
    )
    out["bench_attack_form_xga_gap"] = (
        _ensure_numeric_column(out, "home_bench_attack_form_xga_sum")
        - _ensure_numeric_column(out, "away_bench_attack_form_xga_sum")
    )
    out["bench_attack_market_value_gap"] = (
        _ensure_numeric_column(out, "home_bench_attack_market_value")
        - _ensure_numeric_column(out, "away_bench_attack_market_value")
    )
    out["lineup_completeness_home"] = (
        _ensure_numeric_column(out, "home_listed_player_count") / 11.0
    ).clip(lower=0.0, upper=2.0)
    out["lineup_completeness_away"] = (
        _ensure_numeric_column(out, "away_listed_player_count") / 11.0
    ).clip(lower=0.0, upper=2.0)
    out["lineup_completeness_gap"] = (
        out["lineup_completeness_home"] - out["lineup_completeness_away"]
    )
    out["lineup_known_share_home"] = np.where(
        _ensure_numeric_column(out, "home_listed_player_count") > 0.0,
        (
            _ensure_numeric_column(out, "home_starter_known_count")
            + _ensure_numeric_column(out, "home_bench_known_count")
        )
        / _ensure_numeric_column(out, "home_listed_player_count"),
        np.nan,
    )
    out["lineup_known_share_away"] = np.where(
        _ensure_numeric_column(out, "away_listed_player_count") > 0.0,
        (
            _ensure_numeric_column(out, "away_starter_known_count")
            + _ensure_numeric_column(out, "away_bench_known_count")
        )
        / _ensure_numeric_column(out, "away_listed_player_count"),
        np.nan,
    )
    out["lineup_known_share_gap"] = (
        _ensure_numeric_column(out, "lineup_known_share_home")
        - _ensure_numeric_column(out, "lineup_known_share_away")
    )

    return out
