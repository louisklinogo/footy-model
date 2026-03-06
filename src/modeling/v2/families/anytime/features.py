from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_numeric_column(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        frame[col] = np.nan
    return pd.to_numeric(frame[col], errors="coerce")


def build_anytime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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

    return out
