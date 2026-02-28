"""
Fixtures-first training pipeline for 27 betting markets.

Reads only v1 fixtures-first tables and trains leakage-safe pre-match models
for the following markets:

TOTALS (6): o15, o25, o35, o45, u15, u25
CORNERS (1): c85
BTTS (1): btts
1X2 (3): 1x2_h, 1x2_d, 1x2_a
DOUBLE CHANCE (3): dc_1x, dc_x2, dc_12
TEAM TOTALS (2): ho15, ao15
COMBO OR (4): home_or_o25, away_or_o25, home_or_o15, away_or_o15
COMBO AND (2): home_and_o25, away_and_o25
ANYTIME LEAD (4): h_1up, a_1up, h_2up, a_2up
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


OUT_DIR = ROOT_DIR / "model_artifacts" / "market_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


TEAM_SNAPSHOT_FEATURES = [
    "sample_size",
    "rolling_xg",
    "rolling_xg_against",
    "rolling_xgot",
    "rolling_xgot_against",
    "rolling_xa",
    "rolling_xa_against",
    "rolling_box_touches",
    "rolling_box_touches_against",
    "rolling_big_chances",
    "rolling_big_chances_against",
    "rolling_crosses",
    "rolling_crosses_against",
    "rolling_sot",
    "rolling_sot_against",
    "rolling_corners",
    "rolling_corners_against",
    "rolling_goals_prevented",
    "rolling_goals_prevented_against",
    # New Phase 2 features
    "rolling_xg_p1",
    "rolling_xg_p1_against",
    "rolling_sot_p1",
    "rolling_sot_p1_against",
    "rolling_xg_h2_delta",
    "rolling_xg_h2_delta_against",
    "rolling_sot_h2_delta",
    "rolling_sot_h2_delta_against",
    "rolling_possession",
    "rolling_possession_against",
    "rolling_errors_lead_to_shot",
    "rolling_errors_lead_to_shot_against",
    "rolling_tackles_pct",
    "rolling_tackles_pct_against",
]


def _latest_odds_expr(row_alias: str, side: str) -> str:
    return (
        "CASE "
        f"WHEN ({row_alias}.odds_json -> 'prices_latest' ->> '{side}') ~ '^[-+]?[0-9]*\\.?[0-9]+$' "
        f"THEN ({row_alias}.odds_json -> 'prices_latest' ->> '{side}')::double precision "
        "ELSE NULL END"
    )


def _json_number_expr(json_col: str, key: str) -> str:
    return (
        "CASE "
        f"WHEN ({json_col} ->> '{key}') ~ '^[-+]?[0-9]*\\.?[0-9]+$' "
        f"THEN ({json_col} ->> '{key}')::double precision "
        "ELSE NULL END"
    )


def fetch_dataset() -> pd.DataFrame:
    query = f"""
    SELECT
        f.fixture_id,
        f.league_code,
        f.match_datetime_utc,
        fr.home_goals,
        fr.away_goals,
        (fr.home_goals + fr.away_goals) AS total_goals,
        CASE
            WHEN sp.h_corners IS NOT NULL AND sp.a_corners IS NOT NULL
            THEN (sp.h_corners + sp.a_corners)
            ELSE NULL
        END AS total_corners,
        {_latest_odds_expr('od15', 'over')} AS odds_over_15,
        {_latest_odds_expr('od15', 'under')} AS odds_under_15,
        {_latest_odds_expr('od25', 'over')} AS odds_over_25,
        {_latest_odds_expr('od25', 'under')} AS odds_under_25,
        GREATEST(od15.snapshot_time_utc, od25.snapshot_time_utc) AS odds_snapshot_time_utc,
        CASE
            WHEN od15.snapshot_type = 'latest_pre_match' OR od25.snapshot_type = 'latest_pre_match'
            THEN 'latest_pre_match'
            ELSE COALESCE(od25.snapshot_type, od15.snapshot_type)
        END AS odds_snapshot_type,
        tph.sample_size AS home_sample_size,
        tph.rolling_xg AS home_rolling_xg,
        tph.rolling_xg_against AS home_rolling_xg_against,
        tph.rolling_xgot AS home_rolling_xgot,
        tph.rolling_xgot_against AS home_rolling_xgot_against,
        tph.rolling_xa AS home_rolling_xa,
        tph.rolling_xa_against AS home_rolling_xa_against,
        tph.rolling_box_touches AS home_rolling_box_touches,
        tph.rolling_box_touches_against AS home_rolling_box_touches_against,
        tph.rolling_big_chances AS home_rolling_big_chances,
        tph.rolling_big_chances_against AS home_rolling_big_chances_against,
        tph.rolling_crosses AS home_rolling_crosses,
        tph.rolling_crosses_against AS home_rolling_crosses_against,
        tph.rolling_sot AS home_rolling_sot,
        tph.rolling_sot_against AS home_rolling_sot_against,
        tph.rolling_corners AS home_rolling_corners,
        tph.rolling_corners_against AS home_rolling_corners_against,
        tph.rolling_goals_prevented AS home_rolling_goals_prevented,
        tph.rolling_goals_prevented_against AS home_rolling_goals_prevented_against,
        -- New Home Phase 2
        tph.rolling_xg_p1 AS home_rolling_xg_p1,
        tph.rolling_xg_p1_against AS home_rolling_xg_p1_against,
        tph.rolling_sot_p1 AS home_rolling_sot_p1,
        tph.rolling_sot_p1_against AS home_rolling_sot_p1_against,
        tph.rolling_xg_h2_delta AS home_rolling_xg_h2_delta,
        tph.rolling_xg_h2_delta_against AS home_rolling_xg_h2_delta_against,
        tph.rolling_sot_h2_delta AS home_rolling_sot_h2_delta,
        tph.rolling_sot_h2_delta_against AS home_rolling_sot_h2_delta_against,
        tph.rolling_possession AS home_rolling_possession,
        tph.rolling_possession_against AS home_rolling_possession_against,
        tph.rolling_errors_lead_to_shot AS home_rolling_errors_lead_to_shot,
        tph.rolling_errors_lead_to_shot_against AS home_rolling_errors_lead_to_shot_against,
        tph.rolling_tackles_pct AS home_rolling_tackles_pct,
        tph.rolling_tackles_pct_against AS home_rolling_tackles_pct_against,

        tpa.sample_size AS away_sample_size,
        tpa.rolling_xg AS away_rolling_xg,
        tpa.rolling_xg_against AS away_rolling_xg_against,
        tpa.rolling_xgot AS away_rolling_xgot,
        tpa.rolling_xgot_against AS away_rolling_xgot_against,
        tpa.rolling_xa AS away_rolling_xa,
        tpa.rolling_xa_against AS away_rolling_xa_against,
        tpa.rolling_box_touches AS away_rolling_box_touches,
        tpa.rolling_box_touches_against AS away_rolling_box_touches_against,
        tpa.rolling_big_chances AS away_rolling_big_chances,
        tpa.rolling_big_chances_against AS away_rolling_big_chances_against,
        tpa.rolling_crosses AS away_rolling_crosses,
        tpa.rolling_crosses_against AS away_rolling_crosses_against,
        tpa.rolling_sot AS away_rolling_sot,
        tpa.rolling_sot_against AS away_rolling_sot_against,
        tpa.rolling_corners AS away_rolling_corners,
        tpa.rolling_corners_against AS away_rolling_corners_against,
        tpa.rolling_goals_prevented AS away_rolling_goals_prevented,
        tpa.rolling_goals_prevented_against AS away_rolling_goals_prevented_against,
        -- New Away Phase 2
        tpa.rolling_xg_p1 AS away_rolling_xg_p1,
        tpa.rolling_xg_p1_against AS away_rolling_xg_p1_against,
        tpa.rolling_sot_p1 AS away_rolling_sot_p1,
        tpa.rolling_sot_p1_against AS away_rolling_sot_p1_against,
        tpa.rolling_xg_h2_delta AS away_rolling_xg_h2_delta,
        tpa.rolling_xg_h2_delta_against AS away_rolling_xg_h2_delta_against,
        tpa.rolling_sot_h2_delta AS away_rolling_sot_h2_delta,
        tpa.rolling_sot_h2_delta_against AS away_rolling_sot_h2_delta_against,
        tpa.rolling_possession AS away_rolling_possession,
        tpa.rolling_possession_against AS away_rolling_possession_against,
        tpa.rolling_errors_lead_to_shot AS away_rolling_errors_lead_to_shot,
        tpa.rolling_errors_lead_to_shot_against AS away_rolling_errors_lead_to_shot_against,
        tpa.rolling_tackles_pct AS away_rolling_tackles_pct,
        tpa.rolling_tackles_pct_against AS away_rolling_tackles_pct_against,

        ff.home_formation,
        ff.away_formation,
        {_json_number_expr('l1h.metadata_json', 'lambda')} AS lambda_home_l1,
        {_json_number_expr('l1a.metadata_json', 'lambda')} AS lambda_away_l1,
        {_json_number_expr('l2h.metadata_json', 'lambda')} AS adj_lambda_home_final,
        {_json_number_expr('l2a.metadata_json', 'lambda')} AS adj_lambda_away_final,
        ils.home_led_by_1_any AS home_led_by_1_any,
        ils.away_led_by_1_any AS away_led_by_1_any,
        ils.home_led_by_2_any AS home_led_by_2_any,
        ils.away_led_by_2_any AS away_led_by_2_any,
        CASE
            WHEN lower(COALESCE(l2h.metadata_json ->> 'rule_layer_applied', 'false')) IN ('true', 't', '1')
            THEN 1 ELSE 0
        END AS rule_fired_home,
        CASE
            WHEN lower(COALESCE(l2a.metadata_json ->> 'rule_layer_applied', 'false')) IN ('true', 't', '1')
            THEN 1 ELSE 0
        END AS rule_fired_away
    FROM fixtures f
    JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
    LEFT JOIN fixture_stats_premium sp ON sp.fixture_id = f.fixture_id
    LEFT JOIN fixture_formations ff ON ff.fixture_id = f.fixture_id
    LEFT JOIN fixture_incident_lead_states ils ON ils.fixture_id = f.fixture_id
    LEFT JOIN team_premium_snapshots tph
        ON tph.fixture_id = f.fixture_id
       AND tph.is_home = true
    LEFT JOIN team_premium_snapshots tpa
        ON tpa.fixture_id = f.fixture_id
       AND tpa.is_home = false
    LEFT JOIN LATERAL (
        SELECT fom.odds_json, fom.snapshot_time_utc, fom.snapshot_type
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.provider = 'sofascore'
          AND fom.market_code = 'ou'
          AND fom.line_num = 1.5
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND fom.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY (fom.snapshot_type = 'latest_pre_match') DESC, fom.snapshot_time_utc DESC
        LIMIT 1
    ) od15 ON true
    LEFT JOIN LATERAL (
        SELECT fom.odds_json, fom.snapshot_time_utc, fom.snapshot_type
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.provider = 'sofascore'
          AND fom.market_code = 'ou'
          AND fom.line_num = 2.5
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND fom.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY (fom.snapshot_type = 'latest_pre_match') DESC, fom.snapshot_time_utc DESC
        LIMIT 1
    ) od25 ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'lambda_xgb'
          AND p.market_code = 'lambda_home'
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l1h ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'lambda_xgb'
          AND p.market_code = 'lambda_away'
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l1a ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'situational_xgb'
          AND p.market_code = 'adj_lambda_home'
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l2h ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'situational_xgb'
          AND p.market_code = 'adj_lambda_away'
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l2a ON true
    WHERE f.status = 'ft'
      AND f.match_datetime_utc IS NOT NULL
      AND fr.home_goals IS NOT NULL
      AND fr.away_goals IS NOT NULL
    ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC;
    """

    conn = connect_db()
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df


def add_targets_and_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    # === TOTALS MARKETS ===
    out["target_o15"] = (out["total_goals"] >= 2).astype(int)
    out["target_o25"] = (out["total_goals"] >= 3).astype(int)
    out["target_o35"] = (out["total_goals"] >= 4).astype(int)
    out["target_o45"] = (out["total_goals"] >= 5).astype(int)
    out["target_u15"] = (out["total_goals"] <= 1).astype(int)
    out["target_u25"] = (out["total_goals"] <= 2).astype(int)
    
    # === CORNERS ===
    out["target_c85"] = np.where(out["total_corners"].notna(), (out["total_corners"] >= 9).astype(int), np.nan)
    
    # === BTTS ===
    out["target_btts"] = ((out["home_goals"] > 0) & (out["away_goals"] > 0)).astype(int)
    
    # === 1X2 ===
    out["target_1x2_h"] = (out["home_goals"] > out["away_goals"]).astype(int)
    out["target_1x2_d"] = (out["home_goals"] == out["away_goals"]).astype(int)
    out["target_1x2_a"] = (out["home_goals"] < out["away_goals"]).astype(int)
    
    # === DOUBLE CHANCE ===
    out["target_dc_1x"] = (out["home_goals"] >= out["away_goals"]).astype(int)  # Home win or Draw
    out["target_dc_x2"] = (out["away_goals"] >= out["home_goals"]).astype(int)  # Draw or Away win
    out["target_dc_12"] = (out["home_goals"] != out["away_goals"]).astype(int)  # Home win or Away win
    
    # === TEAM TOTALS ===
    out["target_ho15"] = (out["home_goals"] >= 2).astype(int)  # Home team over 1.5
    out["target_ao15"] = (out["away_goals"] >= 2).astype(int)  # Away team over 1.5
    
    # === COMBO OR (Home/Away win OR Over X.5) ===
    out["target_home_or_o25"] = ((out["home_goals"] > out["away_goals"]) | (out["total_goals"] >= 3)).astype(int)
    out["target_away_or_o25"] = ((out["away_goals"] > out["home_goals"]) | (out["total_goals"] >= 3)).astype(int)
    out["target_home_or_o15"] = ((out["home_goals"] > out["away_goals"]) | (out["total_goals"] >= 2)).astype(int)
    out["target_away_or_o15"] = ((out["away_goals"] > out["home_goals"]) | (out["total_goals"] >= 2)).astype(int)
    
    # === COMBO AND (Home/Away win AND Over X.5) ===
    out["target_home_and_o25"] = ((out["home_goals"] > out["away_goals"]) & (out["total_goals"] >= 3)).astype(int)
    out["target_away_and_o25"] = ((out["away_goals"] > out["home_goals"]) & (out["total_goals"] >= 3)).astype(int)

    # === ANYTIME LEAD MARKETS (incident timeline derived) ===
    out["target_h_1up"] = out["home_led_by_1_any"].astype(float)
    out["target_a_1up"] = out["away_led_by_1_any"].astype(float)
    out["target_h_2up"] = out["home_led_by_2_any"].astype(float)
    out["target_a_2up"] = out["away_led_by_2_any"].astype(float)

    # === DERIVED FEATURES ===
    out["xg_net_diff"] = out["home_rolling_xg"] - out["away_rolling_xg_against"]
    out["xgot_net_diff"] = out["home_rolling_xgot"] - out["away_rolling_xgot_against"]
    out["xa_net_diff"] = out["home_rolling_xa"] - out["away_rolling_xa_against"]
    out["corners_net_diff"] = out["home_rolling_corners"] - out["away_rolling_corners_against"]
    out["sample_size_diff"] = out["home_sample_size"] - out["away_sample_size"]
    
    # Goal difference feature for 1X2/AH markets
    out["goal_diff_proxy"] = out["home_rolling_xg"] - out["away_rolling_xg"]

    out["implied_over15"] = np.where(out["odds_over_15"] > 1.0, 1.0 / out["odds_over_15"], np.nan)
    out["implied_under15"] = np.where(out["odds_under_15"] > 1.0, 1.0 / out["odds_under_15"], np.nan)
    out["implied_over25"] = np.where(out["odds_over_25"] > 1.0, 1.0 / out["odds_over_25"], np.nan)
    out["implied_under25"] = np.where(out["odds_under_25"] > 1.0, 1.0 / out["odds_under_25"], np.nan)

    out["odds_gap_15"] = out["odds_over_15"] - out["odds_under_15"]
    out["odds_gap_25"] = out["odds_over_25"] - out["odds_under_25"]

    # === FORMATION PARSING ===
    def _parse_form(form_str: object) -> tuple[int, int, int]:
        if not isinstance(form_str, str) or "-" not in form_str:
            return (4, 4, 2)  # Default
        parts = [int(p) for p in form_str.split("-") if p.isdigit()]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
        if len(parts) == 4: # e.g. 4-2-3-1
            return (parts[0], parts[1] + parts[2], parts[3])
        if len(parts) == 5: # e.g. 5-4-1-0 or something exotic
            return (parts[0], parts[1] + parts[2] + parts[3], parts[4])
        return (4, 4, 2)

    for prefix in ("home", "away"):
        defenders, midfielders, forwards = zip(*out[f"{prefix}_formation"].apply(_parse_form))
        out[f"{prefix}_defenders"] = defenders
        out[f"{prefix}_midfielders"] = midfielders
        out[f"{prefix}_forwards"] = forwards
        
        # Calculate style score (higher = more offensive baseline)
        out[f"{prefix}_style_score"] = (
            out[f"{prefix}_defenders"] * 1 +
            out[f"{prefix}_midfielders"] * 2 +
            out[f"{prefix}_forwards"] * 3
        )

    out["style_delta"] = out["home_style_score"] - out["away_style_score"]
    
    return out


def feature_columns() -> list[str]:
    cols: list[str] = []
    for prefix in ("home", "away"):
        cols.extend([f"{prefix}_{col}" for col in TEAM_SNAPSHOT_FEATURES])
    cols.extend(
        [
            "odds_over_15",
            "odds_under_15",
            "odds_over_25",
            "odds_under_25",
            "xg_net_diff",
            "xgot_net_diff",
            "xa_net_diff",
            "corners_net_diff",
            "sample_size_diff",
            "goal_diff_proxy",
            "implied_over15",
            "implied_under15",
            "implied_over25",
            "implied_under25",
            "odds_gap_15",
            "odds_gap_25",
            "lambda_home_l1",
            "lambda_away_l1",
            "adj_lambda_home_final",
            "adj_lambda_away_final",
            "rule_fired_home",
            "rule_fired_away",
            # Formation/Style features
            "home_defenders", "home_midfielders", "home_forwards",
            "away_defenders", "away_midfielders", "away_forwards",
            "style_delta",
        ]
    )
    return cols


def split_time_respecting(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(drop=True)
    split_idx = int(len(ordered) * 0.8)
    if split_idx <= 0 or split_idx >= len(ordered):
        raise ValueError("Need at least 2 rows to make an 80/20 time split")
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def impute_for_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    train_out = train.copy()
    test_out = test.copy()
    league_medians: dict[str, dict[str, float]] = {}
    global_medians: dict[str, float] = {}

    for feat in features:
        league_med_series = train_out.groupby("league_code")[feat].median()
        global_med = train_out[feat].median()

        if pd.isna(global_med):
            global_med = 0.0

        global_medians[feat] = float(global_med)
        league_medians[feat] = {
            str(key): (float(val) if pd.notna(val) else float(global_med))
            for key, val in league_med_series.to_dict().items()
        }

        train_fill = train_out["league_code"].map(league_medians[feat])
        test_fill = test_out["league_code"].map(league_medians[feat])
        train_out[feat] = train_out[feat].fillna(train_fill).fillna(global_med).fillna(0.0)
        test_out[feat] = test_out[feat].fillna(test_fill).fillna(global_med).fillna(0.0)

    return train_out, test_out, {"global_medians": global_medians, "league_medians": league_medians}


def fit_market_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    market_code: str,
    features: list[str],
) -> dict[str, object]:
    train = train_df[train_df[target_col].notna()].copy()
    test = test_df[test_df[target_col].notna()].copy()
    if train.empty or test.empty:
        raise ValueError(f"Insufficient rows for {market_code}: train={len(train)}, test={len(test)}")

    y_train = train[target_col].astype(int)
    y_test = test[target_col].astype(int)

    if y_train.nunique() < 2:
        constant_class = int(y_train.iloc[0])
        model = DummyClassifier(strategy="constant", constant=constant_class)
        model.fit(train[features], y_train)
        probs = np.full(len(test), float(constant_class), dtype=float)
        preds = np.full(len(test), constant_class, dtype=int)
        auc = float("nan")
        calibrated = False
    else:
        base = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=40,
            random_state=42,
        )

        class_counts = y_train.value_counts()
        can_calibrate = len(class_counts) > 1 and int(class_counts.min()) >= 3
        if can_calibrate:
            model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
        else:
            model = base

        model.fit(train[features], y_train)
        probs = np.clip(model.predict_proba(test[features])[:, 1], 0.001, 0.999)
        preds = (probs >= 0.5).astype(int)

        try:
            auc = float(roc_auc_score(y_test, probs))
        except ValueError:
            auc = float("nan")
        calibrated = bool(can_calibrate)

    metrics: dict[str, object] = {
        "market": market_code,
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "base_rate_train": float(y_train.mean()),
        "base_rate_test": float(y_test.mean()),
        "auc": None if np.isnan(auc) else auc,
        "accuracy": float(accuracy_score(y_test, preds)),
        "brier": float(brier_score_loss(y_test, probs)),
        "calibrated": calibrated,
        "train_start_utc": str(train["match_datetime_utc"].min()),
        "train_end_utc": str(train["match_datetime_utc"].max()),
        "test_start_utc": str(test["match_datetime_utc"].min()),
        "test_end_utc": str(test["match_datetime_utc"].max()),
    }

    # Save the FITTED model to disk
    joblib.dump(model, OUT_DIR / f"gbm_{market_code}.pkl")
    return metrics


def main() -> None:
    df = fetch_dataset()
    if df.empty:
        raise RuntimeError("No fixtures-first rows available for training")

    df["match_datetime_utc"] = pd.to_datetime(df["match_datetime_utc"], utc=True, errors="coerce")
    df["odds_snapshot_time_utc"] = pd.to_datetime(df["odds_snapshot_time_utc"], utc=True, errors="coerce")
    post_kickoff_odds = (
        df["odds_snapshot_time_utc"].notna()
        & df["match_datetime_utc"].notna()
        & (df["odds_snapshot_time_utc"] > df["match_datetime_utc"])
    )
    if post_kickoff_odds.any():
        raise RuntimeError(
            f"Found {int(post_kickoff_odds.sum())} rows with post-kickoff odds snapshots; refusing to train"
        )

    df = add_targets_and_derived(df)
    features = feature_columns()
    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        missing_csv = ", ".join(sorted(missing_features))
        raise RuntimeError(f"Feature contract mismatch in training dataset: {missing_csv}")

    train_df, test_df = split_time_respecting(df)
    train_imp, test_imp, imputation = impute_for_split(train_df, test_df, features)

    # Define all markets with their targets
    MARKETS = [
        # Totals
        ("o15", "target_o15"),
        ("o25", "target_o25"),
        ("o35", "target_o35"),
        ("o45", "target_o45"),
        ("u15", "target_u15"),
        ("u25", "target_u25"),
        # Corners
        ("c85", "target_c85"),
        # BTTS
        ("btts", "target_btts"),
        # 1X2
        ("1x2_h", "target_1x2_h"),
        ("1x2_d", "target_1x2_d"),
        ("1x2_a", "target_1x2_a"),
        # Double Chance
        ("dc_1x", "target_dc_1x"),
        ("dc_x2", "target_dc_x2"),
        ("dc_12", "target_dc_12"),
        # Team Totals
        ("ho15", "target_ho15"),
        ("ao15", "target_ao15"),
        # Combo OR
        ("home_or_o25", "target_home_or_o25"),
        ("away_or_o25", "target_away_or_o25"),
        ("home_or_o15", "target_home_or_o15"),
        ("away_or_o15", "target_away_or_o15"),
        # Combo AND
        ("home_and_o25", "target_home_and_o25"),
        ("away_and_o25", "target_away_and_o25"),
        # Anytime Lead
        ("h_1up", "target_h_1up"),
        ("a_1up", "target_a_1up"),
        ("h_2up", "target_h_2up"),
        ("a_2up", "target_a_2up"),
    ]
    
    metrics = []
    for market, target in MARKETS:
        result = fit_market_model(
            train_df=train_imp,
            test_df=test_imp,
            target_col=target,
            market_code=market,
            features=features,
        )
        metrics.append(result)
        print(f"{market}: AUC={result['auc']}, Acc={result['accuracy']:.3f}, Brier={result['brier']:.4f}, test_n={result['test_n']}, base_rate={result['base_rate_test']:.3f}")

    with (OUT_DIR / "features.json").open("w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    with (OUT_DIR / "imputation.json").open("w", encoding="utf-8") as f:
        json.dump(imputation, f, indent=2)

    with (OUT_DIR / "metrics_walkforward.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved fixtures-first artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
