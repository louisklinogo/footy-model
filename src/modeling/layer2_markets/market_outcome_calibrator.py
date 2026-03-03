"""
Fixtures-first training pipeline for focused linchpin markets.

Reads only v1 fixtures-first tables and trains leakage-safe pre-match models
for the following markets:

GOALS (2): o15, u35
CORNERS TOTALS (4): c75, c85, c95, c105
CORNERS HOME TEAM (4): hc25, hc35, hc45, hc55
CORNERS AWAY TEAM (4): ac25, ac35, ac45, ac55
1X2 (3): 1x2_h, 1x2_d, 1x2_a
DOUBLE CHANCE (3): dc_1x, dc_x2, dc_12
TEAM TOTALS (2): ho15, ao15
ANYTIME LEAD (4): h_1up, a_1up, h_2up, a_2up
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
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
]


ALL_MARKETS: list[tuple[str, str]] = [
    # Goals
    ("o15", "target_o15"),
    ("u35", "target_u35"),
    # Corners Totals
    ("c75", "target_c75"),
    ("c85", "target_c85"),
    ("c95", "target_c95"),
    ("c105", "target_c105"),
    # Corners Home Team Totals
    ("hc25", "target_hc25"),
    ("hc35", "target_hc35"),
    ("hc45", "target_hc45"),
    ("hc55", "target_hc55"),
    # Corners Away Team Totals
    ("ac25", "target_ac25"),
    ("ac35", "target_ac35"),
    ("ac45", "target_ac45"),
    ("ac55", "target_ac55"),
    # 1X2
    ("1x2_h", "target_1x2_h"),
    ("1x2_d", "target_1x2_d"),
    ("1x2_a", "target_1x2_a"),
    # Double Chance
    ("dc_1x", "target_dc_1x"),
    ("dc_x2", "target_dc_x2"),
    ("dc_12", "target_dc_12"),
    # Handicap
    ("ah_h05", "target_ah_h05"),
    ("ah_a05", "target_ah_a05"),
    ("ah_h15", "target_ah_h15"),
    ("ah_a15", "target_ah_a15"),
    ("eh_h1", "target_eh_h1"),
    ("eh_a1", "target_eh_a1"),
    # Team Totals
    ("ho15", "target_ho15"),
    ("ao15", "target_ao15"),
    # Anytime Lead
    ("h_1up", "target_h_1up"),
    ("a_1up", "target_a_1up"),
    ("h_2up", "target_h_2up"),
    ("a_2up", "target_a_2up"),
]

FAMILY_TO_MARKETS: dict[str, set[str]] = {
    "weak": {"1x2_d", "dc_12", "o15", "u35", "c75", "c85", "c95", "c105"},
    "1x2_dc": {"1x2_h", "1x2_d", "1x2_a", "dc_1x", "dc_x2", "dc_12"},
    "goals_totals": {"o15", "u35"},
    "corners_totals": {"c75", "c85", "c95", "c105"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fixtures-first market models with walk-forward diagnostics."
    )
    parser.add_argument(
        "--markets",
        type=str,
        default=None,
        help="Optional comma-separated market codes to train.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        choices=tuple(sorted(FAMILY_TO_MARKETS.keys())),
        help="Optional market family shortcut (for example: weak, 1x2_dc).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=6,
        help="Number of expanding walk-forward folds for model selection.",
    )
    parser.add_argument(
        "--min-fold-test-n",
        type=int,
        default=200,
        help="Minimum fold test rows required to include a fold in summary.",
    )
    parser.add_argument(
        "--promotion-baseline-path",
        type=Path,
        default=OUT_DIR / "metrics_walkforward.json",
        help="Baseline metrics path used for promotion gating deltas.",
    )
    parser.add_argument(
        "--promotion-min-auc-delta",
        type=float,
        default=0.02,
        help="Minimum AUC lift required for promotion.",
    )
    return parser.parse_args()


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
        sp.h_corners AS home_corners,
        sp.a_corners AS away_corners,
        {_latest_odds_expr("od15", "over")} AS odds_over_15,
        {_latest_odds_expr("od15", "under")} AS odds_under_15,
        {_latest_odds_expr("od25", "over")} AS odds_over_25,
        {_latest_odds_expr("od25", "under")} AS odds_under_25,
        {_latest_odds_expr("od35", "over")} AS odds_over_35,
        {_latest_odds_expr("od35", "under")} AS odds_under_35,
        {_latest_odds_expr("odc75", "over")} AS odds_c75_over,
        {_latest_odds_expr("odc75", "under")} AS odds_c75_under,
        {_latest_odds_expr("odc85", "over")} AS odds_c85_over,
        {_latest_odds_expr("odc85", "under")} AS odds_c85_under,
        {_latest_odds_expr("odc95", "over")} AS odds_c95_over,
        {_latest_odds_expr("odc95", "under")} AS odds_c95_under,
        {_latest_odds_expr("odc105", "over")} AS odds_c105_over,
        {_latest_odds_expr("odc105", "under")} AS odds_c105_under,
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

        ff.home_formation,
        ff.away_formation,
        {_json_number_expr("l1h.metadata_json", "lambda")} AS lambda_home_l1,
        {_json_number_expr("l1a.metadata_json", "lambda")} AS lambda_away_l1,
        {_json_number_expr("l2h.metadata_json", "lambda")} AS adj_lambda_home_final,
        {_json_number_expr("l2a.metadata_json", "lambda")} AS adj_lambda_away_final,
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
        SELECT fom.odds_json, fom.snapshot_time_utc, fom.snapshot_type
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.provider = 'sofascore'
          AND fom.market_code = 'ou'
          AND fom.line_num = 3.5
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND fom.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY (fom.snapshot_type = 'latest_pre_match') DESC, fom.snapshot_time_utc DESC
        LIMIT 1
    ) od35 ON true
    LEFT JOIN LATERAL (
        SELECT fom.odds_json, fom.snapshot_time_utc, fom.snapshot_type
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.provider = 'sofascore'
          AND fom.market_code = 'corners_ou'
          AND fom.line_num = 7.5
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND fom.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY (fom.snapshot_type = 'latest_pre_match') DESC, fom.snapshot_time_utc DESC
        LIMIT 1
    ) odc75 ON true
    LEFT JOIN LATERAL (
        SELECT fom.odds_json, fom.snapshot_time_utc, fom.snapshot_type
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.provider = 'sofascore'
          AND fom.market_code = 'corners_ou'
          AND fom.line_num = 8.5
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND fom.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY (fom.snapshot_type = 'latest_pre_match') DESC, fom.snapshot_time_utc DESC
        LIMIT 1
    ) odc85 ON true
    LEFT JOIN LATERAL (
        SELECT fom.odds_json, fom.snapshot_time_utc, fom.snapshot_type
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.provider = 'sofascore'
          AND fom.market_code = 'corners_ou'
          AND fom.line_num = 9.5
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND fom.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY (fom.snapshot_type = 'latest_pre_match') DESC, fom.snapshot_time_utc DESC
        LIMIT 1
    ) odc95 ON true
    LEFT JOIN LATERAL (
        SELECT fom.odds_json, fom.snapshot_time_utc, fom.snapshot_type
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.provider = 'sofascore'
          AND fom.market_code = 'corners_ou'
          AND fom.line_num = 10.5
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND fom.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY (fom.snapshot_type = 'latest_pre_match') DESC, fom.snapshot_time_utc DESC
        LIMIT 1
    ) odc105 ON true
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
    out["target_u35"] = (out["total_goals"] <= 3).astype(int)

    # === CORNERS TOTALS ===
    out["target_c75"] = np.where(
        out["total_corners"].notna(), (out["total_corners"] >= 8).astype(int), np.nan
    )
    out["target_c85"] = np.where(
        out["total_corners"].notna(), (out["total_corners"] >= 9).astype(int), np.nan
    )
    out["target_c95"] = np.where(
        out["total_corners"].notna(), (out["total_corners"] >= 10).astype(int), np.nan
    )
    out["target_c105"] = np.where(
        out["total_corners"].notna(), (out["total_corners"] >= 11).astype(int), np.nan
    )

    # === CORNERS TEAM TOTALS ===
    out["target_hc25"] = np.where(
        out["home_corners"].notna(), (out["home_corners"] >= 3).astype(int), np.nan
    )
    out["target_hc35"] = np.where(
        out["home_corners"].notna(), (out["home_corners"] >= 4).astype(int), np.nan
    )
    out["target_hc45"] = np.where(
        out["home_corners"].notna(), (out["home_corners"] >= 5).astype(int), np.nan
    )
    out["target_hc55"] = np.where(
        out["home_corners"].notna(), (out["home_corners"] >= 6).astype(int), np.nan
    )
    out["target_ac25"] = np.where(
        out["away_corners"].notna(), (out["away_corners"] >= 3).astype(int), np.nan
    )
    out["target_ac35"] = np.where(
        out["away_corners"].notna(), (out["away_corners"] >= 4).astype(int), np.nan
    )
    out["target_ac45"] = np.where(
        out["away_corners"].notna(), (out["away_corners"] >= 5).astype(int), np.nan
    )
    out["target_ac55"] = np.where(
        out["away_corners"].notna(), (out["away_corners"] >= 6).astype(int), np.nan
    )

    # === BTTS ===
    out["target_btts"] = ((out["home_goals"] > 0) & (out["away_goals"] > 0)).astype(int)

    # === 1X2 ===
    out["target_1x2_h"] = (out["home_goals"] > out["away_goals"]).astype(int)
    out["target_1x2_d"] = (out["home_goals"] == out["away_goals"]).astype(int)
    out["target_1x2_a"] = (out["home_goals"] < out["away_goals"]).astype(int)

    # === DOUBLE CHANCE ===
    out["target_dc_1x"] = (out["home_goals"] >= out["away_goals"]).astype(
        int
    )  # Home win or Draw
    out["target_dc_x2"] = (out["away_goals"] >= out["home_goals"]).astype(
        int
    )  # Draw or Away win
    out["target_dc_12"] = (out["home_goals"] != out["away_goals"]).astype(
        int
    )  # Home win or Away win

    # === HANDICAP (BINARY, SETTLEMENT PUSH IGNORED) ===
    out["target_ah_h05"] = (out["home_goals"] > out["away_goals"]).astype(int)
    out["target_ah_a05"] = (out["away_goals"] > out["home_goals"]).astype(int)
    out["target_ah_h15"] = ((out["home_goals"] - out["away_goals"]) >= 2).astype(int)
    out["target_ah_a15"] = ((out["away_goals"] - out["home_goals"]) >= 2).astype(int)
    out["target_eh_h1"] = ((out["home_goals"] - out["away_goals"]) >= 2).astype(int)
    out["target_eh_a1"] = ((out["away_goals"] - out["home_goals"]) >= 2).astype(int)

    # === TEAM TOTALS ===
    out["target_ho15"] = (out["home_goals"] >= 2).astype(int)  # Home team over 1.5
    out["target_ao15"] = (out["away_goals"] >= 2).astype(int)  # Away team over 1.5

    # === COMBO OR (Home/Away win OR Over X.5) ===
    out["target_home_or_o25"] = (
        (out["home_goals"] > out["away_goals"]) | (out["total_goals"] >= 3)
    ).astype(int)
    out["target_away_or_o25"] = (
        (out["away_goals"] > out["home_goals"]) | (out["total_goals"] >= 3)
    ).astype(int)
    out["target_home_or_o15"] = (
        (out["home_goals"] > out["away_goals"]) | (out["total_goals"] >= 2)
    ).astype(int)
    out["target_away_or_o15"] = (
        (out["away_goals"] > out["home_goals"]) | (out["total_goals"] >= 2)
    ).astype(int)

    # === COMBO AND (Home/Away win AND Over X.5) ===
    out["target_home_and_o25"] = (
        (out["home_goals"] > out["away_goals"]) & (out["total_goals"] >= 3)
    ).astype(int)
    out["target_away_and_o25"] = (
        (out["away_goals"] > out["home_goals"]) & (out["total_goals"] >= 3)
    ).astype(int)

    # === ANYTIME LEAD MARKETS (incident timeline derived) ===
    out["target_h_1up"] = out["home_led_by_1_any"].astype(float)
    out["target_a_1up"] = out["away_led_by_1_any"].astype(float)
    out["target_h_2up"] = out["home_led_by_2_any"].astype(float)
    out["target_a_2up"] = out["away_led_by_2_any"].astype(float)

    # === DERIVED FEATURES ===
    out["xg_net_diff"] = out["home_rolling_xg"] - out["away_rolling_xg_against"]
    out["xgot_net_diff"] = out["home_rolling_xgot"] - out["away_rolling_xgot_against"]
    out["xa_net_diff"] = out["home_rolling_xa"] - out["away_rolling_xa_against"]
    out["corners_net_diff"] = (
        out["home_rolling_corners"] - out["away_rolling_corners_against"]
    )
    out["sample_size_diff"] = out["home_sample_size"] - out["away_sample_size"]

    # Goal difference feature for 1X2/AH markets
    out["goal_diff_proxy"] = out["home_rolling_xg"] - out["away_rolling_xg"]

    out["implied_over15"] = np.where(
        out["odds_over_15"] > 1.0, 1.0 / out["odds_over_15"], np.nan
    )
    out["implied_under15"] = np.where(
        out["odds_under_15"] > 1.0, 1.0 / out["odds_under_15"], np.nan
    )
    out["implied_over25"] = np.where(
        out["odds_over_25"] > 1.0, 1.0 / out["odds_over_25"], np.nan
    )
    out["implied_under25"] = np.where(
        out["odds_under_25"] > 1.0, 1.0 / out["odds_under_25"], np.nan
    )
    out["implied_over35"] = np.where(
        out["odds_over_35"] > 1.0, 1.0 / out["odds_over_35"], np.nan
    )
    out["implied_under35"] = np.where(
        out["odds_under_35"] > 1.0, 1.0 / out["odds_under_35"], np.nan
    )

    out["odds_gap_15"] = out["odds_over_15"] - out["odds_under_15"]
    out["odds_gap_25"] = out["odds_over_25"] - out["odds_under_25"]
    out["odds_gap_35"] = out["odds_over_35"] - out["odds_under_35"]
    out["corners_gap_75"] = out["odds_c75_over"] - out["odds_c75_under"]
    out["corners_gap_85"] = out["odds_c85_over"] - out["odds_c85_under"]
    out["corners_gap_95"] = out["odds_c95_over"] - out["odds_c95_under"]
    out["corners_gap_105"] = out["odds_c105_over"] - out["odds_c105_under"]

    out["implied_c75_over"] = np.where(
        out["odds_c75_over"] > 1.0, 1.0 / out["odds_c75_over"], np.nan
    )
    out["implied_c85_over"] = np.where(
        out["odds_c85_over"] > 1.0, 1.0 / out["odds_c85_over"], np.nan
    )
    out["implied_c95_over"] = np.where(
        out["odds_c95_over"] > 1.0, 1.0 / out["odds_c95_over"], np.nan
    )
    out["implied_c105_over"] = np.where(
        out["odds_c105_over"] > 1.0, 1.0 / out["odds_c105_over"], np.nan
    )

    # === FORMATION PARSING ===
    def _parse_form(form_str: object) -> tuple[int, int, int]:
        if not isinstance(form_str, str) or "-" not in form_str:
            return (4, 4, 2)  # Default
        parts = [int(p) for p in form_str.split("-") if p.isdigit()]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
        if len(parts) == 4:  # e.g. 4-2-3-1
            return (parts[0], parts[1] + parts[2], parts[3])
        if len(parts) == 5:  # e.g. 5-4-1-0 or something exotic
            return (parts[0], parts[1] + parts[2] + parts[3], parts[4])
        return (4, 4, 2)

    for prefix in ("home", "away"):
        defenders, midfielders, forwards = zip(
            *out[f"{prefix}_formation"].apply(_parse_form)
        )
        out[f"{prefix}_defenders"] = defenders
        out[f"{prefix}_midfielders"] = midfielders
        out[f"{prefix}_forwards"] = forwards

        # Calculate style score (higher = more offensive baseline)
        out[f"{prefix}_style_score"] = (
            out[f"{prefix}_defenders"] * 1
            + out[f"{prefix}_midfielders"] * 2
            + out[f"{prefix}_forwards"] * 3
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
            "odds_over_35",
            "odds_under_35",
            "odds_c75_over",
            "odds_c75_under",
            "odds_c85_over",
            "odds_c85_under",
            "odds_c95_over",
            "odds_c95_under",
            "odds_c105_over",
            "odds_c105_under",
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
            "implied_over35",
            "implied_under35",
            "implied_c75_over",
            "implied_c85_over",
            "implied_c95_over",
            "implied_c105_over",
            "odds_gap_15",
            "odds_gap_25",
            "odds_gap_35",
            "corners_gap_75",
            "corners_gap_85",
            "corners_gap_95",
            "corners_gap_105",
            "lambda_home_l1",
            "lambda_away_l1",
            "adj_lambda_home_final",
            "adj_lambda_away_final",
            "rule_fired_home",
            "rule_fired_away",
            # Formation/Style features
            "home_defenders",
            "home_midfielders",
            "home_forwards",
            "away_defenders",
            "away_midfielders",
            "away_forwards",
            "style_delta",
        ]
    )
    return cols


def split_time_respecting(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(
        drop=True
    )
    split_idx = int(len(ordered) * 0.8)
    if split_idx <= 0 or split_idx >= len(ordered):
        raise ValueError("Need at least 2 rows to make an 80/20 time split")
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def resolve_market_selection(args: argparse.Namespace) -> list[tuple[str, str]]:
    selected = {market for market, _target in ALL_MARKETS}

    if args.family:
        selected &= FAMILY_TO_MARKETS.get(args.family, set())

    if args.markets:
        requested = {
            item.strip() for item in str(args.markets).split(",") if item.strip()
        }
        selected &= requested

    chosen = [(market, target) for market, target in ALL_MARKETS if market in selected]
    if not chosen:
        raise ValueError("No markets selected. Check --family/--markets filters.")
    return chosen


def _build_base_estimator(model_kind: str) -> object:
    if model_kind == "hgbm":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=350,
            min_samples_leaf=40,
            random_state=42,
        )
    return GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=40,
        random_state=42,
    )


def _walkforward_ranges(total_rows: int, folds: int) -> list[tuple[int, int]]:
    if total_rows <= 0:
        return []
    points = np.linspace(0, total_rows, folds + 2, dtype=int)
    ranges: list[tuple[int, int]] = []
    for idx in range(1, len(points) - 1):
        train_end = int(points[idx])
        test_end = int(points[idx + 1])
        if train_end <= 0 or test_end <= train_end:
            continue
        ranges.append((train_end, test_end))
    return ranges


def evaluate_market_candidates_walkforward(
    *,
    frame: pd.DataFrame,
    target_col: str,
    market_code: str,
    features: list[str],
    folds: int,
    min_fold_test_n: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float | int | None]], str]:
    market_frame = frame[frame[target_col].notna()].sort_values(
        ["match_datetime_utc", "fixture_id"]
    )
    model_kinds = ("gbm", "hgbm")
    fold_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float | int | None]] = {}

    if market_frame.empty:
        fallback = "gbm"
        for kind in model_kinds:
            summary[kind] = {
                "auc_mean": None,
                "auc_std": None,
                "brier_mean": None,
                "brier_std": None,
                "folds_used": 0,
            }
        return fold_rows, summary, fallback

    ranges = _walkforward_ranges(len(market_frame), folds)
    for model_kind in model_kinds:
        aucs: list[float] = []
        briers: list[float] = []
        used = 0
        for fold_idx, (train_end, test_end) in enumerate(ranges, start=1):
            train = market_frame.iloc[:train_end]
            test = market_frame.iloc[train_end:test_end]
            if len(test) < min_fold_test_n or train.empty:
                continue
            y_train = train[target_col].astype(int)
            y_test = test[target_col].astype(int)
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            model = _build_base_estimator(model_kind)
            model.fit(train[features], y_train)
            probs = np.clip(model.predict_proba(test[features])[:, 1], 0.001, 0.999)
            try:
                auc = float(roc_auc_score(y_test, probs))
            except ValueError:
                continue
            brier = float(brier_score_loss(y_test, probs))
            aucs.append(auc)
            briers.append(brier)
            used += 1
            fold_rows.append(
                {
                    "market": market_code,
                    "candidate_model": model_kind,
                    "fold": fold_idx,
                    "train_n": int(len(train)),
                    "test_n": int(len(test)),
                    "auc": auc,
                    "brier": brier,
                    "train_start_utc": str(train["match_datetime_utc"].min()),
                    "train_end_utc": str(train["match_datetime_utc"].max()),
                    "test_start_utc": str(test["match_datetime_utc"].min()),
                    "test_end_utc": str(test["match_datetime_utc"].max()),
                }
            )

        if used == 0:
            summary[model_kind] = {
                "auc_mean": None,
                "auc_std": None,
                "brier_mean": None,
                "brier_std": None,
                "folds_used": 0,
            }
        else:
            summary[model_kind] = {
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "brier_mean": float(np.mean(briers)),
                "brier_std": float(np.std(briers)),
                "folds_used": int(used),
            }

    def _rank_key(kind: str) -> tuple[float, float]:
        item = summary[kind]
        auc_mean = item["auc_mean"]
        brier_mean = item["brier_mean"]
        auc_rank = float(auc_mean) if auc_mean is not None else -1e9
        brier_rank = -float(brier_mean) if brier_mean is not None else -1e9
        return auc_rank, brier_rank

    selected_model = max(model_kinds, key=_rank_key)
    return fold_rows, summary, selected_model


def load_baseline_metrics(path: Path) -> dict[str, dict[str, float | None]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, list):
        return {}
    out: dict[str, dict[str, float | None]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        market = str(row.get("market") or "").strip()
        if not market:
            continue
        auc_raw = row.get("auc")
        brier_raw = row.get("brier")
        auc = float(auc_raw) if isinstance(auc_raw, (int, float)) else None
        brier = float(brier_raw) if isinstance(brier_raw, (int, float)) else None
        out[market] = {"auc": auc, "brier": brier}
    return out


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
        train_out[feat] = (
            train_out[feat].fillna(train_fill).fillna(global_med).fillna(0.0)
        )
        test_out[feat] = test_out[feat].fillna(test_fill).fillna(global_med).fillna(0.0)

    return (
        train_out,
        test_out,
        {"global_medians": global_medians, "league_medians": league_medians},
    )


def fit_market_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    market_code: str,
    features: list[str],
    model_kind: str,
) -> dict[str, object]:
    train = train_df[train_df[target_col].notna()].copy()
    test = test_df[test_df[target_col].notna()].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Insufficient rows for {market_code}: train={len(train)}, test={len(test)}"
        )

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
        base = _build_base_estimator(model_kind)

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
        "candidate_model": model_kind,
        "train_start_utc": str(train["match_datetime_utc"].min()),
        "train_end_utc": str(train["match_datetime_utc"].max()),
        "test_start_utc": str(test["match_datetime_utc"].min()),
        "test_end_utc": str(test["match_datetime_utc"].max()),
    }

    # Save the FITTED model to disk
    joblib.dump(model, OUT_DIR / f"gbm_{market_code}.pkl")
    return metrics


def _y_1x2_class(frame: pd.DataFrame) -> np.ndarray:
    home = frame["home_goals"].astype(int).to_numpy()
    away = frame["away_goals"].astype(int).to_numpy()
    out = np.full(len(frame), 2, dtype=int)  # away
    out[home > away] = 0  # home
    out[home == away] = 1  # draw
    return out


def _multiclass_probs(
    model: object, x: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = model.predict_proba(x)
    classes = np.asarray(getattr(model, "classes_", [0, 1, 2]), dtype=int)
    idx_home = int(np.where(classes == 0)[0][0]) if np.any(classes == 0) else None
    idx_draw = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else None
    idx_away = int(np.where(classes == 2)[0][0]) if np.any(classes == 2) else None
    p_home = probs[:, idx_home] if idx_home is not None else np.zeros(len(x), dtype=float)
    p_draw = probs[:, idx_draw] if idx_draw is not None else np.zeros(len(x), dtype=float)
    p_away = probs[:, idx_away] if idx_away is not None else np.zeros(len(x), dtype=float)
    total = p_home + p_draw + p_away
    total = np.where(total <= 0.0, 1.0, total)
    return p_home / total, p_draw / total, p_away / total


def _binary_metric_row(
    *,
    market_code: str,
    y_true: np.ndarray,
    p_true: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate_model: str,
    calibrated: bool,
) -> dict[str, object]:
    preds = (p_true >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, p_true))
    except ValueError:
        auc = float("nan")
    return {
        "market": market_code,
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "base_rate_train": float(np.mean(_binary_target_for_market(market_code, _y_1x2_class(train)))),
        "base_rate_test": float(np.mean(y_true)),
        "auc": None if np.isnan(auc) else auc,
        "accuracy": float(accuracy_score(y_true, preds)),
        "brier": float(brier_score_loss(y_true, p_true)),
        "calibrated": calibrated,
        "candidate_model": candidate_model,
        "train_start_utc": str(train["match_datetime_utc"].min()),
        "train_end_utc": str(train["match_datetime_utc"].max()),
        "test_start_utc": str(test["match_datetime_utc"].min()),
        "test_end_utc": str(test["match_datetime_utc"].max()),
    }


def _binary_target_for_market(market_code: str, y_class: np.ndarray) -> np.ndarray:
    if market_code == "1x2_h":
        return (y_class == 0).astype(int)
    if market_code == "1x2_d":
        return (y_class == 1).astype(int)
    if market_code == "1x2_a":
        return (y_class == 2).astype(int)
    if market_code == "dc_1x":
        return np.isin(y_class, [0, 1]).astype(int)
    if market_code == "dc_x2":
        return np.isin(y_class, [1, 2]).astype(int)
    if market_code == "dc_12":
        return np.isin(y_class, [0, 2]).astype(int)
    raise ValueError(f"unsupported 1x2/dc market: {market_code}")


def _binary_prob_for_market(
    market_code: str, p_home: np.ndarray, p_draw: np.ndarray, p_away: np.ndarray
) -> np.ndarray:
    if market_code == "1x2_h":
        return p_home
    if market_code == "1x2_d":
        return p_draw
    if market_code == "1x2_a":
        return p_away
    if market_code == "dc_1x":
        return p_home + p_draw
    if market_code == "dc_x2":
        return p_draw + p_away
    if market_code == "dc_12":
        return p_home + p_away
    raise ValueError(f"unsupported 1x2/dc market: {market_code}")


def evaluate_1x2_dc_multiclass_walkforward(
    *,
    frame: pd.DataFrame,
    features: list[str],
    selected_markets: set[str],
    folds: int,
    min_fold_test_n: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float | int | None]], str]:
    market_frame = frame[
        frame["home_goals"].notna() & frame["away_goals"].notna()
    ].sort_values(["match_datetime_utc", "fixture_id"])
    model_kinds = ("gbm", "hgbm")
    fold_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float | int | None]] = {}
    ranges = _walkforward_ranges(len(market_frame), folds)

    for model_kind in model_kinds:
        aucs: list[float] = []
        briers: list[float] = []
        used = 0
        for fold_idx, (train_end, test_end) in enumerate(ranges, start=1):
            train = market_frame.iloc[:train_end]
            test = market_frame.iloc[train_end:test_end]
            if len(test) < min_fold_test_n or train.empty:
                continue
            y_train = _y_1x2_class(train)
            y_test = _y_1x2_class(test)
            if len(np.unique(y_train)) < 2:
                continue
            model = _build_base_estimator(model_kind)
            model.fit(train[features], y_train)
            p_home, p_draw, p_away = _multiclass_probs(model, test[features])

            for market in sorted(selected_markets):
                y_bin = _binary_target_for_market(market, y_test)
                p_bin = np.clip(
                    _binary_prob_for_market(market, p_home, p_draw, p_away), 0.001, 0.999
                )
                if len(np.unique(y_bin)) < 2:
                    continue
                auc = float(roc_auc_score(y_bin, p_bin))
                brier = float(brier_score_loss(y_bin, p_bin))
                aucs.append(auc)
                briers.append(brier)
                fold_rows.append(
                    {
                        "market": market,
                        "candidate_model": model_kind,
                        "fold": fold_idx,
                        "train_n": int(len(train)),
                        "test_n": int(len(test)),
                        "auc": auc,
                        "brier": brier,
                        "train_start_utc": str(train["match_datetime_utc"].min()),
                        "train_end_utc": str(train["match_datetime_utc"].max()),
                        "test_start_utc": str(test["match_datetime_utc"].min()),
                        "test_end_utc": str(test["match_datetime_utc"].max()),
                    }
                )
            used += 1

        if used == 0 or not aucs:
            summary[model_kind] = {
                "auc_mean": None,
                "auc_std": None,
                "brier_mean": None,
                "brier_std": None,
                "folds_used": 0,
            }
        else:
            summary[model_kind] = {
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "brier_mean": float(np.mean(briers)),
                "brier_std": float(np.std(briers)),
                "folds_used": int(used),
            }

    def _rank_key(kind: str) -> tuple[float, float]:
        item = summary[kind]
        auc_mean = item["auc_mean"]
        brier_mean = item["brier_mean"]
        auc_rank = float(auc_mean) if auc_mean is not None else -1e9
        brier_rank = -float(brier_mean) if brier_mean is not None else -1e9
        return auc_rank, brier_rank

    selected_model = max(model_kinds, key=_rank_key)
    return fold_rows, summary, selected_model


def fit_1x2_dc_multiclass(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model_kind: str,
    selected_markets: set[str],
) -> tuple[object, dict[str, dict[str, object]]]:
    train = train_df[train_df["home_goals"].notna() & train_df["away_goals"].notna()].copy()
    test = test_df[test_df["home_goals"].notna() & test_df["away_goals"].notna()].copy()
    if train.empty or test.empty:
        raise ValueError("Insufficient rows for 1x2/dc multiclass fit.")

    y_train = _y_1x2_class(train)
    y_test = _y_1x2_class(test)
    model = _build_base_estimator(model_kind)
    model.fit(train[features], y_train)
    p_home, p_draw, p_away = _multiclass_probs(model, test[features])

    markets = ("1x2_h", "1x2_d", "1x2_a", "dc_1x", "dc_x2", "dc_12")
    requested = {market for market in selected_markets if market in markets}
    if not requested:
        raise ValueError("No 1x2/dc markets selected for multiclass fit.")
    metrics: dict[str, dict[str, object]] = {}
    for market in markets:
        if market not in requested:
            continue
        y_bin = _binary_target_for_market(market, y_test)
        p_bin = np.clip(_binary_prob_for_market(market, p_home, p_draw, p_away), 0.001, 0.999)
        metrics[market] = _binary_metric_row(
            market_code=market,
            y_true=y_bin,
            p_true=p_bin,
            train=train,
            test=test,
            candidate_model=f"1x2_multi_{model_kind}",
            calibrated=False,
        )
    return model, metrics


def main() -> None:
    args = parse_args()
    df = fetch_dataset()
    if df.empty:
        raise RuntimeError("No fixtures-first rows available for training")

    df["match_datetime_utc"] = pd.to_datetime(
        df["match_datetime_utc"], utc=True, errors="coerce"
    )
    df["odds_snapshot_time_utc"] = pd.to_datetime(
        df["odds_snapshot_time_utc"], utc=True, errors="coerce"
    )
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
        raise RuntimeError(
            f"Feature contract mismatch in training dataset: {missing_csv}"
        )

    train_df, test_df = split_time_respecting(df)
    train_imp, test_imp, imputation = impute_for_split(train_df, test_df, features)
    selected_markets = resolve_market_selection(args)
    one_x2_dc_markets = {"1x2_h", "1x2_d", "1x2_a", "dc_1x", "dc_x2", "dc_12"}
    selected_1x2_dc = [market for market, _ in selected_markets if market in one_x2_dc_markets]
    selected_non_1x2_dc = [
        (market, target)
        for market, target in selected_markets
        if market not in one_x2_dc_markets
    ]
    market_df = pd.concat([train_imp, test_imp], ignore_index=True)
    baseline_metrics = load_baseline_metrics(args.promotion_baseline_path)

    metrics: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    selection_registry: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    if selected_1x2_dc:
        fold_rows, candidate_summary, selected_model = (
            evaluate_1x2_dc_multiclass_walkforward(
                frame=market_df,
                features=features,
                selected_markets=set(selected_1x2_dc),
                folds=max(1, int(args.folds)),
                min_fold_test_n=max(1, int(args.min_fold_test_n)),
            )
        )
        fold_metrics.extend(fold_rows)
        multiclass_model, multiclass_results = fit_1x2_dc_multiclass(
            train_df=train_imp,
            test_df=test_imp,
            features=features,
            model_kind=selected_model,
            selected_markets=set(selected_1x2_dc),
        )
        joblib.dump(multiclass_model, OUT_DIR / "gbm_1x2_dc_multiclass.pkl")

        for market in selected_1x2_dc:
            result = multiclass_results[market]
            metrics.append(result)

            baseline = baseline_metrics.get(market, {})
            baseline_auc = baseline.get("auc")
            baseline_brier = baseline.get("brier")
            current_auc = result.get("auc")
            current_brier = result.get("brier")
            auc_delta = (
                float(current_auc) - float(baseline_auc)
                if isinstance(current_auc, (int, float))
                and isinstance(baseline_auc, (int, float))
                else None
            )
            brier_delta = (
                float(current_brier) - float(baseline_brier)
                if isinstance(current_brier, (int, float))
                and isinstance(baseline_brier, (int, float))
                else None
            )
            promoted = bool(
                auc_delta is not None
                and brier_delta is not None
                and auc_delta >= float(args.promotion_min_auc_delta)
                and brier_delta <= 0.0
            )
            promotion_reason = (
                "auc_brier_gate_passed" if promoted else "auc_brier_gate_failed_or_missing"
            )
            selection_registry.append(
                {
                    "market_code": market,
                    "candidate_model": f"1x2_multi_{selected_model}",
                    "model_family": "1x2_dc_multiclass",
                    "candidate_summary": candidate_summary,
                    "baseline_auc": baseline_auc,
                    "baseline_brier": baseline_brier,
                    "current_auc": current_auc,
                    "current_brier": current_brier,
                    "auc_delta": auc_delta,
                    "brier_delta": brier_delta,
                    "promoted": promoted,
                    "promotion_reason": promotion_reason,
                }
            )
            summary_rows.append(
                {
                    "market": market,
                    "selected_model": f"1x2_multi_{selected_model}",
                    "holdout_auc": current_auc,
                    "holdout_brier": current_brier,
                    "candidate_auc_mean": candidate_summary[selected_model]["auc_mean"],
                    "candidate_auc_std": candidate_summary[selected_model]["auc_std"],
                    "candidate_brier_mean": candidate_summary[selected_model]["brier_mean"],
                    "candidate_brier_std": candidate_summary[selected_model]["brier_std"],
                    "folds_used": candidate_summary[selected_model]["folds_used"],
                    "promoted": promoted,
                }
            )

            print(
                f"{market}: model=1x2_multi_{selected_model} "
                f"AUC={result['auc']}, Acc={result['accuracy']:.3f}, "
                f"Brier={result['brier']:.4f}, test_n={result['test_n']}, "
                f"base_rate={result['base_rate_test']:.3f}, promoted={promoted}"
            )

    for market, target in selected_non_1x2_dc:
        fold_rows, candidate_summary, selected_model = (
            evaluate_market_candidates_walkforward(
                frame=market_df,
                target_col=target,
                market_code=market,
                features=features,
                folds=max(1, int(args.folds)),
                min_fold_test_n=max(1, int(args.min_fold_test_n)),
            )
        )
        fold_metrics.extend(fold_rows)

        result = fit_market_model(
            train_df=train_imp,
            test_df=test_imp,
            target_col=target,
            market_code=market,
            features=features,
            model_kind=selected_model,
        )
        metrics.append(result)

        baseline = baseline_metrics.get(market, {})
        baseline_auc = baseline.get("auc")
        baseline_brier = baseline.get("brier")
        current_auc = result.get("auc")
        current_brier = result.get("brier")
        auc_delta = (
            float(current_auc) - float(baseline_auc)
            if isinstance(current_auc, (int, float))
            and isinstance(baseline_auc, (int, float))
            else None
        )
        brier_delta = (
            float(current_brier) - float(baseline_brier)
            if isinstance(current_brier, (int, float))
            and isinstance(baseline_brier, (int, float))
            else None
        )
        promoted = bool(
            auc_delta is not None
            and brier_delta is not None
            and auc_delta >= float(args.promotion_min_auc_delta)
            and brier_delta <= 0.0
        )
        promotion_reason = (
            "auc_brier_gate_passed" if promoted else "auc_brier_gate_failed_or_missing"
        )
        selection_registry.append(
            {
                "market_code": market,
                "candidate_model": selected_model,
                "model_family": "binary",
                "candidate_summary": candidate_summary,
                "baseline_auc": baseline_auc,
                "baseline_brier": baseline_brier,
                "current_auc": current_auc,
                "current_brier": current_brier,
                "auc_delta": auc_delta,
                "brier_delta": brier_delta,
                "promoted": promoted,
                "promotion_reason": promotion_reason,
            }
        )
        summary_rows.append(
            {
                "market": market,
                "selected_model": selected_model,
                "holdout_auc": current_auc,
                "holdout_brier": current_brier,
                "candidate_auc_mean": candidate_summary[selected_model]["auc_mean"],
                "candidate_auc_std": candidate_summary[selected_model]["auc_std"],
                "candidate_brier_mean": candidate_summary[selected_model]["brier_mean"],
                "candidate_brier_std": candidate_summary[selected_model]["brier_std"],
                "folds_used": candidate_summary[selected_model]["folds_used"],
                "promoted": promoted,
            }
        )

        print(
            f"{market}: model={selected_model} "
            f"AUC={result['auc']}, Acc={result['accuracy']:.3f}, "
            f"Brier={result['brier']:.4f}, test_n={result['test_n']}, "
            f"base_rate={result['base_rate_test']:.3f}, promoted={promoted}"
        )

    with (OUT_DIR / "features.json").open("w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    with (OUT_DIR / "imputation.json").open("w", encoding="utf-8") as f:
        json.dump(imputation, f, indent=2)

    with (OUT_DIR / "metrics_walkforward.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (OUT_DIR / "metrics_walkforward_folds.json").open("w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)

    with (OUT_DIR / "model_selection_registry.json").open("w", encoding="utf-8") as f:
        json.dump(selection_registry, f, indent=2)

    with (OUT_DIR / "metrics_summary_phase1.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"Saved fixtures-first artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
