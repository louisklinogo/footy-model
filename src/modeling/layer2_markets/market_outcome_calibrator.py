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
try:
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - compatibility across sklearn versions
    FrozenEstimator = None


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.availability_features import (
    availability_feature_select_and_join,
    resolve_availability_feature_config,
)


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


def _add_handicap_target_columns(out: pd.DataFrame) -> pd.DataFrame:
    goal_diff = out["home_goals"] - out["away_goals"]

    # Legacy runtime handicap aliases.
    out["target_ah_h05"] = (goal_diff > 0).astype(int)
    out["target_ah_a05"] = (goal_diff < 0).astype(int)
    out["target_ah_h15"] = (goal_diff >= 2).astype(int)
    out["target_ah_a15"] = (goal_diff <= -2).astype(int)
    out["target_eh_h1"] = (goal_diff >= 2).astype(int)
    out["target_eh_a1"] = (goal_diff <= -2).astype(int)

    # Canonical EH 3-way selections for current one-goal displayed lines.
    out["target_eh3_0_1_home"] = (goal_diff >= 2).astype(int)
    out["target_eh3_0_1_draw"] = (goal_diff == 1).astype(int)
    out["target_eh3_0_1_away"] = (goal_diff <= 0).astype(int)
    out["target_eh3_1_0_home"] = (goal_diff >= 0).astype(int)
    out["target_eh3_1_0_draw"] = (goal_diff == -1).astype(int)
    out["target_eh3_1_0_away"] = (goal_diff <= -2).astype(int)

    # Canonical AH selections for current half-line scope.
    out["target_ah2_home_m05"] = (goal_diff > 0).astype(int)
    out["target_ah2_away_p05"] = (goal_diff <= 0).astype(int)
    out["target_ah2_away_m05"] = (goal_diff < 0).astype(int)
    out["target_ah2_home_p05"] = (goal_diff >= 0).astype(int)
    out["target_ah2_home_m15"] = (goal_diff >= 2).astype(int)
    out["target_ah2_away_p15"] = (goal_diff <= 1).astype(int)
    out["target_ah2_away_m15"] = (goal_diff <= -2).astype(int)
    out["target_ah2_home_p15"] = (goal_diff >= -1).astype(int)
    return out


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
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="Allow selected markets without baseline rows (they will be non-promotable).",
    )
    parser.add_argument(
        "--multiclass-calibration-fraction",
        type=float,
        default=0.2,
        help="Fraction of 1X2/DC training rows reserved for chronological calibration holdout.",
    )
    parser.add_argument(
        "--multiclass-calibration-min-rows",
        type=int,
        default=600,
        help="Minimum 1X2/DC rows reserved for chronological calibration holdout.",
    )
    parser.add_argument(
        "--disable-multiclass-calibration",
        action="store_true",
        help="Disable chrono holdout calibration for the 1X2/DC multiclass model.",
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


def fetch_dataset(prediction_lead_hours: int | None = None) -> pd.DataFrame:
    odds_cutoff_expr = "f.match_datetime_utc"
    prediction_cutoff_clause = ""
    if prediction_lead_hours is not None:
        lead_hours = max(0, int(prediction_lead_hours))
        odds_cutoff_expr = f"(f.match_datetime_utc - INTERVAL '{lead_hours} hours')"
        prediction_cutoff_clause = f"\n          AND p.created_at <= {odds_cutoff_expr}"

    conn = connect_db()
    try:
        availability_config = resolve_availability_feature_config(conn)
        availability_select, availability_join = availability_feature_select_and_join(
            fixture_alias="f",
            match_time_expr="f.match_datetime_utc",
            cutoff_expr=odds_cutoff_expr,
            config=availability_config,
        )
        query = f"""
    SELECT
        f.fixture_id,
        f.league_code,
        f.home_team_id,
        f.away_team_id,
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
        tph.rolling_lead_rate_1up AS home_rolling_lead_rate_1up,
        tph.rolling_lead_rate_1up_against AS home_rolling_lead_rate_1up_against,
        tph.rolling_lead_rate_2up AS home_rolling_lead_rate_2up,
        tph.rolling_lead_rate_2up_against AS home_rolling_lead_rate_2up_against,
        tpa.rolling_lead_rate_1up AS away_rolling_lead_rate_1up,
        tpa.rolling_lead_rate_1up_against AS away_rolling_lead_rate_1up_against,
        tpa.rolling_lead_rate_2up AS away_rolling_lead_rate_2up,
        tpa.rolling_lead_rate_2up_against AS away_rolling_lead_rate_2up_against,

        ff.home_formation,
        ff.away_formation,
        {_json_number_expr("l1h.metadata_json", "lambda")} AS lambda_home_l1,
        {_json_number_expr("l1a.metadata_json", "lambda")} AS lambda_away_l1,
        {_json_number_expr("l2h.metadata_json", "lambda")} AS adj_lambda_home_final,
        {_json_number_expr("l2a.metadata_json", "lambda")} AS adj_lambda_away_final,
        CASE
            WHEN ig.home_goals_inc = fr.home_goals AND ig.away_goals_inc = fr.away_goals
            THEN ig.home_goals_p1
            ELSE NULL
        END AS home_goals_p1,
        CASE
            WHEN ig.home_goals_inc = fr.home_goals AND ig.away_goals_inc = fr.away_goals
            THEN ig.away_goals_p1
            ELSE NULL
        END AS away_goals_p1,
        ils.first_home_lead_minute AS first_home_lead_minute,
        ils.first_away_lead_minute AS first_away_lead_minute,
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
        END AS rule_fired_away,
        {availability_select}
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
        SELECT
            COUNT(*) FILTER (
                WHERE i.incident_type = 'goal'
                  AND i.is_home IS TRUE
                  AND i.minute IS NOT NULL
                  AND i.minute <= 45
            )::int AS home_goals_p1,
            COUNT(*) FILTER (
                WHERE i.incident_type = 'goal'
                  AND i.is_home IS FALSE
                  AND i.minute IS NOT NULL
                  AND i.minute <= 45
            )::int AS away_goals_p1,
            COUNT(*) FILTER (
                WHERE i.incident_type = 'goal'
                  AND i.is_home IS TRUE
            )::int AS home_goals_inc,
            COUNT(*) FILTER (
                WHERE i.incident_type = 'goal'
                  AND i.is_home IS FALSE
            )::int AS away_goals_inc
        FROM fixture_incidents_sofascore i
        WHERE i.fixture_id = f.fixture_id
    ) ig ON true
    LEFT JOIN LATERAL (
        SELECT fom.odds_json, fom.snapshot_time_utc, fom.snapshot_type
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.provider = 'sofascore'
          AND fom.market_code = 'ou'
          AND fom.line_num = 1.5
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND fom.snapshot_time_utc <= {odds_cutoff_expr}
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
          AND fom.snapshot_time_utc <= {odds_cutoff_expr}
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
          AND fom.snapshot_time_utc <= {odds_cutoff_expr}
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
          AND fom.snapshot_time_utc <= {odds_cutoff_expr}
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
          AND fom.snapshot_time_utc <= {odds_cutoff_expr}
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
          AND fom.snapshot_time_utc <= {odds_cutoff_expr}
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
          AND fom.snapshot_time_utc <= {odds_cutoff_expr}
        ORDER BY (fom.snapshot_type = 'latest_pre_match') DESC, fom.snapshot_time_utc DESC
        LIMIT 1
    ) odc105 ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'lambda_xgb'
          AND p.market_code = 'lambda_home'
          {prediction_cutoff_clause}
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l1h ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'lambda_xgb'
          AND p.market_code = 'lambda_away'
          {prediction_cutoff_clause}
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l1a ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'situational_xgb'
          AND p.market_code = 'adj_lambda_home'
          {prediction_cutoff_clause}
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l2h ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'situational_xgb'
          AND p.market_code = 'adj_lambda_away'
          {prediction_cutoff_clause}
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l2a ON true
    {availability_join}
    WHERE f.status = 'ft'
      AND f.match_datetime_utc IS NOT NULL
      AND fr.home_goals IS NOT NULL
      AND fr.away_goals IS NOT NULL
    ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC;
    """
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

    # === HANDICAP ===
    out = _add_handicap_target_columns(out)

    # === TEAM TOTALS ===
    out["target_ho15"] = (out["home_goals"] >= 2).astype(int)  # Home team over 1.5
    out["target_ao15"] = (out["away_goals"] >= 2).astype(int)  # Away team over 1.5

    # === TOTAL MULTIGOALS (range bins) ===
    out["target_mg_0"] = (out["total_goals"] == 0).astype(int)
    out["target_mg_1_2"] = out["total_goals"].between(1, 2).astype(int)
    out["target_mg_1_3"] = out["total_goals"].between(1, 3).astype(int)
    out["target_mg_1_4"] = out["total_goals"].between(1, 4).astype(int)
    out["target_mg_1_5"] = out["total_goals"].between(1, 5).astype(int)
    out["target_mg_1_6"] = out["total_goals"].between(1, 6).astype(int)
    out["target_mg_2_3"] = out["total_goals"].between(2, 3).astype(int)
    out["target_mg_2_4"] = out["total_goals"].between(2, 4).astype(int)
    out["target_mg_2_5"] = out["total_goals"].between(2, 5).astype(int)
    out["target_mg_2_6"] = out["total_goals"].between(2, 6).astype(int)
    out["target_mg_3_4"] = out["total_goals"].between(3, 4).astype(int)
    out["target_mg_3_5"] = out["total_goals"].between(3, 5).astype(int)
    out["target_mg_3_6"] = out["total_goals"].between(3, 6).astype(int)
    out["target_mg_4_5"] = out["total_goals"].between(4, 5).astype(int)
    out["target_mg_4_6"] = out["total_goals"].between(4, 6).astype(int)
    out["target_mg_5_6"] = out["total_goals"].between(5, 6).astype(int)
    out["target_mg_7p"] = (out["total_goals"] >= 7).astype(int)

    # === HOME / AWAY MULTIGOALS ===
    out["target_hmg_0"] = (out["home_goals"] == 0).astype(int)
    out["target_hmg_1_2"] = out["home_goals"].between(1, 2).astype(int)
    out["target_hmg_1_3"] = out["home_goals"].between(1, 3).astype(int)
    out["target_hmg_2_3"] = out["home_goals"].between(2, 3).astype(int)
    out["target_hmg_4p"] = (out["home_goals"] >= 4).astype(int)

    out["target_amg_0"] = (out["away_goals"] == 0).astype(int)
    out["target_amg_1_2"] = out["away_goals"].between(1, 2).astype(int)
    out["target_amg_1_3"] = out["away_goals"].between(1, 3).astype(int)
    out["target_amg_2_3"] = out["away_goals"].between(2, 3).astype(int)
    out["target_amg_4p"] = (out["away_goals"] >= 4).astype(int)

    # === MULTISCORES (grouped exact scorelines) ===
    h = out["home_goals"]
    a = out["away_goals"]
    ms_h_1_0_2_0_3_0 = ((h == 1) & (a == 0)) | ((h == 2) & (a == 0)) | ((h == 3) & (a == 0))
    ms_a_0_1_0_2_0_3 = ((h == 0) & (a == 1)) | ((h == 0) & (a == 2)) | ((h == 0) & (a == 3))
    ms_h_4_0_5_0_6_0 = ((h == 4) & (a == 0)) | ((h == 5) & (a == 0)) | ((h == 6) & (a == 0))
    ms_a_0_4_0_5_0_6 = ((h == 0) & (a == 4)) | ((h == 0) & (a == 5)) | ((h == 0) & (a == 6))
    ms_h_2_1_3_1_4_1 = ((h == 2) & (a == 1)) | ((h == 3) & (a == 1)) | ((h == 4) & (a == 1))
    ms_h_1_2_1_3_1_4 = ((h == 1) & (a == 2)) | ((h == 1) & (a == 3)) | ((h == 1) & (a == 4))
    ms_h_3_2_4_2_5_1 = ((h == 3) & (a == 2)) | ((h == 4) & (a == 2)) | ((h == 5) & (a == 1))
    ms_a_2_3_2_4_1_5 = ((h == 2) & (a == 3)) | ((h == 2) & (a == 4)) | ((h == 1) & (a == 5))
    ms_draw = h == a

    out["target_ms_h_1_0_2_0_3_0"] = ms_h_1_0_2_0_3_0.astype(int)
    out["target_ms_a_0_1_0_2_0_3"] = ms_a_0_1_0_2_0_3.astype(int)
    out["target_ms_h_4_0_5_0_6_0"] = ms_h_4_0_5_0_6_0.astype(int)
    out["target_ms_a_0_4_0_5_0_6"] = ms_a_0_4_0_5_0_6.astype(int)
    out["target_ms_h_2_1_3_1_4_1"] = ms_h_2_1_3_1_4_1.astype(int)
    out["target_ms_h_1_2_1_3_1_4"] = ms_h_1_2_1_3_1_4.astype(int)
    out["target_ms_h_3_2_4_2_5_1"] = ms_h_3_2_4_2_5_1.astype(int)
    out["target_ms_a_2_3_2_4_1_5"] = ms_a_2_3_2_4_1_5.astype(int)
    out["target_ms_draw"] = ms_draw.astype(int)
    out["target_ms_other_homewin"] = (
        (h > a)
        & ~(ms_h_1_0_2_0_3_0 | ms_h_4_0_5_0_6_0 | ms_h_2_1_3_1_4_1 | ms_h_3_2_4_2_5_1)
    ).astype(int)
    out["target_ms_other_awaywin"] = (
        (h < a)
        & ~(ms_a_0_1_0_2_0_3 | ms_a_0_4_0_5_0_6 | ms_h_1_2_1_3_1_4 | ms_a_2_3_2_4_1_5)
    ).astype(int)

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

    out = _add_xg_anchor_features(out)

    return out


def _add_xg_anchor_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fixture_id",
        "league_code",
        "match_datetime_utc",
        "home_team_id",
        "away_team_id",
        "home_rolling_xg",
        "away_rolling_xg",
    }
    if not required.issubset(df.columns):
        out = df.copy()
        for col in (
            "home_season_baseline_xg",
            "away_season_baseline_xg",
            "home_recent_xg_mean_5",
            "away_recent_xg_mean_5",
            "home_recent_vs_baseline_zscore",
            "away_recent_vs_baseline_zscore",
            "home_regressed_recent_xg",
            "away_regressed_recent_xg",
            "regressed_xg_diff",
            "recent_vs_baseline_gap",
            "home_season_baseline_xg_is_missing",
            "away_season_baseline_xg_is_missing",
            "home_recent_xg_mean_5_is_missing",
            "away_recent_xg_mean_5_is_missing",
        ):
            out[col] = np.nan
        return out

    out = df.copy()
    ordered = (
        out.reset_index()
        .rename(columns={"index": "_orig_idx"})
        .sort_values(["match_datetime_utc", "fixture_id"])
        .reset_index(drop=True)
    )
    ordered["match_datetime_utc"] = pd.to_datetime(
        ordered["match_datetime_utc"], utc=True, errors="coerce"
    )
    ordered["home_rolling_xg"] = pd.to_numeric(
        ordered["home_rolling_xg"], errors="coerce"
    )
    ordered["away_rolling_xg"] = pd.to_numeric(
        ordered["away_rolling_xg"], errors="coerce"
    )

    home_rows = ordered[
        ["_orig_idx", "fixture_id", "league_code", "match_datetime_utc", "home_team_id"]
    ].rename(columns={"home_team_id": "team_id"}).copy()
    home_rows["side"] = "home"
    home_rows["team_xg_recent"] = ordered["home_rolling_xg"]

    away_rows = ordered[
        ["_orig_idx", "fixture_id", "league_code", "match_datetime_utc", "away_team_id"]
    ].rename(columns={"away_team_id": "team_id"}).copy()
    away_rows["side"] = "away"
    away_rows["team_xg_recent"] = ordered["away_rolling_xg"]

    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long["team_xg_recent"] = pd.to_numeric(long["team_xg_recent"], errors="coerce")
    long = long.sort_values(
        ["team_id", "match_datetime_utc", "fixture_id", "side"]
    ).reset_index(drop=True)

    by_team = long.groupby("team_id", dropna=False)["team_xg_recent"]
    by_league = long.groupby("league_code", dropna=False)["team_xg_recent"]

    long["team_prior_n"] = (
        long.groupby("team_id", dropna=False).cumcount().astype(float)
    )
    long["season_baseline_xg"] = by_team.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    long["recent_xg_mean_5"] = by_team.transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    long["recent_xg_std"] = by_team.transform(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )
    long["league_baseline_xg"] = by_league.transform(
        lambda s: s.shift(1).expanding(min_periods=5).mean()
    )

    global_baseline = float(long["team_xg_recent"].median(skipna=True))
    if np.isnan(global_baseline):
        global_baseline = 1.2

    long["season_baseline_xg"] = (
        long["season_baseline_xg"]
        .fillna(long["league_baseline_xg"])
        .fillna(global_baseline)
    )
    long["recent_xg_mean_5"] = (
        long["recent_xg_mean_5"]
        .fillna(long["season_baseline_xg"])
        .fillna(global_baseline)
    )
    long["recent_xg_std"] = long["recent_xg_std"].fillna(0.35).clip(lower=0.15)

    long["recent_vs_baseline_zscore"] = (
        (long["recent_xg_mean_5"] - long["season_baseline_xg"]) / long["recent_xg_std"]
    ).clip(-4.0, 4.0)
    prior_weight = (long["team_prior_n"] / (long["team_prior_n"] + 10.0)).clip(0.0, 1.0)
    long["regressed_recent_xg"] = (
        prior_weight * long["recent_xg_mean_5"]
        + (1.0 - prior_weight) * long["season_baseline_xg"]
    )

    long["season_baseline_xg_is_missing"] = (
        long["team_prior_n"] <= 0.0
    ).astype(float)
    long["recent_xg_mean_5_is_missing"] = (
        long["team_prior_n"] <= 0.0
    ).astype(float)

    for side in ("home", "away"):
        side_frame = long[long["side"] == side].set_index("_orig_idx")
        out.loc[
            side_frame.index, f"{side}_season_baseline_xg"
        ] = side_frame["season_baseline_xg"].to_numpy(dtype=float)
        out.loc[
            side_frame.index, f"{side}_recent_xg_mean_5"
        ] = side_frame["recent_xg_mean_5"].to_numpy(dtype=float)
        out.loc[
            side_frame.index, f"{side}_recent_vs_baseline_zscore"
        ] = side_frame["recent_vs_baseline_zscore"].to_numpy(dtype=float)
        out.loc[
            side_frame.index, f"{side}_regressed_recent_xg"
        ] = side_frame["regressed_recent_xg"].to_numpy(dtype=float)
        out.loc[
            side_frame.index, f"{side}_season_baseline_xg_is_missing"
        ] = side_frame["season_baseline_xg_is_missing"].to_numpy(dtype=float)
        out.loc[
            side_frame.index, f"{side}_recent_xg_mean_5_is_missing"
        ] = side_frame["recent_xg_mean_5_is_missing"].to_numpy(dtype=float)

    out["regressed_xg_diff"] = (
        out["home_regressed_recent_xg"] - out["away_regressed_recent_xg"]
    )
    out["recent_vs_baseline_gap"] = (
        out["home_recent_vs_baseline_zscore"] - out["away_recent_vs_baseline_zscore"]
    )
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
            "home_season_baseline_xg",
            "away_season_baseline_xg",
            "home_recent_xg_mean_5",
            "away_recent_xg_mean_5",
            "home_recent_vs_baseline_zscore",
            "away_recent_vs_baseline_zscore",
            "home_regressed_recent_xg",
            "away_regressed_recent_xg",
            "regressed_xg_diff",
            "recent_vs_baseline_gap",
            "home_season_baseline_xg_is_missing",
            "away_season_baseline_xg_is_missing",
            "home_recent_xg_mean_5_is_missing",
            "away_recent_xg_mean_5_is_missing",
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


def validate_baseline_coverage(
    *,
    selected_market_codes: list[str],
    baseline_metrics: dict[str, dict[str, float | None]],
    baseline_path: Path,
    allow_missing: bool,
) -> set[str]:
    missing = sorted(
        {
            market
            for market in selected_market_codes
            if market not in baseline_metrics
        }
    )
    if missing and not allow_missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            "Promotion baseline is missing required markets: "
            f"{missing_csv}. Baseline path={baseline_path}. "
            "Provide a complete baseline or pass --allow-missing-baseline."
        )
    return set(missing)


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


def _mean_binary_brier_for_markets(
    *,
    y_class: np.ndarray,
    p_home: np.ndarray,
    p_draw: np.ndarray,
    p_away: np.ndarray,
    markets: set[str],
) -> float:
    briers: list[float] = []
    for market in sorted(markets):
        y_bin = _binary_target_for_market(market, y_class)
        p_bin = np.clip(_binary_prob_for_market(market, p_home, p_draw, p_away), 0.001, 0.999)
        if len(np.unique(y_bin)) < 2:
            continue
        briers.append(float(brier_score_loss(y_bin, p_bin)))
    if not briers:
        return float("nan")
    return float(np.mean(briers))


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
    calibration_fraction: float,
    calibration_min_rows: int,
    enable_calibration: bool,
) -> tuple[object, dict[str, dict[str, object]], dict[str, object]]:
    train = train_df[train_df["home_goals"].notna() & train_df["away_goals"].notna()].copy()
    test = test_df[test_df["home_goals"].notna() & test_df["away_goals"].notna()].copy()
    if train.empty or test.empty:
        raise ValueError("Insufficient rows for 1x2/dc multiclass fit.")

    train = train.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(drop=True)
    test = test.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(drop=True)
    y_train = _y_1x2_class(train)
    y_test = _y_1x2_class(test)
    model: object | None = None

    markets = ("1x2_h", "1x2_d", "1x2_a", "dc_1x", "dc_x2", "dc_12")
    requested = {market for market in selected_markets if market in markets}
    if not requested:
        raise ValueError("No 1x2/dc markets selected for multiclass fit.")

    calibration_info: dict[str, object] = {
        "enabled": bool(enable_calibration),
        "method": None,
        "brier_eval": None,
        "core_train_n": int(len(train)),
        "calibration_fit_n": 0,
        "calibration_eval_n": 0,
        "status": "not_attempted",
    }
    candidate_model_label = f"1x2_multi_{model_kind}"
    calibrated = False

    calibration_fraction = float(np.clip(calibration_fraction, 0.05, 0.5))
    min_rows = max(1, int(calibration_min_rows))
    if enable_calibration and len(train) >= (min_rows + 200):
        holdout_n = max(min_rows, int(len(train) * calibration_fraction))
        max_holdout = max(0, len(train) - 200)
        holdout_n = min(holdout_n, max_holdout)
        if holdout_n >= 80:
            core = train.iloc[:-holdout_n]
            calibration_pool = train.iloc[-holdout_n:]
            split_idx = int(len(calibration_pool) * 0.5)
            calib_fit = calibration_pool.iloc[:split_idx]
            calib_eval = calibration_pool.iloc[split_idx:]
            y_core = _y_1x2_class(core)
            y_calib_fit = _y_1x2_class(calib_fit)
            y_calib_eval = _y_1x2_class(calib_eval)
            calibration_info.update(
                {
                    "core_train_n": int(len(core)),
                    "calibration_fit_n": int(len(calib_fit)),
                    "calibration_eval_n": int(len(calib_eval)),
                }
            )

            if (
                len(core) > 0
                and len(calib_fit) > 0
                and len(calib_eval) > 0
                and len(np.unique(y_core)) >= 2
                and len(np.unique(y_calib_fit)) >= 2
            ):
                base_for_cal = _build_base_estimator(model_kind)
                base_for_cal.fit(core[features], y_core)
                best_model: object | None = None
                best_method: str | None = None
                best_brier: float | None = None

                for method in ("isotonic", "sigmoid"):
                    try:
                        if FrozenEstimator is not None:
                            calibrated_model = CalibratedClassifierCV(
                                estimator=FrozenEstimator(base_for_cal),
                                method=method,
                                cv=None,
                            )
                        else:
                            calibrated_model = CalibratedClassifierCV(
                                estimator=base_for_cal,
                                method=method,
                                cv="prefit",
                            )
                        calibrated_model.fit(calib_fit[features], y_calib_fit)
                        p_h_eval, p_d_eval, p_a_eval = _multiclass_probs(
                            calibrated_model, calib_eval[features]
                        )
                        eval_brier = _mean_binary_brier_for_markets(
                            y_class=y_calib_eval,
                            p_home=p_h_eval,
                            p_draw=p_d_eval,
                            p_away=p_a_eval,
                            markets=requested,
                        )
                        if not np.isfinite(eval_brier):
                            continue
                        if best_brier is None or eval_brier < best_brier:
                            best_brier = float(eval_brier)
                            best_method = method
                            best_model = calibrated_model
                    except Exception:
                        continue

                if best_model is not None and best_method is not None and best_brier is not None:
                    model = best_model
                    calibrated = True
                    candidate_model_label = f"1x2_multi_{model_kind}_{best_method}"
                    calibration_info.update(
                        {
                            "method": best_method,
                            "brier_eval": float(best_brier),
                            "status": "selected",
                        }
                    )
                else:
                    calibration_info["status"] = "no_valid_method"
            else:
                calibration_info["status"] = "insufficient_class_diversity"
        else:
            calibration_info["status"] = "holdout_too_small"
    elif enable_calibration:
        calibration_info["status"] = "insufficient_rows"
    else:
        calibration_info["status"] = "disabled"

    if model is None:
        model = _build_base_estimator(model_kind)
        model.fit(train[features], y_train)

    p_home, p_draw, p_away = _multiclass_probs(model, test[features])

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
            candidate_model=candidate_model_label,
            calibrated=calibrated,
        )
        metrics[market]["calibration_method"] = calibration_info["method"]
        metrics[market]["calibration_status"] = calibration_info["status"]
        metrics[market]["calibration_eval_brier"] = calibration_info["brier_eval"]
    return model, metrics, calibration_info


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
    selected_market_codes = [market for market, _ in selected_markets]
    one_x2_dc_markets = {"1x2_h", "1x2_d", "1x2_a", "dc_1x", "dc_x2", "dc_12"}
    selected_1x2_dc = [market for market, _ in selected_markets if market in one_x2_dc_markets]
    selected_non_1x2_dc = [
        (market, target)
        for market, target in selected_markets
        if market not in one_x2_dc_markets
    ]
    market_df = pd.concat([train_imp, test_imp], ignore_index=True)
    baseline_metrics = load_baseline_metrics(args.promotion_baseline_path)
    missing_baselines = validate_baseline_coverage(
        selected_market_codes=selected_market_codes,
        baseline_metrics=baseline_metrics,
        baseline_path=args.promotion_baseline_path,
        allow_missing=bool(args.allow_missing_baseline),
    )
    if missing_baselines:
        print(
            "Warning: missing baseline rows for markets (non-promotable): "
            + ", ".join(sorted(missing_baselines))
        )

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
        multiclass_model, multiclass_results, multiclass_calibration = fit_1x2_dc_multiclass(
            train_df=train_imp,
            test_df=test_imp,
            features=features,
            model_kind=selected_model,
            selected_markets=set(selected_1x2_dc),
            calibration_fraction=float(args.multiclass_calibration_fraction),
            calibration_min_rows=int(args.multiclass_calibration_min_rows),
            enable_calibration=not bool(args.disable_multiclass_calibration),
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
            baseline_missing = market in missing_baselines
            if baseline_missing:
                promotion_reason = "baseline_missing"
            elif promoted:
                promotion_reason = "auc_brier_gate_passed"
            else:
                promotion_reason = "auc_brier_gate_failed"
            selection_registry.append(
                {
                    "market_code": market,
                    "candidate_model": result["candidate_model"],
                    "model_family": "1x2_dc_multiclass",
                    "multiclass_calibration": multiclass_calibration,
                    "candidate_summary": candidate_summary,
                    "baseline_auc": baseline_auc,
                    "baseline_brier": baseline_brier,
                    "current_auc": current_auc,
                    "current_brier": current_brier,
                    "auc_delta": auc_delta,
                    "brier_delta": brier_delta,
                    "baseline_missing": baseline_missing,
                    "promoted": promoted,
                    "promotion_reason": promotion_reason,
                }
            )
            summary_rows.append(
                {
                    "market": market,
                    "selected_model": result["candidate_model"],
                    "holdout_auc": current_auc,
                    "holdout_brier": current_brier,
                    "candidate_auc_mean": candidate_summary[selected_model]["auc_mean"],
                    "candidate_auc_std": candidate_summary[selected_model]["auc_std"],
                    "candidate_brier_mean": candidate_summary[selected_model]["brier_mean"],
                    "candidate_brier_std": candidate_summary[selected_model]["brier_std"],
                    "folds_used": candidate_summary[selected_model]["folds_used"],
                    "calibrated": result["calibrated"],
                    "calibration_method": result.get("calibration_method"),
                    "calibration_status": result.get("calibration_status"),
                    "baseline_missing": baseline_missing,
                    "promoted": promoted,
                }
            )

            print(
                f"{market}: model={result['candidate_model']} "
                f"AUC={result['auc']}, Acc={result['accuracy']:.3f}, "
                f"Brier={result['brier']:.4f}, test_n={result['test_n']}, "
                f"base_rate={result['base_rate_test']:.3f}, "
                f"calibration={result.get('calibration_status')}, promoted={promoted}"
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
        baseline_missing = market in missing_baselines
        if baseline_missing:
            promotion_reason = "baseline_missing"
        elif promoted:
            promotion_reason = "auc_brier_gate_passed"
        else:
            promotion_reason = "auc_brier_gate_failed"
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
                "baseline_missing": baseline_missing,
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
                "baseline_missing": baseline_missing,
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
