"""
Generate fixtures-first pre-match predictions and upsert into predictions.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.availability_features import (
    availability_feature_select_and_join,
    resolve_availability_feature_config,
)
from src.pricing.markov import MarkovPricer


MODEL_DIR = Path("model_artifacts/market_models")
MODEL_NAME = "market_outcome_gbm"
MODEL_VERSION = "fixtures_first_prematch_v1"
LEGACY_HANDICAP_MARKETS = (
    "ah_h05",
    "ah_a05",
    "ah_h15",
    "ah_a15",
    "eh_h1",
    "eh_a1",
)
CANONICAL_HANDICAP_MARKETS = (
    "ah2_home_m05",
    "ah2_away_p05",
    "ah2_away_m05",
    "ah2_home_p05",
    "ah2_home_m15",
    "ah2_away_p15",
    "ah2_away_m15",
    "ah2_home_p15",
    "eh3_0_1_home",
    "eh3_0_1_draw",
    "eh3_0_1_away",
    "eh3_1_0_home",
    "eh3_1_0_draw",
    "eh3_1_0_away",
)
MARKETS = (
    # Goals
    "o15",
    "u35",
    # Corners Totals
    "c75",
    "c85",
    "c95",
    "c105",
    # Corners Home Team
    "hc25",
    "hc35",
    "hc45",
    "hc55",
    # Corners Away Team
    "ac25",
    "ac35",
    "ac45",
    "ac55",
    # 1X2
    "1x2_h",
    "1x2_d",
    "1x2_a",
    # Double Chance
    "dc_1x",
    "dc_x2",
    "dc_12",
    # Team Totals
    "ho15",
    "ao15",
    # Legacy Handicap Markets
    *LEGACY_HANDICAP_MARKETS,
    # Canonical Handicap Markets
    *CANONICAL_HANDICAP_MARKETS,
    # Anytime Lead Markets
    "h_1up",
    "a_1up",
    "h_2up",
    "a_2up",
)
MULTICLASS_MARKETS = ("1x2_h", "1x2_d", "1x2_a", "dc_1x", "dc_x2", "dc_12")

_MARKOV_PRICER = MarkovPricer(max_goals=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict fixtures-first markets and write to DB"
    )
    parser.add_argument(
        "--league", type=str, default=None, help="Optional league_code filter"
    )
    parser.add_argument(
        "--days", type=int, default=3, help="Prediction horizon in days"
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=None,
        help="If set, backfill FT fixtures within the last N days instead of scheduled fixtures.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max fixtures")
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


def load_artifacts() -> tuple[
    list[str],
    dict[str, dict[str, dict[str, float] | float]],
    dict[str, object],
    dict[str, str],
    object | None,
    set[str],
]:
    with (MODEL_DIR / "features.json").open("r", encoding="utf-8") as f:
        features = json.load(f)
    with (MODEL_DIR / "imputation.json").open("r", encoding="utf-8") as f:
        imputation = json.load(f)
    models: dict[str, object] = {}
    model_failures: dict[str, str] = {}
    for market in MARKETS:
        model_path = MODEL_DIR / f"gbm_{market}.pkl"
        if not model_path.exists():
            model_failures[market] = "missing_artifact"
            continue
        try:
            models[market] = joblib.load(model_path)
        except Exception as exc:
            model_failures[market] = f"load_error:{exc.__class__.__name__}"
    multiclass_model: object | None = None
    multiclass_path = MODEL_DIR / "gbm_1x2_dc_multiclass.pkl"
    if multiclass_path.exists():
        try:
            multiclass_model = joblib.load(multiclass_path)
        except Exception:
            multiclass_model = None
    multiclass_markets = _load_multiclass_markets_from_registry()
    return (
        features,
        imputation,
        models,
        model_failures,
        multiclass_model,
        multiclass_markets,
    )


def _load_multiclass_markets_from_registry() -> set[str]:
    path = MODEL_DIR / "model_selection_registry.json"
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, list):
        return set()
    selected: set[str] = set()
    allowed = set(MULTICLASS_MARKETS)
    for row in payload:
        if not isinstance(row, dict):
            continue
        market = str(row.get("market_code") or "").strip()
        if market not in allowed:
            continue
        family = str(row.get("model_family") or "").strip()
        candidate = str(row.get("candidate_model") or "").strip()
        if family == "1x2_dc_multiclass" or candidate.startswith("1x2_multi_"):
            selected.add(market)
    return selected


def fetch_candidate_fixtures(
    days: int,
    league: str | None,
    limit: int | None,
    backfill_days: int | None,
) -> pd.DataFrame:
    conn = connect_db()
    try:
        availability_config = resolve_availability_feature_config(conn)
        availability_select, availability_join = availability_feature_select_and_join(
            fixture_alias="f",
            match_time_expr="f.match_datetime_utc",
            cutoff_expr="f.match_datetime_utc",
            config=availability_config,
        )
        base_query = f"""
    SELECT
        f.fixture_id,
        f.league_code,
        f.home_team_id,
        f.away_team_id,
        f.match_datetime_utc,
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
        tph.rolling_errors_lead_to_shot AS home_rolling_errors_lead_to_shot,
        tph.rolling_errors_lead_to_shot_against AS home_rolling_errors_lead_to_shot_against,
        tph.rolling_tackles_pct AS home_rolling_tackles_pct,
        tph.rolling_tackles_pct_against AS home_rolling_tackles_pct_against,
        tph.rolling_possession AS home_rolling_possession,
        tph.rolling_possession_against AS home_rolling_possession_against,
        tph.rolling_xg_p1 AS home_rolling_xg_p1,
        tph.rolling_xg_p1_against AS home_rolling_xg_p1_against,
        tph.rolling_sot_p1 AS home_rolling_sot_p1,
        tph.rolling_sot_p1_against AS home_rolling_sot_p1_against,
        tph.rolling_xg_h2_delta AS home_rolling_xg_h2_delta,
        tph.rolling_xg_h2_delta_against AS home_rolling_xg_h2_delta_against,
        tph.rolling_sot_h2_delta AS home_rolling_sot_h2_delta,
        tph.rolling_sot_h2_delta_against AS home_rolling_sot_h2_delta_against,
        tph.rolling_rest_days AS home_rolling_rest_days,
        tph.rolling_lead_rate_1up AS home_rolling_lead_rate_1up,
        tph.rolling_lead_rate_1up_against AS home_rolling_lead_rate_1up_against,
        tph.rolling_lead_rate_2up AS home_rolling_lead_rate_2up,
        tph.rolling_lead_rate_2up_against AS home_rolling_lead_rate_2up_against,
        tph.fidelity_score AS home_fidelity_score,
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
        tpa.rolling_errors_lead_to_shot AS away_rolling_errors_lead_to_shot,
        tpa.rolling_errors_lead_to_shot_against AS away_rolling_errors_lead_to_shot_against,
        tpa.rolling_tackles_pct AS away_rolling_tackles_pct,
        tpa.rolling_tackles_pct_against AS away_rolling_tackles_pct_against,
        tpa.rolling_possession AS away_rolling_possession,
        tpa.rolling_possession_against AS away_rolling_possession_against,
        tpa.rolling_xg_p1 AS away_rolling_xg_p1,
        tpa.rolling_xg_p1_against AS away_rolling_xg_p1_against,
        tpa.rolling_sot_p1 AS away_rolling_sot_p1,
        tpa.rolling_sot_p1_against AS away_rolling_sot_p1_against,
        tpa.rolling_xg_h2_delta AS away_rolling_xg_h2_delta,
        tpa.rolling_xg_h2_delta_against AS away_rolling_xg_h2_delta_against,
        tpa.rolling_sot_h2_delta AS away_rolling_sot_h2_delta,
        tpa.rolling_sot_h2_delta_against AS away_rolling_sot_h2_delta_against,
        tpa.rolling_rest_days AS away_rolling_rest_days,
        tpa.rolling_lead_rate_1up AS away_rolling_lead_rate_1up,
        tpa.rolling_lead_rate_1up_against AS away_rolling_lead_rate_1up_against,
        tpa.rolling_lead_rate_2up AS away_rolling_lead_rate_2up,
        tpa.rolling_lead_rate_2up_against AS away_rolling_lead_rate_2up_against,
        tpa.fidelity_score AS away_fidelity_score,
        {_json_number_expr("l1h.metadata_json", "lambda")} AS lambda_home_l1,
        {_json_number_expr("l1a.metadata_json", "lambda")} AS lambda_away_l1,
        {_json_number_expr("l2h.metadata_json", "lambda")} AS adj_lambda_home_final,
        {_json_number_expr("l2a.metadata_json", "lambda")} AS adj_lambda_away_final,
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
    LEFT JOIN team_premium_snapshots tph
        ON tph.fixture_id = f.fixture_id
       AND tph.is_home = true
    LEFT JOIN team_premium_snapshots tpa
        ON tpa.fixture_id = f.fixture_id
       AND tpa.is_home = false
    LEFT JOIN LATERAL (
        SELECT fom.snapshot_time_utc, fom.snapshot_type, fom.odds_json
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
        SELECT fom.snapshot_time_utc, fom.snapshot_type, fom.odds_json
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
        SELECT fom.snapshot_time_utc, fom.snapshot_type, fom.odds_json
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
        SELECT fom.snapshot_time_utc, fom.snapshot_type, fom.odds_json
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
        SELECT fom.snapshot_time_utc, fom.snapshot_type, fom.odds_json
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
        SELECT fom.snapshot_time_utc, fom.snapshot_type, fom.odds_json
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
        SELECT fom.snapshot_time_utc, fom.snapshot_type, fom.odds_json
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
    {availability_join}
    WHERE f.match_datetime_utc IS NOT NULL
    """
        params: list[object] = []
        if backfill_days is None:
            base_query += """
          AND f.status = 'scheduled'
          AND f.match_datetime_utc > NOW()
          AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
        """
            params.append(days)
        else:
            base_query += """
          AND f.status = 'ft'
          AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
          AND f.match_datetime_utc <= NOW()
        """
            params.append(backfill_days)
        if league:
            base_query += " AND f.league_code = %s"
            params.append(league)
        base_query += " ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC"
        if limit is not None:
            base_query += " LIMIT %s"
            params.append(limit)

        df = pd.read_sql(base_query, conn, params=tuple(params))
    finally:
        conn.close()
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["xg_net_diff"] = out["home_rolling_xg"] - out["away_rolling_xg_against"]
    out["xgot_net_diff"] = out["home_rolling_xgot"] - out["away_rolling_xgot_against"]
    out["xa_net_diff"] = out["home_rolling_xa"] - out["away_rolling_xa_against"]
    out["corners_net_diff"] = (
        out["home_rolling_corners"] - out["away_rolling_corners_against"]
    )
    out["sample_size_diff"] = out["home_sample_size"] - out["away_sample_size"]
    out["lead_rate_1up_diff"] = (
        out["home_rolling_lead_rate_1up"] - out["away_rolling_lead_rate_1up"]
    )
    out["lead_rate_2up_diff"] = (
        out["home_rolling_lead_rate_2up"] - out["away_rolling_lead_rate_2up"]
    )
    out["rest_days_sum"] = out["home_rolling_rest_days"] + out["away_rolling_rest_days"]
    out["rest_days_diff"] = out["home_rolling_rest_days"] - out["away_rolling_rest_days"]
    out["fidelity_score_sum"] = out["home_fidelity_score"] + out["away_fidelity_score"]
    out["fidelity_score_diff"] = out["home_fidelity_score"] - out["away_fidelity_score"]
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
    for feature in (
        "home_rolling_errors_lead_to_shot",
        "away_rolling_errors_lead_to_shot",
        "home_rolling_tackles_pct",
        "away_rolling_tackles_pct",
        "home_rolling_rest_days",
        "away_rolling_rest_days",
        "home_fidelity_score",
        "away_fidelity_score",
        "home_rolling_lead_rate_1up",
        "away_rolling_lead_rate_1up",
        "home_rolling_lead_rate_2up",
        "away_rolling_lead_rate_2up",
    ):
        out[f"{feature}_is_missing"] = out[feature].isna().astype(float)
    return out


def apply_imputation(
    df: pd.DataFrame, features: list[str], imputation: dict[str, object]
) -> pd.DataFrame:
    out = df.copy()
    global_medians = imputation.get("global_medians", {})
    league_medians = imputation.get("league_medians", {})

    for feature in features:
        if feature not in out.columns:
            out[feature] = np.nan

        league_map = league_medians.get(feature, {})
        league_fill = out["league_code"].map(league_map)
        global_fill = global_medians.get(feature, 0.0)
        out[feature] = out[feature].fillna(league_fill).fillna(global_fill).fillna(0.0)

    return out


def _safe_float(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out):
        return default
    return out


def _resolve_backbone_lambdas(fixture: pd.Series) -> tuple[float, float, str]:
    adj_home = _safe_float(fixture.get("adj_lambda_home_final"), float("nan"))
    adj_away = _safe_float(fixture.get("adj_lambda_away_final"), float("nan"))
    if (
        np.isfinite(adj_home)
        and np.isfinite(adj_away)
        and adj_home > 0.0
        and adj_away > 0.0
    ):
        return max(0.01, adj_home), max(0.01, adj_away), "adj_lambda_final"

    l1_home = _safe_float(fixture.get("lambda_home_l1"), float("nan"))
    l1_away = _safe_float(fixture.get("lambda_away_l1"), float("nan"))
    if (
        np.isfinite(l1_home)
        and np.isfinite(l1_away)
        and l1_home > 0.0
        and l1_away > 0.0
    ):
        return max(0.01, l1_home), max(0.01, l1_away), "lambda_l1"

    home_proxy = max(0.01, _safe_float(fixture.get("home_rolling_xg"), 1.35))
    away_proxy = max(0.01, _safe_float(fixture.get("away_rolling_xg"), 1.10))
    return home_proxy, away_proxy, "rolling_xg_proxy"


def _score_matrix(
    lambda_home: float, lambda_away: float, max_goals: int = 10
) -> np.ndarray:
    goals = np.arange(max_goals + 1, dtype=float)
    factorials = np.array([math.factorial(int(g)) for g in goals], dtype=float)
    home_pmf = np.exp(-lambda_home) * np.power(lambda_home, goals) / factorials
    away_pmf = np.exp(-lambda_away) * np.power(lambda_away, goals) / factorials
    mat = np.outer(home_pmf, away_pmf)
    mass = float(mat.sum())
    if mass <= 0.0:
        return np.full(
            (max_goals + 1, max_goals + 1), 1.0 / ((max_goals + 1) ** 2), dtype=float
        )
    return mat / mass


def _poisson_over(mu: float, line_half: float) -> float:
    threshold = int(line_half) + 1
    cdf = 0.0
    for k in range(threshold):
        cdf += math.exp(-mu) * (mu**k) / math.factorial(k)
    return float(max(0.0, min(1.0, 1.0 - cdf)))


@lru_cache(maxsize=4096)
def _markov_anytime_probs(lambda_home: float, lambda_away: float) -> dict[str, float]:
    return _MARKOV_PRICER.calculate_lead_probs(lambda_home, lambda_away)


def _fallback_market_probabilities(
    fixture: pd.Series,
) -> tuple[dict[str, float], dict[str, object]]:
    lambda_home, lambda_away, lambda_source = _resolve_backbone_lambdas(fixture)
    score = _score_matrix(lambda_home, lambda_away, max_goals=10)
    home_idx, away_idx = np.indices(score.shape)
    goal_diff = home_idx - away_idx

    p_home = float(score[home_idx > away_idx].sum())
    p_draw = float(score[home_idx == away_idx].sum())
    p_away = float(score[home_idx < away_idx].sum())

    p_total_ge2 = float(score[(home_idx + away_idx) >= 2].sum())
    p_total_ge3 = float(score[(home_idx + away_idx) >= 3].sum())
    p_total_ge4 = float(score[(home_idx + away_idx) >= 4].sum())
    p_total_ge5 = float(score[(home_idx + away_idx) >= 5].sum())
    p_total_le1 = float(score[(home_idx + away_idx) <= 1].sum())
    p_total_le2 = float(score[(home_idx + away_idx) <= 2].sum())

    p_home_ge2 = float(score[home_idx >= 2].sum())
    p_away_ge2 = float(score[away_idx >= 2].sum())
    p_home_by2_final = float(score[goal_diff >= 2].sum())
    p_home_by1_final = float(score[goal_diff == 1].sum())
    p_away_by2_final = float(score[goal_diff <= -2].sum())
    p_away_by1_final = float(score[goal_diff == -1].sum())
    p_home_not_lose = float(score[goal_diff >= 0].sum())
    p_away_not_lose = float(score[goal_diff <= 0].sum())
    p_home_plus15_cover = float(score[goal_diff >= -1].sum())
    p_away_plus15_cover = float(score[goal_diff <= 1].sum())
    p_btts = float(score[(home_idx >= 1) & (away_idx >= 1)].sum())

    markov = _markov_anytime_probs(
        round(float(lambda_home), 4), round(float(lambda_away), 4)
    )
    p_h_1up = float(markov.get("h_1up", p_home))
    p_a_1up = float(markov.get("a_1up", p_away))
    p_h_2up = float(markov.get("h_2up", p_home_by2_final))
    p_a_2up = float(markov.get("a_2up", p_away_by2_final))

    p_home_and_o25 = float(
        score[(home_idx > away_idx) & ((home_idx + away_idx) >= 3)].sum()
    )
    p_away_and_o25 = float(
        score[(home_idx < away_idx) & ((home_idx + away_idx) >= 3)].sum()
    )
    p_home_or_o25 = p_home + p_total_ge3 - p_home_and_o25
    p_away_or_o25 = p_away + p_total_ge3 - p_away_and_o25

    p_home_and_o15 = float(
        score[(home_idx > away_idx) & ((home_idx + away_idx) >= 2)].sum()
    )
    p_away_and_o15 = float(
        score[(home_idx < away_idx) & ((home_idx + away_idx) >= 2)].sum()
    )
    p_home_or_o15 = p_home + p_total_ge2 - p_home_and_o15
    p_away_or_o15 = p_away + p_total_ge2 - p_away_and_o15

    home_corners = _safe_float(fixture.get("home_rolling_corners"), float("nan"))
    away_corners = _safe_float(fixture.get("away_rolling_corners"), float("nan"))
    home_corners_mu = (
        home_corners if np.isfinite(home_corners) and home_corners > 0.0 else 5.0
    )
    away_corners_mu = (
        away_corners if np.isfinite(away_corners) and away_corners > 0.0 else 4.5
    )
    corners_mu = home_corners_mu + away_corners_mu
    corners_source = (
        "rolling_corners_split"
        if np.isfinite(home_corners) and np.isfinite(away_corners)
        else "default_mu_split_5_0_4_5"
    )

    raw_probs = {
        "o15": p_total_ge2,
        "o25": p_total_ge3,
        "o35": p_total_ge4,
        "o45": p_total_ge5,
        "u15": p_total_le1,
        "u25": p_total_le2,
        "u35": 1.0 - p_total_ge4,
        "c75": _poisson_over(corners_mu, 7.5),
        "c85": _poisson_over(corners_mu, 8.5),
        "c95": _poisson_over(corners_mu, 9.5),
        "c105": _poisson_over(corners_mu, 10.5),
        "hc25": _poisson_over(home_corners_mu, 2.5),
        "hc35": _poisson_over(home_corners_mu, 3.5),
        "hc45": _poisson_over(home_corners_mu, 4.5),
        "hc55": _poisson_over(home_corners_mu, 5.5),
        "ac25": _poisson_over(away_corners_mu, 2.5),
        "ac35": _poisson_over(away_corners_mu, 3.5),
        "ac45": _poisson_over(away_corners_mu, 4.5),
        "ac55": _poisson_over(away_corners_mu, 5.5),
        "btts": p_btts,
        "1x2_h": p_home,
        "1x2_d": p_draw,
        "1x2_a": p_away,
        "dc_1x": p_home + p_draw,
        "dc_x2": p_away + p_draw,
        "dc_12": p_home + p_away,
        "ho15": p_home_ge2,
        "ao15": p_away_ge2,
        # Legacy runtime handicap aliases.
        "ah_h05": p_home,
        "ah_a05": p_away,
        "ah_h15": p_home_by2_final,
        "ah_a15": p_away_by2_final,
        "eh_h1": p_home_by2_final,
        "eh_a1": p_away_by2_final,
        # Canonical AH selections.
        "ah2_home_m05": p_home,
        "ah2_away_p05": p_away_not_lose,
        "ah2_away_m05": p_away,
        "ah2_home_p05": p_home_not_lose,
        "ah2_home_m15": p_home_by2_final,
        "ah2_away_p15": p_away_plus15_cover,
        "ah2_away_m15": p_away_by2_final,
        "ah2_home_p15": p_home_plus15_cover,
        # Canonical EH 3-way selections.
        "eh3_0_1_home": p_home_by2_final,
        "eh3_0_1_draw": p_home_by1_final,
        "eh3_0_1_away": p_away_not_lose,
        "eh3_1_0_home": p_home_not_lose,
        "eh3_1_0_draw": p_away_by1_final,
        "eh3_1_0_away": p_away_by2_final,
        "h_1up": p_h_1up,
        "a_1up": p_a_1up,
        "h_2up": p_h_2up,
        "a_2up": p_a_2up,
        "home_or_o25": p_home_or_o25,
        "away_or_o25": p_away_or_o25,
        "home_or_o15": p_home_or_o15,
        "away_or_o15": p_away_or_o15,
        "home_and_o25": p_home_and_o25,
        "away_and_o25": p_away_and_o25,
    }
    clipped = {
        market: float(np.clip(prob, 0.001, 0.999)) for market, prob in raw_probs.items()
    }
    trace = {
        "lambda_source": lambda_source,
        "lambda_home": float(lambda_home),
        "lambda_away": float(lambda_away),
        "anytime_source": "markov_ctmc",
        "anytime_markov_max_goals": 8,
        "corners_source": corners_source,
        "corners_mu": float(corners_mu),
        "home_corners_mu": float(home_corners_mu),
        "away_corners_mu": float(away_corners_mu),
    }
    return clipped, trace


def build_prediction_rows(
    scored: pd.DataFrame,
    features: list[str],
    models: dict[str, object],
    model_failures: dict[str, str] | None = None,
    multiclass_model: object | None = None,
    multiclass_markets: set[str] | None = None,
) -> tuple[list[tuple[int, str, str, str, float, str]], int]:
    x_mat = scored[features]
    rows: list[tuple[int, str, str, str, float, str]] = []
    fallback_rows = 0
    failures = model_failures or {}
    configured_multiclass_markets = set(multiclass_markets or set())
    configured_multiclass_markets &= set(MULTICLASS_MARKETS)

    for _, fixture in scored.iterrows():
        fixture_id = int(fixture["fixture_id"])
        fallback_probs, fallback_trace = _fallback_market_probabilities(fixture)
        base_metadata = {
            "features_missing_count": int(fixture["features_missing_count"]),
            "home_sample_size": None
            if pd.isna(fixture["home_sample_size"])
            else float(fixture["home_sample_size"]),
            "away_sample_size": None
            if pd.isna(fixture["away_sample_size"])
            else float(fixture["away_sample_size"]),
            "home_played": None
            if pd.isna(fixture["home_sample_size"])
            else float(fixture["home_sample_size"]),
            "away_played": None
            if pd.isna(fixture["away_sample_size"])
            else float(fixture["away_sample_size"]),
            "lambda_home_l1": None
            if pd.isna(fixture.get("lambda_home_l1"))
            else float(fixture["lambda_home_l1"]),
            "lambda_away_l1": None
            if pd.isna(fixture.get("lambda_away_l1"))
            else float(fixture["lambda_away_l1"]),
            "adj_lambda_home_final": None
            if pd.isna(fixture.get("adj_lambda_home_final"))
            else float(fixture["adj_lambda_home_final"]),
            "adj_lambda_away_final": None
            if pd.isna(fixture.get("adj_lambda_away_final"))
            else float(fixture["adj_lambda_away_final"]),
            "rule_fired_home": int(fixture["rule_fired_home"])
            if not pd.isna(fixture.get("rule_fired_home"))
            else 0,
            "rule_fired_away": int(fixture["rule_fired_away"])
            if not pd.isna(fixture.get("rule_fired_away"))
            else 0,
            "odds_snapshot_time_utc": None
            if pd.isna(fixture["odds_snapshot_time_utc"])
            else fixture["odds_snapshot_time_utc"].isoformat(),
        }
        row_df = x_mat.loc[[fixture.name]]
        multiclass_probs: dict[str, float] = {}
        if multiclass_model is not None and configured_multiclass_markets:
            try:
                p_home, p_draw, p_away = multiclass_probabilities(multiclass_model, row_df)
                for market in configured_multiclass_markets:
                    multiclass_probs[market] = multiclass_probability_for_market(
                        market, p_home=p_home, p_draw=p_draw, p_away=p_away
                    )
            except Exception:
                multiclass_probs = {}
        for market in MARKETS:
            fallback_used = False
            fallback_reason: str | None = None
            prediction_model_family = "binary"
            if market in multiclass_probs:
                p_model = multiclass_probs[market]
                prediction_model_family = "1x2_dc_multiclass"
            else:
                model = models.get(market)
                if model is None:
                    fallback_used = True
                    fallback_reason = failures.get(market, "missing_artifact")
                    p_model = fallback_probs[market]
                else:
                    try:
                        p_model = positive_class_probability(model=model, x_row=row_df)
                    except Exception as exc:
                        fallback_used = True
                        fallback_reason = f"predict_error:{exc.__class__.__name__}"
                        p_model = fallback_probs[market]

            if fallback_used:
                fallback_rows += 1

            metadata = dict(base_metadata)
            metadata.update(
                {
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                    "fallback_lambda_source": fallback_trace["lambda_source"],
                    "fallback_lambda_home": fallback_trace["lambda_home"],
                    "fallback_lambda_away": fallback_trace["lambda_away"],
                    "fallback_anytime_source": fallback_trace["anytime_source"],
                    "fallback_anytime_markov_max_goals": fallback_trace[
                        "anytime_markov_max_goals"
                    ],
                    "fallback_corners_source": fallback_trace["corners_source"],
                    "fallback_corners_mu": fallback_trace["corners_mu"],
                    "prediction_model_family": prediction_model_family,
                }
            )
            rows.append(
                (
                    fixture_id,
                    market,
                    MODEL_NAME,
                    MODEL_VERSION,
                    float(np.clip(p_model, 0.001, 0.999)),
                    json.dumps(metadata),
                )
            )

    return rows, fallback_rows


def _align_x_row_to_model_features(model: object, x_row: pd.DataFrame) -> pd.DataFrame:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return x_row
    return x_row.reindex(columns=list(feature_names))


def positive_class_probability(model: object, x_row: pd.DataFrame) -> float:
    x_row = _align_x_row_to_model_features(model, x_row)
    probs = model.predict_proba(x_row)
    classes = getattr(model, "classes_", None)

    if classes is None:
        return float(np.clip(probs[0, 1], 0.001, 0.999))

    classes_arr = np.asarray(classes)
    if len(classes_arr) == 1:
        only_class = int(classes_arr[0])
        return 0.999 if only_class == 1 else 0.001

    idx = int(np.where(classes_arr == 1)[0][0]) if np.any(classes_arr == 1) else 1
    return float(np.clip(probs[0, idx], 0.001, 0.999))


def multiclass_probabilities(
    model: object, x_row: pd.DataFrame
) -> tuple[float, float, float]:
    probs = model.predict_proba(x_row)
    classes = np.asarray(getattr(model, "classes_", [0, 1, 2]), dtype=int)
    idx_home = int(np.where(classes == 0)[0][0]) if np.any(classes == 0) else None
    idx_draw = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else None
    idx_away = int(np.where(classes == 2)[0][0]) if np.any(classes == 2) else None
    p_home = float(probs[0, idx_home]) if idx_home is not None else 0.0
    p_draw = float(probs[0, idx_draw]) if idx_draw is not None else 0.0
    p_away = float(probs[0, idx_away]) if idx_away is not None else 0.0
    total = p_home + p_draw + p_away
    if total <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (p_home / total, p_draw / total, p_away / total)


def multiclass_probability_for_market(
    market: str, *, p_home: float, p_draw: float, p_away: float
) -> float:
    if market == "1x2_h":
        return float(np.clip(p_home, 0.001, 0.999))
    if market == "1x2_d":
        return float(np.clip(p_draw, 0.001, 0.999))
    if market == "1x2_a":
        return float(np.clip(p_away, 0.001, 0.999))
    if market == "dc_1x":
        return float(np.clip(p_home + p_draw, 0.001, 0.999))
    if market == "dc_x2":
        return float(np.clip(p_draw + p_away, 0.001, 0.999))
    if market == "dc_12":
        return float(np.clip(p_home + p_away, 0.001, 0.999))
    raise ValueError(f"unsupported multiclass market: {market}")


def upsert_predictions(rows: list[tuple[int, str, str, str, float, str]]) -> int:
    if not rows:
        return 0

    query = """
    INSERT INTO predictions (
        fixture_id,
        market_code,
        model_name,
        model_version,
        p_model,
        metadata_json,
        created_at
    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, NOW())
    ON CONFLICT (fixture_id, market_code, model_name, model_version)
    DO UPDATE SET
        p_model = EXCLUDED.p_model,
        metadata_json = EXCLUDED.metadata_json,
        created_at = NOW();
    """

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.executemany(query, rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def main() -> None:
    args = parse_args()
    (
        features,
        imputation,
        models,
        model_failures,
        multiclass_model,
        multiclass_markets,
    ) = load_artifacts()

    fixtures = fetch_candidate_fixtures(
        days=args.days,
        league=args.league,
        limit=args.limit,
        backfill_days=args.backfill_days,
    )
    fixtures["match_datetime_utc"] = pd.to_datetime(
        fixtures["match_datetime_utc"], utc=True, errors="coerce"
    )
    fixtures["odds_snapshot_time_utc"] = pd.to_datetime(
        fixtures["odds_snapshot_time_utc"], utc=True, errors="coerce"
    )
    post_kickoff_mask = (
        fixtures["match_datetime_utc"].notna()
        & fixtures["odds_snapshot_time_utc"].notna()
        & (fixtures["odds_snapshot_time_utc"] > fixtures["match_datetime_utc"])
    )
    if post_kickoff_mask.any():
        bad_rows = int(post_kickoff_mask.sum())
        raise RuntimeError(
            f"Detected {bad_rows} fixture(s) with post-kickoff odds snapshots; re-ingest odds or fix snapshot timing semantics."
        )
    if fixtures.empty:
        print("No eligible fixtures found.")
        return

    featured = add_derived_features(fixtures)
    featured["features_missing_count"] = (
        featured.reindex(columns=features).isna().sum(axis=1)
    )
    scored = apply_imputation(featured, features, imputation)

    prediction_rows, fallback_rows = build_prediction_rows(
        scored=scored,
        features=features,
        models=models,
        model_failures=model_failures,
        multiclass_model=multiclass_model,
        multiclass_markets=multiclass_markets,
    )
    written = upsert_predictions(prediction_rows)

    expected = len(scored) * len(MARKETS)
    print(
        f"Processed {len(scored)} fixtures, upserted {written} market predictions ({expected} expected). "
        f"fallback_rows={fallback_rows}."
    )


if __name__ == "__main__":
    main()
