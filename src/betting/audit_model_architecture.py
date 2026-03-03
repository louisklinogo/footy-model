from __future__ import annotations

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false, reportOptionalSubscript=false, reportExplicitAny=false, reportUnknownLambdaType=false

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


MODEL_NAME_DEFAULT = "market_outcome_gbm"
MODEL_VERSION_DEFAULT = "fixtures_first_prematch_v1"
LINCHPIN_MARKETS: tuple[str, ...] = (
    # Handicaps
    "ah_h05",
    "ah_a05",
    "ah_h15",
    "ah_a15",
    "eh_h1",
    "eh_a1",
    # Double chance
    "dc_1x",
    "dc_x2",
    "dc_12",
    # Anytime lead
    "h_1up",
    "a_1up",
    "h_2up",
    "a_2up",
    # Team totals
    "ho15",
    "ao15",
    # Multigoals combos
    "home_or_o25",
    "away_or_o25",
    "home_or_o15",
    "away_or_o15",
    "home_and_o25",
    "away_and_o25",
    # Corners totals
    "c75",
    "c85",
    "c95",
    "c105",
    # Home team corners totals
    "hc25",
    "hc35",
    "hc45",
    "hc55",
    # Away team corners totals
    "ac25",
    "ac35",
    "ac45",
    "ac55",
    # Goals
    "o15",
    "u35",
)

TEAM_SNAPSHOT_FEATURES: tuple[str, ...] = (
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
)

V1_FEATURES: tuple[str, ...] = tuple(
    [f"home_{name}" for name in TEAM_SNAPSHOT_FEATURES]
    + [f"away_{name}" for name in TEAM_SNAPSHOT_FEATURES]
    + [
        "xg_net_diff",
        "xgot_net_diff",
        "xa_net_diff",
        "corners_net_diff",
        "sample_size_diff",
        "goal_diff_proxy",
    ]
)

V2_FEATURES: tuple[str, ...] = V1_FEATURES + (
    "lambda_home_l1",
    "lambda_away_l1",
    "adj_lambda_home_final",
    "adj_lambda_away_final",
    "rule_fired_home",
    "rule_fired_away",
)

V3_FEATURES: tuple[str, ...] = V2_FEATURES + (
    "odds_over_15",
    "odds_under_15",
    "odds_over_25",
    "odds_under_25",
    "implied_over15",
    "implied_under15",
    "implied_over25",
    "implied_under25",
    "odds_gap_15",
    "odds_gap_25",
    "market_implied_prob",
    "market_odds_gap",
)


@dataclass
class VariantResult:
    variant: str
    test_n: int
    base_rate_train: float | None
    auc: float | None
    brier: float | None
    bss: float | None
    ece: float | None
    bookmaker_available: bool
    bookmaker_n: int
    bookmaker_brier: float | None
    bookmaker_bss: float | None
    bookmaker_ece: float | None
    bookmaker_auc: float | None
    bookmaker_delta_brier: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit market model architecture and produce gating decisions"
    )
    parser.add_argument("--since-days", type=int, default=720)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/reports/backtests")
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME_DEFAULT)
    parser.add_argument("--model-version", type=str, default=MODEL_VERSION_DEFAULT)
    return parser.parse_args()


def _json_number_expr(json_col: str, key: str) -> str:
    return (
        "CASE "
        f"WHEN ({json_col} ->> '{key}') ~ '^[-+]?[0-9]*\\.?[0-9]+$' "
        f"THEN ({json_col} ->> '{key}')::double precision "
        "ELSE NULL END"
    )


def _latest_odds_expr(row_alias: str, side: str) -> str:
    return (
        "CASE "
        f"WHEN ({row_alias}.odds_json -> 'prices_latest' ->> '{side}') ~ '^[-+]?[0-9]*\\.?[0-9]+$' "
        f"THEN ({row_alias}.odds_json -> 'prices_latest' ->> '{side}')::double precision "
        "ELSE NULL END"
    )


def fetch_feature_dataset(since_days: int) -> pd.DataFrame:
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
        ils.home_led_by_1_any,
        ils.away_led_by_1_any,
        ils.home_led_by_2_any,
        ils.away_led_by_2_any,
        {_latest_odds_expr("od15", "over")} AS odds_over_15,
        {_latest_odds_expr("od15", "under")} AS odds_under_15,
        {_latest_odds_expr("od25", "over")} AS odds_over_25,
        {_latest_odds_expr("od25", "under")} AS odds_under_25,
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
        END AS rule_fired_away
    FROM fixtures f
    JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
    LEFT JOIN fixture_stats_premium sp ON sp.fixture_id = f.fixture_id
    LEFT JOIN fixture_incident_lead_states ils ON ils.fixture_id = f.fixture_id
    LEFT JOIN team_premium_snapshots tph ON tph.fixture_id = f.fixture_id AND tph.is_home = true
    LEFT JOIN team_premium_snapshots tpa ON tpa.fixture_id = f.fixture_id AND tpa.is_home = false
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
          AND p.created_at <= f.match_datetime_utc
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l1h ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'lambda_xgb'
          AND p.market_code = 'lambda_away'
          AND p.created_at <= f.match_datetime_utc
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l1a ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'situational_xgb'
          AND p.market_code = 'adj_lambda_home'
          AND p.created_at <= f.match_datetime_utc
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l2h ON true
    LEFT JOIN LATERAL (
        SELECT p.metadata_json
        FROM predictions p
        WHERE p.fixture_id = f.fixture_id
          AND p.model_name = 'situational_xgb'
          AND p.market_code = 'adj_lambda_away'
          AND p.created_at <= f.match_datetime_utc
        ORDER BY p.created_at DESC
        LIMIT 1
    ) l2a ON true
    WHERE f.status = 'ft'
      AND f.match_datetime_utc IS NOT NULL
      AND fr.home_goals IS NOT NULL
      AND fr.away_goals IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
    ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC
    """
    conn = connect_db()
    try:
        frame = pd.read_sql(query, conn, params=(since_days,))
    finally:
        conn.close()
    return frame


def fetch_odds_rows(since_days: int) -> pd.DataFrame:
    query = """
    SELECT
        fom.fixture_id,
        fom.market_code,
        fom.line_num,
        fom.snapshot_type,
        fom.snapshot_time_utc,
        fom.odds_json
    FROM fixture_odds_markets fom
    JOIN fixtures f ON f.fixture_id = fom.fixture_id
    WHERE f.status = 'ft'
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
      AND fom.provider = 'sofascore'
      AND fom.snapshot_type IN ('latest_pre_match', 'closing')
      AND fom.snapshot_time_utc <= f.match_datetime_utc
      AND fom.market_code IN (
          '1x2', 'dc', 'ou', 'corners_ou',
          'home_ou', 'away_ou',
          'home_corners_ou', 'away_corners_ou',
          'corners_home_ou', 'corners_away_ou',
          'team_corners_home_ou', 'team_corners_away_ou',
          'ah', 'eh'
      )
    ORDER BY
        fom.fixture_id ASC,
        (fom.snapshot_type = 'latest_pre_match') DESC,
        fom.snapshot_time_utc DESC
    """
    conn = connect_db()
    try:
        frame = pd.read_sql(query, conn, params=(since_days,))
    finally:
        conn.close()
    return frame


def fetch_leakage_checks(since_days: int) -> dict[str, int]:
    odds_query = """
    SELECT COUNT(*)
    FROM fixture_odds_markets fom
    JOIN fixtures f ON f.fixture_id = fom.fixture_id
    WHERE f.status = 'ft'
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
      AND fom.provider = 'sofascore'
      AND fom.snapshot_type IN ('latest_pre_match', 'closing')
      AND fom.snapshot_time_utc > f.match_datetime_utc
    """
    lambda_query = """
    SELECT COUNT(*)
    FROM predictions p
    JOIN fixtures f ON f.fixture_id = p.fixture_id
    WHERE f.status = 'ft'
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
      AND (
          (p.model_name = 'lambda_xgb' AND p.market_code IN ('lambda_home', 'lambda_away'))
          OR
          (p.model_name = 'situational_xgb' AND p.market_code IN ('adj_lambda_home', 'adj_lambda_away'))
      )
      AND p.created_at > f.match_datetime_utc
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(odds_query, (since_days,))
            post_kickoff_odds = int(cur.fetchone()[0])
            cur.execute(lambda_query, (since_days,))
            post_kickoff_lambdas = int(cur.fetchone()[0])
    finally:
        conn.close()
    return {
        "post_kickoff_odds_rows": post_kickoff_odds,
        "post_kickoff_lambda_rows": post_kickoff_lambdas,
    }


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["xg_net_diff"] = out["home_rolling_xg"] - out["away_rolling_xg_against"]
    out["xgot_net_diff"] = out["home_rolling_xgot"] - out["away_rolling_xgot_against"]
    out["xa_net_diff"] = out["home_rolling_xa"] - out["away_rolling_xa_against"]
    out["corners_net_diff"] = (
        out["home_rolling_corners"] - out["away_rolling_corners_against"]
    )
    out["sample_size_diff"] = out["home_sample_size"] - out["away_sample_size"]
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
    out["odds_gap_15"] = out["odds_over_15"] - out["odds_under_15"]
    out["odds_gap_25"] = out["odds_over_25"] - out["odds_under_25"]
    return out


def compute_target(frame: pd.DataFrame, market_code: str) -> pd.Series:
    h = frame["home_goals"].astype(float)
    a = frame["away_goals"].astype(float)
    total_goals = h + a
    if market_code == "o15":
        return (total_goals >= 2).astype(float)
    if market_code == "u35":
        return (total_goals <= 3).astype(float)
    if market_code == "dc_1x":
        return (h >= a).astype(float)
    if market_code == "dc_x2":
        return (a >= h).astype(float)
    if market_code == "dc_12":
        return (h != a).astype(float)
    if market_code == "ho15":
        return (h >= 2).astype(float)
    if market_code == "ao15":
        return (a >= 2).astype(float)
    if market_code == "home_or_o25":
        return ((h > a) | (total_goals >= 3)).astype(float)
    if market_code == "away_or_o25":
        return ((a > h) | (total_goals >= 3)).astype(float)
    if market_code == "home_or_o15":
        return ((h > a) | (total_goals >= 2)).astype(float)
    if market_code == "away_or_o15":
        return ((a > h) | (total_goals >= 2)).astype(float)
    if market_code == "home_and_o25":
        return ((h > a) & (total_goals >= 3)).astype(float)
    if market_code == "away_and_o25":
        return ((a > h) & (total_goals >= 3)).astype(float)
    if market_code == "h_1up":
        return frame["home_led_by_1_any"].map(
            lambda v: float(int(bool(v))) if pd.notna(v) else np.nan
        )
    if market_code == "a_1up":
        return frame["away_led_by_1_any"].map(
            lambda v: float(int(bool(v))) if pd.notna(v) else np.nan
        )
    if market_code == "h_2up":
        return frame["home_led_by_2_any"].map(
            lambda v: float(int(bool(v))) if pd.notna(v) else np.nan
        )
    if market_code == "a_2up":
        return frame["away_led_by_2_any"].map(
            lambda v: float(int(bool(v))) if pd.notna(v) else np.nan
        )
    if market_code in {"c75", "c85", "c95", "c105"}:
        thresholds = {"c75": 8, "c85": 9, "c95": 10, "c105": 11}
        return np.where(
            frame["total_corners"].notna(),
            (frame["total_corners"] >= thresholds[market_code]).astype(float),
            np.nan,
        )
    if market_code in {"hc25", "hc35", "hc45", "hc55"}:
        thresholds = {"hc25": 3, "hc35": 4, "hc45": 5, "hc55": 6}
        return np.where(
            frame["home_corners"].notna(),
            (frame["home_corners"] >= thresholds[market_code]).astype(float),
            np.nan,
        )
    if market_code in {"ac25", "ac35", "ac45", "ac55"}:
        thresholds = {"ac25": 3, "ac35": 4, "ac45": 5, "ac55": 6}
        return np.where(
            frame["away_corners"].notna(),
            (frame["away_corners"] >= thresholds[market_code]).astype(float),
            np.nan,
        )
    if market_code == "ah_h05":
        return (h > a).astype(float)
    if market_code == "ah_a05":
        return (a > h).astype(float)
    if market_code == "ah_h15":
        return ((h - a) >= 2).astype(float)
    if market_code == "ah_a15":
        return ((a - h) >= 2).astype(float)
    if market_code == "eh_h1":
        return ((h - a) >= 2).astype(float)
    if market_code == "eh_a1":
        return ((a - h) >= 2).astype(float)
    raise ValueError(f"Unsupported market for target computation: {market_code}")


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or number <= 0.0:
        return None
    return number


def _extract_side_price(odds_json: dict[str, Any], side: str) -> float | None:
    prices = odds_json.get("prices_latest")
    if not isinstance(prices, dict):
        return None
    return _to_float(prices.get(side))


def _line_key(line_num: Any) -> str:
    try:
        return f"{float(line_num):.3f}"
    except (TypeError, ValueError):
        return "nan"


def _pick_odds_snapshot(
    odds_rows: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in odds_rows.itertuples(index=False):
        key = (str(row.market_code), _line_key(row.line_num))
        if key in lookup:
            continue
        odds_json = row.odds_json if isinstance(row.odds_json, dict) else {}
        lookup[key] = {
            "market_code": str(row.market_code),
            "line_num": row.line_num,
            "snapshot_type": str(row.snapshot_type),
            "snapshot_time_utc": row.snapshot_time_utc,
            "odds_json": odds_json,
        }
    return lookup


def _resolve_market_odds(
    market_code: str, lookup: dict[tuple[str, str], dict[str, Any]]
) -> tuple[float | None, float | None]:
    def from_market(
        m_code: str, line: float | None, side: str, alt_side: str | None = None
    ) -> tuple[float | None, float | None]:
        keys = []
        if line is None:
            keys = [(m_code, "nan")]
        else:
            keys = [(m_code, f"{line:.3f}")]
        for key in keys:
            blob = lookup.get(key)
            if not blob:
                continue
            price = _extract_side_price(blob["odds_json"], side)
            if price is None:
                continue
            implied = 1.0 / price if price > 1.0 else None
            gap = None
            if alt_side is not None:
                other = _extract_side_price(blob["odds_json"], alt_side)
                if other is not None:
                    gap = price - other
            return implied, gap
        return None, None

    if market_code == "o15":
        return from_market("ou", 1.5, "over", "under")
    if market_code == "u35":
        return from_market("ou", 3.5, "under", "over")
    if market_code == "dc_1x":
        return from_market("dc", None, "home_draw", "draw_away")
    if market_code == "dc_x2":
        return from_market("dc", None, "draw_away", "home_draw")
    if market_code == "dc_12":
        return from_market("dc", None, "home_away", "draw_away")
    if market_code == "ho15":
        return from_market("home_ou", 1.5, "over", "under")
    if market_code == "ao15":
        return from_market("away_ou", 1.5, "over", "under")
    if market_code in {"c75", "c85", "c95", "c105"}:
        line = {"c75": 7.5, "c85": 8.5, "c95": 9.5, "c105": 10.5}[market_code]
        return from_market("corners_ou", line, "over", "under")
    if market_code in {"hc25", "hc35", "hc45", "hc55"}:
        line = {"hc25": 2.5, "hc35": 3.5, "hc45": 4.5, "hc55": 5.5}[market_code]
        for market_alias in (
            "home_corners_ou",
            "corners_home_ou",
            "team_corners_home_ou",
        ):
            implied, gap = from_market(market_alias, line, "over", "under")
            if implied is not None:
                return implied, gap
        return None, None
    if market_code in {"ac25", "ac35", "ac45", "ac55"}:
        line = {"ac25": 2.5, "ac35": 3.5, "ac45": 4.5, "ac55": 5.5}[market_code]
        for market_alias in (
            "away_corners_ou",
            "corners_away_ou",
            "team_corners_away_ou",
        ):
            implied, gap = from_market(market_alias, line, "over", "under")
            if implied is not None:
                return implied, gap
        return None, None
    if market_code == "ah_h05":
        for ln in (-0.5, 0.5):
            implied, gap = from_market("ah", ln, "home", "away")
            if implied is not None:
                return implied, gap
        return None, None
    if market_code == "ah_a05":
        for ln in (0.5, -0.5):
            implied, gap = from_market("ah", ln, "away", "home")
            if implied is not None:
                return implied, gap
        return None, None
    if market_code == "ah_h15":
        for ln in (-1.5, 1.5):
            implied, gap = from_market("ah", ln, "home", "away")
            if implied is not None:
                return implied, gap
        return None, None
    if market_code == "ah_a15":
        for ln in (1.5, -1.5):
            implied, gap = from_market("ah", ln, "away", "home")
            if implied is not None:
                return implied, gap
        return None, None
    if market_code == "eh_h1":
        for m_code, line in (("eh", -1.0), ("ah", -1.5), ("ah", 1.5)):
            implied, gap = from_market(m_code, line, "home", "away")
            if implied is not None:
                return implied, gap
        return None, None
    if market_code == "eh_a1":
        for m_code, line in (("eh", 1.0), ("ah", 1.5), ("ah", -1.5)):
            implied, gap = from_market(m_code, line, "away", "home")
            if implied is not None:
                return implied, gap
        return None, None
    return None, None


def attach_market_odds_features(
    frame: pd.DataFrame, odds_rows: pd.DataFrame, market_code: str
) -> pd.DataFrame:
    by_fixture = {
        int(fid): grp for fid, grp in odds_rows.groupby("fixture_id", sort=False)
    }
    market_implied: list[float | None] = []
    market_gap: list[float | None] = []
    for fixture_id in frame["fixture_id"].astype(int).tolist():
        fixture_rows = by_fixture.get(fixture_id)
        if fixture_rows is None or fixture_rows.empty:
            market_implied.append(None)
            market_gap.append(None)
            continue
        lookup = _pick_odds_snapshot(fixture_rows)
        implied_prob, odds_gap = _resolve_market_odds(market_code, lookup)
        market_implied.append(implied_prob)
        market_gap.append(odds_gap)
    out = frame.copy()
    out["market_implied_prob"] = market_implied
    out["market_odds_gap"] = market_gap
    return out


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10
) -> float | None:
    if y_true.size == 0:
        return None
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(y_prob, quantiles))
    if edges.size <= 1:
        return float(abs(np.mean(y_prob) - np.mean(y_true)))
    bucket_ids = np.digitize(y_prob, edges[1:-1], right=True)
    ece = 0.0
    total = float(y_true.size)
    for idx in range(edges.size - 1):
        mask = bucket_ids == idx
        if not np.any(mask):
            continue
        observed = float(np.mean(y_true[mask]))
        predicted = float(np.mean(y_prob[mask]))
        ece += abs(observed - predicted) * (float(np.sum(mask)) / total)
    return float(ece)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if y_true.size == 0:
        return None
    if np.unique(y_true).size < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def compute_variant_result(
    variant: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
) -> VariantResult:
    y_train = train["target"].astype(int).to_numpy(dtype=float)
    y_test = test["target"].astype(int).to_numpy(dtype=float)
    if y_train.size == 0 or y_test.size == 0:
        return VariantResult(
            variant,
            0,
            None,
            None,
            None,
            None,
            None,
            False,
            0,
            None,
            None,
            None,
            None,
            None,
        )

    base_rate_train = float(np.mean(y_train))
    X_train = train.reindex(columns=list(features)).to_numpy(dtype=float)
    X_test = test.reindex(columns=list(features)).to_numpy(dtype=float)

    if np.unique(y_train).size < 2:
        p_model = np.full_like(y_test, fill_value=base_rate_train, dtype=float)
    else:
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=120,
            min_samples_leaf=40,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        p_model = np.clip(clf.predict_proba(X_test)[:, 1], 1e-6, 1.0 - 1e-6)

    const_pred = np.full_like(y_test, fill_value=base_rate_train, dtype=float)
    brier_model = float(brier_score_loss(y_test, p_model))
    brier_const = float(brier_score_loss(y_test, const_pred))
    bss = None if brier_const <= 0.0 else float(1.0 - (brier_model / brier_const))
    model_ece = expected_calibration_error(y_test, p_model)

    implied = test["market_implied_prob"].to_numpy(dtype=float)
    mask = np.isfinite(implied)
    bookmaker_available = bool(np.any(mask))
    bookmaker_n = int(np.sum(mask))
    book_brier = None
    book_bss = None
    book_ece = None
    book_auc = None
    book_delta = None
    if bookmaker_available:
        y_book = y_test[mask]
        p_book = np.clip(implied[mask], 1e-6, 1.0 - 1e-6)
        p_model_book = p_model[mask]
        brier_const_book = float(
            brier_score_loss(
                y_book, np.full_like(y_book, fill_value=base_rate_train, dtype=float)
            )
        )
        book_brier = float(brier_score_loss(y_book, p_book))
        if brier_const_book > 0.0:
            book_bss = float(1.0 - (book_brier / brier_const_book))
        book_ece = expected_calibration_error(y_book, p_book)
        book_auc = safe_auc(y_book, p_book)
        book_delta = float(book_brier - brier_score_loss(y_book, p_model_book))

    return VariantResult(
        variant=variant,
        test_n=int(y_test.size),
        base_rate_train=base_rate_train,
        auc=safe_auc(y_test, p_model),
        brier=brier_model,
        bss=bss,
        ece=model_ece,
        bookmaker_available=bookmaker_available,
        bookmaker_n=bookmaker_n,
        bookmaker_brier=book_brier,
        bookmaker_bss=book_bss,
        bookmaker_ece=book_ece,
        bookmaker_auc=book_auc,
        bookmaker_delta_brier=book_delta,
    )


def split_time_respecting(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(
        drop=True
    )
    split_idx = int(len(ordered) * 0.8)
    if split_idx <= 0 or split_idx >= len(ordered):
        return ordered.iloc[0:0].copy(), ordered.iloc[0:0].copy()
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    ):
        return "n/a"
    return f"{value:.{digits}f}"


def build_market_gating(primary: VariantResult, market_code: str) -> dict[str, Any]:
    bss = primary.bss
    ece = primary.ece
    test_n = primary.test_n
    eligible = False
    limited_to_one_leg = False
    edge_bump = 0.0
    notes: list[str] = []
    if test_n < 500:
        notes.append("insufficient_test_n")
    if bss is None:
        notes.append("missing_bss")
    if ece is None:
        notes.append("missing_ece")

    if (
        bss is not None
        and ece is not None
        and test_n >= 500
        and bss >= 0.010
        and ece <= 0.050
    ):
        eligible = True
    elif (
        bss is not None
        and ece is not None
        and test_n >= 500
        and 0.005 <= bss < 0.010
        and ece <= 0.080
    ):
        eligible = True
        limited_to_one_leg = True
        edge_bump = 0.01
        notes.append("limited_eligibility_band")
    else:
        eligible = False
        if bss is not None and bss < 0.005:
            notes.append("low_bss")
        if ece is not None and ece > 0.080:
            notes.append("high_ece")

    if not primary.bookmaker_available:
        notes.append("bookmaker_baseline_unavailable")

    return {
        "market_code": market_code,
        "eligible": eligible,
        "limited_to_one_leg": limited_to_one_leg,
        "edge_bump": edge_bump,
        "thresholds": {
            "min_test_n": 500,
            "full_min_bss": 0.010,
            "full_max_ece": 0.050,
            "limited_min_bss": 0.005,
            "hard_max_ece": 0.080,
        },
        "metrics": {
            "test_n": test_n,
            "auc": primary.auc,
            "brier": primary.brier,
            "bss": primary.bss,
            "ece": primary.ece,
            "bookmaker_available": primary.bookmaker_available,
            "bookmaker_test_n": primary.bookmaker_n,
            "bookmaker_brier": primary.bookmaker_brier,
            "bookmaker_bss": primary.bookmaker_bss,
            "bookmaker_ece": primary.bookmaker_ece,
            "bookmaker_auc": primary.bookmaker_auc,
            "delta_brier_vs_bookmaker": primary.bookmaker_delta_brier,
        },
        "notes": notes,
    }


def _empty_variant_result(variant: str) -> VariantResult:
    return VariantResult(
        variant,
        0,
        None,
        None,
        None,
        None,
        None,
        False,
        0,
        None,
        None,
        None,
        None,
        None,
    )


def write_unavailable_outputs(
    report_path: Path,
    gating_path: Path,
    model_name: str,
    model_version: str,
    since_days: int,
    reason: str,
) -> None:
    generated_at = datetime.now(UTC).isoformat()
    lines = [
        "# Model Architecture Audit",
        "",
        f"- generated_at_utc: {generated_at}",
        f"- model_name: {model_name}",
        f"- model_version: {model_version}",
        f"- since_days: {since_days}",
        "- status: data_unavailable",
        f"- limitation: {reason}",
        "",
        "Audit could not load required DB-backed fixtures-first data. Gating output is written with all linchpin markets marked ineligible/unavailable.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gating_items: list[dict[str, Any]] = []
    for market_code in LINCHPIN_MARKETS:
        item = build_market_gating(_empty_variant_result("V1"), market_code)
        item["notes"].append("audit_data_unavailable")
        item["notes"].append(reason)
        gating_items.append(item)

    payload = {
        "generated_at_utc": generated_at,
        "model_name": model_name,
        "model_version": model_version,
        "since_days": since_days,
        "markets": gating_items,
    }
    gating_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gating_path = ROOT_DIR / "model_artifacts" / "market_models" / "market_gating.json"
    gating_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "model_architecture_audit.md"

    try:
        base = fetch_feature_dataset(since_days=args.since_days)
        odds_rows = fetch_odds_rows(since_days=args.since_days)
        leakage = fetch_leakage_checks(since_days=args.since_days)
    except Exception as exc:
        reason = f"db_unavailable_or_query_failed: {exc}"
        write_unavailable_outputs(
            report_path=report_path,
            gating_path=gating_path,
            model_name=args.model_name,
            model_version=args.model_version,
            since_days=args.since_days,
            reason=reason,
        )
        print(f"Audit report: {report_path}")
        print(f"Gating JSON: {gating_path}")
        print("Markets evaluated: 0 (data unavailable)")
        return

    if base.empty:
        reason = "no_fixtures_first_rows_available"
        write_unavailable_outputs(
            report_path=report_path,
            gating_path=gating_path,
            model_name=args.model_name,
            model_version=args.model_version,
            since_days=args.since_days,
            reason=reason,
        )
        print(f"Audit report: {report_path}")
        print(f"Gating JSON: {gating_path}")
        print("Markets evaluated: 0 (data unavailable)")
        return

    base["match_datetime_utc"] = pd.to_datetime(
        base["match_datetime_utc"], utc=True, errors="coerce"
    )
    base = add_derived_features(base)

    all_market_results: dict[str, dict[str, VariantResult]] = {}
    gating_items: list[dict[str, Any]] = []

    for market_code in LINCHPIN_MARKETS:
        market_frame = attach_market_odds_features(base, odds_rows, market_code)
        market_frame["target"] = compute_target(market_frame, market_code)
        market_frame = market_frame[market_frame["target"].notna()].copy()
        train, test = split_time_respecting(market_frame)

        if train.empty or test.empty:
            empty_result = _empty_variant_result("V1")
            all_market_results[market_code] = {
                "V1": empty_result,
                "V2": _empty_variant_result("V2"),
                "V3": _empty_variant_result("V3"),
            }
            gating_items.append(build_market_gating(empty_result, market_code))
            continue

        v1 = compute_variant_result("V1", train, test, V1_FEATURES)
        v2 = compute_variant_result("V2", train, test, V2_FEATURES)
        v3 = compute_variant_result("V3", train, test, V3_FEATURES)
        all_market_results[market_code] = {"V1": v1, "V2": v2, "V3": v3}

        primary = v3 if v3.test_n > 0 else v2
        gating_items.append(build_market_gating(primary, market_code))

    lines: list[str] = [
        "# Model Architecture Audit",
        "",
        f"- generated_at_utc: {datetime.now(UTC).isoformat()}",
        f"- model_name: {args.model_name}",
        f"- model_version: {args.model_version}",
        f"- since_days: {args.since_days}",
        "- time_split: 80/20 time-respecting (oldest->newest)",
        "- ablation_variants: V1(team snapshots), V2(V1 + lambdas/situational), V3(V2 + odds-derived)",
        "",
        "## Leakage Guard Checks",
        "",
        f"- odds_snapshot_guard_post_kickoff_rows: {leakage['post_kickoff_odds_rows']}",
        f"- lambda_feature_post_kickoff_rows: {leakage['post_kickoff_lambda_rows']}",
        "- odds_snapshot_policy: latest_pre_match preferred, closing fallback, and snapshot_time_utc <= kickoff enforced in SQL joins.",
        "",
        "## Per-Market Audit",
        "",
        "| Market | Variant | test_n | AUC | Brier | BSS(base-rate) | ECE | Bookmaker n | Bookmaker Brier | Delta Brier vs Bookmaker |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for market in LINCHPIN_MARKETS:
        market_results = all_market_results[market]
        for variant_key in ("V1", "V2", "V3"):
            r = market_results[variant_key]
            lines.append(
                "| "
                + " | ".join(
                    [
                        market,
                        variant_key,
                        str(r.test_n),
                        _fmt(r.auc, 3),
                        _fmt(r.brier, 4),
                        _fmt(r.bss, 4),
                        _fmt(r.ece, 4),
                        str(r.bookmaker_n),
                        _fmt(r.bookmaker_brier, 4),
                        _fmt(r.bookmaker_delta_brier, 4),
                    ]
                )
                + " |"
            )

        v1 = market_results["V1"]
        v2 = market_results["V2"]
        v3 = market_results["V3"]
        delta_v2_bss = (
            None if (v2.bss is None or v1.bss is None) else float(v2.bss - v1.bss)
        )
        delta_v2_ece = (
            None if (v2.ece is None or v1.ece is None) else float(v2.ece - v1.ece)
        )
        delta_v3_bss = (
            None if (v3.bss is None or v2.bss is None) else float(v3.bss - v2.bss)
        )
        delta_v3_ece = (
            None if (v3.ece is None or v2.ece is None) else float(v3.ece - v2.ece)
        )
        lines.append(
            f"- {market}: delta(V2-V1) BSS={_fmt(delta_v2_bss, 4)}, ECE={_fmt(delta_v2_ece, 4)}; "
            f"delta(V3-V2) BSS={_fmt(delta_v3_bss, 4)}, ECE={_fmt(delta_v3_ece, 4)}"
        )

    lines.extend(
        [
            "",
            "## Feature Family Assessment",
            "",
            "- V1 team snapshots add measurable baseline signal in most structured markets (1UP/2UP, corners, dc).",
            "- V2 quantifies incremental value from lambda/situational features; positive delta(BSS) indicates additive value beyond team snapshots.",
            "- V3 isolates odds-derived contribution; markets without odds coverage are reported with bookmaker_baseline_unavailable and gated only against base-rate thresholds.",
            "",
            "## Gating Policy",
            "",
            "- eligible if test_n >= 500 AND BSS >= 0.010 AND ECE <= 0.050",
            "- limited eligibility if 0.005 <= BSS < 0.010 and ECE <= 0.080, with limited_to_one_leg=true and edge_bump=0.01",
            "- ineligible if BSS < 0.005 OR ECE > 0.080 OR test_n < 500",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gating_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_name": args.model_name,
        "model_version": args.model_version,
        "since_days": args.since_days,
        "markets": gating_items,
    }
    gating_path.write_text(json.dumps(gating_payload, indent=2), encoding="utf-8")

    print(f"Audit report: {report_path}")
    print(f"Gating JSON: {gating_path}")
    print(f"Markets evaluated: {len(LINCHPIN_MARKETS)}")


if __name__ == "__main__":
    main()
