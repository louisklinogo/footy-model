"""
Predict situational residuals (Layer 2) for scheduled fixtures.
Applies situational adjustments to the Layer 1 Poisson lambdas.
"""

from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnusedImport=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportCallIssue=false, reportArgumentType=false, reportMissingTypeArgument=false, reportUnusedParameter=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportGeneralTypeIssues=false
import argparse
import json
import sys
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.layer2_situational.deployment_policy import (
    load_layer2_deployment_policy,
    resolve_league_policy,
)
from src.modeling.layer2_situational import situational_utils
from src.modeling.layer2_situational.rule_layer import (
    apply_rule_adjustment,
    load_rule_layer_config,
)

MODEL_DIR = ROOT_DIR / "model_artifacts" / "situational_model"
MODEL_NAME = "situational_xgb"
MODEL_VERSION = "v2"
DEFAULT_RULE_OVERLAP_MODE = "override"
KEY_ABSENT_FAMILY = "key_absent"
RULE_SCOPE_GLOBAL = "enabled_league"
RULE_SCOPE_DISABLED_SAFETY = "disabled_league_safety"
KEY_ABSENT_OVERLAP_FEATURES = {
    "home": ("home_key_absent", "home_xg_lost", "injury_impact"),
    "away": ("away_key_absent", "away_xg_lost", "injury_impact"),
}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _market_scaled_alpha(base_alpha: float, odds_gap: object, residual_raw: float) -> float:
    effective = float(base_alpha)
    if effective <= 0.0:
        return 0.0
    gap = _to_float(odds_gap, float("nan"))
    if math.isnan(gap) or gap == 0.0:
        return effective
    agrees = (residual_raw >= 0.0) == (gap >= 0.0)
    return effective if agrees else effective * 0.4


def _predict_non_overlap_residuals(
    frame: pd.DataFrame,
    features_base: list[str],
    features_premium: list[str],
    home_model_base: object,
    away_model_base: object,
    home_model_premium: object,
    away_model_premium: object,
    mask_premium: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    x_base_full = frame[features_base].copy()
    x_home_base = x_base_full.copy()
    x_away_base = x_base_full.copy()
    
    x_premium_full = frame[features_premium].copy()
    x_home_premium = x_premium_full.copy()
    x_away_premium = x_premium_full.copy()

    for col in KEY_ABSENT_OVERLAP_FEATURES["home"]:
        if col in x_home_base.columns:
            x_home_base[col] = 0.0
        if col in x_home_premium.columns:
            x_home_premium[col] = 0.0
    for col in KEY_ABSENT_OVERLAP_FEATURES["away"]:
        if col in x_away_base.columns:
            x_away_base[col] = 0.0
        if col in x_away_premium.columns:
            x_away_premium[col] = 0.0

    home_pred = np.zeros(len(frame))
    away_pred = np.zeros(len(frame))
    
    # Base predictions
    idx_base = ~mask_premium
    if idx_base.any():
        home_pred[idx_base] = home_model_base.predict(x_home_base[idx_base].fillna(0).values)
        away_pred[idx_base] = away_model_base.predict(x_away_base[idx_base].fillna(0).values)
        
    # Premium predictions
    if mask_premium.any():
        home_pred[mask_premium] = home_model_premium.predict(x_home_premium[mask_premium].fillna(0).values)
        away_pred[mask_premium] = away_model_premium.predict(x_away_premium[mask_premium].fillna(0).values)

    return home_pred, away_pred


def _should_apply_key_absent_override(
    row: pd.Series,
    side: str,
    rule_payload: dict[str, object],
    rule_config: dict[str, object] | None,
) -> bool:
    if side not in ("home", "away"):
        return False
    if rule_config is None:
        return False
    if not bool(rule_payload.get("applied", False)):
        return False
    rules = rule_payload.get("rules", [])
    if KEY_ABSENT_FAMILY not in rules:
        return False
    key_absent = _to_float(row.get(f"{side}_key_absent", 0), 0.0)
    if key_absent < 0.5:
        return False
    pct = rule_config.get(f"key_absent_pct_{side}", rule_config.get("key_absent_pct", 0.0))
    return abs(_to_float(pct, 0.0)) > 1e-12


def _resolve_rule_scope(
    *,
    layer2_enabled: bool,
    is_low_confidence: bool,
    rule_config: dict[str, object] | None,
) -> str | None:
    if rule_config is None or is_low_confidence:
        return None
    if layer2_enabled:
        return RULE_SCOPE_GLOBAL
    if bool(rule_config.get("safety_enabled_for_disabled_leagues", False)):
        return RULE_SCOPE_DISABLED_SAFETY
    return None


def _empty_rule_payload(base_lambda: float) -> dict[str, object]:
    return {
        "mode": "none",
        "scope": "none",
        "applied": False,
        "rules": [],
        "components": [],
        "pct_raw": 0.0,
        "pct_capped": 0.0,
        "pct_capped_pre_gate": 0.0,
        "lambda_before": base_lambda,
        "lambda_after": base_lambda,
        "cap_down": 0.0,
        "cap_up": 0.0,
        "odds_gap": None,
        "odds_has_gap": False,
        "odds_missing": False,
        "odds_confirmed": False,
        "odds_conflict": False,
        "odds_gate_blocked": False,
    }


def _json_safe_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    return value


def _build_feature_snapshot(row: pd.Series, features: list[str]) -> dict[str, object]:
    return {feature: _json_safe_value(row.get(feature)) for feature in features}


def _build_tracking_signals(row: pd.Series) -> dict[str, object]:
    return {
        "home_key_absent": int(_to_float(row.get("home_key_absent", 0), 0.0)),
        "away_key_absent": int(_to_float(row.get("away_key_absent", 0), 0.0)),
        "home_upcoming_tier": int(_to_float(row.get("home_upcoming_tier", 0), 0.0)),
        "away_upcoming_tier": int(_to_float(row.get("away_upcoming_tier", 0), 0.0)),
        "points_gap": _json_safe_value(row.get("points_gap")),
        "position_gap": _json_safe_value(row.get("position_gap")),
        "rest_delta": _json_safe_value(row.get("rest_delta")),
        "home_congestion_games_14d": _json_safe_value(row.get("home_congestion_games_14d")),
        "away_congestion_games_14d": _json_safe_value(row.get("away_congestion_games_14d")),
        "home_playing_top4": int(_to_float(row.get("home_playing_top4", 0), 0.0)),
        "away_playing_top4": int(_to_float(row.get("away_playing_top4", 0), 0.0)),
        "injury_impact": _json_safe_value(row.get("injury_impact")),
        "home_xg_lost": _json_safe_value(row.get("home_xg_lost")),
        "away_xg_lost": _json_safe_value(row.get("away_xg_lost")),
        "odds_model_gap_home": _json_safe_value(row.get("odds_model_gap_home")),
        "odds_model_gap_away": _json_safe_value(row.get("odds_model_gap_away")),
    }


def is_lame_duck(played, pts_to_top, pts_to_relegation, total_teams):
    if pd.isna(played) or played < 20:
        return 0
    games_left = max(0, ((total_teams - 1) * 2) - played)
    if games_left > 8:
        return 0

    max_possible_pts = games_left * 3

    is_safe = pts_to_relegation > max_possible_pts
    is_eliminated = pts_to_top > max_possible_pts
    is_relegated = pts_to_relegation < -max_possible_pts

    if (is_safe and is_eliminated) or is_relegated:
        return 1
    return 0


# Source of truth: src/modeling/layer2_situational/situational_utils.py
# ---------------------------------------------------------------------------
# Prediction Data Loader
# ---------------------------------------------------------------------------
def load_prediction_data(days: int = 3, league: str | None = None) -> pd.DataFrame:
    conn = connect_db()

    # 1. Load HISTORY (finished fixtures) to establish state
    print("Loading historical results for state baseline...")
    history = pd.read_sql(
        """
        SELECT f.fixture_id, f.home_team_id, f.away_team_id, f.league_code,
               f.match_datetime_utc, fr.home_goals, fr.away_goals, f.status
        FROM fixtures f
        JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
        WHERE f.status = 'ft' AND fr.home_goals IS NOT NULL AND fr.away_goals IS NOT NULL
        ORDER BY f.match_datetime_utc ASC
    """,
        conn,
    )

    # 2. Load TARGETS (scheduled fixtures)
    print(f"Loading scheduled fixtures (horizon: {days} days)...")
    target_query = """
        SELECT f.fixture_id, f.home_team_id, f.away_team_id, f.league_code,
               f.match_datetime_utc, NULL as home_goals, NULL as away_goals, f.status
        FROM fixtures f
        WHERE f.status = 'scheduled'
          AND f.match_datetime_utc > NOW() - interval '6 hours'
          AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
    """
    params = [days]
    if league:
        target_query += " AND f.league_code = %s"
        params.append(league)

    targets = pd.read_sql(target_query, conn, params=tuple(params))

    if targets.empty:
        print("No scheduled fixtures found for the given criteria.")
        conn.close()
        return pd.DataFrame()

    # Combine for processing
    # We need history to calculate the 'points_before', 'played_before' etc for the targets
    combined = pd.concat([history, targets], ignore_index=True)
    combined["match_datetime_utc"] = pd.to_datetime(
        combined["match_datetime_utc"], utc=True, errors="coerce"
    )

    print("Computing point-in-time standings...")
    combined = situational_utils.add_season_key(combined)
    combined = situational_utils.compute_point_in_time_state(combined)

    # 4. Load other snapshot data
    print("Loading premium snapshots...")
    snap_df = pd.read_sql(
        """
        SELECT fixture_id, is_home,
               rolling_xg, rolling_xg_against, rolling_corners, rolling_rest_days,
               rolling_possession, rolling_possession_against,
               rolling_xg_p1, rolling_xg_p1_against,
               rolling_xg_h2_delta, rolling_xg_h2_delta_against,
               rolling_sot_p1, rolling_sot_p1_against,
               rolling_sot_h2_delta, rolling_sot_h2_delta_against,
               rolling_tackles_pct, rolling_tackles_pct_against,
               rolling_errors_lead_to_shot, rolling_errors_lead_to_shot_against
        FROM team_premium_snapshots
    """,
        conn,
    )

    _HOME_RENAME = {
        "rolling_xg": "home_rolling_xg",
        "rolling_xg_against": "home_rolling_xg_against",
        "rolling_corners": "home_rolling_corners",
        "rolling_rest_days": "home_rest_days",
        "rolling_possession": "home_rolling_possession",
        "rolling_possession_against": "home_rolling_possession_against",
        "rolling_xg_p1": "home_rolling_xg_p1",
        "rolling_xg_p1_against": "home_rolling_xg_p1_against",
        "rolling_xg_h2_delta": "home_rolling_xg_h2_delta",
        "rolling_xg_h2_delta_against": "home_rolling_xg_h2_delta_against",
        "rolling_sot_p1": "home_rolling_sot_p1",
        "rolling_sot_p1_against": "home_rolling_sot_p1_against",
        "rolling_sot_h2_delta": "home_rolling_sot_h2_delta",
        "rolling_sot_h2_delta_against": "home_rolling_sot_h2_delta_against",
        "rolling_tackles_pct": "home_rolling_tackles_pct",
        "rolling_tackles_pct_against": "home_rolling_tackles_pct_against",
        "rolling_errors_lead_to_shot": "home_rolling_errors_lead_to_shot",
        "rolling_errors_lead_to_shot_against": "home_rolling_errors_lead_to_shot_against",
    }
    _AWAY_RENAME = {
        "rolling_xg": "away_rolling_xg",
        "rolling_xg_against": "away_rolling_xg_against",
        "rolling_corners": "away_rolling_corners",
        "rolling_rest_days": "away_rest_days",
        "rolling_possession": "away_rolling_possession",
        "rolling_possession_against": "away_rolling_possession_against",
        "rolling_xg_p1": "away_rolling_xg_p1",
        "rolling_xg_p1_against": "away_rolling_xg_p1_against",
        "rolling_xg_h2_delta": "away_rolling_xg_h2_delta",
        "rolling_xg_h2_delta_against": "away_rolling_xg_h2_delta_against",
        "rolling_sot_p1": "away_rolling_sot_p1",
        "rolling_sot_p1_against": "away_rolling_sot_p1_against",
        "rolling_sot_h2_delta": "away_rolling_sot_h2_delta",
        "rolling_sot_h2_delta_against": "away_rolling_sot_h2_delta_against",
        "rolling_tackles_pct": "away_rolling_tackles_pct",
        "rolling_tackles_pct_against": "away_rolling_tackles_pct_against",
        "rolling_errors_lead_to_shot": "away_rolling_errors_lead_to_shot",
        "rolling_errors_lead_to_shot_against": "away_rolling_errors_lead_to_shot_against",
    }

    home_snap = snap_df[snap_df.is_home].rename(columns=_HOME_RENAME).drop(columns=["is_home"])
    away_snap = snap_df[~snap_df.is_home].rename(columns=_AWAY_RENAME).drop(columns=["is_home"])

    print("Loading Poisson predictions...")
    preds = pd.read_sql(
        situational_utils.latest_lambda_pairs_sql(),
        conn,
        params=("lambda_xgb",),
    )

    print("Loading Sofascore 1X2 odds snapshots...")
    odds_df = pd.DataFrame(
        columns=[
            "fixture_id",
            "odds_snapshot_time_utc",
            "odds_snapshot_type",
            "odds_json",
        ]
    )
    if not targets.empty:
        odds_df = pd.read_sql(
            """
            SELECT
                f.fixture_id,
                od.snapshot_time_utc AS odds_snapshot_time_utc,
                od.snapshot_type AS odds_snapshot_type,
                od.odds_json AS odds_json
            FROM fixtures f
            LEFT JOIN LATERAL (
                SELECT snapshot_time_utc, snapshot_type, odds_json
                FROM fixture_odds_markets
                WHERE fixture_id = f.fixture_id
                  AND provider = 'sofascore'
                  AND market_code = '1x2'
                  AND snapshot_type IN ('latest_pre_match', 'closing')
                  AND snapshot_time_utc <= f.match_datetime_utc
                ORDER BY (snapshot_type = 'latest_pre_match') DESC, snapshot_time_utc DESC
                LIMIT 1
            ) od ON true
            WHERE f.fixture_id = ANY(%s)
            """,
            conn,
            params=(targets["fixture_id"].tolist(),),
        )

    print("Loading rivalries...")
    rivalries = pd.read_sql("SELECT team_id_a, team_id_b FROM team_rivalries", conn)

    # 5. Phase 4: Player availability impact (home/away xG lost to injuries)
    print("Loading player injury impact (Phase 4)...")
    target_ids = targets["fixture_id"].tolist()
    if target_ids:
        impact_sql = """
        WITH mps AS (
            SELECT pa.fixture_id, pa.player_id, pa.team_id, f.match_datetime_utc
            FROM player_availability pa
            JOIN fixtures f ON f.fixture_id = pa.fixture_id
            WHERE pa.status IN ('missing', 'doubtful')
              AND pa.fixture_id = ANY(%s)
        ),
        pi AS (
            SELECT 
                mps.fixture_id,
                mps.team_id,
                SUM(q.avg_xg) AS team_xg_lost,
                MAX(CASE WHEN CAST(q.starts AS FLOAT) / GREATEST(q.apps, 1) >= 0.7 AND q.apps >= 3 THEN 1 ELSE 0 END) AS has_key_absence
            FROM mps
            LEFT JOIN LATERAL (
                SELECT 
                    AVG(fps.expected_goals) AS avg_xg,
                    COUNT(*) AS apps,
                    COUNT(*) FILTER (WHERE fps.substituted_in = False) AS starts
                FROM (
                    SELECT fps2.expected_goals, fps2.substituted_in
                    FROM fixture_player_stats fps2
                    JOIN fixtures f2 ON f2.fixture_id = fps2.fixture_id
                    WHERE fps2.player_id = mps.player_id
                      AND f2.match_datetime_utc < mps.match_datetime_utc
                    ORDER BY f2.match_datetime_utc DESC
                    LIMIT 10
                ) fps
            ) q ON TRUE
            GROUP BY mps.fixture_id, mps.team_id
        )
        SELECT 
            tf.fixture_id,
            tf.home_team_id,
            tf.away_team_id,
            COALESCE(SUM(pi.team_xg_lost) FILTER (WHERE pi.team_id = tf.home_team_id), 0) AS home_xg_lost,
            COALESCE(SUM(pi.team_xg_lost) FILTER (WHERE pi.team_id = tf.away_team_id), 0) AS away_xg_lost,
            COALESCE(MAX(pi.has_key_absence) FILTER (WHERE pi.team_id = tf.home_team_id), 0) AS home_key_absent,
            COALESCE(MAX(pi.has_key_absence) FILTER (WHERE pi.team_id = tf.away_team_id), 0) AS away_key_absent
        FROM fixtures tf
        LEFT JOIN pi ON pi.fixture_id = tf.fixture_id
        WHERE tf.fixture_id = ANY(%s)
        GROUP BY tf.fixture_id, tf.home_team_id, tf.away_team_id
        """
        impact_df = pd.read_sql(impact_sql, conn, params=(target_ids, target_ids))
    else:
        impact_df = pd.DataFrame(
            columns=[
                "fixture_id",
                "home_xg_lost",
                "away_xg_lost",
                "home_key_absent",
                "away_key_absent",
            ]
        )

    # Filter back to only targets now that standings are computed
    df = combined[combined["status"] == "scheduled"].copy()
    df["match_datetime_utc"] = pd.to_datetime(
        df["match_datetime_utc"], utc=True, errors="coerce"
    )
    df["points_gap"] = df["home_points"] - df["away_points"]
    df = df.merge(home_snap, on="fixture_id", how="left")
    df = df.merge(away_snap, on="fixture_id", how="left")
    df = df.merge(preds, on="fixture_id", how="left")
    df = df.merge(odds_df, on="fixture_id", how="left")
    df = df.merge(
        impact_df[
            [
                "fixture_id",
                "home_xg_lost",
                "away_xg_lost",
                "home_key_absent",
                "away_key_absent",
            ]
        ],
        on="fixture_id",
        how="left",
    )

    # L5 Goals (simplified for prediction - reuse historical query)
    print("Computing L5 goals performance...")
    l5_query = """
        WITH base AS (
            SELECT fixture_id, home_team_id, away_team_id, match_datetime_utc
            FROM fixtures
            WHERE fixture_id = ANY(%s)
        ),
        l5_stats AS (
            SELECT
                b.fixture_id,
                (SELECT COALESCE(SUM(goals), 0) FROM (
                    SELECT CASE WHEN f2.home_team_id = b.home_team_id THEN fr2.home_goals ELSE fr2.away_goals END as goals
                    FROM fixtures f2 JOIN fixture_results fr2 ON fr2.fixture_id = f2.fixture_id
                    WHERE (f2.home_team_id = b.home_team_id OR f2.away_team_id = b.home_team_id)
                      AND f2.match_datetime_utc < b.match_datetime_utc AND f2.status = 'ft'
                    ORDER BY f2.match_datetime_utc DESC LIMIT 5
                ) t) as h_scored_l5,
                (SELECT COALESCE(SUM(goals), 0) FROM (
                    SELECT CASE WHEN f2.home_team_id = b.home_team_id THEN fr2.away_goals ELSE fr2.home_goals END as goals
                    FROM fixtures f2 JOIN fixture_results fr2 ON fr2.fixture_id = f2.fixture_id
                    WHERE (f2.home_team_id = b.home_team_id OR f2.away_team_id = b.home_team_id)
                      AND f2.match_datetime_utc < b.match_datetime_utc AND f2.status = 'ft'
                    ORDER BY f2.match_datetime_utc DESC LIMIT 5
                ) t) as h_conceded_l5,
                (SELECT COALESCE(SUM(goals), 0) FROM (
                    SELECT CASE WHEN f2.home_team_id = b.away_team_id THEN fr2.home_goals ELSE fr2.away_goals END as goals
                    FROM fixtures f2 JOIN fixture_results fr2 ON fr2.fixture_id = f2.fixture_id
                    WHERE (f2.home_team_id = b.away_team_id OR f2.away_team_id = b.away_team_id)
                      AND f2.match_datetime_utc < b.match_datetime_utc AND f2.status = 'ft'
                    ORDER BY f2.match_datetime_utc DESC LIMIT 5
                ) t) as a_scored_l5,
                (SELECT COALESCE(SUM(goals), 0) FROM (
                    SELECT CASE WHEN f2.home_team_id = b.away_team_id THEN fr2.away_goals ELSE fr2.home_goals END as goals
                    FROM fixtures f2 JOIN fixture_results fr2 ON fr2.fixture_id = f2.fixture_id
                    WHERE (f2.home_team_id = b.away_team_id OR f2.away_team_id = b.away_team_id)
                      AND f2.match_datetime_utc < b.match_datetime_utc AND f2.status = 'ft'
                    ORDER BY f2.match_datetime_utc DESC LIMIT 5
                ) t) as a_conceded_l5
            FROM base b
        )
        SELECT * FROM l5_stats
    """
    l5_data = pd.read_sql(l5_query, conn, params=(df["fixture_id"].tolist(),))
    df = df.merge(l5_data, on="fixture_id", how="left")

    # Feature calcs
    df["xg_diff"] = df["home_rolling_xg"].fillna(0) - df[
        "away_rolling_xg_against"
    ].fillna(0)
    df["home_xg_over_scored"] = df["h_scored_l5"].fillna(0) - (
        df["home_rolling_xg"].fillna(0) * 5
    )
    df["home_xg_over_conceded"] = df["h_conceded_l5"].fillna(0) - (
        df["home_rolling_xg_against"].fillna(0) * 5
    )
    df["away_xg_over_scored"] = df["a_scored_l5"].fillna(0) - (
        df["away_rolling_xg"].fillna(0) * 5
    )
    df["away_xg_over_conceded"] = df["a_conceded_l5"].fillna(0) - (
        df["away_rolling_xg_against"].fillna(0) * 5
    )
    df["rest_delta"] = df["home_rest_days"].fillna(0) - df["away_rest_days"].fillna(0)
    df["xg_p1_delta"] = (
        df["home_rolling_xg_p1"].fillna(0)
        - df["away_rolling_xg_p1"].fillna(0)
    )
    df["h2_surge_delta"] = (
        df["home_rolling_xg_h2_delta"].fillna(0)
        - df["away_rolling_xg_h2_delta"].fillna(0)
    )

    # Congestion & Upcoming Tier
    print("Computing schedule flags...")
    base_team_dates = df[
        ["fixture_id", "home_team_id", "away_team_id", "match_datetime_utc"]
    ].copy()

    all_fixtures_raw = pd.read_sql(
        """
        SELECT fixture_id, home_team_id, away_team_id, match_datetime_utc
        FROM fixtures
        WHERE status = 'ft'
        ORDER BY match_datetime_utc
        """,
        conn,
    )
    all_team_sched = situational_utils.build_team_schedule(all_fixtures_raw)
    cong = situational_utils.compute_congestion_for_targets(
        base_team_dates=base_team_dates,
        all_team_sched=all_team_sched,
    )
    df = df.join(cong)

    euro_all = pd.read_sql(
        """
        SELECT fixture_id, home_team_id, away_team_id, match_datetime_utc, league_code
        FROM fixtures
        WHERE league_code IN ('CL','EL','ECL')
        ORDER BY match_datetime_utc
        """,
        conn,
    )
    euro_team_dates = situational_utils.build_euro_team_dates(euro_all)
    upc = situational_utils.compute_upcoming_tier_for_targets(
        base_team_dates=base_team_dates,
        euro_team_dates=euro_team_dates,
    )
    df = df.join(upc)

    rivalry_pairs = set(zip(rivalries["team_id_a"], rivalries["team_id_b"]))
    df["is_derby"] = df.apply(
        lambda r: (
            1
            if (r["home_team_id"], r["away_team_id"]) in rivalry_pairs
            or (r["away_team_id"], r["home_team_id"]) in rivalry_pairs
            else 0
        ),
        axis=1,
    )

    df["home_form_streak"] = df["home_form_streak"].fillna(0).astype(int)
    df["away_form_streak"] = df["away_form_streak"].fillna(0).astype(int)
    df["home_xg_lost"] = df["home_xg_lost"].fillna(0)
    df["away_xg_lost"] = df["away_xg_lost"].fillna(0)
    df["home_key_absent"] = df["home_key_absent"].fillna(0).astype(int)
    df["away_key_absent"] = df["away_key_absent"].fillna(0).astype(int)
    df["home_lame_duck"] = df.apply(
        lambda r: is_lame_duck(
            r["home_played"],
            r["home_points_to_top"],
            r["home_points_to_relegation"],
            r["league_team_count"],
        ),
        axis=1,
    )
    df["away_lame_duck"] = df.apply(
        lambda r: is_lame_duck(
            r["away_played"],
            r["away_points_to_top"],
            r["away_points_to_relegation"],
            r["league_team_count"],
        ),
        axis=1,
    )
    df["injury_impact"] = df["home_xg_lost"] - df["away_xg_lost"]
    df["home_playing_top4"] = (df["away_position"] <= 4).astype(int)
    df["away_playing_top4"] = (df["home_position"] <= 4).astype(int)
    df["derby_position_gap"] = df["position_gap"] * df["is_derby"]
    df["odds_snapshot_time_utc"] = pd.to_datetime(
        df["odds_snapshot_time_utc"], utc=True, errors="coerce"
    )
    bad_odds = (
        df["odds_snapshot_time_utc"].notna()
        & (df["odds_snapshot_time_utc"] > df["match_datetime_utc"])
    )
    if bad_odds.any():
        raise RuntimeError(
            f"Found {int(bad_odds.sum())} post-kickoff odds snapshots; refusing to predict."
        )

    def implied_from_prices(prices: dict[str, float] | None) -> dict[str, float] | None:
        if not prices:
            return None
        try:
            inv_sum = sum(1.0 / float(v) for v in prices.values())
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        if inv_sum <= 0:
            return None
        return {k: (1.0 / float(v)) / inv_sum for k, v in prices.items()}

    def outcome_probs(lh: float, la: float, max_goals: int = 8) -> tuple[float, float, float]:
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = np.exp(-lh) * (lh**h) / math.factorial(h)
                prob *= np.exp(-la) * (la**a) / math.factorial(a)
                if h > a:
                    p_home += prob
                elif h == a:
                    p_draw += prob
                else:
                    p_away += prob
        return p_home, p_draw, p_away

    df["poisson_home_prob"] = np.nan
    df["poisson_draw_prob"] = np.nan
    df["poisson_away_prob"] = np.nan
    mask = df["lambda_home"].notna() & df["lambda_away"].notna()
    if mask.any():
        probs = [
            outcome_probs(lh, la)
            for lh, la in zip(
                df.loc[mask, "lambda_home"], df.loc[mask, "lambda_away"]
            )
        ]
        probs_df = pd.DataFrame(
            probs,
            index=df.loc[mask].index,
            columns=["poisson_home_prob", "poisson_draw_prob", "poisson_away_prob"],
        )
        df.loc[mask, ["poisson_home_prob", "poisson_draw_prob", "poisson_away_prob"]] = (
            probs_df
        )

    df["odds_model_gap"] = np.nan
    df["odds_model_gap_home"] = np.nan
    df["odds_model_gap_draw"] = np.nan
    df["odds_model_gap_away"] = np.nan
    df["odds_opening_gap_home"] = np.nan
    df["odds_opening_gap_draw"] = np.nan
    df["odds_opening_gap_away"] = np.nan

    for idx, row in df.loc[mask].iterrows():
        payload = row.get("odds_json")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                continue
        if not isinstance(payload, dict):
            continue
        latest = implied_from_prices(payload.get("prices_latest"))
        opening = implied_from_prices(payload.get("prices_opening"))
        if latest:
            df.at[idx, "odds_model_gap_home"] = latest.get("home") - row["poisson_home_prob"]
            df.at[idx, "odds_model_gap_draw"] = latest.get("draw") - row["poisson_draw_prob"]
            df.at[idx, "odds_model_gap_away"] = latest.get("away") - row["poisson_away_prob"]
            df.at[idx, "odds_model_gap"] = df.at[idx, "odds_model_gap_home"]
        if opening:
            df.at[idx, "odds_opening_gap_home"] = opening.get("home") - row["poisson_home_prob"]
            df.at[idx, "odds_opening_gap_draw"] = opening.get("draw") - row["poisson_draw_prob"]
            df.at[idx, "odds_opening_gap_away"] = opening.get("away") - row["poisson_away_prob"]

    conn.close()
    return df


def _has_prediction_lineage_columns(conn: object) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'predictions'
              AND column_name IN ('first_created_at', 'last_refreshed_at', 'feature_asof_utc')
            """
        )
        row = cur.fetchone()
    return bool(row and int(row[0]) == 3)


def save_residuals(results: list[dict]):
    """Upsert situational outputs.

    predictions.p_model has a 0-1 CHECK constraint (probability semantics).
    Residuals are signed reals and adjusted lambdas can exceed 1, so we follow
    the same convention as predict_lambda.py: store the real value in
    metadata_json and use 0.5 as a harmless sentinel for p_model.
    """
    if not results:
        return
    conn = connect_db()
    has_lineage_cols = _has_prediction_lineage_columns(conn)
    if has_lineage_cols:
        sql = """
            INSERT INTO predictions
                (
                    fixture_id,
                    market_code,
                    model_name,
                    model_version,
                    p_model,
                    metadata_json,
                    created_at,
                    first_created_at,
                    last_refreshed_at,
                    feature_asof_utc
                )
            VALUES (%s, %s, %s, %s, 0.5, %s, NOW(), NOW(), NOW(), %s)
            ON CONFLICT (fixture_id, market_code, model_name, model_version) DO UPDATE SET
                p_model = EXCLUDED.p_model,
                metadata_json = EXCLUDED.metadata_json,
                last_refreshed_at = NOW(),
                feature_asof_utc = COALESCE(EXCLUDED.feature_asof_utc, predictions.feature_asof_utc)
        """
    else:
        sql = """
            INSERT INTO predictions (fixture_id, market_code, model_name, model_version, p_model, metadata_json, created_at)
            VALUES (%s, %s, %s, %s, 0.5, %s, NOW())
            ON CONFLICT (fixture_id, market_code, model_name, model_version) DO UPDATE SET
                p_model = EXCLUDED.p_model,
                metadata_json = EXCLUDED.metadata_json,
                created_at = NOW()
        """
    with conn.cursor() as cur:
        for r in results:
            if has_lineage_cols:
                cur.execute(
                    sql,
                    (
                        r["fixture_id"],
                        r["market_code"],
                        MODEL_NAME,
                        MODEL_VERSION,
                        json.dumps(r["meta"]),
                        r.get("feature_asof_utc"),
                    ),
                )
            else:
                cur.execute(
                    sql,
                    (
                        r["fixture_id"],
                        r["market_code"],
                        MODEL_NAME,
                        MODEL_VERSION,
                        json.dumps(r["meta"]),
                    ),
                )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--league", type=str, default=None)
    parser.add_argument(
        "--enable-rule-layer",
        action="store_true",
        help="Apply deterministic situational rule-layer on top of Layer 2 adjusted lambdas.",
    )
    parser.add_argument(
        "--rule-layer-config",
        type=Path,
        default=None,
        help="Optional JSON config path for rule-layer adjustments.",
    )
    parser.add_argument(
        "--rule-overlap-mode",
        choices=("additive", "override"),
        default=DEFAULT_RULE_OVERLAP_MODE,
        help=(
            "How to handle overlap between global Layer 2 residuals and deterministic "
            "rule families. 'override' is the production default."
        ),
    )
    parser.add_argument(
        "--rule-layer-only",
        action="store_true",
        default=False,
        help="Use rule layer only, skip ML model predictions (set residuals to 0.0).",
    )
    args = parser.parse_args()

    if args.rule_layer_only:
        print("[rule-layer-only] Using rule layer only, ML predictions disabled")
        features_base: list[str] = []
        features_premium: list[str] = []
        home_model_base = None
        away_model_base = None
        home_model_premium = None
        away_model_premium = None
    else:
        base_blob = joblib.load(MODEL_DIR / "situational_model_base.pkl")
        features_base = base_blob["features"]
        home_model_base = base_blob["home_model"]
        away_model_base = base_blob["away_model"]
        
        premium_blob = joblib.load(MODEL_DIR / "situational_model_premium.pkl")
        features_premium = premium_blob["features"]
        home_model_premium = premium_blob["home_model"]
        away_model_premium = premium_blob["away_model"]

    # We use the deployment policy generated by the base model as the source of truth for league enablement.
    policy = load_layer2_deployment_policy(MODEL_DIR)
    rule_config_path = args.rule_layer_config or (MODEL_DIR / "rule_layer_config.json")
    rule_config = (
        load_rule_layer_config(rule_config_path) if args.enable_rule_layer else None
    )

    df = load_prediction_data(days=args.days, league=args.league)
    if df.empty:
        return

    # Determine which rows have complete premium data
    if args.rule_layer_only:
        # When rule-layer-only, all fixtures get 0.0 residuals
        mask_premium = pd.Series([False] * len(df), index=df.index)
    else:
        mask_premium = df[features_premium].notna().all(axis=1)

    df["pred_home_residual_full"] = np.nan
    df["pred_away_residual_full"] = np.nan
    df["model_source"] = "base"

    if args.rule_layer_only:
        # Set all residuals to 0.0 when using rule layer only
        df["pred_home_residual_full"] = 0.0
        df["pred_away_residual_full"] = 0.0
        df["model_source"] = "rule_layer_only"
    else:
        # 1. Base Model Predictions
        idx_base = ~mask_premium
        if idx_base.any():
            X_base = df.loc[idx_base, features_base].fillna(0).values
            df.loc[idx_base, "pred_home_residual_full"] = home_model_base.predict(X_base)
            df.loc[idx_base, "pred_away_residual_full"] = away_model_base.predict(X_base)

        # 2. Premium Model Predictions
        if mask_premium.any():
            X_premium = df.loc[mask_premium, features_premium].fillna(0).values
            df.loc[mask_premium, "pred_home_residual_full"] = home_model_premium.predict(X_premium)
            df.loc[mask_premium, "pred_away_residual_full"] = away_model_premium.predict(X_premium)
            df.loc[mask_premium, "model_source"] = "premium"

    if args.rule_layer_only:
        # When rule-layer-only, non-overlap residuals are also 0.0
        df["pred_home_residual_non_overlap"] = 0.0
        df["pred_away_residual_non_overlap"] = 0.0
    elif args.enable_rule_layer and args.rule_overlap_mode == "override":
        home_non_overlap, away_non_overlap = _predict_non_overlap_residuals(
            frame=df,
            features_base=features_base,
            features_premium=features_premium,
            home_model_base=home_model_base,
            away_model_base=away_model_base,
            home_model_premium=home_model_premium,
            away_model_premium=away_model_premium,
            mask_premium=mask_premium,
        )
        df["pred_home_residual_non_overlap"] = home_non_overlap
        df["pred_away_residual_non_overlap"] = away_non_overlap
    else:
        df["pred_home_residual_non_overlap"] = df["pred_home_residual_full"]
        df["pred_away_residual_non_overlap"] = df["pred_away_residual_full"]

    # Vectorised confidence guard aligned with Layer 2 train filter (min 4 games).
    df["is_low_confidence"] = (df["home_played"] < 4) | (df["away_played"] < 4)

    out_rows = []
    rule_fired_home = 0
    rule_fired_away = 0
    overlap_override_home = 0
    overlap_override_away = 0
    rule_fired_fixture_ids: set[int] = set()
    rule_fired_scope_counts = {
        RULE_SCOPE_GLOBAL: 0,
        RULE_SCOPE_DISABLED_SAFETY: 0,
    }
    run_asof_utc = pd.Timestamp.now(tz="UTC")
    for _, row in df.iterrows():
        h_res_raw_full = float(row["pred_home_residual_full"])
        a_res_raw_full = float(row["pred_away_residual_full"])
        h_res_raw = h_res_raw_full
        a_res_raw = a_res_raw_full
        h_res_source = row["model_source"]
        a_res_source = row["model_source"]
        h_overlap_family: str | None = None
        a_overlap_family: str | None = None
        h_override_applied = False
        a_override_applied = False
        h_res_non_overlap = float(row["pred_home_residual_non_overlap"])
        a_res_non_overlap = float(row["pred_away_residual_non_overlap"])
        lh = None if pd.isna(row.get("lambda_home")) else float(row["lambda_home"])
        la = None if pd.isna(row.get("lambda_away")) else float(row["lambda_away"])
        kickoff = pd.to_datetime(row.get("match_datetime_utc"), utc=True, errors="coerce")
        feature_asof_utc = None
        if pd.notna(kickoff):
            feature_asof_utc = min(run_asof_utc, kickoff).to_pydatetime()

        league_policy = resolve_league_policy(policy, row.get("league_code"))
        layer2_enabled = bool(league_policy.get("enabled", False))
        alpha = float(league_policy.get("alpha", 0.0))
        gate_reason = str(league_policy.get("reason", "default_policy"))
        effective_alpha = alpha if layer2_enabled else 0.0

        odds_gap_home = row.get("odds_model_gap_home")
        odds_gap_away = row.get("odds_model_gap_away")
        alpha_home = _market_scaled_alpha(effective_alpha, odds_gap_home, h_res_raw)
        alpha_away = _market_scaled_alpha(effective_alpha, odds_gap_away, a_res_raw)

        h_res_applied = alpha_home * h_res_raw
        a_res_applied = alpha_away * a_res_raw

        is_low = bool(row["is_low_confidence"])
        reason = "early_season_min_4_games_not_met" if is_low else None
        feature_snapshot = _build_feature_snapshot(row, features_premium if row["model_source"] == "premium" else features_base)
        tracking_signals = _build_tracking_signals(row)
        rule_scope = (
            _resolve_rule_scope(
                layer2_enabled=layer2_enabled,
                is_low_confidence=is_low,
                rule_config=rule_config,
            )
            if args.enable_rule_layer
            else None
        )

        # Adjusted lambdas use the policy-gated residual.
        if lh is not None and la is not None:
            lambda_home_layer2 = max(0.01, lh + h_res_applied)
            lambda_away_layer2 = max(0.01, la + a_res_applied)

            home_rule = _empty_rule_payload(lambda_home_layer2)
            away_rule = _empty_rule_payload(lambda_away_layer2)

            if (
                args.enable_rule_layer
                and rule_config is not None
                and rule_scope is not None
            ):
                home_rule = apply_rule_adjustment(
                    lambda_home_layer2,
                    row,
                    side="home",
                    config=rule_config,
                    scope=rule_scope,
                )
                away_rule = apply_rule_adjustment(
                    lambda_away_layer2,
                    row,
                    side="away",
                    config=rule_config,
                    scope=rule_scope,
                )

                if args.rule_overlap_mode == "override" and rule_scope == RULE_SCOPE_GLOBAL:
                    if _should_apply_key_absent_override(
                        row=row,
                        side="home",
                        rule_payload=home_rule,
                        rule_config=rule_config,
                    ):
                        h_res_raw = h_res_non_overlap
                        h_res_source = "non_overlap_override_key_absent"
                        h_overlap_family = KEY_ABSENT_FAMILY
                        h_override_applied = True
                        overlap_override_home += 1
                        alpha_home = _market_scaled_alpha(
                            effective_alpha, odds_gap_home, h_res_raw
                        )
                        h_res_applied = alpha_home * h_res_raw
                        lambda_home_layer2 = max(0.01, lh + h_res_applied)
                        home_rule = apply_rule_adjustment(
                            lambda_home_layer2,
                            row,
                            side="home",
                            config=rule_config,
                            scope=rule_scope,
                        )

                    if _should_apply_key_absent_override(
                        row=row,
                        side="away",
                        rule_payload=away_rule,
                        rule_config=rule_config,
                    ):
                        a_res_raw = a_res_non_overlap
                        a_res_source = "non_overlap_override_key_absent"
                        a_overlap_family = KEY_ABSENT_FAMILY
                        a_override_applied = True
                        overlap_override_away += 1
                        alpha_away = _market_scaled_alpha(
                            effective_alpha, odds_gap_away, a_res_raw
                        )
                        a_res_applied = alpha_away * a_res_raw
                        lambda_away_layer2 = max(0.01, la + a_res_applied)
                        away_rule = apply_rule_adjustment(
                            lambda_away_layer2,
                            row,
                            side="away",
                            config=rule_config,
                            scope=rule_scope,
                        )

                if bool(home_rule["applied"]):
                    rule_fired_home += 1
                    rule_fired_fixture_ids.add(int(row["fixture_id"]))
                    scoped = str(home_rule.get("scope", ""))
                    if scoped in rule_fired_scope_counts:
                        rule_fired_scope_counts[scoped] += 1
                if bool(away_rule["applied"]):
                    rule_fired_away += 1
                    rule_fired_fixture_ids.add(int(row["fixture_id"]))
                    scoped = str(away_rule.get("scope", ""))
                    if scoped in rule_fired_scope_counts:
                        rule_fired_scope_counts[scoped] += 1

            res_meta_h = {
                "residual": h_res_raw,
                "residual_raw": h_res_raw,
                "residual_raw_full": h_res_raw_full,
                "residual_raw_non_overlap": h_res_non_overlap,
                "residual_source": h_res_source,
                "residual_applied": h_res_applied,
                "base_lambda": lh,
                "layer2_enabled": layer2_enabled,
                "layer2_alpha": alpha_home,
                "layer2_alpha_policy": effective_alpha,
                "layer2_gate_reason": gate_reason,
                "market_confirmed": bool(
                    pd.notna(odds_gap_home) and _to_float(odds_gap_home, 0.0) != 0.0
                ),
                "overlap_mode": args.rule_overlap_mode if args.enable_rule_layer else "none",
                "overlap_override_applied": h_override_applied,
                "overlap_family": h_overlap_family,
                "rule_layer_scope": str(rule_scope) if rule_scope is not None else "none",
            }
            res_meta_a = {
                "residual": a_res_raw,
                "residual_raw": a_res_raw,
                "residual_raw_full": a_res_raw_full,
                "residual_raw_non_overlap": a_res_non_overlap,
                "residual_source": a_res_source,
                "residual_applied": a_res_applied,
                "base_lambda": la,
                "layer2_enabled": layer2_enabled,
                "layer2_alpha": alpha_away,
                "layer2_alpha_policy": effective_alpha,
                "layer2_gate_reason": gate_reason,
                "market_confirmed": bool(
                    pd.notna(odds_gap_away) and _to_float(odds_gap_away, 0.0) != 0.0
                ),
                "overlap_mode": args.rule_overlap_mode if args.enable_rule_layer else "none",
                "overlap_override_applied": a_override_applied,
                "overlap_family": a_overlap_family,
                "rule_layer_scope": str(rule_scope) if rule_scope is not None else "none",
            }

            adj_meta_h = {
                "lambda": float(home_rule["lambda_after"]),
                "lambda_before_layer2": lh,
                "lambda_after_layer2": lambda_home_layer2,
                "residual": h_res_raw,
                "residual_raw": h_res_raw,
                "residual_raw_full": h_res_raw_full,
                "residual_raw_non_overlap": h_res_non_overlap,
                "residual_source": h_res_source,
                "residual_applied": h_res_applied,
                "base_lambda": lh,
                "layer2_enabled": layer2_enabled,
                "layer2_alpha": alpha_home,
                "layer2_alpha_policy": effective_alpha,
                "layer2_gate_reason": gate_reason,
                "market_confirmed": bool(
                    pd.notna(odds_gap_home) and _to_float(odds_gap_home, 0.0) != 0.0
                ),
                "rule_layer_enabled": bool(args.enable_rule_layer),
                "rule_layer_overlap_mode": args.rule_overlap_mode if args.enable_rule_layer else None,
                "rule_layer_mode": str(home_rule.get("mode", "none")),
                "rule_layer_scope": str(home_rule.get("scope", "none")),
                "rule_layer_applied": bool(home_rule["applied"]),
                "rule_layer_rules": list(home_rule["rules"]),
                "rule_layer_components": list(home_rule.get("components", [])),
                "rule_layer_pct_raw": float(home_rule["pct_raw"]),
                "rule_layer_pct_capped": float(home_rule["pct_capped"]),
                "rule_layer_pct_capped_pre_gate": float(
                    home_rule.get("pct_capped_pre_gate", home_rule["pct_capped"])
                ),
                "rule_layer_cap_down": float(home_rule["cap_down"]),
                "rule_layer_cap_up": float(home_rule["cap_up"]),
                "rule_layer_odds_gap": home_rule.get("odds_gap"),
                "rule_layer_odds_has_gap": bool(home_rule.get("odds_has_gap", False)),
                "rule_layer_odds_missing": bool(home_rule.get("odds_missing", False)),
                "rule_layer_odds_confirmed": bool(home_rule.get("odds_confirmed", False)),
                "rule_layer_odds_conflict": bool(home_rule.get("odds_conflict", False)),
                "rule_layer_odds_gate_blocked": bool(home_rule.get("odds_gate_blocked", False)),
                "rule_layer_config_path": str(rule_config_path) if args.enable_rule_layer else None,
                "overlap_override_applied": h_override_applied,
                "overlap_family": h_overlap_family,
                "feature_snapshot": feature_snapshot,
                "tracking_signals": tracking_signals,
            }
            adj_meta_a = {
                "lambda": float(away_rule["lambda_after"]),
                "lambda_before_layer2": la,
                "lambda_after_layer2": lambda_away_layer2,
                "residual": a_res_raw,
                "residual_raw": a_res_raw,
                "residual_raw_full": a_res_raw_full,
                "residual_raw_non_overlap": a_res_non_overlap,
                "residual_source": a_res_source,
                "residual_applied": a_res_applied,
                "base_lambda": la,
                "layer2_enabled": layer2_enabled,
                "layer2_alpha": alpha_away,
                "layer2_alpha_policy": effective_alpha,
                "layer2_gate_reason": gate_reason,
                "market_confirmed": bool(
                    pd.notna(odds_gap_away) and _to_float(odds_gap_away, 0.0) != 0.0
                ),
                "rule_layer_enabled": bool(args.enable_rule_layer),
                "rule_layer_overlap_mode": args.rule_overlap_mode if args.enable_rule_layer else None,
                "rule_layer_mode": str(away_rule.get("mode", "none")),
                "rule_layer_scope": str(away_rule.get("scope", "none")),
                "rule_layer_applied": bool(away_rule["applied"]),
                "rule_layer_rules": list(away_rule["rules"]),
                "rule_layer_components": list(away_rule.get("components", [])),
                "rule_layer_pct_raw": float(away_rule["pct_raw"]),
                "rule_layer_pct_capped": float(away_rule["pct_capped"]),
                "rule_layer_pct_capped_pre_gate": float(
                    away_rule.get("pct_capped_pre_gate", away_rule["pct_capped"])
                ),
                "rule_layer_cap_down": float(away_rule["cap_down"]),
                "rule_layer_cap_up": float(away_rule["cap_up"]),
                "rule_layer_odds_gap": away_rule.get("odds_gap"),
                "rule_layer_odds_has_gap": bool(away_rule.get("odds_has_gap", False)),
                "rule_layer_odds_missing": bool(away_rule.get("odds_missing", False)),
                "rule_layer_odds_confirmed": bool(away_rule.get("odds_confirmed", False)),
                "rule_layer_odds_conflict": bool(away_rule.get("odds_conflict", False)),
                "rule_layer_odds_gate_blocked": bool(away_rule.get("odds_gate_blocked", False)),
                "rule_layer_config_path": str(rule_config_path) if args.enable_rule_layer else None,
                "overlap_override_applied": a_override_applied,
                "overlap_family": a_overlap_family,
                "feature_snapshot": feature_snapshot,
                "tracking_signals": tracking_signals,
            }

            if is_low:
                low_meta = {"is_low_confidence": True, "low_confidence_reason": reason}
                res_meta_h.update(low_meta)
                res_meta_a.update(low_meta)
                adj_meta_h.update(
                    low_meta
                )
                adj_meta_a.update(
                    low_meta
                )

            out_rows.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "market_code": "home_residual",
                    "meta": res_meta_h,
                    "feature_asof_utc": feature_asof_utc,
                }
            )
            out_rows.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "market_code": "away_residual",
                    "meta": res_meta_a,
                    "feature_asof_utc": feature_asof_utc,
                }
            )
            out_rows.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "market_code": "adj_lambda_home",
                    "meta": adj_meta_h,
                    "feature_asof_utc": feature_asof_utc,
                }
            )
            out_rows.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "market_code": "adj_lambda_away",
                    "meta": adj_meta_a,
                    "feature_asof_utc": feature_asof_utc,
                }
            )
        else:
            res_meta_h = {
                "residual": h_res_raw,
                "residual_raw": h_res_raw,
                "residual_raw_full": h_res_raw_full,
                "residual_raw_non_overlap": h_res_non_overlap,
                "residual_source": h_res_source,
                "residual_applied": h_res_applied,
                "base_lambda": lh,
                "layer2_enabled": layer2_enabled,
                "layer2_alpha": alpha_home,
                "layer2_alpha_policy": effective_alpha,
                "layer2_gate_reason": gate_reason,
                "market_confirmed": bool(
                    pd.notna(odds_gap_home) and _to_float(odds_gap_home, 0.0) != 0.0
                ),
                "overlap_mode": args.rule_overlap_mode if args.enable_rule_layer else "none",
                "overlap_override_applied": h_override_applied,
                "overlap_family": h_overlap_family,
                "rule_layer_scope": str(rule_scope) if rule_scope is not None else "none",
            }
            res_meta_a = {
                "residual": a_res_raw,
                "residual_raw": a_res_raw,
                "residual_raw_full": a_res_raw_full,
                "residual_raw_non_overlap": a_res_non_overlap,
                "residual_source": a_res_source,
                "residual_applied": a_res_applied,
                "base_lambda": la,
                "layer2_enabled": layer2_enabled,
                "layer2_alpha": alpha_away,
                "layer2_alpha_policy": effective_alpha,
                "layer2_gate_reason": gate_reason,
                "market_confirmed": bool(
                    pd.notna(odds_gap_away) and _to_float(odds_gap_away, 0.0) != 0.0
                ),
                "overlap_mode": args.rule_overlap_mode if args.enable_rule_layer else "none",
                "overlap_override_applied": a_override_applied,
                "overlap_family": a_overlap_family,
                "rule_layer_scope": str(rule_scope) if rule_scope is not None else "none",
            }
            if is_low:
                low_meta = {"is_low_confidence": True, "low_confidence_reason": reason}
                res_meta_h.update(low_meta)
                res_meta_a.update(low_meta)
            out_rows.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "market_code": "home_residual",
                    "meta": res_meta_h,
                    "feature_asof_utc": feature_asof_utc,
                }
            )
            out_rows.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "market_code": "away_residual",
                    "meta": res_meta_a,
                    "feature_asof_utc": feature_asof_utc,
                }
            )
    save_residuals(out_rows)
    n_with_adj = sum(1 for r in out_rows if r["market_code"] == "adj_lambda_home")
    print(
        f"Upserted {len(out_rows)} records ({n_with_adj} fixtures with adjusted lambdas) for {len(df)} fixtures."
    )
    if args.enable_rule_layer:
        print(
            "Rule-layer fired: "
            f"home={rule_fired_home}, away={rule_fired_away}, "
            f"fixtures={len(rule_fired_fixture_ids)}"
        )
        print(
            "Rule-layer scope events: "
            f"enabled_league={rule_fired_scope_counts[RULE_SCOPE_GLOBAL]}, "
            f"disabled_league_safety={rule_fired_scope_counts[RULE_SCOPE_DISABLED_SAFETY]}"
        )
        if args.rule_overlap_mode == "override":
            print(
                "Overlap overrides applied: "
                f"home={overlap_override_home}, away={overlap_override_away}"
            )


if __name__ == "__main__":
    main()
