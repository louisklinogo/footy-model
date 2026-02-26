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
               rolling_xg, rolling_xg_against, rolling_corners, rolling_rest_days
        FROM team_premium_snapshots
    """,
        conn,
    )
    home_snap = (
        snap_df[snap_df.is_home]
        .rename(
            columns={
                "rolling_xg": "home_rolling_xg",
                "rolling_xg_against": "home_rolling_xg_against",
                "rolling_corners": "home_rolling_corners",
                "rolling_rest_days": "home_rest_days",
            }
        )
        .drop(columns=["is_home"])
    )
    away_snap = (
        snap_df[~snap_df.is_home]
        .rename(
            columns={
                "rolling_xg": "away_rolling_xg",
                "rolling_xg_against": "away_rolling_xg_against",
                "rolling_corners": "away_rolling_corners",
                "rolling_rest_days": "away_rest_days",
            }
        )
        .drop(columns=["is_home"])
    )

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
    args = parser.parse_args()

    model_blob = joblib.load(MODEL_DIR / "situational_model.pkl")
    features = model_blob["features"]
    home_model = model_blob["home_model"]
    away_model = model_blob["away_model"]
    policy = load_layer2_deployment_policy(MODEL_DIR)
    rule_config_path = args.rule_layer_config or (MODEL_DIR / "rule_layer_config.json")
    rule_config = (
        load_rule_layer_config(rule_config_path) if args.enable_rule_layer else None
    )

    df = load_prediction_data(days=args.days, league=args.league)
    if df.empty:
        return

    X = df[features].fillna(0).values
    df["pred_home_residual"] = home_model.predict(X)
    df["pred_away_residual"] = away_model.predict(X)

    # Vectorised confidence guard aligned with Layer 2 train filter (min 4 games).
    df["is_low_confidence"] = (df["home_played"] < 4) | (df["away_played"] < 4)

    out_rows = []
    rule_fired_home = 0
    rule_fired_away = 0
    rule_fired_fixture_ids: set[int] = set()
    for _, row in df.iterrows():
        h_res_raw = float(row["pred_home_residual"])
        a_res_raw = float(row["pred_away_residual"])
        lh = None if pd.isna(row.get("lambda_home")) else float(row["lambda_home"])
        la = None if pd.isna(row.get("lambda_away")) else float(row["lambda_away"])

        league_policy = resolve_league_policy(policy, row.get("league_code"))
        layer2_enabled = bool(league_policy.get("enabled", False))
        alpha = float(league_policy.get("alpha", 0.0))
        gate_reason = str(league_policy.get("reason", "default_policy"))
        effective_alpha = alpha if layer2_enabled else 0.0

        # -----------------------------------------------------------
        # Market-informed alpha scaling
        # When the market's implied direction agrees with our residual
        # direction → trust the model more (full alpha).
        # When they disagree → the market knows something we don't;
        # shrink to 40% of alpha as a conservative fallback.
        # Falls back to base alpha when odds are unavailable.
        # -----------------------------------------------------------
        odds_gap_home = row.get("odds_model_gap_home")
        odds_gap_away = row.get("odds_model_gap_away")

        if effective_alpha > 0 and pd.notna(odds_gap_home) and float(odds_gap_home) != 0:
            home_agree = (h_res_raw >= 0) == (float(odds_gap_home) >= 0)
            alpha_home = effective_alpha if home_agree else effective_alpha * 0.4
        else:
            alpha_home = effective_alpha

        if effective_alpha > 0 and pd.notna(odds_gap_away) and float(odds_gap_away) != 0:
            away_agree = (a_res_raw >= 0) == (float(odds_gap_away) >= 0)
            alpha_away = effective_alpha if away_agree else effective_alpha * 0.4
        else:
            alpha_away = effective_alpha

        h_res_applied = alpha_home * h_res_raw
        a_res_applied = alpha_away * a_res_raw

        is_low = bool(row["is_low_confidence"])
        reason = "early_season_min_4_games_not_met" if is_low else None

        # Raw residuals live in metadata_json["residual"] for backward compatibility.
        res_meta_h = {
            "residual": h_res_raw,
            "residual_raw": h_res_raw,
            "residual_applied": h_res_applied,
            "base_lambda": lh,
            "layer2_enabled": layer2_enabled,
            "layer2_alpha": alpha_home,
            "layer2_alpha_policy": effective_alpha,
            "layer2_gate_reason": gate_reason,
            "market_confirmed": bool(pd.notna(odds_gap_home) and float(odds_gap_home or 0) != 0),
        }
        res_meta_a = {
            "residual": a_res_raw,
            "residual_raw": a_res_raw,
            "residual_applied": a_res_applied,
            "base_lambda": la,
            "layer2_enabled": layer2_enabled,
            "layer2_alpha": alpha_away,
            "layer2_alpha_policy": effective_alpha,
            "layer2_gate_reason": gate_reason,
            "market_confirmed": bool(pd.notna(odds_gap_away) and float(odds_gap_away or 0) != 0),
        }

        if is_low:
            res_meta_h.update(
                {"is_low_confidence": True, "low_confidence_reason": reason}
            )
            res_meta_a.update(
                {"is_low_confidence": True, "low_confidence_reason": reason}
            )

        out_rows.append(
            {
                "fixture_id": int(row["fixture_id"]),
                "market_code": "home_residual",
                "meta": res_meta_h,
            }
        )
        out_rows.append(
            {
                "fixture_id": int(row["fixture_id"]),
                "market_code": "away_residual",
                "meta": res_meta_a,
            }
        )

        # Adjusted lambdas use the policy-gated residual.
        if lh is not None and la is not None:
            lambda_home_layer2 = max(0.01, lh + h_res_applied)
            lambda_away_layer2 = max(0.01, la + a_res_applied)

            home_rule = {
                "applied": False,
                "rules": [],
                "pct_raw": 0.0,
                "pct_capped": 0.0,
                "lambda_before": lambda_home_layer2,
                "lambda_after": lambda_home_layer2,
                "cap_down": 0.0,
                "cap_up": 0.0,
            }
            away_rule = {
                "applied": False,
                "rules": [],
                "pct_raw": 0.0,
                "pct_capped": 0.0,
                "lambda_before": lambda_away_layer2,
                "lambda_after": lambda_away_layer2,
                "cap_down": 0.0,
                "cap_up": 0.0,
            }

            if (
                args.enable_rule_layer
                and rule_config is not None
                and layer2_enabled
                and not is_low
            ):
                home_rule = apply_rule_adjustment(
                    lambda_home_layer2, row, side="home", config=rule_config
                )
                away_rule = apply_rule_adjustment(
                    lambda_away_layer2, row, side="away", config=rule_config
                )
                if bool(home_rule["applied"]):
                    rule_fired_home += 1
                    rule_fired_fixture_ids.add(int(row["fixture_id"]))
                if bool(away_rule["applied"]):
                    rule_fired_away += 1
                    rule_fired_fixture_ids.add(int(row["fixture_id"]))

            adj_meta_h = {
                "lambda": float(home_rule["lambda_after"]),
                "residual": h_res_raw,
                "residual_raw": h_res_raw,
                "residual_applied": h_res_applied,
                "base_lambda": lh,
                "layer2_enabled": layer2_enabled,
                "layer2_alpha": alpha_home,
                "layer2_alpha_policy": effective_alpha,
                "layer2_gate_reason": gate_reason,
                "market_confirmed": bool(pd.notna(odds_gap_home) and float(odds_gap_home or 0) != 0),
                "lambda_after_layer2": lambda_home_layer2,
                "rule_layer_enabled": bool(args.enable_rule_layer),
                "rule_layer_applied": bool(home_rule["applied"]),
                "rule_layer_rules": list(home_rule["rules"]),
                "rule_layer_pct_raw": float(home_rule["pct_raw"]),
                "rule_layer_pct_capped": float(home_rule["pct_capped"]),
                "rule_layer_cap_down": float(home_rule["cap_down"]),
                "rule_layer_cap_up": float(home_rule["cap_up"]),
                "rule_layer_config_path": str(rule_config_path) if args.enable_rule_layer else None,
            }
            adj_meta_a = {
                "lambda": float(away_rule["lambda_after"]),
                "residual": a_res_raw,
                "residual_raw": a_res_raw,
                "residual_applied": a_res_applied,
                "base_lambda": la,
                "layer2_enabled": layer2_enabled,
                "layer2_alpha": alpha_away,
                "layer2_alpha_policy": effective_alpha,
                "layer2_gate_reason": gate_reason,
                "market_confirmed": bool(pd.notna(odds_gap_away) and float(odds_gap_away or 0) != 0),
                "lambda_after_layer2": lambda_away_layer2,
                "rule_layer_enabled": bool(args.enable_rule_layer),
                "rule_layer_applied": bool(away_rule["applied"]),
                "rule_layer_rules": list(away_rule["rules"]),
                "rule_layer_pct_raw": float(away_rule["pct_raw"]),
                "rule_layer_pct_capped": float(away_rule["pct_capped"]),
                "rule_layer_cap_down": float(away_rule["cap_down"]),
                "rule_layer_cap_up": float(away_rule["cap_up"]),
                "rule_layer_config_path": str(rule_config_path) if args.enable_rule_layer else None,
            }

            if is_low:
                adj_meta_h.update(
                    {"is_low_confidence": True, "low_confidence_reason": reason}
                )
                adj_meta_a.update(
                    {"is_low_confidence": True, "low_confidence_reason": reason}
                )

            out_rows.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "market_code": "adj_lambda_home",
                    "meta": adj_meta_h,
                }
            )
            out_rows.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "market_code": "adj_lambda_away",
                    "meta": adj_meta_a,
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


if __name__ == "__main__":
    main()
