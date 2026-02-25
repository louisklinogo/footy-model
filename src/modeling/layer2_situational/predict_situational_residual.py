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
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.layer2_situational import situational_utils

MODEL_DIR = ROOT_DIR / "model_artifacts" / "situational_model"
MODEL_NAME = "situational_xgb"
MODEL_VERSION = "v2"


def is_lame_duck(played, position, pts_gap_to_top, pts_gap_to_relegation, total_teams):
    if pd.isna(played) or played < 20:
        return 0
    games_left = max(0, ((total_teams - 1) * 2) - played)
    if games_left > 8:
        return 0

    max_possible_pts = games_left * 3

    is_safe = pts_gap_to_relegation > max_possible_pts
    is_eliminated = pts_gap_to_top > max_possible_pts
    is_relegated = pts_gap_to_relegation < -max_possible_pts

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
        """
        SELECT 
            p_h.fixture_id, 
            (p_h.metadata_json->>'lambda')::double precision AS lambda_home, 
            (p_a.metadata_json->>'lambda')::double precision AS lambda_away
        FROM predictions p_h
        JOIN predictions p_a ON p_a.fixture_id = p_h.fixture_id
            AND p_a.market_code = 'lambda_away' AND p_a.model_name = p_h.model_name
        WHERE p_h.market_code = 'lambda_home' AND p_h.model_name = 'lambda_xgb'
    """,
        conn,
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
            r["home_played"], r["home_position"], r["points_gap"], 15, 20
        ),
        axis=1,
    )
    df["away_lame_duck"] = df.apply(
        lambda r: is_lame_duck(
            r["away_played"], r["away_position"], -r["points_gap"], 15, 20
        ),
        axis=1,
    )
    df["injury_impact"] = df["home_xg_lost"] - df["away_xg_lost"]
    df["home_playing_top4"] = (df["away_position"] <= 4).astype(int)
    df["away_playing_top4"] = (df["home_position"] <= 4).astype(int)
    df["derby_position_gap"] = df["position_gap"] * df["is_derby"]
    df["odds_model_gap"] = 0.0

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
    args = parser.parse_args()

    model_blob = joblib.load(MODEL_DIR / "situational_model.pkl")
    features = model_blob["features"]
    home_model = model_blob["home_model"]
    away_model = model_blob["away_model"]

    df = load_prediction_data(days=args.days, league=args.league)
    if df.empty:
        return

    X = df[features].fillna(0).values
    df["pred_home_residual"] = home_model.predict(X)
    df["pred_away_residual"] = away_model.predict(X)

    # Vectorised confidence guard: flag early-season fixtures (< 6 games played by either team)
    df["is_low_confidence"] = (df["home_played"] < 6) | (df["away_played"] < 6)

    out_rows = []
    for _, row in df.iterrows():
        h_res = float(row["pred_home_residual"])
        a_res = float(row["pred_away_residual"])
        lh = None if pd.isna(row.get("lambda_home")) else float(row["lambda_home"])
        la = None if pd.isna(row.get("lambda_away")) else float(row["lambda_away"])

        is_low = bool(row["is_low_confidence"])
        reason = "early_season_min_6_games_not_met" if is_low else None

        # Raw residuals — real value lives in metadata_json["residual"]
        res_meta_h = {"residual": h_res, "base_lambda": lh}
        res_meta_a = {"residual": a_res, "base_lambda": la}

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

        # Adjusted lambdas — value lives in metadata_json["lambda"]
        if lh is not None and la is not None:
            adj_meta_h = {"lambda": max(0.01, lh + h_res), "residual": h_res}
            adj_meta_a = {"lambda": max(0.01, la + a_res), "residual": a_res}

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


if __name__ == "__main__":
    main()
