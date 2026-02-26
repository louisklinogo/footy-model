"""
Layer 2: Situational ML Residual Model (v2 - Rebuilt)

Trains a GBM to predict Poisson residuals using 16 situational features:
  Group A: Rolling xG / xA / corners (team_premium_snapshots)
  Group B: Rest delta, congestion, upcoming match tier
  Group C: Position gap, points gap, lame-duck flag
  Group D: Derby / rivalry flag
  Group E: Form streak, league home advantage, odds-model gap

Usage:
  python train_situational_residual.py
  python train_situational_residual.py --train
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.layer2_situational.deployment_policy import (
    POLICY_FILENAME,
    build_layer2_deployment_policy,
)
from src.modeling.layer2_situational import situational_utils

warnings.filterwarnings("ignore")

# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnusedImport=false, reportUnusedVariable=false, reportUnusedParameter=false, reportCallIssue=false, reportArgumentType=false, reportUndefinedVariable=false, reportMissingTypeArgument=false, reportUnknownLambdaType=false, reportIndexIssue=false, reportGeneralTypeIssues=false

OUT_DIR = ROOT_DIR / "model_artifacts" / "situational_model"


# Source of truth: src/modeling/layer2_situational/situational_utils.py
# ---------------------------------------------------------------------------
# Feature Builder
# ---------------------------------------------------------------------------
# Feature Builder
# ---------------------------------------------------------------------------


FEATURE_COLS = [
    "home_rolling_xg",
    "home_rolling_xg_against",
    "away_rolling_xg",
    "away_rolling_xg_against",
    "xg_diff",
    "home_rolling_corners",
    "rest_delta",
    "congestion_flag",
    "home_upcoming_tier",
    "away_upcoming_tier",
    "position_gap",
    "points_gap",
    "home_lame_duck",
    "away_lame_duck",
    "is_derby",
    "home_form_streak",
    "away_form_streak",
    "home_xg_lost",
    "away_xg_lost",
    "home_key_absent",
    "away_key_absent",
    "injury_impact",
    "home_xg_over_scored",
    "home_xg_over_conceded",
    "away_xg_over_scored",
    "away_xg_over_conceded",
    "home_playing_top4",
    "away_playing_top4",
    "derby_position_gap",
    "away_rolling_corners",
    # league_home_adv intentionally excluded from Layer 2.
    # It is a structural baseline feature that Layer 1 already captures.
    # Keeping it here acts as a crutch and steals importance from genuine situational signals.
]

ODDS_FEATURE_COLS = [
    # odds_model_gap dropped: it is an alias for odds_model_gap_home (identical values).
    # Keeping both was a perfectly-collinear duplicate wasting feature budget.
    "odds_model_gap_home",
    "odds_model_gap_draw",
    "odds_model_gap_away",
    "odds_opening_gap_home",
    "odds_opening_gap_draw",
    "odds_opening_gap_away",
]


def load_feature_data() -> pd.DataFrame:
    conn = connect_db()
    print("Loading base fixtures...")
    base = pd.read_sql(
        """
        SELECT f.fixture_id, f.home_team_id, f.away_team_id, f.league_code,
               f.match_datetime_utc, fr.home_goals, fr.away_goals
        FROM fixtures f
        JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
        WHERE f.status = 'ft' AND fr.home_goals IS NOT NULL AND fr.away_goals IS NOT NULL
        ORDER BY f.match_datetime_utc ASC
    """,
        conn,
    )
    print(f"  {len(base)} base fixtures")

    print("Loading premium snapshots...")
    snap = pd.read_sql(
        """
        SELECT fixture_id, is_home,
               rolling_xg, rolling_xg_against, rolling_corners, rolling_rest_days
        FROM team_premium_snapshots
    """,
        conn,
    )
    home_snap = (
        snap[snap.is_home]
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
        snap[~snap.is_home]
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

    print("Computing point-in-time standings...")
    base = situational_utils.add_season_key(base)
    base["status"] = "ft"
    base = situational_utils.compute_point_in_time_state(base)

    print("Loading rivalries...")
    rivalries = pd.read_sql("SELECT team_id_a, team_id_b FROM team_rivalries", conn)

    print("Loading Poisson predictions...")
    preds = pd.read_sql(
        situational_utils.latest_lambda_pairs_sql(),
        conn,
        params=("lambda_xgb",),
    )

    print("Loading Sofascore 1X2 odds snapshots...")
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
        WHERE f.status = 'ft'
        """,
        conn,
    )

    league_adv = pd.read_sql(
        """
        SELECT f.league_code,
               AVG(CASE WHEN fr.home_goals > fr.away_goals THEN 1.0 ELSE 0.0 END) AS league_home_adv
        FROM fixtures f
        JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
        WHERE f.status = 'ft'
        GROUP BY f.league_code
        """,
        conn,
    )

    print("Loading missing player impact (Phase 4)...")
    impact_sql = """
    WITH base_f AS (
        SELECT fixture_id, home_team_id, away_team_id, match_datetime_utc
        FROM fixtures
        WHERE status = 'ft'
    ),
    mps AS (
        SELECT pa.fixture_id, pa.player_id, pa.team_id, b.match_datetime_utc
        FROM player_availability pa
        JOIN base_f b ON b.fixture_id = pa.fixture_id
        WHERE pa.status IN ('missing', 'doubtful')
    ),
    pi AS (
        SELECT 
            mps.fixture_id,
            mps.team_id,
            SUM(q.avg_xg) as team_xg_lost,
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
        bf.fixture_id,
        COALESCE(SUM(pi.team_xg_lost) FILTER (WHERE pi.team_id = bf.home_team_id), 0) as home_xg_lost,
        COALESCE(SUM(pi.team_xg_lost) FILTER (WHERE pi.team_id = bf.away_team_id), 0) as away_xg_lost,
        COALESCE(MAX(pi.has_key_absence) FILTER (WHERE pi.team_id = bf.home_team_id), 0) AS home_key_absent,
        COALESCE(MAX(pi.has_key_absence) FILTER (WHERE pi.team_id = bf.away_team_id), 0) AS away_key_absent
    FROM base_f bf
    LEFT JOIN pi ON pi.fixture_id = bf.fixture_id
    GROUP BY bf.fixture_id
    """
    impact_df = pd.read_sql(impact_sql, conn)

    conn.close()

    # --- Python-side assembly ---
    df = base.copy()
    df = df.merge(home_snap, on="fixture_id", how="left")
    df = df.merge(away_snap, on="fixture_id", how="left")
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
    df["injury_impact"] = df["home_xg_lost"] - df["away_xg_lost"]

    print("Computing last 5 games actual goals...")
    conn_l5 = connect_db()
    l5_goals = pd.read_sql(
        """
        WITH base AS (
            SELECT fixture_id, home_team_id, away_team_id, match_datetime_utc
            FROM fixtures
            WHERE status = 'ft'
        ),
        home_l5_goals AS (
            SELECT
                b.fixture_id,
                q.home_goals_scored_l5,
                q.home_goals_conceded_l5
            FROM base b
            LEFT JOIN LATERAL (
                SELECT 
                    COALESCE(SUM(CASE WHEN f2.home_team_id = b.home_team_id THEN fr2.home_goals ELSE fr2.away_goals END), 0) AS home_goals_scored_l5,
                    COALESCE(SUM(CASE WHEN f2.home_team_id = b.home_team_id THEN fr2.away_goals ELSE fr2.home_goals END), 0) AS home_goals_conceded_l5
                FROM (
                    SELECT fx.fixture_id, fx.home_team_id, fx.away_team_id
                    FROM fixtures fx
                    WHERE (fx.home_team_id = b.home_team_id OR fx.away_team_id = b.home_team_id)
                      AND fx.match_datetime_utc < b.match_datetime_utc
                      AND fx.status = 'ft'
                    ORDER BY fx.match_datetime_utc DESC
                    LIMIT 5
                ) f2
                JOIN fixture_results fr2 ON fr2.fixture_id = f2.fixture_id
            ) q ON TRUE
        ),
        away_l5_goals AS (
            SELECT
                b.fixture_id,
                q.away_goals_scored_l5,
                q.away_goals_conceded_l5
            FROM base b
            LEFT JOIN LATERAL (
                SELECT 
                    COALESCE(SUM(CASE WHEN f2.home_team_id = b.away_team_id THEN fr2.home_goals ELSE fr2.away_goals END), 0) AS away_goals_scored_l5,
                    COALESCE(SUM(CASE WHEN f2.home_team_id = b.away_team_id THEN fr2.away_goals ELSE fr2.home_goals END), 0) AS away_goals_conceded_l5
                FROM (
                    SELECT fx.fixture_id, fx.home_team_id, fx.away_team_id
                    FROM fixtures fx
                    WHERE (fx.home_team_id = b.away_team_id OR fx.away_team_id = b.away_team_id)
                      AND fx.match_datetime_utc < b.match_datetime_utc
                      AND fx.status = 'ft'
                    ORDER BY fx.match_datetime_utc DESC
                    LIMIT 5
                ) f2
                JOIN fixture_results fr2 ON fr2.fixture_id = f2.fixture_id
            ) q ON TRUE
        )
        SELECT h.fixture_id, h.home_goals_scored_l5, h.home_goals_conceded_l5,
               a.away_goals_scored_l5, a.away_goals_conceded_l5
        FROM home_l5_goals h
        JOIN away_l5_goals a ON h.fixture_id = a.fixture_id
    """,
        conn_l5,
    )
    conn_l5.close()

    df = df.merge(l5_goals, on="fixture_id", how="left")

    # xg_diff
    df["xg_diff"] = df["home_rolling_xg"].fillna(0) - df[
        "away_rolling_xg_against"
    ].fillna(0)

    # xG Overperformance Streaks (Actual Goals L5 - Expected Goals L5)
    # The team_premium_snapshots tracks rolling per-game xG. For 5 games, total expected xG = rolling_xg * 5.
    df["home_xg_over_scored"] = df["home_goals_scored_l5"].fillna(0) - (
        df["home_rolling_xg"].fillna(0) * 5
    )
    df["home_xg_over_conceded"] = df["home_goals_conceded_l5"].fillna(0) - (
        df["home_rolling_xg_against"].fillna(0) * 5
    )

    df["away_xg_over_scored"] = df["away_goals_scored_l5"].fillna(0) - (
        df["away_rolling_xg"].fillna(0) * 5
    )
    df["away_xg_over_conceded"] = df["away_goals_conceded_l5"].fillna(0) - (
        df["away_rolling_xg_against"].fillna(0) * 5
    )

    # rest delta
    df["rest_delta"] = df["home_rest_days"].fillna(0) - df["away_rest_days"].fillna(0)

    df["home_key_absent"] = df["home_key_absent"].fillna(0).astype(int)
    df["away_key_absent"] = df["away_key_absent"].fillna(0).astype(int)

    # ---------------------------
    # LAME DUCK CALCULATION
    # ---------------------------
    # We approximate a 38 game season unless we have better data.
    # We now have played games exactly from the point in time engine!

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

    # Derby flag
    rivalry_pairs = set(zip(rivalries["team_id_a"], rivalries["team_id_b"]))

    def check_derby(row):
        a, b = int(row["home_team_id"]), int(row["away_team_id"])
        return 1 if (a, b) in rivalry_pairs or (b, a) in rivalry_pairs else 0

    df["is_derby"] = df.apply(check_derby, axis=1)

    # ---------------------------
    # ADVANCED INTERACTION FEATURES
    # ---------------------------
    # 1. Playing a Top 4 Team (Point-in-Time)
    # Allows the residual model to learn if baseline Lambda under-predicts the
    # difficulty of playing elite teams.
    df["home_playing_top4"] = (df["away_position"] <= 4).astype(int)
    df["away_playing_top4"] = (df["home_position"] <= 4).astype(int)

    # 2. Derby Position Gap Interaction
    # Does the 'form goes out the window in a derby' cliché hold true?
    # This feature multiplies the point-in-time position gap by the derby flag.
    # If derby=0, this is 0. If derby=1, it exposes the raw position gap.
    df["derby_position_gap"] = df["position_gap"] * df["is_derby"]

    # Congestion: rolling 14-day count across ALL competitions (domestic + European)
    # Load all fixtures directly (not just those with fixture_results)
    print("Computing congestion flags...")
    conn_cong = connect_db()
    all_fixtures_raw = pd.read_sql(
        """
        SELECT fixture_id, home_team_id, away_team_id, match_datetime_utc
        FROM fixtures WHERE status = 'ft'
        ORDER BY match_datetime_utc
    """,
        conn_cong,
    )
    conn_cong.close()

    all_team_sched = situational_utils.build_team_schedule(all_fixtures_raw)
    base_team_dates = base[
        ["fixture_id", "home_team_id", "away_team_id", "match_datetime_utc"]
    ].copy()
    cong = situational_utils.compute_congestion_for_targets(
        base_team_dates=base_team_dates,
        all_team_sched=all_team_sched,
    )
    df = df.join(cong)
    df["congestion_flag"] = df["congestion_flag"].fillna(0).astype(int)

    # Upcoming match tier: check if team plays CL(3), EL(3), ECL(2) within next 4 days
    # IMPORTANT: Load European fixtures independently - they are NOT in fixture_results
    # so they don't appear in `base`. We query fixtures directly.
    print("Computing upcoming match tiers...")
    conn_euro = connect_db()
    euro_all = pd.read_sql(
        """
        SELECT fixture_id, home_team_id, away_team_id, match_datetime_utc, league_code
        FROM fixtures
        WHERE league_code IN ('CL','EL','ECL')
        ORDER BY match_datetime_utc
    """,
        conn_euro,
    )
    conn_euro.close()

    euro_team_dates = situational_utils.build_euro_team_dates(euro_all)
    upc = situational_utils.compute_upcoming_tier_for_targets(
        base_team_dates=base_team_dates,
        euro_team_dates=euro_team_dates,
    )
    df = df.join(upc)
    df["home_upcoming_tier"] = df["home_upcoming_tier"].fillna(0).astype(int)
    df["away_upcoming_tier"] = df["away_upcoming_tier"].fillna(0).astype(int)

    # Form streaks are already produced by the point-in-time engine.
    df["home_form_streak"] = df["home_form_streak"].fillna(0)
    df["away_form_streak"] = df["away_form_streak"].fillna(0)

    # League home advantage
    df = df.merge(league_adv, on="league_code", how="left")
    df["league_home_adv"] = df["league_home_adv"].fillna(0.45)

    # Poisson predictions
    df = df.merge(preds, on="fixture_id", how="left")
    df = df.merge(odds_df, on="fixture_id", how="left")

    df["odds_snapshot_time_utc"] = pd.to_datetime(
        df["odds_snapshot_time_utc"], utc=True, errors="coerce"
    )
    bad_odds = (
        df["odds_snapshot_time_utc"].notna()
        & (df["odds_snapshot_time_utc"] > df["match_datetime_utc"])
    )
    if bad_odds.any():
        raise RuntimeError(
            f"Found {int(bad_odds.sum())} post-kickoff odds snapshots; refusing to train."
        )

    print(f"Loaded {len(df)} fixtures with situational features.")
    return df


def add_odds_model_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute odds-model gap: market implied home prob - Poisson home prob.
    Requires lambda_home and lambda_away from Poisson predictions.
    """
    from scipy.stats import poisson as sp_poisson

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

    def outcome_probs(lh, la, max_goals: int = 8) -> tuple[float, float, float]:
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = sp_poisson.pmf(h, lh) * sp_poisson.pmf(a, la)
                if h > a:
                    p_home += prob
                elif h == a:
                    p_draw += prob
                else:
                    p_away += prob
        return p_home, p_draw, p_away

    mask = df["lambda_home"].notna() & df["lambda_away"].notna()
    df["poisson_home_prob"] = np.nan
    df["poisson_draw_prob"] = np.nan
    df["poisson_away_prob"] = np.nan
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

    odds_json = df["odds_json"]
    for idx, row in df.loc[mask].iterrows():
        payload = odds_json.get(idx)
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

    return df


def train_model(
    df: pd.DataFrame, out_dir: Path, feature_override: list[str] | None = None
) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    import hashlib
    import subprocess

    all_features = feature_override or (FEATURE_COLS + ODDS_FEATURE_COLS)

    df = df.sort_values("match_datetime_utc").reset_index(drop=True)

    # Time-respecting split by timestamp (80th percentile)
    split_time = df["match_datetime_utc"].dropna().quantile(0.8)
    train_df = df[df["match_datetime_utc"] <= split_time].copy()
    test_df = df[df["match_datetime_utc"] > split_time].copy()

    if not train_df.empty and not test_df.empty:
        max_train_dt = train_df["match_datetime_utc"].max()
        min_test_dt = test_df["match_datetime_utc"].min()
        if pd.notna(max_train_dt) and pd.notna(min_test_dt) and max_train_dt >= min_test_dt:
            raise RuntimeError("Chronology violation: train set is not strictly before test set.")

    MIN_GAMES = 4
    def _apply_filters(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        pre_filter = len(frame)
        frame = frame[
            (frame["home_played"] >= MIN_GAMES) & (frame["away_played"] >= MIN_GAMES)
        ]
        print(
            f"Min-{MIN_GAMES}-games filter ({label}): {pre_filter} -> {len(frame)} rows"
        )

        pre_lambda = len(frame)
        frame = frame[frame["lambda_home"].notna() & frame["lambda_away"].notna()]
        print(
            f"Lambda-availability filter ({label}): {pre_lambda} -> {len(frame)} rows"
        )
        return frame

    train_df = _apply_filters(train_df, "Train")
    test_df = _apply_filters(test_df, "Test")

    if train_df.empty:
        raise RuntimeError("No training rows remain after lambda availability filter.")
    if test_df.empty:
        raise RuntimeError("No test rows remain after lambda availability filter.")

    # Compute residuals against Layer-1 lambda predictions.
    train_df["home_residual"] = train_df["home_goals"] - train_df["lambda_home"]
    train_df["away_residual"] = train_df["away_goals"] - train_df["lambda_away"]
    test_df["home_residual"] = test_df["home_goals"] - test_df["lambda_home"]
    test_df["away_residual"] = test_df["away_goals"] - test_df["lambda_away"]

    # Odds coverage report
    latest_cov = df["odds_model_gap_home"].notna().mean()
    opening_cov = df["odds_opening_gap_home"].notna().mean()
    print(f"\nOdds coverage: latest={latest_cov:.1%} | opening={opening_cov:.1%}")

    # Temporary threshold: keep sparse-but-useful odds features until full odds backfill is complete.
    COVERAGE_THRESHOLD = 0.10
    # Only use features with >10% coverage (temporary; target is to raise this after odds backfill).
    coverage = train_df[all_features].notna().mean()
    use_features = [f for f in all_features if coverage[f] > COVERAGE_THRESHOLD]
    print(f"\nFeature coverage (train set):")
    for feat in all_features:
        pct = coverage.get(feat, 0)
        flag = "[OK]" if pct > COVERAGE_THRESHOLD else "[NO]"
        print(f"  {flag} {feat}: {pct:.1%}")

    print(
        f"\nUsing {len(use_features)}/{len(all_features)} features with >{COVERAGE_THRESHOLD:.0%} coverage."
    )

    X_train_main = train_df[use_features].fillna(0).values
    X_test_main = test_df[use_features].fillna(0).values
    y_home_train_main = train_df["home_residual"].values
    y_away_train_main = train_df["away_residual"].values
    y_home_test_main = test_df["home_residual"].values
    y_away_test_main = test_df["away_residual"].values
    w_home_train = 1.0 / np.sqrt(np.maximum(train_df["lambda_home"].values, 0.1))
    w_away_train = 1.0 / np.sqrt(np.maximum(train_df["lambda_away"].values, 0.1))

    gbm_params = dict(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=30,
        learning_rate=0.05,
        random_state=42,
    )

    # Rolling temporal CV (3 folds, expanding window).
    # IMPORTANT: CV runs only within train_df (the 80% training slice).
    # Using the full df would leak holdout period data into gate decisions.
    cv_rows: list[dict] = []
    cv_league_rows: list[dict] = []
    segments = 4
    fold_size = max(1, len(train_df) // segments)
    if fold_size > 1:
        for fold_idx in range(1, segments):
            train_end = fold_size * fold_idx
            test_end = fold_size * (fold_idx + 1) if fold_idx < segments - 1 else len(train_df)
            train_fold = train_df.iloc[:train_end].copy()
            test_fold = train_df.iloc[train_end:test_end].copy()
            if train_fold.empty or test_fold.empty:
                continue

            train_fold = _apply_filters(train_fold, f"CV{fold_idx}-Train")
            test_fold = _apply_filters(test_fold, f"CV{fold_idx}-Test")
            if train_fold.empty or test_fold.empty:
                continue

            train_fold["home_residual"] = train_fold["home_goals"] - train_fold["lambda_home"]
            train_fold["away_residual"] = train_fold["away_goals"] - train_fold["lambda_away"]
            test_fold["home_residual"] = test_fold["home_goals"] - test_fold["lambda_home"]
            test_fold["away_residual"] = test_fold["away_goals"] - test_fold["lambda_away"]

            X_train_cv = train_fold[use_features].fillna(0).values
            X_test_cv = test_fold[use_features].fillna(0).values
            y_home_train_cv = train_fold["home_residual"].values
            y_away_train_cv = train_fold["away_residual"].values
            y_home_test_cv = test_fold["home_residual"].values
            y_away_test_cv = test_fold["away_residual"].values
            w_home_train_cv = 1.0 / np.sqrt(
                np.maximum(train_fold["lambda_home"].values, 0.1)
            )
            w_away_train_cv = 1.0 / np.sqrt(
                np.maximum(train_fold["lambda_away"].values, 0.1)
            )

            home_model_cv = GradientBoostingRegressor(**gbm_params)
            home_model_cv.fit(X_train_cv, y_home_train_cv, sample_weight=w_home_train_cv)
            away_model_cv = GradientBoostingRegressor(**gbm_params)
            away_model_cv.fit(X_train_cv, y_away_train_cv, sample_weight=w_away_train_cv)

            y_home_pred_cv = home_model_cv.predict(X_test_cv)
            y_away_pred_cv = away_model_cv.predict(X_test_cv)

            home_rmse = (
                mean_squared_error(y_home_test_cv, y_home_pred_cv) ** 0.5
            )
            away_rmse = (
                mean_squared_error(y_away_test_cv, y_away_pred_cv) ** 0.5
            )
            baseline_home = mean_squared_error(
                y_home_test_cv, np.zeros_like(y_home_test_cv)
            ) ** 0.5
            baseline_away = mean_squared_error(
                y_away_test_cv, np.zeros_like(y_away_test_cv)
            ) ** 0.5

            cv_rows.append(
                {
                    "fold": fold_idx,
                    "train_start": str(train_fold["match_datetime_utc"].min()),
                    "train_end": str(train_fold["match_datetime_utc"].max()),
                    "test_start": str(test_fold["match_datetime_utc"].min()),
                    "test_end": str(test_fold["match_datetime_utc"].max()),
                    "train_n": len(train_fold),
                    "test_n": len(test_fold),
                    "home_rmse": home_rmse,
                    "away_rmse": away_rmse,
                    "baseline_home_rmse": baseline_home,
                    "baseline_away_rmse": baseline_away,
                }
            )

            fold_eval = test_fold[["league_code"]].copy()
            fold_eval["y_home_true"] = y_home_test_cv
            fold_eval["y_away_true"] = y_away_test_cv
            fold_eval["y_home_pred"] = y_home_pred_cv
            fold_eval["y_away_pred"] = y_away_pred_cv
            for league_code, grp in fold_eval.groupby("league_code"):
                y_h_true = grp["y_home_true"].to_numpy(dtype=float)
                y_a_true = grp["y_away_true"].to_numpy(dtype=float)
                y_h_pred = grp["y_home_pred"].to_numpy(dtype=float)
                y_a_pred = grp["y_away_pred"].to_numpy(dtype=float)
                base_h = mean_squared_error(y_h_true, np.zeros_like(y_h_true)) ** 0.5
                base_a = mean_squared_error(y_a_true, np.zeros_like(y_a_true)) ** 0.5
                rmse_h = mean_squared_error(y_h_true, y_h_pred) ** 0.5
                rmse_a = mean_squared_error(y_a_true, y_a_pred) ** 0.5
                lift_h = ((base_h - rmse_h) / base_h) if base_h > 0 else 0.0
                lift_a = ((base_a - rmse_a) / base_a) if base_a > 0 else 0.0
                cv_league_rows.append(
                    {
                        "fold": fold_idx,
                        "league_code": str(league_code),
                        "home_lift": float(lift_h),
                        "away_lift": float(lift_a),
                        "mean_lift": float(np.mean([lift_h, lift_a])),
                        "test_n": int(len(grp)),
                    }
                )

    home_model = GradientBoostingRegressor(**gbm_params)
    home_model.fit(X_train_main, y_home_train_main, sample_weight=w_home_train)

    away_model = GradientBoostingRegressor(**gbm_params)
    away_model.fit(X_train_main, y_away_train_main, sample_weight=w_away_train)

    # Metrics
    home_train_rmse = (
        mean_squared_error(y_home_train_main, home_model.predict(X_train_main)) ** 0.5
    )
    away_train_rmse = (
        mean_squared_error(y_away_train_main, away_model.predict(X_train_main)) ** 0.5
    )
    home_test_rmse = mean_squared_error(
        y_home_test_main, home_model.predict(X_test_main)
    ) ** 0.5
    away_test_rmse = mean_squared_error(
        y_away_test_main, away_model.predict(X_test_main)
    ) ** 0.5

    # Baselines (always predict zero residual)
    baseline_home = mean_squared_error(
        y_home_test_main, np.zeros_like(y_home_test_main)
    ) ** 0.5
    baseline_away = mean_squared_error(
        y_away_test_main, np.zeros_like(y_away_test_main)
    ) ** 0.5

    print(f"\n=== Results ===")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(
        f"\nHome residual RMSE  - Train: {home_train_rmse:.4f} | Test: {home_test_rmse:.4f} | Baseline: {baseline_home:.4f}"
    )
    print(
        f"Away residual RMSE  - Train: {away_train_rmse:.4f} | Test: {away_test_rmse:.4f} | Baseline: {baseline_away:.4f}"
    )

    # R-Squared calculation
    y_h_pred = home_model.predict(X_test_main)
    y_a_pred = away_model.predict(X_test_main)
    r2_h = r2_score(y_home_test_main, y_h_pred)
    r2_a = r2_score(y_away_test_main, y_a_pred)

    # Adjusted R2
    n, p = X_test_main.shape
    adj_r2_h = 1 - (1 - r2_h) * (n - 1) / (max(1, n - p - 1))
    adj_r2_a = 1 - (1 - r2_a) * (n - 1) / (max(1, n - p - 1))

    print(f"\nHome R-Squared: {r2_h:.4f} (Adjusted: {adj_r2_h:.4f})")
    print(f"Away R-Squared: {r2_a:.4f} (Adjusted: {adj_r2_a:.4f})")

    print(
        f"\nHome lift vs baseline: {(baseline_home - home_test_rmse) / baseline_home:.1%}"
    )
    print(
        f"Away lift vs baseline: {(baseline_away - away_test_rmse) / baseline_away:.1%}"
    )

    # Feature importance
    print(f"\n--- Feature Importance (Home model) ---")
    importances = sorted(
        zip(use_features, home_model.feature_importances_), key=lambda x: -x[1]
    )
    for feat, imp in importances:
        print(f"  {feat}: {imp:.4f}")

    result = {
        "features": use_features,
        "home_model": home_model,
        "away_model": away_model,
        "train_n": len(train_df),
        "test_n": len(test_df),
        "home_test_rmse": home_test_rmse,
        "away_test_rmse": away_test_rmse,
        "home_baseline_rmse": baseline_home,
        "away_baseline_rmse": baseline_away,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, out_dir / "situational_model.pkl")
    print(f"\nModel saved to {out_dir / 'situational_model.pkl'}")

    holdout_eval = test_df[["league_code"]].copy()
    holdout_eval["y_home_true"] = y_home_test_main
    holdout_eval["y_away_true"] = y_away_test_main
    holdout_eval["y_home_pred"] = y_h_pred
    holdout_eval["y_away_pred"] = y_a_pred
    # Short-term rollout gate while Sofascore odds backfill is still maturing.
    # Keep CV quality strict, but relax recent holdout sample from 50 -> 40 so
    # strong leagues can start receiving bounded Layer-2 adjustments.
    deployment_policy = build_layer2_deployment_policy(
        holdout_eval,
        cv_league_rows,
        min_recent_n=40,
        required_positive_folds=2,
        required_cv_folds=3,
        model_version="v2",
    )
    policy_path = out_dir / POLICY_FILENAME
    policy_path.write_text(json.dumps(deployment_policy, indent=2), encoding="utf-8")
    enabled_leagues = [
        league
        for league, payload in deployment_policy.get("leagues", {}).items()
        if payload.get("enabled")
    ]
    print(
        f"Layer2 policy saved to {policy_path} (enabled leagues: {len(enabled_leagues)})"
    )

    feature_hash = hashlib.sha256(",".join(use_features).encode("utf-8")).hexdigest()
    git_commit = None
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:
        git_commit = None

    cv_home = [row["home_rmse"] for row in cv_rows]
    cv_away = [row["away_rmse"] for row in cv_rows]
    cv_home_mean = float(np.mean(cv_home)) if cv_home else None
    cv_home_std = float(np.std(cv_home)) if cv_home else None
    cv_away_mean = float(np.mean(cv_away)) if cv_away else None
    cv_away_std = float(np.std(cv_away)) if cv_away else None

    if cv_rows:
        print(
            f"\nTemporal CV (3 folds): home_rmse={cv_home_mean:.4f}±{cv_home_std:.4f}, "
            f"away_rmse={cv_away_mean:.4f}±{cv_away_std:.4f}"
        )
    else:
        print("\nTemporal CV skipped: insufficient data for folds.")

    sidecar = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_range": {
            "start": str(train_df["match_datetime_utc"].min()),
            "end": str(train_df["match_datetime_utc"].max()),
        },
        "test_range": {
            "start": str(test_df["match_datetime_utc"].min()),
            "end": str(test_df["match_datetime_utc"].max()),
        },
        "train_n": len(train_df),
        "test_n": len(test_df),
        "features": use_features,
        "feature_hash": feature_hash,
        "hyperparams": gbm_params,
        "git_commit": git_commit,
        "cv_folds": cv_rows,
        "cv_summary": {
            "home_rmse_mean": cv_home_mean,
            "home_rmse_std": cv_home_std,
            "away_rmse_mean": cv_away_mean,
            "away_rmse_std": cv_away_std,
        },
        "deployment_policy_file": str(policy_path),
        "enabled_leagues": enabled_leagues,
    }
    sidecar_path = out_dir / "situational_model.meta.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Metadata saved to {sidecar_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the GBM model")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--drop-features",
        type=str,
        default="",
        help="Comma-separated feature list to drop before coverage filtering.",
    )
    args = parser.parse_args()

    df = load_feature_data()
    df = add_odds_model_gap(df)

    all_features = FEATURE_COLS + ODDS_FEATURE_COLS
    drop_features = [f.strip() for f in args.drop_features.split(",") if f.strip()]
    unknown = sorted(set(drop_features) - set(all_features))
    if unknown:
        raise ValueError(f"Unknown features in --drop-features: {unknown}")
    selected_features = [f for f in all_features if f not in set(drop_features)]
    if not selected_features:
        raise ValueError("No features remain after applying --drop-features.")
    if drop_features:
        print(f"Dropping {len(drop_features)} features: {drop_features}")
        print(f"Candidate feature pool size: {len(selected_features)}")

    if args.train:
        train_model(df, args.out_dir, feature_override=selected_features)
    else:
        out = args.out_dir / "situational_features.parquet"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"Feature data saved to {out}")
        print(f"Columns: {list(df.columns)}")
        print(df[selected_features].describe())


if __name__ == "__main__":
    main()
