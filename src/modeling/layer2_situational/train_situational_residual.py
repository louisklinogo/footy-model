"""
Layer 2: Situational ML Residual Model (v2 - Rebuilt)

Trains a GBM to predict Poisson residuals using 16 situational features:
  Group A: Rolling xG / xA / corners (team_premium_snapshots)
  Group B: Rest delta, congestion, upcoming match tier
  Group C: Position gap, points gap, lame-duck flag
  Group D: Derby / rivalry flag
  Group E: Form streak, league home advantage, odds-model gap

Usage:
  python train_situational_residual.py --build-features
  python train_situational_residual.py --train
  python train_situational_residual.py --build-features --train
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
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
    # league_home_adv intentionally excluded from Layer 2.
    # It is a structural baseline feature that Layer 1 already captures.
    # Keeping it here acts as a crutch and steals importance from genuine situational signals.
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
        """
        SELECT p_h.fixture_id, p_h.p_model AS lambda_home, p_a.p_model AS lambda_away
        FROM predictions p_h
        JOIN predictions p_a ON p_a.fixture_id = p_h.fixture_id
            AND p_a.market_code = 'lambda_away' AND p_a.model_name = p_h.model_name
        WHERE p_h.market_code = 'lambda_home' AND p_h.model_name = 'lambda_xgb'
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

    def is_lame_duck(
        played, position, pts_gap_to_top, pts_gap_to_relegation, total_teams
    ):
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

    print(f"Loaded {len(df)} fixtures with situational features.")
    return df


def add_odds_model_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute odds-model gap: market implied home prob - Poisson home prob.
    Requires lambda_home and lambda_away from Poisson predictions.
    """
    from scipy.stats import poisson as sp_poisson

    def home_win_prob(lh, la, max_goals=8):
        p = 0.0
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                if h > a:
                    p += sp_poisson.pmf(h, lh) * sp_poisson.pmf(a, la)
        return p

    mask = df["lambda_home"].notna() & df["lambda_away"].notna()
    df["poisson_home_prob"] = np.nan
    df.loc[mask, "poisson_home_prob"] = df.loc[mask].apply(
        lambda r: home_win_prob(r["lambda_home"], r["lambda_away"]), axis=1
    )
    # Odds-model gap: we don't have market odds pre-computed here yet,
    # so we'll leave as 0 for now - it can be joined later.
    df["odds_model_gap"] = 0.0
    return df


def train_model(df: pd.DataFrame, out_dir: Path) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib

    all_features = FEATURE_COLS + ["odds_model_gap"]

    # Time-respecting split - most recent 20% as test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    MIN_GAMES = 6
    pre_filter_train = len(train_df)
    train_df = train_df[
        (train_df["home_played"] >= MIN_GAMES) & (train_df["away_played"] >= MIN_GAMES)
    ]
    print(
        f"Min-{MIN_GAMES}-games filter (Train): {pre_filter_train} -> {len(train_df)} rows"
    )

    pre_filter_test = len(test_df)
    test_df = test_df[
        (test_df["home_played"] >= MIN_GAMES) & (test_df["away_played"] >= MIN_GAMES)
    ]
    print(
        f"Min-{MIN_GAMES}-games filter (Test):  {pre_filter_test} -> {len(test_df)} rows"
    )

    # Compute residuals
    train_df["home_residual"] = train_df["home_goals"] - train_df["lambda_home"].fillna(
        train_df["home_goals"].mean()
    )
    train_df["away_residual"] = train_df["away_goals"] - train_df["lambda_away"].fillna(
        train_df["away_goals"].mean()
    )
    test_df["home_residual"] = test_df["home_goals"] - test_df["lambda_home"].fillna(
        test_df["home_goals"].mean()
    )
    test_df["away_residual"] = test_df["away_goals"] - test_df["lambda_away"].fillna(
        test_df["away_goals"].mean()
    )

    # Only use features with >20% coverage
    coverage = train_df[all_features].notna().mean()
    use_features = [f for f in all_features if coverage[f] > 0.20]
    print(f"\nFeature coverage (train set):")
    for feat in all_features:
        pct = coverage.get(feat, 0)
        flag = "[OK]" if pct > 0.20 else "[NO]"
        print(f"  {flag} {feat}: {pct:.1%}")

    print(
        f"\nUsing {len(use_features)}/{len(all_features)} features with >20% coverage."
    )

    X_train = train_df[use_features].fillna(0).values
    X_test = test_df[use_features].fillna(0).values
    y_home_train = train_df["home_residual"].values
    y_away_train = train_df["away_residual"].values
    y_home_test = test_df["home_residual"].values
    y_away_test = test_df["away_residual"].values

    gbm_params = dict(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=30,
        learning_rate=0.05,
        random_state=42,
    )

    home_model = GradientBoostingRegressor(**gbm_params)
    home_model.fit(X_train, y_home_train)

    away_model = GradientBoostingRegressor(**gbm_params)
    away_model.fit(X_train, y_away_train)

    # Metrics
    home_train_rmse = (
        mean_squared_error(y_home_train, home_model.predict(X_train)) ** 0.5
    )
    away_train_rmse = (
        mean_squared_error(y_away_train, away_model.predict(X_train)) ** 0.5
    )
    home_test_rmse = mean_squared_error(y_home_test, home_model.predict(X_test)) ** 0.5
    away_test_rmse = mean_squared_error(y_away_test, away_model.predict(X_test)) ** 0.5

    # Baselines (always predict zero residual)
    baseline_home = mean_squared_error(y_home_test, np.zeros_like(y_home_test)) ** 0.5
    baseline_away = mean_squared_error(y_away_test, np.zeros_like(y_away_test)) ** 0.5

    print(f"\n=== Results ===")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(
        f"\nHome residual RMSE  - Train: {home_train_rmse:.4f} | Test: {home_test_rmse:.4f} | Baseline: {baseline_home:.4f}"
    )
    print(
        f"Away residual RMSE  - Train: {away_train_rmse:.4f} | Test: {away_test_rmse:.4f} | Baseline: {baseline_away:.4f}"
    )

    # R-Squared calculation
    y_h_pred = home_model.predict(X_test)
    y_a_pred = away_model.predict(X_test)
    r2_h = r2_score(y_home_test, y_h_pred)
    r2_a = r2_score(y_away_test, y_a_pred)

    # Adjusted R2
    n, p = X_test.shape
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
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the GBM model")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    df = load_feature_data()
    df = add_odds_model_gap(df)

    if args.train:
        train_model(df, args.out_dir)
    else:
        out = args.out_dir / "situational_features.parquet"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"Feature data saved to {out}")
        print(f"Columns: {list(df.columns)}")
        print(df[FEATURE_COLS].describe())


if __name__ == "__main__":
    main()
