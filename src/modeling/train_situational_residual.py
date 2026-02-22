"""
Layer 2: Situational ML Residual Model

Computes situational features (rest days, fixture congestion, motivation proxies)
and trains a model to predict Poisson model residuals.

This is Layer 2 of the 3-Layer syndicate architecture:
- Layer 1: Poisson/Dixon-Coles baseline (train_lambda.py)
- Layer 2: Situational ML residual (this script)
- Layer 3: AI Research overlay (apply_research_overlay.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.pricing.poisson import PoissonPricer

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false


def bootstrap_standings_table(conn) -> None:
    """Create team_league_standings table if it doesn't exist."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS team_league_standings (
        id SERIAL PRIMARY KEY,
        team_id INTEGER NOT NULL REFERENCES teams(team_id),
        league_code VARCHAR(10) NOT NULL,
        season VARCHAR(10) NOT NULL,
        matches_played INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        draws INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        goals_for INTEGER DEFAULT 0,
        goals_against INTEGER DEFAULT 0,
        points INTEGER DEFAULT 0,
        computed_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(team_id, league_code, season)
    );
    
    CREATE INDEX IF NOT EXISTS idx_standings_team_league_season 
        ON team_league_standings(team_id, league_code, season);
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
        conn.commit()
    print("Created team_league_standings table")


def compute_standings_from_results(conn, season: str | None = None) -> int:
    """
    Compute league standings from fixture_results.
    
    Returns number of team-season rows inserted/updated.
    """
    query = """
    SELECT 
        f.fixture_id,
        f.league_code,
        f.home_team_id,
        f.away_team_id,
        f.match_datetime_utc,
        fr.home_goals,
        fr.away_goals,
        EXTRACT(YEAR FROM f.match_datetime_utc)::text AS season
    FROM fixtures f
    JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
    WHERE f.status = 'ft'
      AND fr.home_goals IS NOT NULL
      AND fr.away_goals IS NOT NULL
    ORDER BY f.match_datetime_utc
    """
    
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    
    if not rows:
        print("No completed fixtures found")
        return 0
    
    standings: dict[tuple, dict] = {}
    
    for row in rows:
        fixture_id, league_code, home_id, away_id, match_dt, home_goals, away_goals, match_season = row
        
        if season and match_season != season:
            continue
        
        home_key = (home_id, league_code, match_season)
        if home_key not in standings:
            standings[home_key] = {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}
        
        standings[home_key]["matches"] += 1
        standings[home_key]["gf"] += home_goals
        standings[home_key]["ga"] += away_goals
        
        if home_goals > away_goals:
            standings[home_key]["wins"] += 1
        elif home_goals < away_goals:
            standings[home_key]["losses"] += 1
        else:
            standings[home_key]["draws"] += 1
        
        away_key = (away_id, league_code, match_season)
        if away_key not in standings:
            standings[away_key] = {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}
        
        standings[away_key]["matches"] += 1
        standings[away_key]["gf"] += away_goals
        standings[away_key]["ga"] += home_goals
        
        if away_goals > home_goals:
            standings[away_key]["wins"] += 1
        elif away_goals < home_goals:
            standings[away_key]["losses"] += 1
        else:
            standings[away_key]["draws"] += 1
    
    upsert_sql = """
    INSERT INTO team_league_standings 
        (team_id, league_code, season, matches_played, wins, draws, losses, goals_for, goals_against, points, computed_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
    ON CONFLICT (team_id, league_code, season) DO UPDATE SET
        matches_played = EXCLUDED.matches_played,
        wins = EXCLUDED.wins,
        draws = EXCLUDED.draws,
        losses = EXCLUDED.losses,
        goals_for = EXCLUDED.goals_for,
        goals_against = EXCLUDED.goals_against,
        points = EXCLUDED.points,
        computed_at = NOW()
    """
    
    rows_affected = 0
    with conn.cursor() as cur:
        for (team_id, league_code, seas), stats in standings.items():
            points = stats["wins"] * 3 + stats["draws"]
            cur.execute(upsert_sql, (
                team_id, league_code, seas,
                stats["matches"], stats["wins"], stats["draws"], stats["losses"],
                stats["gf"], stats["ga"], points
            ))
            rows_affected += 1
        conn.commit()
    
    print(f"Updated {rows_affected} team-season standings")
    return rows_affected


def ensure_standings_exist(conn) -> bool:
    """Check if standings table has data; bootstrap if needed."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM team_league_standings")
            count = cur.fetchone()[0]
        
        if count > 0:
            print(f"Standings table has {count} rows")
            return True
    except Exception:
        pass
    
    print("Standings table empty or missing, computing from fixture results...")
    return compute_standings_from_results(conn) > 0


def compute_rest_delta(fixture_id: int, conn) -> int | None:
    """
    Compute rest days delta (home rest - away rest).
    
    Returns:
        Positive: Home team has MORE rest
        Negative: Away team has MORE rest
    """
    query = """
    WITH fixture_dates AS (
        SELECT 
            f.fixture_id,
            f.home_team_id,
            f.away_team_id,
            f.match_datetime_utc,
            LAG(f.match_datetime_utc) OVER (
                PARTITION BY f.home_team_id 
                ORDER BY f.match_datetime_utc
            ) AS home_prev_match,
            LAG(f.match_datetime_utc) OVER (
                PARTITION BY f.away_team_id 
                ORDER BY f.match_datetime_utc
            ) AS away_prev_match
        FROM fixtures f
        WHERE f.fixture_id = %s
    )
    SELECT 
        EXTRACT(DAY FROM (match_datetime_utc - home_prev_match)) AS home_rest_days,
        EXTRACT(DAY FROM (match_datetime_utc - away_prev_match)) AS away_rest_days
    FROM fixture_dates
    """
    
    with conn.cursor() as cur:
        cur.execute(query, (fixture_id,))
        row = cur.fetchone()
        if row and row[0] is not None and row[1] is not None:
            return int(row[0]) - int(row[1])
    return None


def compute_congestion_flag(fixture_id: int, conn, window_days: int = 14) -> int:
    """
    Check if either team has played 3+ matches in the last N days.
    
    Returns:
        1 if congested, 0 otherwise
    """
    query = """
    WITH current_fixture AS (
        SELECT 
            fixture_id,
            home_team_id,
            away_team_id,
            match_datetime_utc
        FROM fixtures
        WHERE fixture_id = %s
    ),
    recent_matches AS (
        SELECT 
            cf.fixture_id,
            COUNT(CASE WHEN f.home_team_id = cf.home_team_id OR f.away_team_id = cf.home_team_id THEN 1 END) AS home_recent,
            COUNT(CASE WHEN f.home_team_id = cf.away_team_id OR f.away_team_id = cf.away_team_id THEN 1 END) AS away_recent
        FROM current_fixture cf
        JOIN fixtures f
          ON (f.home_team_id = cf.home_team_id OR f.away_team_id = cf.home_team_id
              OR f.home_team_id = cf.away_team_id OR f.away_team_id = cf.away_team_id)
         AND f.match_datetime_utc < cf.match_datetime_utc
         AND f.match_datetime_utc >= cf.match_datetime_utc - (%s || ' days')::interval
         AND f.status = 'ft'
        GROUP BY cf.fixture_id
    )
    SELECT 
        CASE WHEN home_recent >= 3 OR away_recent >= 3 THEN 1 ELSE 0 END AS congested
    FROM recent_matches
    """
    
    with conn.cursor() as cur:
        cur.execute(query, (fixture_id, window_days))
        row = cur.fetchone()
        return int(row[0]) if row else 0


def compute_motivation_score(fixture_id: int, conn) -> float:
    """
    Compute motivation proxy based on league position proximity to key thresholds.
    
    Factors:
    - Title race (top 3 within 6 points)
    - Champions League spots (top 4-6 within 3 points)
    - Relegation battle (bottom 3 within 3 points)
    
    Returns score from -1 (low motivation) to +1 (high motivation)
    """
    query = """
    WITH current_fixture AS (
        SELECT 
            fixture_id,
            home_team_id,
            away_team_id,
            league_code,
            match_datetime_utc
        FROM fixtures
        WHERE fixture_id = %s
    ),
    league_standings AS (
        SELECT 
            team_id,
            points,
            rank() OVER (ORDER BY points DESC) AS position,
            COUNT(*) OVER () AS total_teams
        FROM team_league_standings tls
        JOIN current_fixture cf ON tls.league_code = cf.league_code
        WHERE tls.season = EXTRACT(YEAR FROM cf.match_datetime_utc)::text
    )
    SELECT 
        home.position AS home_pos,
        home.points AS home_pts,
        away.position AS away_pos,
        away.points AS away_pts,
        (SELECT total_teams FROM league_standings LIMIT 1) AS total_teams
    FROM current_fixture cf
    JOIN league_standings home ON home.team_id = cf.home_team_id
    JOIN league_standings away ON away.team_id = cf.away_team_id
    """
    
    with conn.cursor() as cur:
        cur.execute(query, (fixture_id,))
        row = cur.fetchone()
        if not row:
            return 0.0
        
        home_pos, home_pts, away_pos, away_pts, total_teams = row
        
        if not all(v is not None for v in [home_pos, home_pts, away_pos, away_pts, total_teams]):
            return 0.0
        
        home_pos, away_pos = int(home_pos), int(away_pos)
        total_teams = int(total_teams)
        
        home_motivation = 0.0
        away_motivation = 0.0
        
        if home_pos <= 3:
            home_motivation += 0.5
        if home_pos <= 6:
            home_motivation += 0.3
        if home_pos >= total_teams - 2:
            home_motivation += 0.4
        
        if away_pos <= 3:
            away_motivation += 0.5
        if away_pos <= 6:
            away_motivation += 0.3
        if away_pos >= total_teams - 2:
            away_motivation += 0.4
        
        return home_motivation - away_motivation


def build_situational_features(fixture_id: int, conn) -> dict:
    """Build all situational features for a fixture."""
    return {
        "fixture_id": fixture_id,
        "rest_delta": compute_rest_delta(fixture_id, conn),
        "congestion_flag": compute_congestion_flag(fixture_id, conn),
        "motivation_score": compute_motivation_score(fixture_id, conn),
    }


def fetch_scored_fixtures_with_poisson(
    model_name: str = "lambda_xgb",
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Fetch fixtures with:
    - Actual goals (from fixture_results)
    - Poisson predictions (from predictions or computed)
    - Situational features
    """
    conn = connect_db()
    
    query = """
    SELECT 
        f.fixture_id,
        f.home_team_id,
        f.away_team_id,
        f.match_datetime_utc,
        fr.home_goals,
        fr.away_goals,
        p_h.p_model AS lambda_home,
        p_a.p_model AS lambda_away
    FROM fixtures f
    JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
    LEFT JOIN predictions p_h 
        ON p_h.fixture_id = f.fixture_id 
        AND p_h.market_code = 'lambda_home'
        AND p_h.model_name = %s
    LEFT JOIN predictions p_a 
        ON p_a.fixture_id = f.fixture_id 
        AND p_a.market_code = 'lambda_away'
        AND p_a.model_name = %s
    WHERE f.status = 'ft'
    ORDER BY f.match_datetime_utc DESC
    """
    
    params = [model_name, model_name]
    
    if limit:
        query += " LIMIT %s"
        params.append(limit)
    
    df = pd.read_sql(query, conn, params=tuple(params))
    conn.close()
    
    return df


def compute_poisson_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute residuals: actual_goals - predicted_lambda
    """
    df = df.copy()
    df["home_residual"] = df["home_goals"] - df["lambda_home"]
    df["away_residual"] = df["away_goals"] - df["lambda_away"]
    return df


def train_residual_model(df: pd.DataFrame) -> dict:
    """
    Train a simple model to predict residuals from situational features.
    
    Uses a linear model for interpretability.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    features = ["rest_delta", "congestion_flag", "motivation_score"]
    
    valid_mask = df[features].notna().all(axis=1)
    train_df = df[valid_mask].copy()
    
    if len(train_df) < 100:
        print(f"Warning: Only {len(train_df)} samples for training")
        return {}
    
    X = train_df[features].values
    y_home = train_df["home_residual"].values
    y_away = train_df["away_residual"].values
    
    model_home = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    model_home.fit(X, y_home)
    
    model_away = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    model_away.fit(X, y_away)
    
    return {
        "features": features,
        "home_residual_model": model_home,
        "away_residual_model": model_away,
        "train_n": len(train_df),
        "home_r2": model_home.score(X, y_home),
        "away_r2": model_away.score(X, y_away),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 2 Situational ML Residual Model")
    parser.add_argument("--limit", type=int, default=None, help="Max fixtures to process")
    parser.add_argument("--build-features", action="store_true", help="Build situational features")
    parser.add_argument("--train", action="store_true", help="Train residual model")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap standings table from fixture results")
    parser.add_argument("--output-dir", type=Path, default=Path("models/v2_situational"), help="Output directory")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conn = connect_db()
    
    if args.bootstrap:
        print("Bootstrapping standings table...")
        bootstrap_standings_table(conn)
        compute_standings_from_results(conn)
        conn.close()
        return
    
    ensure_standings_exist(conn)
    conn.close()
    
    if args.build_features:
        print("Building situational features...")
        conn = connect_db()
        
        query = "SELECT fixture_id FROM fixtures WHERE status = 'ft' ORDER BY match_datetime_utc DESC"
        if args.limit:
            query += f" LIMIT {args.limit}"
        
        with conn.cursor() as cur:
            cur.execute(query)
            fixture_ids = [row[0] for row in cur.fetchall()]
        
        print(f"Processing {len(fixture_ids)} fixtures...")
        
        features_list = []
        for i, fid in enumerate(fixture_ids):
            if i % 100 == 0:
                print(f"  {i}/{len(fixture_ids)}")
            feat = build_situational_features(fid, conn)
            features_list.append(feat)
        
        conn.close()
        
        df = pd.DataFrame(features_list)
        out_path = args.output_dir / "situational_features.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} feature rows to {out_path}")
    
    if args.train:
        print("Training residual model...")
        
        df = fetch_scored_fixtures_with_poisson(limit=args.limit)
        print(f"Loaded {len(df)} fixtures with Poisson predictions")
        
        df = compute_poisson_residuals(df)
        
        features_df = pd.read_parquet(args.output_dir / "situational_features.parquet")
        df = df.merge(features_df, on="fixture_id", how="inner")
        
        results = train_residual_model(df)
        
        if results:
            import joblib
            
            model_path = args.output_dir / "residual_model.pkl"
            joblib.dump(results, model_path)
            
            print(f"\n=== Training Results ===")
            print(f"Training samples: {results['train_n']}")
            print(f"Home residual R²: {results['home_r2']:.4f}")
            print(f"Away residual R²: {results['away_r2']:.4f}")
            print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
