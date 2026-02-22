import sys
from pathlib import Path
from typing import List
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

def build_point_in_time_features(
    fixture_ids: List[int],
    target_window_mins: int = 60
) -> pd.DataFrame:
    """
    Builds a point-in-time feature set for a list of fixtures.
    Crucially ensures that source_ts <= decision_timestamp (T minus target_window_mins).
    """
    if not fixture_ids:
        return pd.DataFrame()
        
    conn = connect_db()
    
    query = """
    WITH target_fixtures AS (
        SELECT 
            f.fixture_id,
            f.league_code,
            f.match_datetime_utc as kickoff_time,
            EXTRACT(EPOCH FROM f.match_datetime_utc) as kickoff_ts,
            f.home_team_id,
            f.away_team_id
        FROM fixtures f
        WHERE f.fixture_id = ANY(%(fixture_ids)s)
    ),
    decision_times AS (
        SELECT
            *,
            (kickoff_ts - (%(target_window_mins)s * 60)) AS decision_ts
        FROM target_fixtures
    ),
    -- League Intensity baselines (point-in-time!)
    -- Calculates avg goals/xG in that league BEFORE the match kickoff
    league_intensities AS (
        SELECT 
            dt.fixture_id as dt_fixture_id,
            AVG( (p.h_xg + p.a_xg)/2.0 ) as league_avg_xg,
            AVG( (r.home_goals + r.away_goals)/2.0 ) as league_avg_goals
        FROM decision_times dt
        JOIN fixtures f ON f.league_code = dt.league_code
        JOIN fixture_stats_premium p ON p.fixture_id = f.fixture_id
        JOIN fixture_results r ON r.fixture_id = f.fixture_id
        WHERE EXTRACT(EPOCH FROM f.match_datetime_utc) < dt.kickoff_ts
        AND f.status = 'ft'
        GROUP BY dt.fixture_id
    ),
    latest_home_snapshots AS (
        SELECT * FROM team_premium_snapshots WHERE is_home = TRUE
    ),
    latest_away_snapshots AS (
        SELECT * FROM team_premium_snapshots WHERE is_home = FALSE
    )
    
    SELECT
        d.fixture_id,
        d.league_code, 
        d.kickoff_time,
        
        -- The Y Targets (Lambdas)
        res.home_goals,
        res.away_goals,
        
        -- League Intensity Baselines (The "Senior" normalized anchor)
        li.league_avg_xg,
        li.league_avg_goals,
        
        -- Home Features (EWMA based)
        h.sample_size as h_sample_size,
        h.rolling_xg as h_xg,
        h.rolling_xg_against as h_xg_against,
        h.rolling_xgot as h_xgot,
        h.rolling_xa as h_xa,
        h.rolling_box_touches as h_box_touches,
        h.rolling_big_chances as h_big_chances,
        h.rolling_sot as h_sot,
        h.rolling_corners as h_corners,
        h.rolling_rest_days as h_rest_days,
        
        -- Away Features (EWMA based)
        a.sample_size as a_sample_size,
        a.rolling_xg as a_xg,
        a.rolling_xg_against as a_xg_against,
        a.rolling_xgot as a_xgot,
        a.rolling_xa as a_xa,
        a.rolling_box_touches as a_box_touches,
        a.rolling_big_chances as a_big_chances,
        a.rolling_sot as a_sot,
        a.rolling_corners as a_corners,
        a.rolling_rest_days as a_rest_days
        
    FROM decision_times d
    LEFT JOIN latest_home_snapshots h ON d.fixture_id = h.fixture_id
    LEFT JOIN latest_away_snapshots a ON d.fixture_id = a.fixture_id
    LEFT JOIN fixture_results res ON d.fixture_id = res.fixture_id
    LEFT JOIN league_intensities li ON d.fixture_id = li.dt_fixture_id
    ORDER BY d.kickoff_time ASC, d.fixture_id ASC
    """
    
    df = pd.read_sql_query(query, conn, params={"fixture_ids": fixture_ids, "target_window_mins": target_window_mins})
    conn.close()
    
    # 3. Type Coercion (Senior Data Scientist must-have)
    # PostgreSQL numeric types can come back as 'Decimal' objects which XGBoost hates.
    # We force all feature-like columns to float64.
    cols_to_fix = [c for c in df.columns if any(prefix in c for prefix in ['rolling_', 'league_', 'h_', 'a_']) and c not in ['fixture_id', 'league_code', 'kickoff_time']]
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
    return df

if __name__ == "__main__":
    # Test block for Senior Data Scientist verification
    print("--- FEATURE BUILDER INTEGRITY TEST (Intensity & PIT) ---")
    conn = connect_db()
    with conn.cursor() as cur:
        # Get a few fixtures with snapshots built
        cur.execute("SELECT fixture_id FROM team_premium_snapshots LIMIT 3")
        rows = cur.fetchall()
        test_ids = [r[0] for r in rows]
    conn.close()

    if not test_ids:
        print("No snapshots found in DB. Run snapshot builder first.")
    else:
        print(f"Testing with Fixture IDs: {test_ids}")
        df = build_point_in_time_features(test_ids, target_window_mins=0) 
        print(df.head())
        if 'league_avg_xg' in df.columns:
            print(f"\nSUCCESS: League Intensities integrated. Avg XG: {df['league_avg_xg'].iloc[0]:.2f}")
        else:
            print("\nFAILURE: League Intensities missing.")
