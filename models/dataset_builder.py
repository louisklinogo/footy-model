import sqlite3
import pandas as pd
from typing import List, Dict, Any

from db_utils import get_db_connection

def build_point_in_time_features(
    fixture_ids: List[int],
    target_window_mins: int = 60,
    db_path: str = "data/footy_premium.db"
) -> pd.DataFrame:
    """
    Builds a point-in-time feature set for a list of fixtures.
    Crucially ensures that source_ts <= decision_timestamp.
    
    Args:
        fixture_ids: List of fixture IDs to extract features for.
        target_window_mins: The decision window before kickoff (e.g., 60 for T-60m).
        db_path: Path to the SQLite database.
        
    Returns:
        DataFrame containing the point-in-time features joined to the fixture data.
    """
    
    conn = get_db_connection(db_path)
    
    if not fixture_ids:
        return pd.DataFrame()
        
    ids_str = ",".join(map(str, fixture_ids))
    
    query = f"""
    WITH target_fixtures AS (
        SELECT 
            fixture_id,
            league_id,
            season,
            date as kickoff_time,
            CAST(strftime('%s', date) as integer) as kickoff_ts,
            home_team_id,
            home_team_name,
            away_team_id,
            away_team_name
        FROM fixtures
        WHERE fixture_id IN ({ids_str})
    ),
    decision_times AS (
        SELECT
            *,
            (kickoff_ts - ({target_window_mins} * 60)) AS decision_ts
        FROM target_fixtures
    ),
    valid_home_snapshots AS (
        SELECT 
            dt.fixture_id,
            s.*,
            ROW_NUMBER() OVER(PARTITION BY dt.fixture_id ORDER BY s.built_at DESC) as rn
        FROM decision_times dt
        JOIN team_premium_snapshots s 
          ON s.team_id = dt.home_team_id 
         AND s.league_id = dt.league_id
        WHERE CAST(strftime('%s', s.built_at) as integer) <= dt.decision_ts
    ),
    latest_home_snapshots AS (
        SELECT * FROM valid_home_snapshots WHERE rn = 1
    ),
    valid_away_snapshots AS (
        SELECT 
            dt.fixture_id,
            s.*,
            ROW_NUMBER() OVER(PARTITION BY dt.fixture_id ORDER BY s.built_at DESC) as rn
        FROM decision_times dt
        JOIN team_premium_snapshots s 
          ON s.team_id = dt.away_team_id 
         AND s.league_id = dt.league_id
        WHERE CAST(strftime('%s', s.built_at) as integer) <= dt.decision_ts
    ),
    latest_away_snapshots AS (
        SELECT * FROM valid_away_snapshots WHERE rn = 1
    )
    
    SELECT
        d.fixture_id,
        d.league_id, d.season, d.kickoff_time,
        d.home_team_name, d.away_team_name,
        
        -- Home Features
        h.rolling_xg as home_rolling_xg,
        h.rolling_xgot as home_rolling_xgot,
        h.rolling_xg_against as home_rolling_xg_against,
        h.rolling_xgot_against as home_rolling_xgot_against,
        h.rolling_sot as home_rolling_sot,
        h.rolling_sot_against as home_rolling_sot_against,
        h.rolling_corners as home_rolling_corners,
        h.rolling_corners_against as home_rolling_corners_against,
        h.rolling_possession as home_rolling_possession,
        
        -- Away Features
        a.rolling_xg as away_rolling_xg,
        a.rolling_xgot as away_rolling_xgot,
        a.rolling_xg_against as away_rolling_xg_against,
        a.rolling_xgot_against as away_rolling_xgot_against,
        a.rolling_sot as away_rolling_sot,
        a.rolling_sot_against as away_rolling_sot_against,
        a.rolling_corners as away_rolling_corners,
        a.rolling_corners_against as away_rolling_corners_against,
        a.rolling_possession as away_rolling_possession
        
    FROM decision_times d
    LEFT JOIN latest_home_snapshots h ON d.fixture_id = h.fixture_id
    LEFT JOIN latest_away_snapshots a ON d.fixture_id = a.fixture_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

if __name__ == "__main__":
    # Test with a few known English Premier League fixture IDs
    test_fixture_ids = [1207153, 1207164, 1207165]
    print(f"Building features for fixtures: {test_fixture_ids} at T-60m")
    
    df = build_point_in_time_features(test_fixture_ids, target_window_mins=60)
    print(df.head())
    print("\nShape:", df.shape)
