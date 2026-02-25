import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

def run_audit():
    conn = connect_db()
    
    # 1. League Coverage & Fidelity
    print("--- 1. TEAM SNAPSHOTS GENERATED (RAW VOLUME) ---")
    query1 = """
        SELECT 
            COUNT(*) as total_snapshots,
            COUNT(DISTINCT team_id) as unique_teams,
            AVG(fidelity_score) as avg_fidelity,
            SUM(CASE WHEN fidelity_score >= 0.8 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as pct_high_fidelity
        FROM team_premium_snapshots
    """
    df1 = pd.read_sql_query(query1, conn)
    print(df1.to_markdown())
    
    # 2. Feature Completeness (Null Check)
    print("\n--- 2. REQUIRED ROLLING METRICS COMPLETENESS ---")
    query2 = """
        SELECT
            COUNT(*) as total_rows,
            COUNT(rolling_xg) * 100.0 / NULLIF(COUNT(*), 0) as pct_has_xg,
            COUNT(rolling_xgot) * 100.0 / NULLIF(COUNT(*), 0) as pct_has_xgot,
            COUNT(rolling_box_touches) * 100.0 / NULLIF(COUNT(*), 0) as pct_has_box_touches,
            COUNT(rolling_big_chances) * 100.0 / NULLIF(COUNT(*), 0) as pct_has_big_chances,
            COUNT(rolling_goals_prevented) * 100.0 / NULLIF(COUNT(*), 0) as pct_has_goals_prevented,
            COUNT(rolling_xa) * 100.0 / NULLIF(COUNT(*), 0) as pct_has_xa
        FROM team_premium_snapshots
    """
    df2 = pd.read_sql_query(query2, conn)
    print(df2.to_markdown())
    
    # 3. Checking for Rest Days / Schedule Density in the DB
    print("\n--- 3. CHECKING FOR REST DAYS / MATCH CONGESTION ---")
    query3 = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'team_premium_snapshots'
          AND column_name LIKE '%rest%' OR column_name LIKE '%days%'
    """
    with conn.cursor() as cur:
        cur.execute(query3)
        rest_cols = cur.fetchall()
        if rest_cols:
            print("Found rest/schedule columns in snapshots:", rest_cols)
        else:
            print("WARNING: No rest/days columns found in team_premium_snapshots. We currently have no way to measure schedule congestion.")
            
    # 4. Checking for Elo / Rating in the DB
    print("\n--- 4. CHECKING FOR ELO / BAYESIAN RATINGS ---")
    query4 = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'team_premium_snapshots'
          AND column_name LIKE '%elo%' OR column_name LIKE '%rating%'
    """
    with conn.cursor() as cur:
        cur.execute(query4)
        rating_cols = cur.fetchall()
        if rating_cols:
            print("Found rating columns in snapshots:", rating_cols)
        else:
            print("WARNING: No elo/rating columns found in team_premium_snapshots. XGBoost will be relying purely on recent form (rolling averages) without long-term team strength anchors.")

    conn.close()

if __name__ == "__main__":
    run_audit()
