import os
import psycopg2
from src.db.db_utils import get_database_url

def run_audit():
    url = get_database_url()
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    
    print("--- Fixture Status Counts ---")
    cur.execute("SELECT status, COUNT(*), MIN(match_datetime_utc), MAX(match_datetime_utc) FROM fixtures GROUP BY status;")
    for row in cur.fetchall():
        print(row)
        
    print("\n--- Result Coverage per League ---")
    cur.execute("""
        SELECT 
            l.league_code, 
            COUNT(f.fixture_id) as total_ft, 
            COUNT(r.fixture_id) as with_results,
            ROUND(COUNT(r.fixture_id)::numeric / NULLIF(COUNT(f.fixture_id), 0), 2) as pct
        FROM leagues l
        JOIN fixtures f ON l.league_code = f.league_code
        LEFT JOIN fixture_results r ON f.fixture_id = r.fixture_id
        WHERE f.status = 'ft'
        GROUP BY l.league_code
        ORDER BY pct DESC;
    """)
    for row in cur.fetchall():
        print(row)

    print("\n--- Odds Snapshot Population ---")
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE ou_json IS NOT NULL AND ou_json::text != '{}') as non_empty_ou,
            COUNT(*) FILTER (WHERE one_x_two_json IS NOT NULL AND one_x_two_json::text != '{}') as non_empty_1x2
        FROM fixture_odds_snapshots;
    """)
    print(cur.fetchone())

    print("\n--- Premium Stats Audit ---")
    cur.execute("""
        SELECT 
            l.league_code,
            COUNT(*) as total_stats,
            COUNT(*) FILTER (WHERE h_xg IS NULL OR a_xg IS NULL) as missing_xg,
            COUNT(*) FILTER (WHERE h_xg = 0 AND a_xg = 0) as zero_xg
        FROM fixture_stats_premium s
        JOIN fixtures f ON s.fixture_id = f.fixture_id
        JOIN leagues l ON f.league_code = l.league_code
        GROUP BY l.league_code;
    """)
    for row in cur.fetchall():
        print(row)

    print("\n--- League Registry ---")
    cur.execute("SELECT league_id, league_code, league_name FROM leagues;")
    for row in cur.fetchall():
        print(row)

    print("\n--- Recent Result Samples ---")
    cur.execute("""
        SELECT f.fixture_id, f.league_code, r.home_goals, r.away_goals, f.match_datetime_utc
        FROM fixtures f
        JOIN fixture_results r ON f.fixture_id = r.fixture_id
        ORDER BY f.match_datetime_utc DESC
        LIMIT 5;
    """)
    for row in cur.fetchall():
        print(row)
    
    print("\n--- Snapshot Builder History ---")
    cur.execute("""
        SELECT details_json->>'league' as league, COUNT(*), MAX(ended_at)
        FROM pipeline_runs
        WHERE job_name = 'build_team_premium_snapshots_v1'
        GROUP BY details_json->>'league';
    """)
    for row in cur.fetchall():
        print(row)

    print("\n--- Team Snapshot Counts ---")
    cur.execute("""
        SELECT l.league_code, COUNT(*) 
        FROM team_premium_snapshots s
        JOIN fixtures f ON s.fixture_id = f.fixture_id
        JOIN leagues l ON f.league_code = l.league_code
        GROUP BY l.league_code;
    """)
    for row in cur.fetchall():
        print(row)


if __name__ == "__main__":
    run_audit()
