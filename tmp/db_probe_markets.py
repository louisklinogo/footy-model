import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.db_utils import connect_db

QUERIES = {
    "coverage": """
        WITH base AS (
            SELECT f.fixture_id, f.league_code
            FROM fixtures f
            JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
            WHERE f.status = 'ft'
              AND f.match_datetime_utc IS NOT NULL
              AND fr.home_goals IS NOT NULL
              AND fr.away_goals IS NOT NULL
        )
        SELECT
            COUNT(*) AS ft_fixtures,
            COUNT(*) FILTER (WHERE sp.fixture_id IS NOT NULL) AS stats_rows,
            COUNT(*) FILTER (WHERE sp.h_corners IS NOT NULL AND sp.a_corners IS NOT NULL) AS corners_rows,
            COUNT(*) FILTER (WHERE ff.fixture_id IS NOT NULL) AS formations_rows,
            COUNT(*) FILTER (WHERE tph.fixture_id IS NOT NULL) AS home_snapshot_rows,
            COUNT(*) FILTER (WHERE tpa.fixture_id IS NOT NULL) AS away_snapshot_rows,
            COUNT(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM fixture_incidents_sofascore i WHERE i.fixture_id = b.fixture_id
            )) AS incident_rows,
            COUNT(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM fixture_odds_markets fom
                WHERE fom.fixture_id = b.fixture_id AND fom.provider = 'sofascore'
                  AND fom.market_code = '1x2' AND fom.snapshot_type IN ('latest_pre_match','closing')
            )) AS odds_1x2_rows,
            COUNT(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM fixture_odds_markets fom
                WHERE fom.fixture_id = b.fixture_id AND fom.provider = 'sofascore'
                  AND fom.market_code = 'ou' AND fom.line_num = 1.5 AND fom.snapshot_type IN ('latest_pre_match','closing')
            )) AS odds_o15_rows,
            COUNT(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM fixture_odds_markets fom
                WHERE fom.fixture_id = b.fixture_id AND fom.provider = 'sofascore'
                  AND fom.market_code = 'ou' AND fom.line_num = 3.5 AND fom.snapshot_type IN ('latest_pre_match','closing')
            )) AS odds_u35_rows,
            COUNT(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM fixture_odds_markets fom
                WHERE fom.fixture_id = b.fixture_id AND fom.provider = 'sofascore'
                  AND fom.market_code = 'corners_ou' AND fom.line_num = 7.5 AND fom.snapshot_type IN ('latest_pre_match','closing')
            )) AS odds_c75_rows,
            COUNT(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM fixture_odds_markets fom
                WHERE fom.fixture_id = b.fixture_id AND fom.provider = 'sofascore'
                  AND fom.market_code = 'corners_ou' AND fom.line_num = 10.5 AND fom.snapshot_type IN ('latest_pre_match','closing')
            )) AS odds_c105_rows
        FROM base b
        LEFT JOIN fixture_stats_premium sp ON sp.fixture_id = b.fixture_id
        LEFT JOIN fixture_formations ff ON ff.fixture_id = b.fixture_id
        LEFT JOIN team_premium_snapshots tph ON tph.fixture_id = b.fixture_id AND tph.is_home = true
        LEFT JOIN team_premium_snapshots tpa ON tpa.fixture_id = b.fixture_id AND tpa.is_home = false;
    """,
    "dispersion": """
        SELECT
            COUNT(*) AS n,
            AVG(sp.h_corners::float) AS home_mean,
            VAR_SAMP(sp.h_corners::float) AS home_var,
            AVG(sp.a_corners::float) AS away_mean,
            VAR_SAMP(sp.a_corners::float) AS away_var,
            AVG((sp.h_corners + sp.a_corners)::float) AS total_mean,
            VAR_SAMP((sp.h_corners + sp.a_corners)::float) AS total_var,
            CORR(sp.h_corners::float, sp.a_corners::float) AS home_away_corr,
            COVAR_SAMP(sp.h_corners::float, sp.a_corners::float) AS home_away_cov
        FROM fixture_stats_premium sp
        JOIN fixtures f ON f.fixture_id = sp.fixture_id
        JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
        WHERE f.status = 'ft'
          AND sp.h_corners IS NOT NULL AND sp.a_corners IS NOT NULL
          AND fr.home_goals IS NOT NULL AND fr.away_goals IS NOT NULL;
    """,
    "goals": """
        SELECT
            COUNT(*) AS n,
            AVG((fr.home_goals = fr.away_goals)::int::float) AS draw_rate,
            AVG((fr.home_goals + fr.away_goals)::float) AS total_goals_mean,
            VAR_SAMP((fr.home_goals + fr.away_goals)::float) AS total_goals_var,
            CORR(fr.home_goals::float, fr.away_goals::float) AS home_away_goal_corr
        FROM fixtures f
        JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
        WHERE f.status = 'ft'
          AND f.match_datetime_utc IS NOT NULL
          AND fr.home_goals IS NOT NULL AND fr.away_goals IS NOT NULL;
    """,
    "league_segments": """
        SELECT
            f.league_code,
            COUNT(*) AS n,
            COUNT(*) FILTER (WHERE sp.h_corners IS NOT NULL AND sp.a_corners IS NOT NULL) AS corners_n,
            AVG((fr.home_goals = fr.away_goals)::int::float) AS draw_rate,
            AVG((sp.h_corners + sp.a_corners)::float) FILTER (WHERE sp.h_corners IS NOT NULL AND sp.a_corners IS NOT NULL) AS total_corners_mean,
            VAR_SAMP((sp.h_corners + sp.a_corners)::float) FILTER (WHERE sp.h_corners IS NOT NULL AND sp.a_corners IS NOT NULL) AS total_corners_var,
            CORR(sp.h_corners::float, sp.a_corners::float) FILTER (WHERE sp.h_corners IS NOT NULL AND sp.a_corners IS NOT NULL) AS corners_corr
        FROM fixtures f
        JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
        LEFT JOIN fixture_stats_premium sp ON sp.fixture_id = f.fixture_id
        WHERE f.status = 'ft'
          AND f.match_datetime_utc IS NOT NULL
          AND fr.home_goals IS NOT NULL AND fr.away_goals IS NOT NULL
        GROUP BY f.league_code
        HAVING COUNT(*) >= 100
        ORDER BY n DESC
        LIMIT 25;
    """,
    "odds_markets": """
        SELECT provider, snapshot_type, market_code, line_num, COUNT(DISTINCT fixture_id) AS fixtures
        FROM fixture_odds_markets
        WHERE provider = 'sofascore'
        GROUP BY provider, snapshot_type, market_code, line_num
        ORDER BY market_code, line_num, snapshot_type;
    """,
}


def main() -> None:
    conn = connect_db()
    try:
        out = {}
        with conn.cursor() as cur:
            for name, sql in QUERIES.items():
                cur.execute(sql)
                cols = [d[0] for d in cur.description]
                out[name] = [dict(zip(cols, row)) for row in cur.fetchall()]
        print(json.dumps(out, indent=2, default=str))
    finally:
        conn.close()


if __name__ == "__main__":
    main()

