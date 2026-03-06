import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.db_utils import connect_db


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def independent_poisson_draw_rate(home_mu: float, away_mu: float, max_k: int = 15) -> float:
    return sum(poisson_pmf(k, home_mu) * poisson_pmf(k, away_mu) for k in range(max_k + 1))


def rows(cur, sql: str):
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def main() -> None:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            goals = rows(cur, """
                SELECT
                    AVG(fr.home_goals::float) AS home_mu,
                    AVG(fr.away_goals::float) AS away_mu,
                    AVG((fr.home_goals = fr.away_goals)::int::float) AS draw_rate
                FROM fixtures f
                JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
                WHERE f.status = 'ft'
                  AND f.match_datetime_utc IS NOT NULL
                  AND fr.home_goals IS NOT NULL
                  AND fr.away_goals IS NOT NULL;
            """)[0]
            score_freq = rows(cur, """
                SELECT fr.home_goals, fr.away_goals, COUNT(*) AS n
                FROM fixtures f
                JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
                WHERE f.status = 'ft'
                  AND f.match_datetime_utc IS NOT NULL
                  AND fr.home_goals IS NOT NULL
                  AND fr.away_goals IS NOT NULL
                GROUP BY fr.home_goals, fr.away_goals
                ORDER BY n DESC
                LIMIT 15;
            """)
            corners_prev = rows(cur, """
                SELECT
                    COUNT(*) AS n,
                    AVG(((sp.h_corners + sp.a_corners) >= 8)::int::float) AS c75_rate,
                    AVG(((sp.h_corners + sp.a_corners) >= 9)::int::float) AS c85_rate,
                    AVG(((sp.h_corners + sp.a_corners) >= 10)::int::float) AS c95_rate,
                    AVG(((sp.h_corners + sp.a_corners) >= 11)::int::float) AS c105_rate,
                    AVG((sp.h_corners >= 3)::int::float) AS hc25_rate,
                    AVG((sp.h_corners >= 4)::int::float) AS hc35_rate,
                    AVG((sp.a_corners >= 3)::int::float) AS ac25_rate,
                    AVG((sp.a_corners >= 4)::int::float) AS ac35_rate
                FROM fixture_stats_premium sp
                JOIN fixtures f ON f.fixture_id = sp.fixture_id
                JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
                WHERE f.status = 'ft'
                  AND sp.h_corners IS NOT NULL AND sp.a_corners IS NOT NULL
                  AND fr.home_goals IS NOT NULL AND fr.away_goals IS NOT NULL;
            """)[0]
            odds_focus = rows(cur, """
                SELECT market_code, line_num, snapshot_type, COUNT(DISTINCT fixture_id) AS fixtures
                FROM fixture_odds_markets
                WHERE provider = 'sofascore'
                  AND (
                    (market_code = 'corners_ou' AND line_num IN (7.5, 8.5, 9.5, 10.5)) OR
                    (market_code = 'home_corners_ou' AND line_num IN (2.5, 3.5, 4.5, 5.5)) OR
                    (market_code = 'away_corners_ou' AND line_num IN (2.5, 3.5, 4.5, 5.5))
                  )
                GROUP BY market_code, line_num, snapshot_type
                ORDER BY market_code, line_num, snapshot_type;
            """)
            team_market_presence = rows(cur, """
                SELECT market_code, COUNT(DISTINCT fixture_id) AS fixtures
                FROM fixture_odds_markets
                WHERE provider = 'sofascore'
                  AND market_code IN ('home_ou','away_ou','home_corners_ou','away_corners_ou')
                GROUP BY market_code
                ORDER BY market_code;
            """)
        home_mu = float(goals["home_mu"])
        away_mu = float(goals["away_mu"])
        out = {
            "goals": goals,
            "independent_poisson_draw_rate": independent_poisson_draw_rate(home_mu, away_mu),
            "top_scorelines": score_freq,
            "corners_prevalence": corners_prev,
            "odds_focus": odds_focus,
            "team_market_presence": team_market_presence,
        }
        print(json.dumps(out, indent=2, default=str))
    finally:
        conn.close()


if __name__ == "__main__":
    main()

