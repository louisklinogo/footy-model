"""
Export fixtures-first v3 predictions to v1-compatible CSV.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


DEFAULT_OUT_PATH = Path("data/v1/daily/predictions_v3_fixtures_first.csv")
MODEL_NAME = "fixtures_first_gbm"
MODEL_VERSION = "v3"
MARKETS = ("o15", "o25", "c85")
OUT_COLUMNS = [
    "fixture_id",
    "flashscore_id",
    "match_datetime_utc",
    "league_code",
    "home_team",
    "away_team",
    "p_o15",
    "p_o25",
    "p_c85",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixtures-first v3 predictions to CSV")
    parser.add_argument("--league", type=str, default=None, help="Optional league_code filter")
    parser.add_argument("--days", type=int, default=3, help="Future horizon in days")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Output CSV path")
    return parser.parse_args()


def fetch_export_rows(days: int, league: str | None) -> pd.DataFrame:
    query = """
    SELECT
        f.fixture_id,
        f.flashscore_id,
        f.match_datetime_utc,
        f.league_code,
        th.team_name AS home_team,
        ta.team_name AS away_team,
        MAX(CASE WHEN p.market_code = 'o15' THEN p.p_model END) AS p_o15,
        MAX(CASE WHEN p.market_code = 'o25' THEN p.p_model END) AS p_o25,
        MAX(CASE WHEN p.market_code = 'c85' THEN p.p_model END) AS p_c85
    FROM fixtures f
    JOIN teams th ON th.team_id = f.home_team_id
    JOIN teams ta ON ta.team_id = f.away_team_id
    LEFT JOIN predictions p
        ON p.fixture_id = f.fixture_id
       AND p.model_name = %s
       AND p.model_version = %s
       AND p.market_code IN ('o15', 'o25', 'c85')
    WHERE f.status = 'scheduled'
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc > NOW()
      AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
    """
    params: list[object] = [MODEL_NAME, MODEL_VERSION, days]

    if league:
        query += " AND f.league_code = %s"
        params.append(league)

    query += """
    GROUP BY
        f.fixture_id,
        f.flashscore_id,
        f.match_datetime_utc,
        f.league_code,
        th.team_name,
        ta.team_name
    ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC
    """

    conn = connect_db()
    try:
        return pd.read_sql(query, conn, params=tuple(params))
    finally:
        conn.close()


def export_csv(df: pd.DataFrame, out_path: Path) -> None:
    output = df.reindex(columns=OUT_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    rows = fetch_export_rows(days=args.days, league=args.league)
    export_csv(rows, args.out)
    print(f"Saved {len(rows)} fixtures to {args.out}")


if __name__ == "__main__":
    main()
