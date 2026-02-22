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
MODEL_NAME = "premium_gbm"
MODEL_VERSION = "v3"
MARKETS = (
    # Totals
    "o15", "o25", "o35", "o45", "u15", "u25",
    # Corners
    "c85",
    # BTTS
    "btts",
    # 1X2
    "1x2_h", "1x2_d", "1x2_a",
    # Double Chance
    "dc_1x", "dc_x2", "dc_12",
    # Team Totals
    "ho15", "ao15",
    # Combo OR
    "home_or_o25", "away_or_o25", "home_or_o15", "away_or_o15",
    # Combo AND
    "home_and_o25", "away_and_o25",
)
OUT_COLUMNS = [
    "fixture_id",
    "flashscore_id",
    "match_datetime_utc",
    "league_code",
    "home_team",
    "away_team",
    # Totals
    "p_o15", "p_o25", "p_o35", "p_o45", "p_u15", "p_u25",
    # Corners
    "p_c85",
    # BTTS
    "p_btts",
    # 1X2
    "p_1x2_h", "p_1x2_d", "p_1x2_a",
    # Double Chance
    "p_dc_1x", "p_dc_x2", "p_dc_12",
    # Team Totals
    "p_ho15", "p_ao15",
    # Combo OR
    "p_home_or_o25", "p_away_or_o25", "p_home_or_o15", "p_away_or_o15",
    # Combo AND
    "p_home_and_o25", "p_away_and_o25",
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
        -- Totals
        MAX(CASE WHEN p.market_code = 'o15' THEN p.p_model END) AS p_o15,
        MAX(CASE WHEN p.market_code = 'o25' THEN p.p_model END) AS p_o25,
        MAX(CASE WHEN p.market_code = 'o35' THEN p.p_model END) AS p_o35,
        MAX(CASE WHEN p.market_code = 'o45' THEN p.p_model END) AS p_o45,
        MAX(CASE WHEN p.market_code = 'u15' THEN p.p_model END) AS p_u15,
        MAX(CASE WHEN p.market_code = 'u25' THEN p.p_model END) AS p_u25,
        -- Corners
        MAX(CASE WHEN p.market_code = 'c85' THEN p.p_model END) AS p_c85,
        -- BTTS
        MAX(CASE WHEN p.market_code = 'btts' THEN p.p_model END) AS p_btts,
        -- 1X2
        MAX(CASE WHEN p.market_code = '1x2_h' THEN p.p_model END) AS p_1x2_h,
        MAX(CASE WHEN p.market_code = '1x2_d' THEN p.p_model END) AS p_1x2_d,
        MAX(CASE WHEN p.market_code = '1x2_a' THEN p.p_model END) AS p_1x2_a,
        -- Double Chance
        MAX(CASE WHEN p.market_code = 'dc_1x' THEN p.p_model END) AS p_dc_1x,
        MAX(CASE WHEN p.market_code = 'dc_x2' THEN p.p_model END) AS p_dc_x2,
        MAX(CASE WHEN p.market_code = 'dc_12' THEN p.p_model END) AS p_dc_12,
        -- Team Totals
        MAX(CASE WHEN p.market_code = 'ho15' THEN p.p_model END) AS p_ho15,
        MAX(CASE WHEN p.market_code = 'ao15' THEN p.p_model END) AS p_ao15,
        -- Combo OR
        MAX(CASE WHEN p.market_code = 'home_or_o25' THEN p.p_model END) AS p_home_or_o25,
        MAX(CASE WHEN p.market_code = 'away_or_o25' THEN p.p_model END) AS p_away_or_o25,
        MAX(CASE WHEN p.market_code = 'home_or_o15' THEN p.p_model END) AS p_home_or_o15,
        MAX(CASE WHEN p.market_code = 'away_or_o15' THEN p.p_model END) AS p_away_or_o15,
        -- Combo AND
        MAX(CASE WHEN p.market_code = 'home_and_o25' THEN p.p_model END) AS p_home_and_o25,
        MAX(CASE WHEN p.market_code = 'away_and_o25' THEN p.p_model END) AS p_away_and_o25
    FROM fixtures f
    JOIN teams th ON th.team_id = f.home_team_id
    JOIN teams ta ON ta.team_id = f.away_team_id
    LEFT JOIN predictions p
        ON p.fixture_id = f.fixture_id
       AND p.model_name = %s
       AND p.model_version = %s
       AND p.market_code IN ('o15', 'o25', 'o35', 'o45', 'u15', 'u25', 'c85', 'btts', '1x2_h', '1x2_d', '1x2_a', 'dc_1x', 'dc_x2', 'dc_12', 'ho15', 'ao15', 'home_or_o25', 'away_or_o25', 'home_or_o15', 'away_or_o15', 'home_and_o25', 'away_and_o25')
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
