"""
Export market outcome predictions to standard CSV.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


DEFAULT_OUT_PATH = Path("storage/reports/market_predictions.csv")
MODEL_NAME = "market_outcome_gbm"
MODEL_VERSION = "fixtures_first_prematch_v1"
LEGACY_HANDICAP_MARKETS = (
    "ah_h05",
    "ah_a05",
    "ah_h15",
    "ah_a15",
    "eh_h1",
    "eh_a1",
)
CANONICAL_HANDICAP_MARKETS = (
    "ah2_home_m05",
    "ah2_away_p05",
    "ah2_away_m05",
    "ah2_home_p05",
    "ah2_home_m15",
    "ah2_away_p15",
    "ah2_away_m15",
    "ah2_home_p15",
    "eh3_0_1_home",
    "eh3_0_1_draw",
    "eh3_0_1_away",
    "eh3_1_0_home",
    "eh3_1_0_draw",
    "eh3_1_0_away",
)
MARKETS = (
    # Goals
    "o15", "u35",
    # Corners Totals
    "c75", "c85", "c95", "c105",
    # Corners Home Team
    "hc25", "hc35", "hc45", "hc55",
    # Corners Away Team
    "ac25", "ac35", "ac45", "ac55",
    # 1X2
    "1x2_h", "1x2_d", "1x2_a",
    # Double Chance
    "dc_1x", "dc_x2", "dc_12",
    # Team Totals
    "ho15", "ao15",
    # Legacy Handicap Markets
    *LEGACY_HANDICAP_MARKETS,
    # Canonical Handicap Markets
    *CANONICAL_HANDICAP_MARKETS,
    # Anytime Lead Markets
    "h_1up", "a_1up", "h_2up", "a_2up",
)
OUT_COLUMNS = [
    "fixture_id",
    "flashscore_id",
    "match_datetime_utc",
    "league_code",
    "home_team",
    "away_team",
    "risk_market_code",
    "risk_market_mode",
    "risk_action",
    "risk_score",
    "risk_edge_adjusted",
    "risk_stake_fraction",
    # Goals
    "p_o15", "p_u35",
    # Corners Totals
    "p_c75", "p_c85", "p_c95", "p_c105",
    # Corners Home Team
    "p_hc25", "p_hc35", "p_hc45", "p_hc55",
    # Corners Away Team
    "p_ac25", "p_ac35", "p_ac45", "p_ac55",
    # 1X2
    "p_1x2_h", "p_1x2_d", "p_1x2_a",
    # Double Chance
    "p_dc_1x", "p_dc_x2", "p_dc_12",
    # Team Totals
    "p_ho15", "p_ao15",
    # Legacy Handicap Markets
    "p_ah_h05", "p_ah_a05", "p_ah_h15", "p_ah_a15", "p_eh_h1", "p_eh_a1",
    # Canonical Handicap Markets
    "p_ah2_home_m05", "p_ah2_away_p05", "p_ah2_away_m05", "p_ah2_home_p05",
    "p_ah2_home_m15", "p_ah2_away_p15", "p_ah2_away_m15", "p_ah2_home_p15",
    "p_eh3_0_1_home", "p_eh3_0_1_draw", "p_eh3_0_1_away",
    "p_eh3_1_0_home", "p_eh3_1_0_draw", "p_eh3_1_0_away",
    # Anytime Lead Markets
    "p_h_1up", "p_a_1up", "p_h_2up", "p_a_2up",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export market outcome predictions to CSV")
    parser.add_argument("--league", type=str, default=None, help="Optional league_code filter")
    parser.add_argument("--days", type=int, default=3, help="Future horizon in days")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--version", type=str, default=MODEL_VERSION, help="Model version")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Output CSV path")
    return parser.parse_args()


def _has_risk_table(conn: object) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.prediction_risk_assessments')")
        row = cur.fetchone()
    return bool(row and row[0] is not None)


def fetch_export_rows(
    days: int,
    league: str | None,
    *,
    model: str = MODEL_NAME,
    version: str = MODEL_VERSION,
) -> pd.DataFrame:
    conn = connect_db()
    try:
        has_risk_table = _has_risk_table(conn)
    finally:
        conn.close()

    risk_select = """
        risk.risk_market_code,
        risk.risk_market_mode,
        risk.risk_action,
        risk.risk_score,
        risk.risk_edge_adjusted,
        risk.risk_stake_fraction,
    """ if has_risk_table else """
        NULL::text AS risk_market_code,
        NULL::text AS risk_market_mode,
        NULL::text AS risk_action,
        NULL::double precision AS risk_score,
        NULL::double precision AS risk_edge_adjusted,
        NULL::double precision AS risk_stake_fraction,
    """

    risk_join = """
    LEFT JOIN LATERAL (
        SELECT
            pra.market_code AS risk_market_code,
            COALESCE(pra.metadata_json ->> 'market_mode', 'predict_only') AS risk_market_mode,
            pra.action AS risk_action,
            pra.risk_score AS risk_score,
            pra.edge_adjusted AS risk_edge_adjusted,
            pra.stake_fraction AS risk_stake_fraction
        FROM prediction_risk_assessments pra
        WHERE pra.fixture_id = f.fixture_id
          AND pra.model_name = %s
          AND pra.model_version = %s
        ORDER BY
            CASE pra.action
                WHEN 'bet' THEN 3
                WHEN 'bet_small' THEN 2
                WHEN 'watch' THEN 1
                ELSE 0
            END DESC,
            pra.edge_adjusted DESC NULLS LAST,
            pra.risk_score ASC,
            pra.updated_at DESC
        LIMIT 1
    ) risk ON true
    """ if has_risk_table else ""

    risk_group_by = """
        risk.risk_market_code,
        risk.risk_market_mode,
        risk.risk_action,
        risk.risk_score,
        risk.risk_edge_adjusted,
        risk.risk_stake_fraction
    """ if has_risk_table else ""

    query = """
    SELECT
        f.fixture_id,
        f.flashscore_id,
        f.match_datetime_utc,
        f.league_code,
        th.team_name AS home_team,
        ta.team_name AS away_team,
    """
    query += risk_select
    query += """
        -- Goals
        MAX(CASE WHEN p.market_code = 'o15' THEN p.p_model END) AS p_o15,
        MAX(CASE WHEN p.market_code = 'u35' THEN p.p_model END) AS p_u35,
        -- Corners Totals
        MAX(CASE WHEN p.market_code = 'c75' THEN p.p_model END) AS p_c75,
        MAX(CASE WHEN p.market_code = 'c85' THEN p.p_model END) AS p_c85,
        MAX(CASE WHEN p.market_code = 'c95' THEN p.p_model END) AS p_c95,
        MAX(CASE WHEN p.market_code = 'c105' THEN p.p_model END) AS p_c105,
        -- Corners Home Team
        MAX(CASE WHEN p.market_code = 'hc25' THEN p.p_model END) AS p_hc25,
        MAX(CASE WHEN p.market_code = 'hc35' THEN p.p_model END) AS p_hc35,
        MAX(CASE WHEN p.market_code = 'hc45' THEN p.p_model END) AS p_hc45,
        MAX(CASE WHEN p.market_code = 'hc55' THEN p.p_model END) AS p_hc55,
        -- Corners Away Team
        MAX(CASE WHEN p.market_code = 'ac25' THEN p.p_model END) AS p_ac25,
        MAX(CASE WHEN p.market_code = 'ac35' THEN p.p_model END) AS p_ac35,
        MAX(CASE WHEN p.market_code = 'ac45' THEN p.p_model END) AS p_ac45,
        MAX(CASE WHEN p.market_code = 'ac55' THEN p.p_model END) AS p_ac55,
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
        -- Legacy Handicap Markets
        MAX(CASE WHEN p.market_code = 'ah_h05' THEN p.p_model END) AS p_ah_h05,
        MAX(CASE WHEN p.market_code = 'ah_a05' THEN p.p_model END) AS p_ah_a05,
        MAX(CASE WHEN p.market_code = 'ah_h15' THEN p.p_model END) AS p_ah_h15,
        MAX(CASE WHEN p.market_code = 'ah_a15' THEN p.p_model END) AS p_ah_a15,
        MAX(CASE WHEN p.market_code = 'eh_h1' THEN p.p_model END) AS p_eh_h1,
        MAX(CASE WHEN p.market_code = 'eh_a1' THEN p.p_model END) AS p_eh_a1,
        -- Canonical Handicap Markets
        MAX(CASE WHEN p.market_code = 'ah2_home_m05' THEN p.p_model END) AS p_ah2_home_m05,
        MAX(CASE WHEN p.market_code = 'ah2_away_p05' THEN p.p_model END) AS p_ah2_away_p05,
        MAX(CASE WHEN p.market_code = 'ah2_away_m05' THEN p.p_model END) AS p_ah2_away_m05,
        MAX(CASE WHEN p.market_code = 'ah2_home_p05' THEN p.p_model END) AS p_ah2_home_p05,
        MAX(CASE WHEN p.market_code = 'ah2_home_m15' THEN p.p_model END) AS p_ah2_home_m15,
        MAX(CASE WHEN p.market_code = 'ah2_away_p15' THEN p.p_model END) AS p_ah2_away_p15,
        MAX(CASE WHEN p.market_code = 'ah2_away_m15' THEN p.p_model END) AS p_ah2_away_m15,
        MAX(CASE WHEN p.market_code = 'ah2_home_p15' THEN p.p_model END) AS p_ah2_home_p15,
        MAX(CASE WHEN p.market_code = 'eh3_0_1_home' THEN p.p_model END) AS p_eh3_0_1_home,
        MAX(CASE WHEN p.market_code = 'eh3_0_1_draw' THEN p.p_model END) AS p_eh3_0_1_draw,
        MAX(CASE WHEN p.market_code = 'eh3_0_1_away' THEN p.p_model END) AS p_eh3_0_1_away,
        MAX(CASE WHEN p.market_code = 'eh3_1_0_home' THEN p.p_model END) AS p_eh3_1_0_home,
        MAX(CASE WHEN p.market_code = 'eh3_1_0_draw' THEN p.p_model END) AS p_eh3_1_0_draw,
        MAX(CASE WHEN p.market_code = 'eh3_1_0_away' THEN p.p_model END) AS p_eh3_1_0_away,
        -- Anytime Lead Markets
        MAX(CASE WHEN p.market_code = 'h_1up' THEN p.p_model END) AS p_h_1up,
        MAX(CASE WHEN p.market_code = 'a_1up' THEN p.p_model END) AS p_a_1up,
        MAX(CASE WHEN p.market_code = 'h_2up' THEN p.p_model END) AS p_h_2up,
        MAX(CASE WHEN p.market_code = 'a_2up' THEN p.p_model END) AS p_a_2up
    FROM fixtures f
    JOIN teams th ON th.team_id = f.home_team_id
    JOIN teams ta ON ta.team_id = f.away_team_id
    LEFT JOIN predictions p
       ON p.fixture_id = f.fixture_id
       AND p.model_name = %s
       AND p.model_version = %s
       AND p.market_code IN ('o15', 'u35', 'c75', 'c85', 'c95', 'c105', 'hc25', 'hc35', 'hc45', 'hc55', 'ac25', 'ac35', 'ac45', 'ac55', '1x2_h', '1x2_d', '1x2_a', 'dc_1x', 'dc_x2', 'dc_12', 'ho15', 'ao15', 'ah_h05', 'ah_a05', 'ah_h15', 'ah_a15', 'eh_h1', 'eh_a1', 'ah2_home_m05', 'ah2_away_p05', 'ah2_away_m05', 'ah2_home_p05', 'ah2_home_m15', 'ah2_away_p15', 'ah2_away_m15', 'ah2_home_p15', 'eh3_0_1_home', 'eh3_0_1_draw', 'eh3_0_1_away', 'eh3_1_0_home', 'eh3_1_0_draw', 'eh3_1_0_away', 'h_1up', 'a_1up', 'h_2up', 'a_2up')
    """
    query += risk_join
    query += """
    WHERE f.status = 'scheduled'
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc > NOW()
      AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
    """
    params: list[object] = [model, version]
    if has_risk_table:
        params.extend([model, version])
    params.append(days)

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
    """
    if has_risk_table:
        query += ",\n"
        query += risk_group_by
    query += """
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
    rows = fetch_export_rows(
        days=args.days,
        league=args.league,
        model=args.model,
        version=args.version,
    )
    export_csv(rows, args.out)
    print(f"Saved {len(rows)} fixtures to {args.out}")


if __name__ == "__main__":
    main()
