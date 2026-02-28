"""
Score settled market outcome predictions into prediction_scores.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


MODEL_NAME = "market_outcome_gbm"
MODEL_VERSION = "fixtures_first_prematch_v1"
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
    # Anytime Lead Markets
    "h_1up", "a_1up", "h_2up", "a_2up",
    # Combo OR
    "home_or_o25", "away_or_o25", "home_or_o15", "away_or_o15",
    # Combo AND
    "home_and_o25", "away_and_o25",
)
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score settled market outcome predictions")
    parser.add_argument("--league", type=str, default=None, help="Optional league_code filter")
    parser.add_argument("--limit", type=int, default=None, help="Optional max predictions to score")
    parser.add_argument(
        "--since-days",
        type=int,
        default=30,
        help="Only include fixtures within this many days back",
    )
    return parser.parse_args()


def _latest_over_odds_expr(row_alias: str) -> str:
    return (
        "CASE "
        f"WHEN ({row_alias}.odds_json -> 'prices_latest' ->> 'over') ~ '^[-+]?[0-9]*\\.?[0-9]+$' "
        f"THEN ({row_alias}.odds_json -> 'prices_latest' ->> 'over')::double precision "
        "ELSE NULL END"
    )


def fetch_unscored_predictions(
    league: str | None,
    limit: int | None,
    since_days: int | None,
) -> list[dict[str, object]]:
    query = f"""
    SELECT
        p.prediction_id,
        p.market_code,
        p.p_model,
        f.fixture_id,
        f.league_code,
        f.status,
        f.match_datetime_utc,
        fr.home_goals,
        fr.away_goals,
        fs.h_corners,
        fs.a_corners,
        ils.home_led_by_1_any,
        ils.away_led_by_1_any,
        ils.home_led_by_2_any,
        ils.away_led_by_2_any,
        CASE
            WHEN p.market_code = 'o15' THEN {_latest_over_odds_expr('od15')}
            WHEN p.market_code = 'o25' THEN {_latest_over_odds_expr('od25')}
            ELSE NULL
        END AS odds_used
    FROM predictions p
    JOIN fixtures f
      ON f.fixture_id = p.fixture_id
    JOIN fixture_results fr
      ON fr.fixture_id = f.fixture_id
    LEFT JOIN fixture_stats_premium fs
      ON fs.fixture_id = f.fixture_id
    LEFT JOIN fixture_incident_lead_states ils
      ON ils.fixture_id = f.fixture_id
    LEFT JOIN LATERAL (
        SELECT fom.snapshot_time_utc, fom.snapshot_type, fom.odds_json
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.provider = 'sofascore'
          AND fom.market_code = 'ou'
          AND fom.line_num = 1.5
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND fom.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY (fom.snapshot_type = 'latest_pre_match') DESC, fom.snapshot_time_utc DESC
        LIMIT 1
    ) od15 ON true
    LEFT JOIN LATERAL (
        SELECT fom.snapshot_time_utc, fom.snapshot_type, fom.odds_json
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.provider = 'sofascore'
          AND fom.market_code = 'ou'
          AND fom.line_num = 2.5
          AND fom.snapshot_type IN ('latest_pre_match', 'closing')
          AND fom.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY (fom.snapshot_type = 'latest_pre_match') DESC, fom.snapshot_time_utc DESC
        LIMIT 1
    ) od25 ON true
    LEFT JOIN prediction_scores ps
      ON ps.prediction_id = p.prediction_id
    WHERE p.model_name = %s
      AND p.model_version = %s
      AND p.market_code IN ('o15', 'o25', 'o35', 'o45', 'u15', 'u25', 'c85', 'btts', '1x2_h', '1x2_d', '1x2_a', 'dc_1x', 'dc_x2', 'dc_12', 'ho15', 'ao15', 'h_1up', 'a_1up', 'h_2up', 'a_2up', 'home_or_o25', 'away_or_o25', 'home_or_o15', 'away_or_o15', 'home_and_o25', 'away_and_o25')
      AND f.status = 'ft'
      AND f.status NOT IN ('postponed', 'cancelled', 'abandoned')
      AND ps.prediction_id IS NULL
    """
    params: list[object] = [MODEL_NAME, MODEL_VERSION]

    if since_days is not None:
        query += " AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval"
        params.append(since_days)

    if league:
        query += " AND f.league_code = %s"
        params.append(league)

    query += " ORDER BY f.match_datetime_utc DESC, p.prediction_id ASC"

    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            description = cur.description or []
            cols = [desc[0] for desc in description]
    finally:
        conn.close()

    return [dict(zip(cols, row)) for row in rows]


def compute_actual(row: dict[str, object]) -> float | None:
    market = str(row["market_code"])
    home_goals = row.get("home_goals")
    away_goals = row.get("away_goals")

    if home_goals is None or away_goals is None:
        return None

    h = int(home_goals)
    a = int(away_goals)
    total_goals = h + a

    # === TOTALS MARKETS ===
    if market == "o15":
        return 1.0 if total_goals >= 2 else 0.0
    if market == "o25":
        return 1.0 if total_goals >= 3 else 0.0
    if market == "o35":
        return 1.0 if total_goals >= 4 else 0.0
    if market == "o45":
        return 1.0 if total_goals >= 5 else 0.0
    if market == "u15":
        return 1.0 if total_goals <= 1 else 0.0
    if market == "u25":
        return 1.0 if total_goals <= 2 else 0.0

    # === CORNERS ===
    if market == "c85":
        h_corners = row.get("h_corners")
        a_corners = row.get("a_corners")
        if h_corners is None or a_corners is None:
            return None
        return 1.0 if (int(h_corners) + int(a_corners)) >= 9 else 0.0

    # === BTTS ===
    if market == "btts":
        return 1.0 if (h > 0 and a > 0) else 0.0

    # === 1X2 ===
    if market == "1x2_h":
        return 1.0 if h > a else 0.0
    if market == "1x2_d":
        return 1.0 if h == a else 0.0
    if market == "1x2_a":
        return 1.0 if h < a else 0.0

    # === DOUBLE CHANCE ===
    if market == "dc_1x":
        return 1.0 if h >= a else 0.0
    if market == "dc_x2":
        return 1.0 if a >= h else 0.0
    if market == "dc_12":
        return 1.0 if h != a else 0.0

    # === TEAM TOTALS ===
    if market == "ho15":
        return 1.0 if h >= 2 else 0.0
    if market == "ao15":
        return 1.0 if a >= 2 else 0.0

    # === ANYTIME LEAD MARKETS (incident timeline derived) ===
    if market == "h_1up":
        val = row.get("home_led_by_1_any")
        return float(int(bool(val))) if val is not None else None
    if market == "a_1up":
        val = row.get("away_led_by_1_any")
        return float(int(bool(val))) if val is not None else None
    if market == "h_2up":
        val = row.get("home_led_by_2_any")
        return float(int(bool(val))) if val is not None else None
    if market == "a_2up":
        val = row.get("away_led_by_2_any")
        return float(int(bool(val))) if val is not None else None

    # === COMBO OR ===
    if market == "home_or_o25":
        return 1.0 if (h > a or total_goals >= 3) else 0.0
    if market == "away_or_o25":
        return 1.0 if (a > h or total_goals >= 3) else 0.0
    if market == "home_or_o15":
        return 1.0 if (h > a or total_goals >= 2) else 0.0
    if market == "away_or_o15":
        return 1.0 if (a > h or total_goals >= 2) else 0.0

    # === COMBO AND ===
    if market == "home_and_o25":
        return 1.0 if (h > a and total_goals >= 3) else 0.0
    if market == "away_and_o25":
        return 1.0 if (a > h and total_goals >= 3) else 0.0

    return None


def build_score_rows(candidates: list[dict[str, object]]) -> tuple[list[tuple[object, ...]], int]:
    rows: list[tuple[object, ...]] = []
    skipped = 0

    for candidate in candidates:
        actual = compute_actual(candidate)
        if actual is None:
            skipped += 1
            continue

        p_raw = float(candidate["p_model"])
        p_clipped = min(max(p_raw, EPS), 1.0 - EPS)
        brier = (p_clipped - actual) ** 2
        log_loss = -(
            actual * math.log(p_clipped)
            + (1.0 - actual) * math.log(1.0 - p_clipped)
        )
        hit = (p_clipped >= 0.5) == (actual == 1.0)

        odds_used_raw = candidate.get("odds_used")
        odds_used: float | None = None
        edge: float | None = None
        roi_unit: float | None = None

        if odds_used_raw is not None:
            parsed_odds = float(odds_used_raw)
            if parsed_odds > 0.0:
                odds_used = parsed_odds
                if odds_used > 1.0:
                    edge = p_clipped - (1.0 / odds_used)
                roi_unit = (odds_used - 1.0) if actual == 1.0 else -1.0

        rows.append(
            (
                int(candidate["prediction_id"]),
                actual,
                brier,
                log_loss,
                hit,
                odds_used,
                edge,
                roi_unit,
            )
        )

    return rows, skipped


def upsert_scores(rows: list[tuple[object, ...]]) -> int:
    if not rows:
        return 0

    query = """
    INSERT INTO prediction_scores (
        prediction_id,
        actual,
        brier,
        log_loss,
        hit,
        odds_used,
        edge,
        roi_unit,
        scored_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
    ON CONFLICT (prediction_id)
    DO UPDATE SET
        actual = EXCLUDED.actual,
        brier = EXCLUDED.brier,
        log_loss = EXCLUDED.log_loss,
        hit = EXCLUDED.hit,
        odds_used = EXCLUDED.odds_used,
        edge = EXCLUDED.edge,
        roi_unit = EXCLUDED.roi_unit,
        scored_at = NOW();
    """

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.executemany(query, rows)
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def main() -> None:
    args = parse_args()
    candidates = fetch_unscored_predictions(
        league=args.league,
        limit=args.limit,
        since_days=args.since_days,
    )

    if not candidates:
        print("No eligible unscored predictions found.")
        return

    rows, skipped = build_score_rows(candidates)
    written = upsert_scores(rows)

    print(
        f"Scanned {len(candidates)} predictions, scored {written}, "
        f"skipped {skipped} (missing required settlement inputs)."
    )


if __name__ == "__main__":
    main()
