"""
Score settled fixtures-first v3 predictions into prediction_scores.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


MODEL_NAME = "fixtures_first_gbm"
MODEL_VERSION = "v3"
MARKETS = ("o15", "o25", "c85")
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score settled fixtures-first v3 predictions")
    parser.add_argument("--league", type=str, default=None, help="Optional league_code filter")
    parser.add_argument("--limit", type=int, default=None, help="Optional max predictions to score")
    parser.add_argument(
        "--since-days",
        type=int,
        default=30,
        help="Only include fixtures within this many days back",
    )
    return parser.parse_args()


def _latest_over_odds_expr(line: str) -> str:
    return (
        "CASE "
        f"WHEN (od.ou_json -> '{line}' ->> 'over') ~ '^[-+]?[0-9]*\\.?[0-9]+$' "
        f"THEN (od.ou_json -> '{line}' ->> 'over')::double precision "
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
        CASE
            WHEN p.market_code = 'o15' THEN {_latest_over_odds_expr('1.5')}
            WHEN p.market_code = 'o25' THEN {_latest_over_odds_expr('2.5')}
            ELSE NULL
        END AS odds_used
    FROM predictions p
    JOIN fixtures f
      ON f.fixture_id = p.fixture_id
    JOIN fixture_results fr
      ON fr.fixture_id = f.fixture_id
    LEFT JOIN fixture_stats_premium fs
      ON fs.fixture_id = f.fixture_id
    LEFT JOIN LATERAL (
        SELECT fos.ou_json
        FROM fixture_odds_snapshots fos
        WHERE fos.fixture_id = f.fixture_id
          AND fos.snapshot_time_utc <= f.match_datetime_utc
        ORDER BY fos.snapshot_time_utc DESC
        LIMIT 1
    ) od ON true
    LEFT JOIN prediction_scores ps
      ON ps.prediction_id = p.prediction_id
    WHERE p.model_name = %s
      AND p.model_version = %s
      AND p.market_code IN ('o15', 'o25', 'c85')
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

    total_goals = int(home_goals) + int(away_goals)

    if market == "o15":
        return 1.0 if total_goals >= 2 else 0.0
    if market == "o25":
        return 1.0 if total_goals >= 3 else 0.0
    if market == "c85":
        h_corners = row.get("h_corners")
        a_corners = row.get("a_corners")
        if h_corners is None or a_corners is None:
            return None
        return 1.0 if (int(h_corners) + int(a_corners)) >= 9 else 0.0
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
