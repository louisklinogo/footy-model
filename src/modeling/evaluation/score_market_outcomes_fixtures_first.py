"""
Score settled market outcome predictions into prediction_scores.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.settlement import settle_market
from src.betting.odds_resolver import OddsResolution, resolve_odds_for_market
from src.db.db_utils import connect_db
from src.modeling.v2.families.scoreline.derive_markets import (
    MULTIGOALS_AWAY_RANGES,
    MULTIGOALS_HOME_RANGES,
    MULTIGOALS_TOTAL_RANGES,
    MULTISCORE_GROUPS,
)


MODEL_NAME = "market_outcome_gbm"
MODEL_VERSION = "fixtures_first_prematch_v1"
CORNERS_MARKETS = (
    "c75",
    "c85",
    "c95",
    "c105",
    "hc25",
    "hc35",
    "hc45",
    "hc55",
    "ac25",
    "ac35",
    "ac45",
    "ac55",
)
ANYTIME_MARKETS = (
    "h_1up",
    "a_1up",
    "h_2up",
    "a_2up",
)
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
SCORELINE_MULTIGOALS_TOTAL_MARKETS = tuple(MULTIGOALS_TOTAL_RANGES.keys())
SCORELINE_MULTIGOALS_HOME_MARKETS = tuple(MULTIGOALS_HOME_RANGES.keys())
SCORELINE_MULTIGOALS_AWAY_MARKETS = tuple(MULTIGOALS_AWAY_RANGES.keys())
SCORELINE_MULTISCORE_MARKETS = (
    *tuple(MULTISCORE_GROUPS.keys()),
    "ms_draw",
    "ms_other_homewin",
    "ms_other_awaywin",
)
_MULTISCORE_HOME_GROUP_SCORES = {
    score
    for scores in MULTISCORE_GROUPS.values()
    for score in scores
    if score[0] > score[1]
}
_MULTISCORE_AWAY_GROUP_SCORES = {
    score
    for scores in MULTISCORE_GROUPS.values()
    for score in scores
    if score[0] < score[1]
}
MARKETS = (
    # Goals
    "o15",
    "u35",
    # Corners Totals
    "c75",
    "c85",
    "c95",
    "c105",
    # Corners Home Team
    "hc25",
    "hc35",
    "hc45",
    "hc55",
    # Corners Away Team
    "ac25",
    "ac35",
    "ac45",
    "ac55",
    # 1X2
    "1x2_h",
    "1x2_d",
    "1x2_a",
    # Double Chance
    "dc_1x",
    "dc_x2",
    "dc_12",
    # Team Totals
    "ho15",
    "ao15",
    # Scoreline multigoals / multiscore
    *SCORELINE_MULTIGOALS_TOTAL_MARKETS,
    *SCORELINE_MULTIGOALS_HOME_MARKETS,
    *SCORELINE_MULTIGOALS_AWAY_MARKETS,
    *SCORELINE_MULTISCORE_MARKETS,
    # Legacy Handicap Markets
    *LEGACY_HANDICAP_MARKETS,
    # Canonical Handicap Markets
    *CANONICAL_HANDICAP_MARKETS,
    # Anytime Lead Markets
    "h_1up",
    "a_1up",
    "h_2up",
    "a_2up",
)
ODDS_MARKET_CODES = (
    "1x2",
    "dc",
    "ou",
    "home_ou",
    "away_ou",
    "corners_ou",
    "home_corners_ou",
    "corners_home_ou",
    "team_corners_home_ou",
    "away_corners_ou",
    "corners_away_ou",
    "team_corners_away_ou",
    "ah",
    "eh",
)
EPS = 1e-6


def _is_handicap_market_code(market: str) -> bool:
    return market.startswith(("ah_", "eh_", "ah2_", "eh3_"))


def _settle_handicap_for_scoring(market: str, home_goals: int, away_goals: int):
    return settle_market(
        market,
        odds=2.0,
        home_goals=home_goals,
        away_goals=away_goals,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score settled market outcome predictions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Model name to score from predictions table",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=MODEL_VERSION,
        help="Model version to score from predictions table",
    )
    parser.add_argument(
        "--league", type=str, default=None, help="Optional league_code filter"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional max predictions to score"
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=30,
        help="Only include fixtures within this many days back",
    )
    return parser.parse_args()


def ensure_prediction_scores_schema() -> None:
    query = """
    ALTER TABLE prediction_scores
    ADD COLUMN IF NOT EXISTS metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb;
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()
    finally:
        conn.close()


def fetch_unscored_predictions(
    model_name: str,
    model_version: str,
    league: str | None,
    limit: int | None,
    since_days: int | None,
) -> list[dict[str, object]]:
    query = """
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
        od.odds_rows
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
        SELECT COALESCE(
            jsonb_agg(
                jsonb_build_object(
                    'provider', fom.provider,
                    'snapshot_type', fom.snapshot_type,
                    'snapshot_time_utc', fom.snapshot_time_utc,
                    'market_code', fom.market_code,
                    'line_num', fom.line_num,
                    'odds_json', fom.odds_json
                )
                ORDER BY fom.snapshot_time_utc DESC
            ),
            '[]'::jsonb
        ) AS odds_rows
        FROM fixture_odds_markets fom
        WHERE fom.fixture_id = f.fixture_id
          AND fom.market_code = ANY(%s)
          AND fom.snapshot_time_utc <= f.match_datetime_utc
    ) od ON true
    LEFT JOIN prediction_scores ps
      ON ps.prediction_id = p.prediction_id
    WHERE p.model_name = %s
      AND p.model_version = %s
      AND p.market_code = ANY(%s)
      AND f.status = 'ft'
      AND f.status NOT IN ('postponed', 'cancelled', 'abandoned')
      AND (
          ps.prediction_id IS NULL
          OR (
              ps.odds_used IS NULL
              AND od.odds_rows <> '[]'::jsonb
          )
      )
      AND (
          (
              p.market_code = ANY(%s)
              AND fs.h_corners IS NOT NULL
              AND fs.a_corners IS NOT NULL
          )
          OR (
              p.market_code = ANY(%s)
              AND (
                  ils.home_led_by_1_any IS NOT NULL
                  OR ils.away_led_by_1_any IS NOT NULL
                  OR ils.home_led_by_2_any IS NOT NULL
                  OR ils.away_led_by_2_any IS NOT NULL
              )
          )
          OR (
              p.market_code <> ALL(%s)
              AND p.market_code <> ALL(%s)
              AND fr.home_goals IS NOT NULL
              AND fr.away_goals IS NOT NULL
          )
      )
    """
    params: list[object] = [
        list(ODDS_MARKET_CODES),
        model_name,
        model_version,
        list(MARKETS),
        list(CORNERS_MARKETS),
        list(ANYTIME_MARKETS),
        list(CORNERS_MARKETS),
        list(ANYTIME_MARKETS),
    ]

    if since_days is not None:
        query += " AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval"
        params.append(since_days)

    if league:
        query += " AND f.league_code = %s"
        params.append(league)

    query += """
     ORDER BY
        CASE
            WHEN ps.prediction_id IS NOT NULL AND ps.odds_used IS NULL THEN 0
            ELSE 1
        END ASC,
        f.match_datetime_utc DESC,
        p.prediction_id ASC
    """

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
    if market == "u35":
        return 1.0 if total_goals <= 3 else 0.0

    # === CORNERS TOTALS ===
    if market in {"c75", "c85", "c95", "c105"}:
        h_corners = row.get("h_corners")
        a_corners = row.get("a_corners")
        if h_corners is None or a_corners is None:
            return None
        total_corners = int(h_corners) + int(a_corners)
        threshold = {
            "c75": 8,
            "c85": 9,
            "c95": 10,
            "c105": 11,
        }[market]
        return 1.0 if total_corners >= threshold else 0.0

    # === CORNERS TEAM TOTALS ===
    if market in {"hc25", "hc35", "hc45", "hc55"}:
        h_corners = row.get("h_corners")
        if h_corners is None:
            return None
        threshold = {
            "hc25": 3,
            "hc35": 4,
            "hc45": 5,
            "hc55": 6,
        }[market]
        return 1.0 if int(h_corners) >= threshold else 0.0

    if market in {"ac25", "ac35", "ac45", "ac55"}:
        a_corners = row.get("a_corners")
        if a_corners is None:
            return None
        threshold = {
            "ac25": 3,
            "ac35": 4,
            "ac45": 5,
            "ac55": 6,
        }[market]
        return 1.0 if int(a_corners) >= threshold else 0.0

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

    # === SCORELINE MULTIGOALS / MULTISCORE ===
    if market in MULTIGOALS_TOTAL_RANGES:
        low, high = MULTIGOALS_TOTAL_RANGES[market]
        return 1.0 if _is_in_range(total_goals, low=low, high=high) else 0.0
    if market in MULTIGOALS_HOME_RANGES:
        low, high = MULTIGOALS_HOME_RANGES[market]
        return 1.0 if _is_in_range(h, low=low, high=high) else 0.0
    if market in MULTIGOALS_AWAY_RANGES:
        low, high = MULTIGOALS_AWAY_RANGES[market]
        return 1.0 if _is_in_range(a, low=low, high=high) else 0.0
    if market in MULTISCORE_GROUPS:
        return 1.0 if (h, a) in MULTISCORE_GROUPS[market] else 0.0
    if market == "ms_draw":
        return 1.0 if h == a else 0.0
    if market == "ms_other_homewin":
        return 1.0 if h > a and (h, a) not in _MULTISCORE_HOME_GROUP_SCORES else 0.0
    if market == "ms_other_awaywin":
        return 1.0 if h < a and (h, a) not in _MULTISCORE_AWAY_GROUP_SCORES else 0.0

    # === HANDICAP MARKETS ===
    if _is_handicap_market_code(market):
        try:
            return _settle_handicap_for_scoring(market, h, a).actual
        except ValueError:
            pass

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


def compute_return_factor(
    actual: float | None,
    odds_used: float | None,
    is_push: bool,
) -> float | None:
    if odds_used is None:
        return None
    if is_push:
        return 1.0
    if actual is None:
        return None
    return odds_used if actual == 1.0 else 0.0


def is_push_outcome(row: dict[str, object]) -> bool:
    market = str(row["market_code"])
    if not _is_handicap_market_code(market):
        return False

    home_goals = row.get("home_goals")
    away_goals = row.get("away_goals")
    if home_goals is None or away_goals is None:
        return False

    try:
        result = _settle_handicap_for_scoring(market, int(home_goals), int(away_goals))
    except ValueError:
        return False
    return result.outcome == "push"


def _is_in_range(value: int, *, low: int, high: int | None) -> bool:
    if high is None:
        return value >= low
    return low <= value <= high


def _isoformat_or_none(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _parse_odds_rows(raw_rows: object) -> list[dict[str, object]]:
    if isinstance(raw_rows, list):
        return [item for item in raw_rows if isinstance(item, dict)]
    if isinstance(raw_rows, str):
        try:
            parsed = json.loads(raw_rows)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _build_metadata_json(resolution: OddsResolution) -> dict[str, object]:
    return {
        "odds_trace": {
            "provider": resolution.provider,
            "snapshot_type": resolution.snapshot_type,
            "snapshot_time_utc": _isoformat_or_none(resolution.snapshot_time_utc),
            "line_num": resolution.line_num,
            "odds_field": resolution.odds_field,
            "fallback_used": resolution.fallback_used,
        }
    }


def build_score_rows(
    candidates: list[dict[str, object]],
) -> tuple[list[tuple[object, ...]], int]:
    rows: list[tuple[object, ...]] = []
    skipped = 0

    for candidate in candidates:
        actual = compute_actual(candidate)
        is_push = is_push_outcome(candidate)
        if actual is None and not is_push:
            skipped += 1
            continue

        p_raw = float(candidate["p_model"])
        p_clipped = min(max(p_raw, EPS), 1.0 - EPS)
        brier: float | None
        log_loss: float | None
        hit: bool | None
        if actual in (0.0, 1.0):
            brier = (p_clipped - actual) ** 2
            log_loss = -(
                actual * math.log(p_clipped)
                + (1.0 - actual) * math.log(1.0 - p_clipped)
            )
            hit = (p_clipped >= 0.5) == (actual == 1.0)
        else:
            brier = None
            log_loss = None
            hit = None

        odds_rows = _parse_odds_rows(candidate.get("odds_rows"))
        resolution = resolve_odds_for_market(
            fixture_kickoff_utc=candidate["match_datetime_utc"],
            odds_rows=odds_rows,
            market_code=str(candidate["market_code"]),
        )

        odds_used: float | None = None
        edge: float | None = None
        return_factor: float | None = None
        roi_unit: float | None = None

        if resolution.odds_used is not None:
            parsed_odds = float(resolution.odds_used)
            if parsed_odds > 0.0:
                odds_used = parsed_odds
                if odds_used > 1.0:
                    edge = p_clipped - (1.0 / odds_used)
                return_factor = compute_return_factor(actual, odds_used, is_push)
                if return_factor is not None:
                    roi_unit = return_factor - 1.0

        metadata_json = json.dumps(_build_metadata_json(resolution))

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
                metadata_json,
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
        metadata_json,
        scored_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
    ON CONFLICT (prediction_id)
    DO UPDATE SET
        actual = EXCLUDED.actual,
        brier = EXCLUDED.brier,
        log_loss = EXCLUDED.log_loss,
        hit = EXCLUDED.hit,
        odds_used = EXCLUDED.odds_used,
        edge = EXCLUDED.edge,
        roi_unit = EXCLUDED.roi_unit,
        metadata_json = EXCLUDED.metadata_json,
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
    ensure_prediction_scores_schema()
    candidates = fetch_unscored_predictions(
        model_name=str(args.model),
        model_version=str(args.version),
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
