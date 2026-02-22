from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from psycopg2.extensions import cursor as PsycopgCursor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.common.pipeline_logging import create_pipeline_run, finalize_pipeline_run


JOB_NAME = "build_team_premium_snapshots_v1"
WINDOW_SIZE = 10
EWMA_ALPHA = 2 / (WINDOW_SIZE + 1)  # Standard alpha for span=10

METRICS = (
    "xg",
    "xgot",
    "xa",
    "box_touches",
    "big_chances",
    "crosses",
    "sot",
    "corners",
    "goals_prevented",
)


@dataclass(frozen=True)
class FixtureRow:
    fixture_id: int
    league_code: str | None
    home_team_id: int | None
    away_team_id: int | None
    match_datetime_utc: datetime
    status: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-free team premium snapshots with EWMA and Rest Days"
    )
    _ = parser.add_argument("--league", help="Optional league_code filter")
    _ = parser.add_argument("--limit", type=int, help="Optional fixture processing limit")
    return parser.parse_args()


def _ewma(values: Sequence[float | None], alpha: float) -> float | None:
    """Calculate Exponentially Weighted Moving Average for the given values."""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    
    # We want to give higher weight to more RECENT values (end of list)
    # Recursion: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
    weighted_mean = filtered[0]
    for i in range(1, len(filtered)):
        weighted_mean = alpha * filtered[i] + (1 - alpha) * weighted_mean
    return float(weighted_mean)


def _fetch_fixtures(cur: PsycopgCursor, league: str | None, limit: int | None) -> list[FixtureRow]:
    where_sql = ["f.match_datetime_utc IS NOT NULL", "f.home_team_id IS NOT NULL", "f.away_team_id IS NOT NULL"]
    params: list[object] = []

    if league:
        where_sql.append("f.league_code = %s")
        params.append(league)

    limit_sql = ""
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        limit_sql = " LIMIT %s"
        params.append(limit)

    cur.execute(
        f"""
        SELECT
            f.fixture_id,
            f.league_code,
            f.home_team_id,
            f.away_team_id,
            f.match_datetime_utc,
            f.status
        FROM fixtures f
        WHERE {' AND '.join(where_sql)}
        ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC
        {limit_sql}
        """,
        params,
    )

    rows: list[FixtureRow] = []
    for raw_row in cast(list[tuple[object, ...]], cur.fetchall()):
        fixture_id = cast(int, raw_row[0])
        league_code = cast(str | None, raw_row[1])
        home_team_id = cast(int | None, raw_row[2])
        away_team_id = cast(int | None, raw_row[3])
        kickoff = raw_row[4]
        status = raw_row[5]
        if not isinstance(kickoff, datetime):
            continue
        rows.append(
            FixtureRow(
                fixture_id=fixture_id,
                league_code=league_code,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                match_datetime_utc=kickoff,
                status=str(status) if status is not None else "",
            )
        )
    return rows


def _fetch_premium_stats(
    cur: PsycopgCursor, fixture_ids: Sequence[int]
) -> dict[int, dict[str, float | None]]:
    if not fixture_ids:
        return {}

    cur.execute(
        """
        SELECT
            p.fixture_id,
            p.h_xg, p.a_xg,
            p.h_xgot, p.a_xgot,
            p.h_xa, p.a_xa,
            p.h_box_touches, p.a_box_touches,
            p.h_big_chances, p.a_big_chances,
            p.h_crosses, p.a_crosses,
            p.h_sot, p.a_sot,
            p.h_corners, p.a_corners,
            p.h_goals_prevented, p.a_goals_prevented,
            p.fidelity_score
        FROM fixture_stats_premium p
        WHERE p.fixture_id = ANY(%s)
        """,
        (list(fixture_ids),),
    )

    output: dict[int, dict[str, float | None]] = {}
    for row in cast(list[tuple[object, ...]], cur.fetchall()):
        output[int(cast(int, row[0]))] = {
            "h_xg": cast(float | None, row[1]),
            "a_xg": cast(float | None, row[2]),
            "h_xgot": cast(float | None, row[3]),
            "a_xgot": cast(float | None, row[4]),
            "h_xa": cast(float | None, row[5]),
            "a_xa": cast(float | None, row[6]),
            "h_box_touches": cast(float | None, row[7]),
            "a_box_touches": cast(float | None, row[8]),
            "h_big_chances": cast(float | None, row[9]),
            "a_big_chances": cast(float | None, row[10]),
            "h_crosses": cast(float | None, row[11]),
            "a_crosses": cast(float | None, row[12]),
            "h_sot": cast(float | None, row[13]),
            "a_sot": cast(float | None, row[14]),
            "h_corners": cast(float | None, row[15]),
            "a_corners": cast(float | None, row[16]),
            "h_goals_prevented": cast(float | None, row[17]),
            "a_goals_prevented": cast(float | None, row[18]),
            "fidelity_score": cast(float | None, row[19]),
        }
    return output


def _empty_team_history() -> dict[str, list[float | None] | datetime | None]:
    out: dict[str, list[float | None] | datetime | None] = {
        **{f"{metric}_for": [] for metric in METRICS},
        **{f"{metric}_against": [] for metric in METRICS},
        "rest_days": [],
        "fidelity": [],
        "last_kickoff": None,
    }
    return out


def _team_metric_values(
    premium_row: dict[str, float | None],
    metric: str,
    is_home_side: bool,
) -> tuple[float | None, float | None]:
    if is_home_side:
        return premium_row.get(f"h_{metric}"), premium_row.get(f"a_{metric}")
    return premium_row.get(f"a_{metric}"), premium_row.get(f"h_{metric}")


def build_team_premium_snapshots(league: str | None = None, limit: int | None = None) -> dict[str, int]:
    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                print(f"Fetching fixtures for league: {league or 'ALL'}")
                fixtures = _fetch_fixtures(cur, league=league, limit=limit)
                fixture_ids = [fixture.fixture_id for fixture in fixtures]
                
                print(f"Fetching premium stats for {len(fixture_ids)} fixtures...")
                premium_by_fixture = _fetch_premium_stats(cur, fixture_ids)

                team_history: dict[tuple[str | None, int], dict[str, any]] = defaultdict(
                    _empty_team_history
                )

                upsert_rows: list[tuple[object, ...]] = []
                processed_fixtures = 0
                index = 0
                
                print(f"Building snapshots for {len(fixtures)} fixtures...")
                while index < len(fixtures):
                    if processed_fixtures % 50 == 0 and processed_fixtures > 0:
                        print(f"  Progress: {processed_fixtures}/{len(fixtures)} fixtures...")
                    
                    kickoff = fixtures[index].match_datetime_utc
                    same_kickoff: list[FixtureRow] = []
                    while index < len(fixtures) and fixtures[index].match_datetime_utc == kickoff:
                        same_kickoff.append(fixtures[index])
                        index += 1

                    for fixture in same_kickoff:
                        processed_fixtures += 1

                        for team_id, is_home in (
                            (fixture.home_team_id, True),
                            (fixture.away_team_id, False),
                        ):
                            if team_id is None:
                                continue

                            # Identify the opponent
                            opponent_id = fixture.away_team_id if is_home else fixture.home_team_id

                            history_key = (fixture.league_code, team_id)
                            history = team_history[history_key]

                            # 1. Calculate Rest Days (P0 proxy)
                            current_rest = None
                            if history["last_kickoff"]:
                                delta = fixture.match_datetime_utc - history["last_kickoff"]
                                current_rest = min(21.0, delta.total_seconds() / 86400.0)
                            
                            rolling: dict[str, float | None] = {}
                            # 2. EWMA for metrics
                            for metric in METRICS:
                                rolling[f"rolling_{metric}"] = _ewma(
                                    history[f"{metric}_for"][-WINDOW_SIZE:], EWMA_ALPHA
                                )
                                rolling[f"rolling_{metric}_against"] = _ewma(
                                    history[f"{metric}_against"][-WINDOW_SIZE:], EWMA_ALPHA
                                )
                            
                            rolling_rest = _ewma(history["rest_days"][-WINDOW_SIZE:], EWMA_ALPHA)
                            fidelity_score = _ewma(history["fidelity"][-WINDOW_SIZE:], EWMA_ALPHA)
                            
                            # 3. Opponent rest days (from opponent's history)
                            opp_rolling_rest = None
                            if opponent_id is not None:
                                opp_key = (fixture.league_code, opponent_id)
                                opp_history = team_history[opp_key]
                                opp_rolling_rest = _ewma(
                                    opp_history["rest_days"][-WINDOW_SIZE:], EWMA_ALPHA
                                )
                            
                            # We keep simple sample size for gating
                            sample_size = len([v for v in history["xg_for"] if v is not None])

                            upsert_rows.append(
                                (
                                    fixture.fixture_id,
                                    team_id,
                                    is_home,
                                    sample_size,
                                    rolling["rolling_xg"],
                                    rolling["rolling_xg_against"],
                                    rolling["rolling_xgot"],
                                    rolling["rolling_xgot_against"],
                                    rolling["rolling_xa"],
                                    rolling["rolling_xa_against"],
                                    rolling["rolling_box_touches"],
                                    rolling["rolling_box_touches_against"],
                                    rolling["rolling_big_chances"],
                                    rolling["rolling_big_chances_against"],
                                    rolling["rolling_crosses"],
                                    rolling["rolling_crosses_against"],
                                    rolling["rolling_sot"],
                                    rolling["rolling_sot_against"],
                                    rolling["rolling_corners"],
                                    rolling["rolling_corners_against"],
                                    rolling["rolling_goals_prevented"],
                                    rolling["rolling_goals_prevented_against"],
                                    fidelity_score,
                                    rolling_rest,
                                    opp_rolling_rest,
                                )
                            )
                            
                            # Update rest history BEFORE moving to the 'settling' loop
                            # Note: we append current_rest to history *after* the snapshot calculation
                            # to ensure point-in-time (snapshot is as-of BEFORE this kickoff)
                            if current_rest is not None:
                                history["rest_days"].append(current_rest)
                            history["last_kickoff"] = fixture.match_datetime_utc

                    # Settle finished fixtures into history for future rounds
                    for fixture in same_kickoff:
                        if fixture.status != "ft":
                            continue
                        premium = premium_by_fixture.get(fixture.fixture_id)
                        if premium is None:
                            continue

                        for team_id, is_home in (
                            (fixture.home_team_id, True),
                            (fixture.away_team_id, False),
                        ):
                            if team_id is None:
                                continue

                            history_key = (fixture.league_code, team_id)
                            history = team_history[history_key]
                            for metric in METRICS:
                                value_for, value_against = _team_metric_values(
                                    premium_row=premium,
                                    metric=metric,
                                    is_home_side=is_home,
                                )
                                history[f"{metric}_for"].append(value_for)
                                history[f"{metric}_against"].append(value_against)
                            history["fidelity"].append(premium.get("fidelity_score"))

                print(f"Batch upserting {len(upsert_rows)} snapshot rows...")
                cur.executemany(
                    """
                    INSERT INTO team_premium_snapshots (
                        fixture_id, team_id, is_home, sample_size,
                        rolling_xg, rolling_xg_against,
                        rolling_xgot, rolling_xgot_against,
                        rolling_xa, rolling_xa_against,
                        rolling_box_touches, rolling_box_touches_against,
                        rolling_big_chances, rolling_big_chances_against,
                        rolling_crosses, rolling_crosses_against,
                        rolling_sot, rolling_sot_against,
                        rolling_corners, rolling_corners_against,
                        rolling_goals_prevented, rolling_goals_prevented_against,
                        fidelity_score, rolling_rest_days, rolling_rest_days_against, built_at
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, NOW()
                    )
                    ON CONFLICT (fixture_id, team_id, is_home) DO UPDATE
                    SET
                        sample_size = EXCLUDED.sample_size,
                        rolling_xg = EXCLUDED.rolling_xg,
                        rolling_xg_against = EXCLUDED.rolling_xg_against,
                        rolling_xgot = EXCLUDED.rolling_xgot,
                        rolling_xgot_against = EXCLUDED.rolling_xgot_against,
                        rolling_xa = EXCLUDED.rolling_xa,
                        rolling_xa_against = EXCLUDED.rolling_xa_against,
                        rolling_box_touches = EXCLUDED.rolling_box_touches,
                        rolling_box_touches_against = EXCLUDED.rolling_box_touches_against,
                        rolling_big_chances = EXCLUDED.rolling_big_chances,
                        rolling_big_chances_against = EXCLUDED.rolling_big_chances_against,
                        rolling_crosses = EXCLUDED.rolling_crosses,
                        rolling_crosses_against = EXCLUDED.rolling_crosses_against,
                        rolling_sot = EXCLUDED.rolling_sot,
                        rolling_sot_against = EXCLUDED.rolling_sot_against,
                        rolling_corners = EXCLUDED.rolling_corners,
                        rolling_corners_against = EXCLUDED.rolling_corners_against,
                        rolling_goals_prevented = EXCLUDED.rolling_goals_prevented,
                        rolling_goals_prevented_against = EXCLUDED.rolling_goals_prevented_against,
                        fidelity_score = EXCLUDED.fidelity_score,
                        rolling_rest_days = EXCLUDED.rolling_rest_days,
                        rolling_rest_days_against = EXCLUDED.rolling_rest_days_against,
                        built_at = NOW()
                    """,
                    upsert_rows,
                )

                return {
                    "fixtures_processed": processed_fixtures,
                    "rows_upserted": len(upsert_rows),
                    "premium_rows_seen": len(premium_by_fixture),
                }
    finally:
        conn.close()


def main() -> int:
    args = _parse_args()
    league = args.league
    limit = args.limit
    run_details = {"league": league, "limit": limit, "window_size": WINDOW_SIZE, "method": "EWMA"}
    run_id = create_pipeline_run(
        job_name=JOB_NAME,
        message="Building high-fidelity team premium snapshots (EWMA + Rest Days)",
        details_json=run_details,
    )

    try:
        result = build_team_premium_snapshots(league=league, limit=limit)
        finalize_pipeline_run(
            run_id=run_id,
            status="success",
            message=f"Built EWMA snapshots for {result['rows_upserted']} team rows",
            details_json={**run_details, **result},
        )
        print(f"COMPLETED: {json.dumps(result, indent=2)}")
        return 0
    except Exception as exc:
        finalize_pipeline_run(
            run_id=run_id,
            status="fail",
            message=f"Snapshot build failed: {exc}",
            details_json=run_details,
        )
        print(f"FAILED: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
