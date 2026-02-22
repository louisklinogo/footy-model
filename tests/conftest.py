from __future__ import annotations

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportInvalidTypeForm=false, reportReturnType=false, reportUnusedParameter=false

import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db, get_database_url


@dataclass
class TestDbCase:
    conn: object
    league_code: str
    suffix: str
    created_files: list[Path] = field(default_factory=list)


def _cleanup_test_rows(conn: object, league_code: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT fixture_id FROM fixtures WHERE league_code = %s", (league_code,))
        fixture_ids = [int(row[0]) for row in cur.fetchall()]

        if fixture_ids:
            cur.execute(
                """
                DELETE FROM prediction_scores
                WHERE prediction_id IN (
                    SELECT prediction_id FROM predictions WHERE fixture_id = ANY(%s)
                )
                """,
                (fixture_ids,),
            )
            cur.execute("DELETE FROM predictions WHERE fixture_id = ANY(%s)", (fixture_ids,))
            cur.execute("DELETE FROM team_premium_snapshots WHERE fixture_id = ANY(%s)", (fixture_ids,))
            cur.execute("DELETE FROM fixture_stats_premium WHERE fixture_id = ANY(%s)", (fixture_ids,))
            cur.execute("DELETE FROM fixture_odds_snapshots WHERE fixture_id = ANY(%s)", (fixture_ids,))
            cur.execute("DELETE FROM fixture_results WHERE fixture_id = ANY(%s)", (fixture_ids,))
            cur.execute("DELETE FROM fixtures WHERE fixture_id = ANY(%s)", (fixture_ids,))

        cur.execute("SELECT team_id FROM teams WHERE league_code = %s", (league_code,))
        team_ids = [int(row[0]) for row in cur.fetchall()]
        if team_ids:
            cur.execute("DELETE FROM team_aliases WHERE team_id = ANY(%s)", (team_ids,))

        cur.execute("DELETE FROM teams WHERE league_code = %s", (league_code,))
        cur.execute("DELETE FROM leagues WHERE league_code = %s", (league_code,))
        cur.execute(
            """
            DELETE FROM pipeline_runs
            WHERE details_json ->> 'league' = %s
               OR details_json ->> 'league_code' = %s
            """,
            (league_code, league_code),
        )

    conn.commit()


@pytest.fixture
def db_case() -> TestDbCase:
    try:
        _ = get_database_url()
    except RuntimeError:
        pytest.skip("Skipping DB regression tests: DATABASE_URL/DEV_DATABASE_URL not configured")

    conn = connect_db()
    suffix = uuid.uuid4().hex[:10]
    case = TestDbCase(conn=conn, league_code=f"TST_{suffix}", suffix=suffix)

    try:
        yield case
    finally:
        for path in case.created_files:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        _cleanup_test_rows(conn, case.league_code)
        conn.close()


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )


def insert_league(cur: object, league_code: str) -> None:
    cur.execute(
        """
        INSERT INTO leagues (league_code, league_name, country)
        VALUES (%s, %s, %s)
        ON CONFLICT (league_code) DO UPDATE
        SET league_name = EXCLUDED.league_name,
            country = EXCLUDED.country,
            updated_at = NOW()
        """,
        (league_code, f"League {league_code}", "Testland"),
    )


def insert_team(cur: object, league_code: str, team_name: str) -> int:
    cur.execute(
        """
        INSERT INTO teams (team_name, league_code)
        VALUES (%s, %s)
        ON CONFLICT (league_code, team_name) DO UPDATE
        SET updated_at = NOW()
        RETURNING team_id
        """,
        (team_name, league_code),
    )
    return int(cur.fetchone()[0])


def insert_fixture(
    cur: object,
    league_code: str,
    flashscore_id: str,
    home_team_id: int,
    away_team_id: int,
    kickoff: datetime,
    status: str,
) -> int:
    cur.execute(
        """
        INSERT INTO fixtures (
            flashscore_id,
            league_code,
            home_team_id,
            away_team_id,
            match_datetime_utc,
            status,
            flashscore_url
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING fixture_id
        """,
        (
            flashscore_id,
            league_code,
            home_team_id,
            away_team_id,
            kickoff,
            status,
            f"https://www.flashscore.com/match/{flashscore_id}/",
        ),
    )
    return int(cur.fetchone()[0])


def insert_premium_stats(cur: object, fixture_id: int, h_xg: float, a_xg: float) -> None:
    cur.execute(
        """
        INSERT INTO fixture_stats_premium (
            fixture_id,
            h_xg,
            a_xg,
            h_xgot,
            a_xgot,
            h_xa,
            a_xa,
            h_box_touches,
            a_box_touches,
            h_big_chances,
            a_big_chances,
            h_crosses,
            a_crosses,
            h_sot,
            a_sot,
            h_corners,
            a_corners,
            h_goals_prevented,
            a_goals_prevented,
            fidelity_score,
            raw_json
        )
        VALUES (
            %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            1.0,
            '{}'::jsonb
        )
        ON CONFLICT (fixture_id) DO UPDATE
        SET h_xg = EXCLUDED.h_xg,
            a_xg = EXCLUDED.a_xg,
            h_xgot = EXCLUDED.h_xgot,
            a_xgot = EXCLUDED.a_xgot,
            h_xa = EXCLUDED.h_xa,
            a_xa = EXCLUDED.a_xa,
            h_box_touches = EXCLUDED.h_box_touches,
            a_box_touches = EXCLUDED.a_box_touches,
            h_big_chances = EXCLUDED.h_big_chances,
            a_big_chances = EXCLUDED.a_big_chances,
            h_crosses = EXCLUDED.h_crosses,
            a_crosses = EXCLUDED.a_crosses,
            h_sot = EXCLUDED.h_sot,
            a_sot = EXCLUDED.a_sot,
            h_corners = EXCLUDED.h_corners,
            a_corners = EXCLUDED.a_corners,
            h_goals_prevented = EXCLUDED.h_goals_prevented,
            a_goals_prevented = EXCLUDED.a_goals_prevented,
            fidelity_score = EXCLUDED.fidelity_score
        """,
        (
            fixture_id,
            h_xg,
            a_xg,
            h_xg,
            a_xg,
            h_xg,
            a_xg,
            int(h_xg * 10),
            int(a_xg * 10),
            int(h_xg * 3),
            int(a_xg * 3),
            int(h_xg * 5),
            int(a_xg * 5),
            int(h_xg * 2),
            int(a_xg * 2),
            int(h_xg * 2),
            int(a_xg * 2),
            h_xg / 10.0,
            a_xg / 10.0,
        ),
    )


def insert_odds_snapshot(cur: object, fixture_id: int, snapshot_time: datetime) -> None:
    cur.execute(
        """
        INSERT INTO fixture_odds_snapshots (
            fixture_id,
            snapshot_time_utc,
            snapshot_type,
            ou_json,
            ah_json,
            one_x_two_json
        )
        VALUES (
            %s,
            %s,
            'latest_pre_match',
            %s::jsonb,
            '{}'::jsonb,
            '{}'::jsonb
        )
        ON CONFLICT (fixture_id, snapshot_time_utc, snapshot_type) DO NOTHING
        """,
        (
            fixture_id,
            snapshot_time,
            '{"1.5": {"over": 1.85, "under": 1.95}, "2.5": {"over": 2.05, "under": 1.75}}',
        ),
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def days_from_now(days: int) -> datetime:
    return utc_now() + timedelta(days=days)
