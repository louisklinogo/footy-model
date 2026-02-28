from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psycopg2.extras import execute_batch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixture-level lead-state features from SofaScore incidents."
    )
    parser.add_argument("--league", type=str, default=None, help="Optional league filter.")
    parser.add_argument(
        "--status",
        type=str,
        default="ft",
        choices=["scheduled", "ft", "all"],
        help="Fixture status filter (default: ft).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max fixtures to process (0 means no limit).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute only, do not write DB.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/market_reconciliation"),
        help="Output directory for quality report artifacts.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="incident_timeline_quality",
        help="Output artifact stem.",
    )
    parser.add_argument(
        "--fixture-ids-file",
        type=str,
        default=None,
        help="Optional CSV/TXT file with fixture_id values (first column).",
    )
    return parser.parse_args()


def ensure_schema() -> None:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS fixture_incident_lead_states (
                    fixture_id BIGINT PRIMARY KEY REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
                    league_code TEXT,
                    fixture_status TEXT,
                    total_goal_events SMALLINT NOT NULL DEFAULT 0,
                    home_goal_events SMALLINT NOT NULL DEFAULT 0,
                    away_goal_events SMALLINT NOT NULL DEFAULT 0,
                    first_home_lead_minute SMALLINT,
                    first_home_lead_added SMALLINT,
                    first_away_lead_minute SMALLINT,
                    first_away_lead_added SMALLINT,
                    max_home_lead SMALLINT NOT NULL DEFAULT 0,
                    max_away_lead SMALLINT NOT NULL DEFAULT 0,
                    home_led_by_1_any BOOLEAN NOT NULL DEFAULT FALSE,
                    home_led_by_2_any BOOLEAN NOT NULL DEFAULT FALSE,
                    away_led_by_1_any BOOLEAN NOT NULL DEFAULT FALSE,
                    away_led_by_2_any BOOLEAN NOT NULL DEFAULT FALSE,
                    incidents_final_home_score SMALLINT,
                    incidents_final_away_score SMALLINT,
                    ft_home_goals SMALLINT,
                    ft_away_goals SMALLINT,
                    scoreline_match BOOLEAN,
                    source_goal_count SMALLINT NOT NULL DEFAULT 0,
                    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fixture_incident_lead_states_league_status
                ON fixture_incident_lead_states (league_code, fixture_status);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fixture_incident_lead_states_flags
                ON fixture_incident_lead_states (
                    home_led_by_1_any,
                    away_led_by_1_any,
                    home_led_by_2_any,
                    away_led_by_2_any
                );
                """
            )
        conn.commit()
    finally:
        conn.close()


def fetch_fixture_scope(league: str | None, status: str, limit: int) -> list[int]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT DISTINCT i.fixture_id
                FROM fixture_incidents_sofascore i
                JOIN fixtures f ON f.fixture_id = i.fixture_id
                WHERE 1 = 1
            """
            params: list[Any] = []
            if league:
                query += " AND f.league_code = %s"
                params.append(league)
            if status != "all":
                query += " AND f.status = %s"
                params.append(status)
            query += " ORDER BY i.fixture_id DESC"
            if limit > 0:
                query += " LIMIT %s"
                params.append(limit)
            cur.execute(query, tuple(params))
            return [int(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def load_fixture_ids(path_str: str | None) -> list[int]:
    if not path_str:
        return []
    fixture_ids: list[int] = []
    for raw_line in Path(path_str).read_text(encoding="utf-8").splitlines():
        token = raw_line.split(",")[0].strip()
        if not token or token.lower() == "fixture_id":
            continue
        try:
            fixture_ids.append(int(token))
        except ValueError:
            continue
    return fixture_ids


def compute_lead_state_rows(fixture_ids: list[int]) -> list[dict[str, Any]]:
    if not fixture_ids:
        return []

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH scope AS (
                    SELECT
                        i.fixture_id,
                        i.incident_uid,
                        i.incident_id,
                        i.incident_type,
                        i.is_home,
                        i.minute,
                        i.added_time,
                        i.home_score,
                        i.away_score
                    FROM fixture_incidents_sofascore i
                    WHERE i.fixture_id = ANY(%s)
                ),
                goals AS (
                    SELECT
                        s.fixture_id,
                        s.incident_uid,
                        COALESCE(s.incident_id, 0) AS incident_id_sort,
                        s.is_home,
                        s.minute,
                        s.added_time,
                        COALESCE(s.added_time, 0) AS added_sort,
                        s.home_score,
                        s.away_score,
                        COALESCE(s.home_score, 0) - COALESCE(s.away_score, 0) AS lead
                    FROM scope s
                    WHERE s.incident_type = 'goal'
                ),
                goals_reg AS (
                    SELECT *
                    FROM goals
                    WHERE minute IS NOT NULL
                      AND minute <= 90
                ),
                home_first AS (
                    SELECT DISTINCT ON (g.fixture_id)
                        g.fixture_id,
                        g.minute AS first_home_lead_minute,
                        g.added_time AS first_home_lead_added
                    FROM goals_reg g
                    WHERE g.lead > 0
                    ORDER BY g.fixture_id, g.minute ASC, g.added_sort ASC, g.incident_id_sort ASC, g.incident_uid ASC
                ),
                away_first AS (
                    SELECT DISTINCT ON (g.fixture_id)
                        g.fixture_id,
                        g.minute AS first_away_lead_minute,
                        g.added_time AS first_away_lead_added
                    FROM goals_reg g
                    WHERE g.lead < 0
                    ORDER BY g.fixture_id, g.minute ASC, g.added_sort ASC, g.incident_id_sort ASC, g.incident_uid ASC
                ),
                agg_all AS (
                    SELECT
                        g.fixture_id,
                        COUNT(*)::int AS total_goal_events,
                        COUNT(*) FILTER (WHERE g.is_home IS TRUE)::int AS home_goal_events,
                        COUNT(*) FILTER (WHERE g.is_home IS FALSE)::int AS away_goal_events,
                        MAX(g.home_score)::int AS incidents_final_home_score,
                        MAX(g.away_score)::int AS incidents_final_away_score
                    FROM goals g
                    GROUP BY g.fixture_id
                ),
                agg_reg AS (
                    SELECT
                        g.fixture_id,
                        GREATEST(COALESCE(MAX(g.lead), 0), 0)::int AS max_home_lead,
                        GREATEST(COALESCE(MAX(-g.lead), 0), 0)::int AS max_away_lead
                    FROM goals_reg g
                    GROUP BY g.fixture_id
                ),
                base AS (
                    SELECT UNNEST(%s::bigint[]) AS fixture_id
                )
                SELECT
                    b.fixture_id,
                    f.league_code,
                    f.status AS fixture_status,
                    COALESCE(aa.total_goal_events, 0) AS total_goal_events,
                    COALESCE(aa.home_goal_events, 0) AS home_goal_events,
                    COALESCE(aa.away_goal_events, 0) AS away_goal_events,
                    hf.first_home_lead_minute,
                    hf.first_home_lead_added,
                    af.first_away_lead_minute,
                    af.first_away_lead_added,
                    COALESCE(ar.max_home_lead, 0) AS max_home_lead,
                    COALESCE(ar.max_away_lead, 0) AS max_away_lead,
                    (COALESCE(ar.max_home_lead, 0) >= 1) AS home_led_by_1_any,
                    (COALESCE(ar.max_home_lead, 0) >= 2) AS home_led_by_2_any,
                    (COALESCE(ar.max_away_lead, 0) >= 1) AS away_led_by_1_any,
                    (COALESCE(ar.max_away_lead, 0) >= 2) AS away_led_by_2_any,
                    aa.incidents_final_home_score,
                    aa.incidents_final_away_score,
                    fr.home_goals AS ft_home_goals,
                    fr.away_goals AS ft_away_goals
                FROM base b
                JOIN fixtures f ON f.fixture_id = b.fixture_id
                LEFT JOIN agg_all aa ON aa.fixture_id = b.fixture_id
                LEFT JOIN agg_reg ar ON ar.fixture_id = b.fixture_id
                LEFT JOIN home_first hf ON hf.fixture_id = b.fixture_id
                LEFT JOIN away_first af ON af.fixture_id = b.fixture_id
                LEFT JOIN fixture_results fr ON fr.fixture_id = b.fixture_id
                ORDER BY b.fixture_id DESC;
                """,
                (fixture_ids, fixture_ids),
            )
            cols = [desc[0] for desc in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    for row in rows:
        ft_home = row.get("ft_home_goals")
        ft_away = row.get("ft_away_goals")
        if ft_home is None or ft_away is None:
            row["scoreline_match"] = None
            continue
        inc_home = row.get("incidents_final_home_score")
        inc_away = row.get("incidents_final_away_score")
        row["scoreline_match"] = (
            int(0 if inc_home is None else inc_home) == int(ft_home)
            and int(0 if inc_away is None else inc_away) == int(ft_away)
        )
    return rows


def upsert_lead_state_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            execute_batch(
                cur,
                """
                INSERT INTO fixture_incident_lead_states (
                    fixture_id,
                    league_code,
                    fixture_status,
                    total_goal_events,
                    home_goal_events,
                    away_goal_events,
                    first_home_lead_minute,
                    first_home_lead_added,
                    first_away_lead_minute,
                    first_away_lead_added,
                    max_home_lead,
                    max_away_lead,
                    home_led_by_1_any,
                    home_led_by_2_any,
                    away_led_by_1_any,
                    away_led_by_2_any,
                    incidents_final_home_score,
                    incidents_final_away_score,
                    ft_home_goals,
                    ft_away_goals,
                    scoreline_match,
                    source_goal_count,
                    computed_at,
                    updated_at
                )
                VALUES (
                    %(fixture_id)s,
                    %(league_code)s,
                    %(fixture_status)s,
                    %(total_goal_events)s,
                    %(home_goal_events)s,
                    %(away_goal_events)s,
                    %(first_home_lead_minute)s,
                    %(first_home_lead_added)s,
                    %(first_away_lead_minute)s,
                    %(first_away_lead_added)s,
                    %(max_home_lead)s,
                    %(max_away_lead)s,
                    %(home_led_by_1_any)s,
                    %(home_led_by_2_any)s,
                    %(away_led_by_1_any)s,
                    %(away_led_by_2_any)s,
                    %(incidents_final_home_score)s,
                    %(incidents_final_away_score)s,
                    %(ft_home_goals)s,
                    %(ft_away_goals)s,
                    %(scoreline_match)s,
                    %(total_goal_events)s,
                    NOW(),
                    NOW()
                )
                ON CONFLICT (fixture_id) DO UPDATE SET
                    league_code = EXCLUDED.league_code,
                    fixture_status = EXCLUDED.fixture_status,
                    total_goal_events = EXCLUDED.total_goal_events,
                    home_goal_events = EXCLUDED.home_goal_events,
                    away_goal_events = EXCLUDED.away_goal_events,
                    first_home_lead_minute = EXCLUDED.first_home_lead_minute,
                    first_home_lead_added = EXCLUDED.first_home_lead_added,
                    first_away_lead_minute = EXCLUDED.first_away_lead_minute,
                    first_away_lead_added = EXCLUDED.first_away_lead_added,
                    max_home_lead = EXCLUDED.max_home_lead,
                    max_away_lead = EXCLUDED.max_away_lead,
                    home_led_by_1_any = EXCLUDED.home_led_by_1_any,
                    home_led_by_2_any = EXCLUDED.home_led_by_2_any,
                    away_led_by_1_any = EXCLUDED.away_led_by_1_any,
                    away_led_by_2_any = EXCLUDED.away_led_by_2_any,
                    incidents_final_home_score = EXCLUDED.incidents_final_home_score,
                    incidents_final_away_score = EXCLUDED.incidents_final_away_score,
                    ft_home_goals = EXCLUDED.ft_home_goals,
                    ft_away_goals = EXCLUDED.ft_away_goals,
                    scoreline_match = EXCLUDED.scoreline_match,
                    source_goal_count = EXCLUDED.source_goal_count,
                    computed_at = EXCLUDED.computed_at,
                    updated_at = NOW()
                """,
                rows,
                page_size=200,
            )
        conn.commit()
    finally:
        conn.close()


def compute_timeline_quality(fixture_ids: list[int]) -> dict[str, Any]:
    if not fixture_ids:
        return {}
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS goal_rows,
                    COUNT(*) FILTER (WHERE minute IS NULL) AS goal_minute_null,
                    COUNT(*) FILTER (WHERE is_home IS NULL) AS goal_is_home_null,
                    COUNT(*) FILTER (WHERE home_score IS NULL OR away_score IS NULL) AS goal_score_null,
                    COUNT(*) FILTER (WHERE player_name IS NULL) AS goal_player_name_null
                FROM fixture_incidents_sofascore
                WHERE fixture_id = ANY(%s)
                  AND incident_type = 'goal'
                """,
                (fixture_ids,),
            )
            goal_quality = cur.fetchone()

            cur.execute(
                """
                SELECT
                    COUNT(*) AS goal_rows,
                    COUNT(*) FILTER (WHERE minute IS NULL) AS goal_minute_null,
                    COUNT(*) FILTER (WHERE is_home IS NULL) AS goal_is_home_null,
                    COUNT(*) FILTER (WHERE home_score IS NULL OR away_score IS NULL) AS goal_score_null,
                    COUNT(*) FILTER (WHERE player_name IS NULL) AS goal_player_name_null
                FROM fixture_incidents_sofascore
                WHERE fixture_id = ANY(%s)
                  AND incident_type = 'goal'
                  AND minute IS NOT NULL
                  AND minute <= 90
                """,
                (fixture_ids,),
            )
            goal_quality_reg = cur.fetchone()

            cur.execute(
                """
                WITH goals AS (
                    SELECT
                        fixture_id,
                        incident_uid,
                        COALESCE(incident_id, 0) AS incident_id_sort,
                        minute,
                        COALESCE(added_time, 0) AS added_sort,
                        home_score,
                        away_score,
                        LAG(home_score) OVER (
                            PARTITION BY fixture_id
                            ORDER BY minute ASC, COALESCE(added_time, 0) ASC, COALESCE(incident_id, 0) ASC, incident_uid ASC
                        ) AS prev_home_score,
                        LAG(away_score) OVER (
                            PARTITION BY fixture_id
                            ORDER BY minute ASC, COALESCE(added_time, 0) ASC, COALESCE(incident_id, 0) ASC, incident_uid ASC
                        ) AS prev_away_score
                    FROM fixture_incidents_sofascore
                    WHERE fixture_id = ANY(%s)
                      AND incident_type = 'goal'
                ),
                deltas AS (
                    SELECT
                        fixture_id,
                        COALESCE(home_score - prev_home_score, home_score) AS dh,
                        COALESCE(away_score - prev_away_score, away_score) AS da
                    FROM goals
                )
                SELECT
                    COUNT(*) AS goal_rows,
                    COUNT(*) FILTER (
                        WHERE (dh = 1 AND da = 0) OR (dh = 0 AND da = 1)
                    ) AS valid_progress_rows,
                    COUNT(*) FILTER (
                        WHERE NOT ((dh = 1 AND da = 0) OR (dh = 0 AND da = 1))
                    ) AS suspicious_progress_rows
                FROM deltas
                """,
                (fixture_ids,),
            )
            progression = cur.fetchone()

            cur.execute(
                """
                WITH goals AS (
                    SELECT
                        fixture_id,
                        incident_uid,
                        COALESCE(incident_id, 0) AS incident_id_sort,
                        minute,
                        COALESCE(added_time, 0) AS added_sort,
                        home_score,
                        away_score,
                        LAG(home_score) OVER (
                            PARTITION BY fixture_id
                            ORDER BY minute ASC, COALESCE(added_time, 0) ASC, COALESCE(incident_id, 0) ASC, incident_uid ASC
                        ) AS prev_home_score,
                        LAG(away_score) OVER (
                            PARTITION BY fixture_id
                            ORDER BY minute ASC, COALESCE(added_time, 0) ASC, COALESCE(incident_id, 0) ASC, incident_uid ASC
                        ) AS prev_away_score
                    FROM fixture_incidents_sofascore
                    WHERE fixture_id = ANY(%s)
                      AND incident_type = 'goal'
                      AND minute IS NOT NULL
                      AND minute <= 90
                ),
                deltas AS (
                    SELECT
                        fixture_id,
                        COALESCE(home_score - prev_home_score, home_score) AS dh,
                        COALESCE(away_score - prev_away_score, away_score) AS da
                    FROM goals
                )
                SELECT
                    COUNT(*) AS goal_rows,
                    COUNT(*) FILTER (
                        WHERE (dh = 1 AND da = 0) OR (dh = 0 AND da = 1)
                    ) AS valid_progress_rows,
                    COUNT(*) FILTER (
                        WHERE NOT ((dh = 1 AND da = 0) OR (dh = 0 AND da = 1))
                    ) AS suspicious_progress_rows
                FROM deltas
                """,
                (fixture_ids,),
            )
            progression_reg = cur.fetchone()

            cur.execute(
                """
                SELECT
                    incident_type,
                    COUNT(*) AS n
                FROM fixture_incidents_sofascore
                WHERE fixture_id = ANY(%s)
                  AND minute IS NOT NULL
                  AND (minute < 0 OR minute > 130)
                GROUP BY incident_type
                ORDER BY n DESC
                """,
                (fixture_ids,),
            )
            minute_outliers = [{"incident_type": r[0], "count": int(r[1])} for r in cur.fetchall()]
    finally:
        conn.close()

    return {
        "goal_quality": {
            "goal_rows": int(goal_quality[0] or 0),
            "goal_minute_null": int(goal_quality[1] or 0),
            "goal_is_home_null": int(goal_quality[2] or 0),
            "goal_score_null": int(goal_quality[3] or 0),
            "goal_player_name_null": int(goal_quality[4] or 0),
        },
        "goal_quality_regulation_90": {
            "goal_rows": int(goal_quality_reg[0] or 0),
            "goal_minute_null": int(goal_quality_reg[1] or 0),
            "goal_is_home_null": int(goal_quality_reg[2] or 0),
            "goal_score_null": int(goal_quality_reg[3] or 0),
            "goal_player_name_null": int(goal_quality_reg[4] or 0),
        },
        "progression_quality": {
            "goal_rows": int(progression[0] or 0),
            "valid_progress_rows": int(progression[1] or 0),
            "suspicious_progress_rows": int(progression[2] or 0),
        },
        "progression_quality_regulation_90": {
            "goal_rows": int(progression_reg[0] or 0),
            "valid_progress_rows": int(progression_reg[1] or 0),
            "suspicious_progress_rows": int(progression_reg[2] or 0),
        },
        "minute_outlier_incidents": minute_outliers,
    }


def build_report(
    rows: list[dict[str, Any]],
    quality: dict[str, Any],
    league: str | None,
    status: str,
    limit: int,
) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "scope": {"league": league, "status": status, "limit": limit},
            "summary": {"fixtures": 0},
            "quality": quality,
        }

    ft_rows = [r for r in rows if r.get("ft_home_goals") is not None and r.get("ft_away_goals") is not None]
    scoreline_match_count = sum(1 for r in ft_rows if bool(r.get("scoreline_match")))
    scoreline_mismatch_count = sum(1 for r in ft_rows if r.get("scoreline_match") is False)

    by_league: dict[str, int] = {}
    for row in rows:
        key = str(row.get("league_code") or "UNKNOWN")
        by_league[key] = by_league.get(key, 0) + 1
    by_league_sorted = sorted(by_league.items(), key=lambda kv: (-kv[1], kv[0]))

    summary = {
        "fixtures": n,
        "fixtures_with_goals": int(sum(1 for r in rows if int(r.get("total_goal_events") or 0) > 0)),
        "fixtures_home_led_by_1_any": int(sum(1 for r in rows if bool(r.get("home_led_by_1_any")))),
        "fixtures_away_led_by_1_any": int(sum(1 for r in rows if bool(r.get("away_led_by_1_any")))),
        "fixtures_home_led_by_2_any": int(sum(1 for r in rows if bool(r.get("home_led_by_2_any")))),
        "fixtures_away_led_by_2_any": int(sum(1 for r in rows if bool(r.get("away_led_by_2_any")))),
        "avg_total_goal_events": float(sum(int(r.get("total_goal_events") or 0) for r in rows) / max(n, 1)),
        "scoreline_check": {
            "fixtures_with_ft_result": int(len(ft_rows)),
            "scoreline_match_count": int(scoreline_match_count),
            "scoreline_mismatch_count": int(scoreline_mismatch_count),
            "scoreline_match_rate": float(scoreline_match_count / len(ft_rows)) if ft_rows else None,
        },
        "league_fixture_counts": [{"league_code": k, "fixtures": int(v)} for k, v in by_league_sorted],
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": {"league": league, "status": status, "limit": limit},
        "summary": summary,
        "quality": quality,
    }


def write_report(payload: dict[str, Any], output_dir: Path, output_stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"{output_stem}_{ts}.json"
    md_path = output_dir / f"{output_stem}_{ts}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = payload.get("summary", {})
    scoreline = summary.get("scoreline_check", {})
    quality = payload.get("quality", {})
    goal_quality = quality.get("goal_quality", {})
    goal_quality_reg = quality.get("goal_quality_regulation_90", {})
    progression = quality.get("progression_quality", {})
    progression_reg = quality.get("progression_quality_regulation_90", {})

    lines: list[str] = []
    lines.append("# Incident Timeline Quality")
    lines.append("")
    lines.append(f"Generated: {payload.get('generated_at')}")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- league: `{payload.get('scope', {}).get('league')}`")
    lines.append(f"- status: `{payload.get('scope', {}).get('status')}`")
    lines.append(f"- limit: `{payload.get('scope', {}).get('limit')}`")
    lines.append("- lead-state rule: `regulation_only (minute <= 90)`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- fixtures: {summary.get('fixtures')}")
    lines.append(f"- fixtures with goals: {summary.get('fixtures_with_goals')}")
    lines.append(f"- home_led_by_1_any fixtures: {summary.get('fixtures_home_led_by_1_any')}")
    lines.append(f"- away_led_by_1_any fixtures: {summary.get('fixtures_away_led_by_1_any')}")
    lines.append(f"- home_led_by_2_any fixtures: {summary.get('fixtures_home_led_by_2_any')}")
    lines.append(f"- away_led_by_2_any fixtures: {summary.get('fixtures_away_led_by_2_any')}")
    lines.append(f"- avg total goal events / fixture: {summary.get('avg_total_goal_events')}")
    lines.append("")
    lines.append("## Scoreline Check")
    lines.append(f"- fixtures with FT result: {scoreline.get('fixtures_with_ft_result')}")
    lines.append(f"- scoreline match count: {scoreline.get('scoreline_match_count')}")
    lines.append(f"- scoreline mismatch count: {scoreline.get('scoreline_mismatch_count')}")
    lines.append(f"- scoreline match rate: {scoreline.get('scoreline_match_rate')}")
    lines.append("")
    lines.append("## Goal Timeline Quality")
    lines.append(f"- goal rows: {goal_quality.get('goal_rows')}")
    lines.append(f"- goal minute null: {goal_quality.get('goal_minute_null')}")
    lines.append(f"- goal is_home null: {goal_quality.get('goal_is_home_null')}")
    lines.append(f"- goal score null: {goal_quality.get('goal_score_null')}")
    lines.append(f"- goal player_name null: {goal_quality.get('goal_player_name_null')}")
    lines.append("")
    lines.append("## Goal Timeline Quality (Regulation <=90)")
    lines.append(f"- goal rows: {goal_quality_reg.get('goal_rows')}")
    lines.append(f"- goal minute null: {goal_quality_reg.get('goal_minute_null')}")
    lines.append(f"- goal is_home null: {goal_quality_reg.get('goal_is_home_null')}")
    lines.append(f"- goal score null: {goal_quality_reg.get('goal_score_null')}")
    lines.append(f"- goal player_name null: {goal_quality_reg.get('goal_player_name_null')}")
    lines.append("")
    lines.append("## Progression Quality")
    lines.append(f"- goal rows checked: {progression.get('goal_rows')}")
    lines.append(f"- valid progression rows: {progression.get('valid_progress_rows')}")
    lines.append(f"- suspicious progression rows: {progression.get('suspicious_progress_rows')}")
    lines.append("")
    lines.append("## Progression Quality (Regulation <=90)")
    lines.append(f"- goal rows checked: {progression_reg.get('goal_rows')}")
    lines.append(f"- valid progression rows: {progression_reg.get('valid_progress_rows')}")
    lines.append(f"- suspicious progression rows: {progression_reg.get('suspicious_progress_rows')}")
    lines.append("")
    lines.append("## Minute Outlier Incidents")
    for row in quality.get("minute_outlier_incidents", []):
        lines.append(f"- {row['incident_type']}: {row['count']}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()

    ensure_schema()
    fixture_ids = load_fixture_ids(args.fixture_ids_file)
    if not fixture_ids:
        fixture_ids = fetch_fixture_scope(league=args.league, status=args.status, limit=args.limit)
    if not fixture_ids:
        print("No fixture incidents found for requested scope.")
        return

    rows = compute_lead_state_rows(fixture_ids)
    if not args.dry_run:
        upsert_lead_state_rows(rows)

    quality = compute_timeline_quality(fixture_ids)
    payload = build_report(
        rows=rows,
        quality=quality,
        league=args.league,
        status=args.status,
        limit=args.limit,
    )
    json_path, md_path = write_report(payload, args.output_dir, args.output_stem)

    print(
        f"Processed fixtures: {len(rows)} | dry_run={args.dry_run} | "
        f"scoreline_match_rate={payload.get('summary', {}).get('scoreline_check', {}).get('scoreline_match_rate')}"
    )
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
