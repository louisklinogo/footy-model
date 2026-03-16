from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.league import League
from sofascore_wrapper.match import Match
from sofascore_wrapper.player import Player
from sofascore_wrapper.team import Team

from src.db.db_utils import connect_db
from src.ingest.bootstrap_fixtures_schema_v1 import bootstrap_schema


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest external context snapshots from SofaScore.")
    parser.add_argument(
        "--context-type",
        choices=["team", "match", "player", "all"],
        default="all",
        help="Context family to ingest.",
    )
    parser.add_argument("--league", type=str, default=None, help="Optional league_code filter.")
    parser.add_argument("--limit", type=int, default=10, help="Max entities to process per context family.")
    parser.add_argument(
        "--status",
        choices=["scheduled", "ft", "all"],
        default="scheduled",
        help="Fixture status filter for match context ingestion.",
    )
    parser.add_argument(
        "--scheduled-start-hours",
        type=int,
        default=0,
        help="For scheduled scope, include fixtures from now plus this many hours.",
    )
    parser.add_argument(
        "--scheduled-end-hours",
        type=int,
        default=72,
        help="For scheduled scope, include fixtures through now plus this many hours.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip schema bootstrap when the tables are already present.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and normalize but do not write to DB.")
    return parser.parse_args()


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _score_diff_to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _form_sequence_to_text(form: list[Any] | None) -> str | None:
    if not isinstance(form, list) or not form:
        return None
    values = [str(item).strip().upper()[:1] for item in form if str(item).strip()]
    if not values:
        return None
    return "".join(values)


def _form_points(form: list[Any] | None) -> int | None:
    if not isinstance(form, list) or not form:
        return None
    score_map = {"W": 3, "D": 1, "L": 0}
    points = 0
    found = False
    for item in form:
        key = str(item).strip().upper()[:1]
        if key in score_map:
            points += score_map[key]
            found = True
    return points if found else None


def _form_result_counts(form: list[Any] | None) -> tuple[int | None, int | None, int | None]:
    if not isinstance(form, list) or not form:
        return None, None, None
    wins = draws = losses = 0
    found = False
    for item in form:
        key = str(item).strip().upper()[:1]
        if key == "W":
            wins += 1
            found = True
        elif key == "D":
            draws += 1
            found = True
        elif key == "L":
            losses += 1
            found = True
    if not found:
        return None, None, None
    return wins, draws, losses


def _extract_standings_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    standings = payload.get("standings")
    if isinstance(standings, list) and standings:
        first = standings[0]
        rows = first.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _extract_standing_metrics(row: dict[str, Any]) -> dict[str, int | None]:
    return {
        "rank": _safe_int(row.get("position")),
        "points": _safe_int(row.get("points")),
        "played": _safe_int(row.get("matches")),
        "wins": _safe_int(row.get("wins")),
        "draws": _safe_int(row.get("draws")),
        "losses": _safe_int(row.get("losses")),
        "goals_for": _safe_int(row.get("scoresFor")),
        "goals_against": _safe_int(row.get("scoresAgainst")),
        "goal_diff": _score_diff_to_int(row.get("scoreDiffFormatted")),
    }


def _summarize_performance_graph(payload: dict[str, Any]) -> dict[str, int | float | None]:
    graph_data = payload.get("graphData")
    if not isinstance(graph_data, list):
        return {
            "performance_graph_points_avg": None,
            "performance_graph_goal_diff_avg": None,
            "performance_graph_samples": None,
        }
    points: list[float] = []
    goal_diffs: list[float] = []
    for item in graph_data:
        if not isinstance(item, dict):
            continue
        point_value = _safe_float(_first_present(item.get("points"), item.get("value")))
        if point_value is not None:
            points.append(point_value)
        goal_diff_value = _safe_float(_first_present(item.get("scoreDiff"), item.get("goalDiff")))
        if goal_diff_value is not None:
            goal_diffs.append(goal_diff_value)
    return {
        "performance_graph_points_avg": (sum(points) / len(points)) if points else None,
        "performance_graph_goal_diff_avg": (sum(goal_diffs) / len(goal_diffs)) if goal_diffs else None,
        "performance_graph_samples": len(graph_data),
    }


def _extract_team_overview_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    pregame_form = payload.get("pregameForm")
    if not isinstance(pregame_form, dict):
        return {
            "form_sequence": None,
            "form_points_last5": None,
            "form_wins_last5": None,
            "form_draws_last5": None,
            "form_losses_last5": None,
            "pregame_avg_rating": None,
            "pregame_position": None,
            "pregame_value": None,
        }
    form = pregame_form.get("form")
    wins, draws, losses = _form_result_counts(form)
    return {
        "form_sequence": _form_sequence_to_text(form),
        "form_points_last5": _form_points(form),
        "form_wins_last5": wins,
        "form_draws_last5": draws,
        "form_losses_last5": losses,
        "pregame_avg_rating": _safe_float(pregame_form.get("avgRating")),
        "pregame_position": _safe_int(pregame_form.get("position")),
        "pregame_value": _safe_float(pregame_form.get("value")),
    }


def _extract_team_league_stats_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    stats = payload.get("statistics")
    if not isinstance(stats, dict):
        return {}
    return {
        "league_stats_matches": _safe_int(stats.get("matches")),
        "league_stats_goals_scored": _safe_int(stats.get("goalsScored")),
        "league_stats_goals_conceded": _safe_int(stats.get("goalsConceded")),
        "league_stats_big_chances": _safe_int(stats.get("bigChances")),
        "league_stats_shots_on_target": _safe_int(stats.get("shotsOnTarget")),
        "league_stats_corners": _safe_int(stats.get("corners")),
        "league_stats_average_ball_possession": _safe_float(stats.get("averageBallPossession")),
        "league_stats_accurate_passes_percentage": _safe_float(stats.get("accuratePassesPercentage")),
        "league_stats_accurate_long_balls_percentage": _safe_float(stats.get("accurateLongBallsPercentage")),
        "league_stats_accurate_crosses_percentage": _safe_float(stats.get("accurateCrossesPercentage")),
        "league_stats_clean_sheets": _safe_int(stats.get("cleanSheets")),
        "league_stats_tackles": _safe_int(stats.get("tackles")),
        "league_stats_interceptions": _safe_int(stats.get("interceptions")),
        "league_stats_saves": _safe_int(stats.get("saves")),
        "league_stats_errors_leading_to_shot": _safe_int(stats.get("errorsLeadingToShot")),
        "league_stats_total_duels": _safe_int(stats.get("totalDuels")),
        "league_stats_duels_won_percentage": _safe_float(stats.get("duelsWonPercentage")),
        "league_stats_total_aerial_duels": _safe_int(stats.get("totalAerialDuels")),
        "league_stats_aerial_duels_won_percentage": _safe_float(stats.get("aerialDuelsWonPercentage")),
        "league_stats_possession_lost": _safe_int(stats.get("possessionLost")),
        "league_stats_offsides": _safe_int(stats.get("offsides")),
        "league_stats_fouls": _safe_int(stats.get("fouls")),
        "league_stats_yellow_cards": _safe_int(stats.get("yellowCards")),
        "league_stats_red_cards": _safe_int(stats.get("redCards")),
    }


def _extract_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    event = payload.get("event")
    return event if isinstance(event, dict) else payload


def _extract_match_form_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    home = payload.get("homeTeam") if isinstance(payload.get("homeTeam"), dict) else {}
    away = payload.get("awayTeam") if isinstance(payload.get("awayTeam"), dict) else {}
    home_form = home.get("form")
    away_form = away.get("form")
    return {
        "home_form_sequence": _form_sequence_to_text(home_form),
        "away_form_sequence": _form_sequence_to_text(away_form),
        "home_form_points_last5": _form_points(home_form),
        "away_form_points_last5": _form_points(away_form),
        "home_avg_rating": _safe_float(home.get("avgRating")),
        "away_avg_rating": _safe_float(away.get("avgRating")),
        "home_position": _safe_int(home.get("position")),
        "away_position": _safe_int(away.get("position")),
        "home_value": _safe_float(home.get("value")),
        "away_value": _safe_float(away.get("value")),
    }


def _extract_streak_metrics(payload: dict[str, Any]) -> dict[str, int | None]:
    if not isinstance(payload, dict):
        return {}
    home = payload.get("homeTeam") if isinstance(payload.get("homeTeam"), dict) else {}
    away = payload.get("awayTeam") if isinstance(payload.get("awayTeam"), dict) else {}
    home_values = home if home else payload
    away_values = away if away else payload
    return {
        "home_streak_win": _safe_int(_first_present(home_values.get("wins"), home_values.get("winStreak"))),
        "away_streak_win": _safe_int(_first_present(away_values.get("wins"), away_values.get("winStreak"))),
        "home_streak_unbeaten": _safe_int(_first_present(home_values.get("unbeaten"), home_values.get("unbeatenStreak"))),
        "away_streak_unbeaten": _safe_int(_first_present(away_values.get("unbeaten"), away_values.get("unbeatenStreak"))),
    }


def _summarize_h2h_results(payload: dict[str, Any], current_home_id: str | None, current_away_id: str | None) -> dict[str, int | None]:
    events = payload.get("events")
    if not isinstance(events, list) or not current_home_id or not current_away_id:
        return {
            "h2h_home_wins_last_n": None,
            "h2h_draws_last_n": None,
            "h2h_away_wins_last_n": None,
            "h2h_matches_count": None,
        }
    home_wins = 0
    away_wins = 0
    draws = 0
    counted = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        home_team = event.get("homeTeam") if isinstance(event.get("homeTeam"), dict) else {}
        away_team = event.get("awayTeam") if isinstance(event.get("awayTeam"), dict) else {}
        hist_home_id = str(home_team.get("id")) if home_team.get("id") is not None else None
        hist_away_id = str(away_team.get("id")) if away_team.get("id") is not None else None
        home_score = event.get("homeScore") if isinstance(event.get("homeScore"), dict) else {}
        away_score = event.get("awayScore") if isinstance(event.get("awayScore"), dict) else {}
        hist_home_goals = _safe_int(_first_present(home_score.get("current"), home_score.get("display"), home_score.get("normaltime")))
        hist_away_goals = _safe_int(_first_present(away_score.get("current"), away_score.get("display"), away_score.get("normaltime")))
        if hist_home_id is None or hist_away_id is None or hist_home_goals is None or hist_away_goals is None:
            continue
        counted += 1
        if hist_home_goals == hist_away_goals:
            draws += 1
            continue
        winner_id = hist_home_id if hist_home_goals > hist_away_goals else hist_away_id
        if winner_id == current_home_id:
            home_wins += 1
        elif winner_id == current_away_id:
            away_wins += 1
    return {
        "h2h_home_wins_last_n": home_wins if counted else None,
        "h2h_draws_last_n": draws if counted else None,
        "h2h_away_wins_last_n": away_wins if counted else None,
        "h2h_matches_count": counted if counted else None,
    }


def _extract_win_probability(payload: dict[str, Any]) -> dict[str, float | None]:
    if not isinstance(payload, dict):
        return {}
    return {
        "win_probability_home": _safe_float(_first_present(payload.get("home"), payload.get("homeWin"))),
        "win_probability_draw": _safe_float(payload.get("draw")),
        "win_probability_away": _safe_float(_first_present(payload.get("away"), payload.get("awayWin"))),
    }


def _build_fixture_scope_clause(
    status: str,
    scheduled_start_hours: int,
    scheduled_end_hours: int,
    fixture_alias: str = "f",
) -> tuple[str, list[Any], str]:
    clauses: list[str] = []
    params: list[Any] = []
    order_direction = "DESC"

    if status == "scheduled":
        clauses.extend(
            [
                f"{fixture_alias}.status = %s",
                f"{fixture_alias}.match_datetime_utc >= NOW() + (%s * INTERVAL '1 hour')",
                f"{fixture_alias}.match_datetime_utc <= NOW() + (%s * INTERVAL '1 hour')",
            ]
        )
        params.extend(["scheduled", scheduled_start_hours, scheduled_end_hours])
        order_direction = "ASC"
    elif status == "ft":
        clauses.append(f"{fixture_alias}.status = %s")
        params.append("ft")

    return (" AND ".join(clauses), params, order_direction)


def _extract_player_attributes_metrics(payload: dict[str, Any], fallback_position: str | None, fallback_market_value: int | None) -> dict[str, Any]:
    overview = None
    candidates = payload.get("playerAttributeOverviews")
    if isinstance(candidates, list) and candidates:
        overview = candidates[0]
    if not isinstance(overview, dict):
        candidates = payload.get("averageAttributeOverviews")
        if isinstance(candidates, list) and candidates:
            overview = candidates[0]
    if not isinstance(overview, dict):
        overview = {}
    return {
        "position_group": overview.get("position") or fallback_position,
        "attribute_attacking": _safe_int(overview.get("attacking")),
        "attribute_technical": _safe_int(overview.get("technical")),
        "attribute_tactical": _safe_int(overview.get("tactical")),
        "attribute_defending": _safe_int(overview.get("defending")),
        "attribute_creativity": _safe_int(overview.get("creativity")),
        "market_value_euro_snapshot": fallback_market_value,
    }


def _extract_player_league_stats_metrics(payload: dict[str, Any], fallback_position: str | None, fallback_market_value: int | None) -> dict[str, Any]:
    stats = payload.get("statistics")
    if not isinstance(stats, dict):
        stats = {}
    return {
        "position_group": fallback_position,
        "rating_avg": _safe_float(stats.get("rating")),
        "minutes_played": _safe_int(stats.get("minutesPlayed")),
        "goals": _safe_int(stats.get("goals")),
        "assists": _safe_int(stats.get("assists")),
        "expected_goals": _safe_float(stats.get("expectedGoals")),
        "expected_assists": _safe_float(stats.get("expectedAssists")),
        "market_value_euro_snapshot": fallback_market_value,
    }


def _upsert_row(conn, table: str, conflict_cols: list[str], row: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        logger.info("[DRY RUN] Would upsert %s row into %s", row.get("context_type"), table)
        return
    columns = list(row.keys())
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(
        f"{col} = EXCLUDED.{col}" for col in columns if col not in set(conflict_cols) and col != "context_id"
    )
    query = f"""
        INSERT INTO {table} ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT ({", ".join(conflict_cols)}) DO UPDATE SET
            {updates},
            ingested_at = NOW()
    """
    with conn.cursor() as cur:
        cur.execute(query, [json.dumps(value) if isinstance(value, (dict, list)) else value for value in row.values()])


def fetch_team_targets(
    conn,
    league_code: str | None,
    status: str,
    limit: int,
    scheduled_start_hours: int,
    scheduled_end_hours: int,
) -> list[dict[str, Any]]:
    scope_sql, scope_params, order_direction = _build_fixture_scope_clause(
        status=status,
        scheduled_start_hours=scheduled_start_hours,
        scheduled_end_hours=scheduled_end_hours,
        fixture_alias="f",
    )
    fixture_where = ["f.sofascore_id IS NOT NULL", "slot.team_id IS NOT NULL"]
    params: list[Any] = []
    if scope_sql:
        fixture_where.append(scope_sql)
        params.extend(scope_params)
    if league_code:
        fixture_where.append("f.league_code = %s")
        params.append(league_code)

    query = f"""
        WITH scoped_teams AS (
            SELECT
                slot.team_id,
                f.league_code,
                f.match_datetime_utc
            FROM fixtures f
            CROSS JOIN LATERAL (
                VALUES (f.home_team_id), (f.away_team_id)
            ) AS slot(team_id)
            WHERE {" AND ".join(fixture_where)}
        ),
        ranked_teams AS (
            SELECT DISTINCT ON (st.team_id)
                st.team_id,
                st.league_code,
                st.match_datetime_utc
            FROM scoped_teams st
            ORDER BY st.team_id, st.match_datetime_utc {order_direction} NULLS LAST
        )
        SELECT
            t.team_id,
            t.team_name,
            t.sofascore_id,
            t.league_code,
            l.sofascore_league_id,
            l.sofascore_season_id
        FROM ranked_teams rt
        JOIN teams t ON t.team_id = rt.team_id
        JOIN leagues l ON l.league_code = t.league_code
        WHERE t.sofascore_id IS NOT NULL
          AND l.sofascore_league_id IS NOT NULL
          AND l.sofascore_season_id IS NOT NULL
        ORDER BY rt.match_datetime_utc {order_direction} NULLS LAST, t.team_id
        LIMIT %s
    """
    params.append(limit)
    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_match_targets(
    conn,
    league_code: str | None,
    status: str,
    limit: int,
    scheduled_start_hours: int,
    scheduled_end_hours: int,
) -> list[dict[str, Any]]:
    scope_sql, scope_params, order_direction = _build_fixture_scope_clause(
        status=status,
        scheduled_start_hours=scheduled_start_hours,
        scheduled_end_hours=scheduled_end_hours,
        fixture_alias="f",
    )
    query = """
        SELECT fixture_id, league_code, season, sofascore_id, status
        FROM fixtures f
        WHERE sofascore_id IS NOT NULL
    """
    params: list[Any] = []
    if scope_sql:
        query += f" AND {scope_sql}"
        params.extend(scope_params)
    if league_code:
        query += " AND league_code = %s"
        params.append(league_code)
    query += f" ORDER BY match_datetime_utc {order_direction} NULLS LAST, fixture_id {order_direction} LIMIT %s"
    params.append(limit)
    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_player_targets(
    conn,
    league_code: str | None,
    status: str,
    limit: int,
    scheduled_start_hours: int,
    scheduled_end_hours: int,
) -> list[dict[str, Any]]:
    scope_sql, scope_params, order_direction = _build_fixture_scope_clause(
        status=status,
        scheduled_start_hours=scheduled_start_hours,
        scheduled_end_hours=scheduled_end_hours,
        fixture_alias="f_target",
    )
    target_where = ["f_target.sofascore_id IS NOT NULL", "slot.team_id IS NOT NULL"]
    params: list[Any] = []
    if scope_sql:
        target_where.append(scope_sql)
        params.extend(scope_params)
    if league_code:
        target_where.append("f_target.league_code = %s")
        params.append(league_code)

    query = f"""
        WITH target_teams AS (
            SELECT DISTINCT ON (slot.team_id)
                slot.team_id,
                f_target.league_code,
                f_target.season,
                f_target.match_datetime_utc
            FROM fixtures f_target
            CROSS JOIN LATERAL (
                VALUES (f_target.home_team_id), (f_target.away_team_id)
            ) AS slot(team_id)
            WHERE {" AND ".join(target_where)}
            ORDER BY slot.team_id, f_target.match_datetime_utc {order_direction} NULLS LAST
        )
        SELECT DISTINCT ON (p.player_id)
            p.player_id,
            p.sofascore_id,
            p.position,
            p.market_value_euro,
            tt.league_code,
            tt.season,
            l.sofascore_league_id,
            l.sofascore_season_id
        FROM target_teams tt
        JOIN fixture_player_stats fps ON fps.team_id = tt.team_id
        JOIN fixtures f_hist ON f_hist.fixture_id = fps.fixture_id
        JOIN players p ON p.player_id = fps.player_id
        JOIN leagues l ON l.league_code = tt.league_code
        WHERE p.sofascore_id IS NOT NULL
          AND l.sofascore_league_id IS NOT NULL
          AND l.sofascore_season_id IS NOT NULL
        ORDER BY p.player_id, tt.match_datetime_utc {order_direction} NULLS LAST, f_hist.match_datetime_utc DESC NULLS LAST
        LIMIT %s
    """
    params.append(limit)
    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


async def ingest_team_context(
    api: SofascoreAPI,
    conn,
    league_code: str | None,
    status: str,
    limit: int,
    scheduled_start_hours: int,
    scheduled_end_hours: int,
    dry_run: bool,
) -> int:
    targets = fetch_team_targets(conn, league_code, status, limit, scheduled_start_hours, scheduled_end_hours)
    if not targets:
        logger.info("No team targets found for external context ingestion.")
        return 0

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in targets:
        grouped[(row["league_code"], int(row["sofascore_league_id"]), int(row["sofascore_season_id"]))].append(row)

    rows_written = 0
    snapshot_time = datetime.now(UTC)
    for (team_league_code, provider_league_id, provider_season_id), league_targets in grouped.items():
        league = League(api, provider_league_id)
        standings_payloads: dict[str, dict[str, Any]] = {}
        for context_type, fetcher in {
            "standings_total": league.standings,
            "standings_home": league.standings_home,
            "standings_away": league.standings_away,
        }.items():
            try:
                standings_payloads[context_type] = await fetcher(provider_season_id)
            except Exception as exc:
                logger.warning("Failed to fetch %s for league %s: %s", context_type, team_league_code, exc)
        standings_maps: dict[str, dict[str, dict[str, Any]]] = {}
        for context_type, payload in standings_payloads.items():
            standings_maps[context_type] = {}
            for row in _extract_standings_rows(payload):
                team = row.get("team") if isinstance(row.get("team"), dict) else {}
                provider_team_id = str(team.get("id")) if team.get("id") is not None else None
                if provider_team_id:
                    standings_maps[context_type][provider_team_id] = row

        for target in league_targets:
            provider_team_id = str(target["sofascore_id"])
            season_label = str(provider_season_id)
            for context_type, metrics_prefix in {
                "standings_total": "overall",
                "standings_home": "home",
                "standings_away": "away",
            }.items():
                standing_row = standings_maps.get(context_type, {}).get(provider_team_id)
                if not standing_row:
                    continue
                standing_metrics = _extract_standing_metrics(standing_row)
                team_row = {
                    "provider": "sofascore",
                    "context_type": context_type,
                    "team_id": target["team_id"],
                    "provider_team_id": provider_team_id,
                    "league_code": team_league_code,
                    "provider_league_id": provider_league_id,
                    "season_label": season_label,
                    "provider_season_id": provider_season_id,
                    "snapshot_time_utc": snapshot_time,
                    f"{metrics_prefix}_rank": standing_metrics["rank"],
                    f"{metrics_prefix}_points": standing_metrics["points"],
                    f"{metrics_prefix}_played": standing_metrics["played"],
                    f"{metrics_prefix}_wins": standing_metrics["wins"],
                    f"{metrics_prefix}_draws": standing_metrics["draws"],
                    f"{metrics_prefix}_losses": standing_metrics["losses"],
                    f"{metrics_prefix}_goals_for": standing_metrics["goals_for"],
                    f"{metrics_prefix}_goals_against": standing_metrics["goals_against"],
                    f"{metrics_prefix}_goal_diff": standing_metrics["goal_diff"],
                    "raw_json": standing_row,
                }
                _upsert_row(
                    conn,
                    "team_external_context",
                    ["provider", "context_type", "provider_team_id", "league_code", "season_label", "snapshot_time_utc"],
                    team_row,
                    dry_run,
                )
                rows_written += 1

            team_client = Team(api, int(provider_team_id))
            for context_type, fetcher in {
                "team_overview": team_client.get_team,
                "league_stats": lambda: team_client.league_stats(provider_league_id, provider_season_id),
                "performance_graph": lambda: team_client.performance_graph(provider_league_id, provider_season_id),
            }.items():
                try:
                    payload = await fetcher()
                except Exception as exc:
                    logger.warning("Failed to fetch %s for team %s: %s", context_type, provider_team_id, exc)
                    continue
                row = {
                    "provider": "sofascore",
                    "context_type": context_type,
                    "team_id": target["team_id"],
                    "provider_team_id": provider_team_id,
                    "league_code": team_league_code,
                    "provider_league_id": provider_league_id,
                    "season_label": season_label,
                    "provider_season_id": provider_season_id,
                    "snapshot_time_utc": snapshot_time,
                    "raw_json": payload,
                }
                if context_type == "team_overview":
                    row.update(_extract_team_overview_metrics(payload))
                elif context_type == "league_stats":
                    row.update(_extract_team_league_stats_metrics(payload))
                elif context_type == "performance_graph":
                    row.update(_summarize_performance_graph(payload))
                _upsert_row(
                    conn,
                    "team_external_context",
                    ["provider", "context_type", "provider_team_id", "league_code", "season_label", "snapshot_time_utc"],
                    row,
                    dry_run,
                )
                rows_written += 1
    if not dry_run:
        conn.commit()
    return rows_written


async def ingest_match_context(
    api: SofascoreAPI,
    conn,
    league_code: str | None,
    status: str,
    limit: int,
    scheduled_start_hours: int,
    scheduled_end_hours: int,
    dry_run: bool,
) -> int:
    targets = fetch_match_targets(conn, league_code, status, limit, scheduled_start_hours, scheduled_end_hours)
    if not targets:
        logger.info("No match targets found for external context ingestion.")
        return 0
    rows_written = 0
    snapshot_time = datetime.now(UTC)
    for target in targets:
        provider_fixture_id = str(target["sofascore_id"])
        match = Match(api, int(provider_fixture_id))
        event_payload: dict[str, Any] = {}
        provider_match_code = None
        current_home_id = None
        current_away_id = None
        try:
            event_payload = _extract_event_payload(await match.get_match())
            provider_match_code = event_payload.get("customId")
            home_team = event_payload.get("homeTeam") if isinstance(event_payload.get("homeTeam"), dict) else {}
            away_team = event_payload.get("awayTeam") if isinstance(event_payload.get("awayTeam"), dict) else {}
            current_home_id = str(home_team.get("id")) if home_team.get("id") is not None else None
            current_away_id = str(away_team.get("id")) if away_team.get("id") is not None else None
        except Exception as exc:
            logger.warning("Failed to fetch match overview for fixture %s: %s", provider_fixture_id, exc)

        for context_type, fetcher in {
            "pre_match_form": match.pre_match_form,
            "team_streaks": match.team_streaks,
            "win_probability": match.win_probability,
        }.items():
            try:
                payload = await fetcher()
            except Exception as exc:
                logger.warning("Failed to fetch %s for fixture %s: %s", context_type, provider_fixture_id, exc)
                continue
            row = {
                "provider": "sofascore",
                "context_type": context_type,
                "fixture_id": target["fixture_id"],
                "provider_fixture_id": provider_fixture_id,
                "snapshot_time_utc": snapshot_time,
                "provider_match_code": provider_match_code,
                "raw_json": payload,
            }
            if context_type == "pre_match_form":
                row.update(_extract_match_form_metrics(payload))
            elif context_type == "team_streaks":
                row.update(_extract_streak_metrics(payload))
            elif context_type == "win_probability":
                row.update(_extract_win_probability(payload))
            _upsert_row(
                conn,
                "match_external_context",
                ["provider", "context_type", "provider_fixture_id", "snapshot_time_utc"],
                row,
                dry_run,
            )
            rows_written += 1

        if provider_match_code:
            try:
                payload = await match.h2h_results(str(provider_match_code))
                row = {
                    "provider": "sofascore",
                    "context_type": "h2h_results",
                    "fixture_id": target["fixture_id"],
                    "provider_fixture_id": provider_fixture_id,
                    "snapshot_time_utc": snapshot_time,
                    "provider_match_code": str(provider_match_code),
                    "raw_json": payload,
                }
                row.update(_summarize_h2h_results(payload, current_home_id, current_away_id))
                _upsert_row(
                    conn,
                    "match_external_context",
                    ["provider", "context_type", "provider_fixture_id", "snapshot_time_utc"],
                    row,
                    dry_run,
                )
                rows_written += 1
            except Exception as exc:
                logger.warning("Failed to fetch h2h_results for fixture %s: %s", provider_fixture_id, exc)
    if not dry_run:
        conn.commit()
    return rows_written


async def ingest_player_context(
    api: SofascoreAPI,
    conn,
    league_code: str | None,
    status: str,
    limit: int,
    scheduled_start_hours: int,
    scheduled_end_hours: int,
    dry_run: bool,
) -> int:
    targets = fetch_player_targets(conn, league_code, status, limit, scheduled_start_hours, scheduled_end_hours)
    if not targets:
        logger.info("No player targets found for external context ingestion.")
        return 0
    rows_written = 0
    snapshot_time = datetime.now(UTC)
    for target in targets:
        provider_player_id = str(target["sofascore_id"])
        player = Player(api, int(provider_player_id))
        season_label = target.get("season") or str(target["sofascore_season_id"])
        fallback_position = target.get("position")
        fallback_market_value = _safe_int(target.get("market_value_euro"))
        for context_type, fetcher in {
            "player_overview": player.get_player,
            "attributes": player.attributes,
            "league_stats": lambda: player.league_stats(int(target["sofascore_league_id"]), int(target["sofascore_season_id"])),
        }.items():
            try:
                payload = await fetcher()
            except Exception as exc:
                logger.warning("Failed to fetch %s for player %s: %s", context_type, provider_player_id, exc)
                continue
            row = {
                "provider": "sofascore",
                "context_type": context_type,
                "player_id": target["player_id"],
                "provider_player_id": provider_player_id,
                "league_code": target["league_code"],
                "provider_league_id": target["sofascore_league_id"],
                "season_label": season_label,
                "provider_season_id": target["sofascore_season_id"],
                "snapshot_time_utc": snapshot_time,
                "raw_json": payload,
            }
            if context_type == "attributes":
                row.update(_extract_player_attributes_metrics(payload, fallback_position, fallback_market_value))
            elif context_type == "league_stats":
                row.update(_extract_player_league_stats_metrics(payload, fallback_position, fallback_market_value))
            else:
                row.update(
                    {
                        "position_group": fallback_position,
                        "market_value_euro_snapshot": fallback_market_value,
                    }
                )
            _upsert_row(
                conn,
                "player_external_context",
                ["provider", "context_type", "provider_player_id", "league_code", "season_label", "snapshot_time_utc"],
                row,
                dry_run,
            )
            rows_written += 1
    if not dry_run:
        conn.commit()
    return rows_written


async def main() -> None:
    args = parse_args()
    if not args.skip_bootstrap:
        bootstrap_schema()
    api = SofascoreAPI()
    conn = connect_db()
    try:
        total_rows = 0
        if args.context_type in {"team", "all"}:
            total_rows += await ingest_team_context(
                api,
                conn,
                args.league,
                args.status,
                args.limit,
                args.scheduled_start_hours,
                args.scheduled_end_hours,
                args.dry_run,
            )
        if args.context_type in {"match", "all"}:
            total_rows += await ingest_match_context(
                api,
                conn,
                args.league,
                args.status,
                args.limit,
                args.scheduled_start_hours,
                args.scheduled_end_hours,
                args.dry_run,
            )
        if args.context_type in {"player", "all"}:
            total_rows += await ingest_player_context(
                api,
                conn,
                args.league,
                args.status,
                args.limit,
                args.scheduled_start_hours,
                args.scheduled_end_hours,
                args.dry_run,
            )
        logger.info("External context ingestion complete. rows_written=%s dry_run=%s", total_rows, args.dry_run)
    finally:
        conn.close()
        await api.close()


if __name__ == "__main__":
    asyncio.run(main())
