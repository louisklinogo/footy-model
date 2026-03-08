from __future__ import annotations

from dataclasses import dataclass


AVAILABILITY_TIMING_COLUMNS = (
    "event_recorded_at",
    "first_recorded_at",
    "recorded_at",
)


@dataclass(frozen=True)
class AvailabilityFeatureConfig:
    table_available: bool
    timing_column: str | None
    players_table_available: bool


def _table_exists(cur, table_name: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
        )
        """,
        (table_name,),
    )
    row = cur.fetchone()
    return bool(row[0]) if row else False


def _column_exists(cur, table_name: str, column_name: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
        )
        """,
        (table_name, column_name),
    )
    row = cur.fetchone()
    return bool(row[0]) if row else False


def resolve_availability_feature_config(conn) -> AvailabilityFeatureConfig:
    with conn.cursor() as cur:
        table_available = _table_exists(cur, "player_availability")
        players_table_available = _table_exists(cur, "players")
        if not table_available:
            return AvailabilityFeatureConfig(
                table_available=False,
                timing_column=None,
                players_table_available=players_table_available,
            )

        timing_column = next(
            (
                column
                for column in AVAILABILITY_TIMING_COLUMNS
                if _column_exists(cur, "player_availability", column)
            ),
            None,
        )
        return AvailabilityFeatureConfig(
            table_available=True,
            timing_column=timing_column,
            players_table_available=players_table_available,
        )


def _fallback_select_sql() -> str:
    int_columns = [
        "home_availability_known",
        "away_availability_known",
        "home_lineup_known",
        "away_lineup_known",
        "home_missing_players",
        "away_missing_players",
        "home_missing_defenders",
        "home_missing_midfielders",
        "home_missing_forwards",
        "away_missing_defenders",
        "away_missing_midfielders",
        "away_missing_forwards",
        "home_availability_freshness_hours_is_missing",
        "away_availability_freshness_hours_is_missing",
        "home_lineup_freshness_hours_is_missing",
        "away_lineup_freshness_hours_is_missing",
    ]
    float_columns = [
        "home_availability_freshness_hours",
        "away_availability_freshness_hours",
        "home_lineup_freshness_hours",
        "away_lineup_freshness_hours",
    ]
    ts_columns = [
        "home_availability_last_seen_utc",
        "away_availability_last_seen_utc",
        "home_lineup_last_seen_utc",
        "away_lineup_last_seen_utc",
    ]
    parts = [f"0::int AS {column}" for column in int_columns]
    parts.extend(f"NULL::double precision AS {column}" for column in float_columns)
    parts.extend(f"NULL::timestamptz AS {column}" for column in ts_columns)
    return ",\n        ".join(parts)


def availability_feature_select_and_join(
    *,
    fixture_alias: str,
    match_time_expr: str,
    cutoff_expr: str,
    config: AvailabilityFeatureConfig,
) -> tuple[str, str]:
    if not config.table_available or not config.timing_column:
        return _fallback_select_sql(), ""

    home_team_expr = f"{fixture_alias}.home_team_id"
    away_team_expr = f"{fixture_alias}.away_team_id"
    timing_expr = f"pa.{config.timing_column}"
    player_join = ""
    role_counts = """
            0::int AS home_missing_defenders,
            0::int AS home_missing_midfielders,
            0::int AS home_missing_forwards,
            0::int AS away_missing_defenders,
            0::int AS away_missing_midfielders,
            0::int AS away_missing_forwards,
    """
    if config.players_table_available:
        player_join = "LEFT JOIN players pl ON pl.player_id = pa.player_id"
        role_counts = f"""
            COUNT(*) FILTER (
                WHERE pa.team_id = {home_team_expr}
                  AND pa.status IN ('missing', 'doubtful')
                  AND upper(COALESCE(pl.position, '')) IN ('D', 'DEF', 'DEFENDER')
            )::int AS home_missing_defenders,
            COUNT(*) FILTER (
                WHERE pa.team_id = {home_team_expr}
                  AND pa.status IN ('missing', 'doubtful')
                  AND upper(COALESCE(pl.position, '')) IN ('M', 'MID', 'MIDFIELDER')
            )::int AS home_missing_midfielders,
            COUNT(*) FILTER (
                WHERE pa.team_id = {home_team_expr}
                  AND pa.status IN ('missing', 'doubtful')
                  AND upper(COALESCE(pl.position, '')) IN ('F', 'FW', 'ATT', 'FORWARD', 'STRIKER')
            )::int AS home_missing_forwards,
            COUNT(*) FILTER (
                WHERE pa.team_id = {away_team_expr}
                  AND pa.status IN ('missing', 'doubtful')
                  AND upper(COALESCE(pl.position, '')) IN ('D', 'DEF', 'DEFENDER')
            )::int AS away_missing_defenders,
            COUNT(*) FILTER (
                WHERE pa.team_id = {away_team_expr}
                  AND pa.status IN ('missing', 'doubtful')
                  AND upper(COALESCE(pl.position, '')) IN ('M', 'MID', 'MIDFIELDER')
            )::int AS away_missing_midfielders,
            COUNT(*) FILTER (
                WHERE pa.team_id = {away_team_expr}
                  AND pa.status IN ('missing', 'doubtful')
                  AND upper(COALESCE(pl.position, '')) IN ('F', 'FW', 'ATT', 'FORWARD', 'STRIKER')
            )::int AS away_missing_forwards,
        """

    join_sql = f"""
    LEFT JOIN LATERAL (
        SELECT
            MAX(CASE WHEN pa.team_id = {home_team_expr} THEN 1 ELSE 0 END)::int AS home_availability_known,
            MAX(CASE WHEN pa.team_id = {away_team_expr} THEN 1 ELSE 0 END)::int AS away_availability_known,
            MAX(CASE WHEN pa.team_id = {home_team_expr} AND pa.status IN ('starter', 'bench') THEN 1 ELSE 0 END)::int AS home_lineup_known,
            MAX(CASE WHEN pa.team_id = {away_team_expr} AND pa.status IN ('starter', 'bench') THEN 1 ELSE 0 END)::int AS away_lineup_known,
            COUNT(*) FILTER (
                WHERE pa.team_id = {home_team_expr}
                  AND pa.status IN ('missing', 'doubtful')
            )::int AS home_missing_players,
            COUNT(*) FILTER (
                WHERE pa.team_id = {away_team_expr}
                  AND pa.status IN ('missing', 'doubtful')
            )::int AS away_missing_players,
{role_counts}
            MAX({timing_expr}) FILTER (
                WHERE pa.team_id = {home_team_expr}
            ) AS home_availability_last_seen_utc,
            MAX({timing_expr}) FILTER (
                WHERE pa.team_id = {away_team_expr}
            ) AS away_availability_last_seen_utc,
            MAX({timing_expr}) FILTER (
                WHERE pa.team_id = {home_team_expr}
                  AND pa.status IN ('starter', 'bench')
            ) AS home_lineup_last_seen_utc,
            MAX({timing_expr}) FILTER (
                WHERE pa.team_id = {away_team_expr}
                  AND pa.status IN ('starter', 'bench')
            ) AS away_lineup_last_seen_utc
        FROM player_availability pa
        {player_join}
        WHERE pa.fixture_id = {fixture_alias}.fixture_id
          AND pa.team_id IN ({home_team_expr}, {away_team_expr})
          AND {timing_expr} IS NOT NULL
          AND {timing_expr} <= {cutoff_expr}
    ) avail ON true
    """

    select_sql = f"""
        COALESCE(avail.home_availability_known, 0) AS home_availability_known,
        COALESCE(avail.away_availability_known, 0) AS away_availability_known,
        COALESCE(avail.home_lineup_known, 0) AS home_lineup_known,
        COALESCE(avail.away_lineup_known, 0) AS away_lineup_known,
        COALESCE(avail.home_missing_players, 0) AS home_missing_players,
        COALESCE(avail.away_missing_players, 0) AS away_missing_players,
        COALESCE(avail.home_missing_defenders, 0) AS home_missing_defenders,
        COALESCE(avail.home_missing_midfielders, 0) AS home_missing_midfielders,
        COALESCE(avail.home_missing_forwards, 0) AS home_missing_forwards,
        COALESCE(avail.away_missing_defenders, 0) AS away_missing_defenders,
        COALESCE(avail.away_missing_midfielders, 0) AS away_missing_midfielders,
        COALESCE(avail.away_missing_forwards, 0) AS away_missing_forwards,
        CASE
            WHEN avail.home_availability_last_seen_utc IS NULL THEN NULL
            ELSE EXTRACT(EPOCH FROM ({match_time_expr} - avail.home_availability_last_seen_utc)) / 3600.0
        END AS home_availability_freshness_hours,
        CASE
            WHEN avail.away_availability_last_seen_utc IS NULL THEN NULL
            ELSE EXTRACT(EPOCH FROM ({match_time_expr} - avail.away_availability_last_seen_utc)) / 3600.0
        END AS away_availability_freshness_hours,
        CASE
            WHEN avail.home_lineup_last_seen_utc IS NULL THEN NULL
            ELSE EXTRACT(EPOCH FROM ({match_time_expr} - avail.home_lineup_last_seen_utc)) / 3600.0
        END AS home_lineup_freshness_hours,
        CASE
            WHEN avail.away_lineup_last_seen_utc IS NULL THEN NULL
            ELSE EXTRACT(EPOCH FROM ({match_time_expr} - avail.away_lineup_last_seen_utc)) / 3600.0
        END AS away_lineup_freshness_hours,
        CASE WHEN avail.home_availability_last_seen_utc IS NULL THEN 1 ELSE 0 END AS home_availability_freshness_hours_is_missing,
        CASE WHEN avail.away_availability_last_seen_utc IS NULL THEN 1 ELSE 0 END AS away_availability_freshness_hours_is_missing,
        CASE WHEN avail.home_lineup_last_seen_utc IS NULL THEN 1 ELSE 0 END AS home_lineup_freshness_hours_is_missing,
        CASE WHEN avail.away_lineup_last_seen_utc IS NULL THEN 1 ELSE 0 END AS away_lineup_freshness_hours_is_missing,
        avail.home_availability_last_seen_utc AS home_availability_last_seen_utc,
        avail.away_availability_last_seen_utc AS away_availability_last_seen_utc,
        avail.home_lineup_last_seen_utc AS home_lineup_last_seen_utc,
        avail.away_lineup_last_seen_utc AS away_lineup_last_seen_utc
    """
    return select_sql.strip(), join_sql.rstrip()