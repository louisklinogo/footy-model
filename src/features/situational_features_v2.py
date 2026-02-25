"""
Situational Features V2 - Enhanced context for match predictions.

Features computed:
1. rest_delta - Days rest differential (home - away)
2. congestion_flag - 3+ matches in 14 days for either team
3. position_gap - Points difference between the two teams (1-pt = volatile)
4. games_remaining_home - Matches left for home team this season
5. games_remaining_away - Matches left for away team this season
6. season_pct_complete - % of season completed
7. games_played_home - Games played by home team (for filtering)
8. games_played_away - Games played by away team (for filtering)
9. has_upcoming_big_match_home - UCL/UEL within 7 days (requires European fixtures)
10. has_upcoming_big_match_away - UCL/UEL within 7 days (requires European fixtures)
11. motivation_tier_home - 0-3 based on league position stakes
12. motivation_tier_away - 0-3 based on league position stakes
13. missing_players_home - Count (requires fixture_availability data)
14. missing_players_away - Count (requires fixture_availability data)
15. is_valid_for_training - Flag: both teams played >= MIN_GAMES_FOR_TRAINING

CRITICAL: Early Season Problem
-----------------------------
Position/motivation features are NOISY early in season.
Training should filter to fixtures where both teams have >= 6 games played.
Use `is_valid_for_training` column to filter training data.

Design notes:
- Features 1-8: Work with existing data
- Features 9-10: Require European cup fixtures in DB
- Features 11-12: Derived from standings
- Features 13-14: Require fixture_availability populated (use heuristics for prediction)

Usage:
    python src/features/situational_features_v2.py --output data/v1/situational_features_v2.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

# European cup league codes for "big match" detection
EUROPEAN_CUP_CODES = ["CL", "EL", "ECL"]

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false


# =============================================================================
# FEATURE COMPUTATION FUNCTIONS
# =============================================================================


def compute_rest_delta(fixture_id: int, conn: Any) -> float | None:
    """
    Compute rest days delta (home rest - away rest).
    """
    query = """
    WITH current_match AS (
        SELECT home_team_id, away_team_id, match_datetime_utc
        FROM fixtures WHERE fixture_id = %s
    ),
    home_prev AS (
        SELECT f.match_datetime_utc 
        FROM fixtures f, current_match cm
        WHERE (f.home_team_id = cm.home_team_id OR f.away_team_id = cm.home_team_id)
          AND f.match_datetime_utc < cm.match_datetime_utc
          AND f.status = 'ft'
        ORDER BY f.match_datetime_utc DESC LIMIT 1
    ),
    away_prev AS (
        SELECT f.match_datetime_utc 
        FROM fixtures f, current_match cm
        WHERE (f.home_team_id = cm.away_team_id OR f.away_team_id = cm.away_team_id)
          AND f.match_datetime_utc < cm.match_datetime_utc
          AND f.status = 'ft'
        ORDER BY f.match_datetime_utc DESC LIMIT 1
    )
    SELECT 
        EXTRACT(DAY FROM (cm.match_datetime_utc - hp.match_datetime_utc)) AS home_rest,
        EXTRACT(DAY FROM (cm.match_datetime_utc - ap.match_datetime_utc)) AS away_rest
    FROM current_match cm
    LEFT JOIN home_prev hp ON true
    LEFT JOIN away_prev ap ON true
    """

    with conn.cursor() as cur:
        cur.execute(query, (fixture_id,))
        row = cur.fetchone()
        if row and row[0] is not None and row[1] is not None:
            return float(row[0]) - float(row[1])
    return None


def compute_congestion_flag(fixture_id: int, conn: Any, window_days: int = 14) -> int:
    """
    Check if either team has played 3+ matches in the last N days.

    Returns 1 if congested, 0 otherwise.
    """
    query = """
    WITH current_fixture AS (
        SELECT 
            fixture_id,
            home_team_id,
            away_team_id,
            match_datetime_utc
        FROM fixtures
        WHERE fixture_id = %s
    ),
    recent_matches AS (
        SELECT 
            cf.fixture_id,
            COUNT(CASE WHEN f.home_team_id = cf.home_team_id OR f.away_team_id = cf.home_team_id THEN 1 END) AS home_recent,
            COUNT(CASE WHEN f.home_team_id = cf.away_team_id OR f.away_team_id = cf.away_team_id THEN 1 END) AS away_recent
        FROM current_fixture cf
        JOIN fixtures f
          ON (f.home_team_id = cf.home_team_id OR f.away_team_id = cf.home_team_id
              OR f.home_team_id = cf.away_team_id OR f.away_team_id = cf.away_team_id)
         AND f.match_datetime_utc < cf.match_datetime_utc
         AND f.match_datetime_utc >= cf.match_datetime_utc - (%s || ' days')::interval
         AND f.status = 'ft'
        GROUP BY cf.fixture_id
    )
    SELECT 
        CASE WHEN home_recent >= 3 OR away_recent >= 3 THEN 1 ELSE 0 END AS congested
    FROM recent_matches
    """

    with conn.cursor() as cur:
        cur.execute(query, (fixture_id, window_days))
        row = cur.fetchone()
        return int(row[0]) if row else 0


def compute_position_gap(fixture_id: int, conn: Any) -> float | None:
    """
    Compute the points gap between the two teams.

    KEY INSIGHT: Teams separated by 1 point are VOLATILE (your observation).

    Returns:
        Positive = Home has more points
        Negative = Away has more points
        Small absolute value (0-3) = Volatile matchup
    """
    query = """
    WITH current_fixture AS (
        SELECT 
            fixture_id,
            home_team_id,
            away_team_id,
            league_code,
            season,
            match_datetime_utc
        FROM fixtures
        WHERE fixture_id = %s
    ),
    standings AS (
        SELECT 
            team_id,
            points,
            rank() OVER (ORDER BY points DESC) AS position
        FROM team_league_standings tls
        JOIN current_fixture cf ON tls.league_code = cf.league_code
        WHERE tls.season = cf.season
    )
    SELECT 
        home.points AS home_pts,
        away.points AS away_pts
    FROM current_fixture cf
    JOIN standings home ON home.team_id = cf.home_team_id
    JOIN standings away ON away.team_id = cf.away_team_id
    """

    with conn.cursor() as cur:
        cur.execute(query, (fixture_id,))
        row = cur.fetchone()
        if row and row[0] is not None and row[1] is not None:
            return float(row[0]) - float(row[1])
    return None


def compute_games_remaining(
    fixture_id: int, conn: Any
) -> tuple[int | None, int | None, float | None]:
    """
    Compute games remaining for each team and season completion %.

    Returns: (home_remaining, away_remaining, season_pct_complete)
    """
    query = """
    WITH current_fixture AS (
        SELECT 
            fixture_id,
            home_team_id,
            away_team_id,
            league_code,
            season,
            match_datetime_utc
        FROM fixtures
        WHERE fixture_id = %s
    ),
    league_season AS (
        SELECT 
            f.league_code,
            f.season,
            COUNT(DISTINCT f.fixture_id) AS total_fixtures,
            COUNT(DISTINCT CASE WHEN f.status = 'ft' THEN f.fixture_id END) AS played_fixtures
        FROM fixtures f
        JOIN current_fixture cf ON f.league_code = cf.league_code
        WHERE f.season = cf.season
        GROUP BY f.league_code, f.season
    ),
    team_games AS (
        SELECT 
            cf.home_team_id AS team_id,
            COUNT(CASE WHEN f.status = 'ft' THEN 1 END) AS games_played
        FROM current_fixture cf
        JOIN fixtures f ON (f.home_team_id = cf.home_team_id OR f.away_team_id = cf.home_team_id)
        WHERE f.season = cf.season
          AND f.league_code = cf.league_code
        GROUP BY cf.home_team_id
        UNION ALL
        SELECT 
            cf.away_team_id AS team_id,
            COUNT(CASE WHEN f.status = 'ft' THEN 1 END) AS games_played
        FROM current_fixture cf
        JOIN fixtures f ON (f.home_team_id = cf.away_team_id OR f.away_team_id = cf.away_team_id)
        WHERE f.season = cf.season
          AND f.league_code = cf.league_code
        GROUP BY cf.away_team_id
    ),
    team_totals AS (
        SELECT team_id, MAX(games_played) AS games_played
        FROM team_games
        GROUP BY team_id
    )
    SELECT 
        tt_home.games_played AS home_played,
        tt_away.games_played AS away_played,
        ls.total_fixtures,
        ls.played_fixtures,
        ls.total_fixtures::float / NULLIF(
            (SELECT COUNT(DISTINCT team_id) FROM team_league_standings WHERE league_code = cf.league_code) / 2, 0
        ) AS expected_team_games
    FROM current_fixture cf
    JOIN league_season ls ON ls.league_code = cf.league_code
    LEFT JOIN team_totals tt_home ON tt_home.team_id = cf.home_team_id
    LEFT JOIN team_totals tt_away ON tt_away.team_id = cf.away_team_id
    """

    with conn.cursor() as cur:
        cur.execute(query, (fixture_id,))
        row = cur.fetchone()
        if row and row[2] is not None:
            total_league_fixtures = row[2]
            # Estimate games per team (total fixtures * 2 teams / number of teams)
            # For a 20-team league: 380 fixtures, each team plays 38 games
            # Simplified: assume each team plays total_fixtures / (teams/2) games
            # We'll use a standard approach: most leagues have each team play (n-1)*2 games
            # For now, use played as proxy and estimate remaining
            home_played = row[0] if row[0] else 0
            away_played = row[1] if row[1] else 0

            # Season % complete from league perspective
            season_pct = row[3] / row[2] if row[2] else None

            # Estimate total games per team (typically 38 for major leagues, 34 for others)
            # Use 38 as default, could be refined per league
            estimated_total_games = 38

            home_remaining = max(0, estimated_total_games - home_played)
            away_remaining = max(0, estimated_total_games - away_played)

            return home_remaining, away_remaining, season_pct, home_played, away_played

    return None, None, None, 0, 0


def compute_upcoming_big_match(
    fixture_id: int, conn: Any, days_ahead: int = 7
) -> tuple[int, int]:
    """
    Check if either team has a European cup match within N days after this fixture.

    Returns: (home_has_big_match, away_has_big_match)
    """
    query = """
    WITH current_fixture AS (
        SELECT 
            fixture_id,
            home_team_id,
            away_team_id,
            match_datetime_utc
        FROM fixtures
        WHERE fixture_id = %s
    ),
    upcoming_european AS (
        SELECT 
            cf.fixture_id AS current_fixture_id,
            CASE WHEN eu.home_team_id = cf.home_team_id OR eu.away_team_id = cf.home_team_id THEN 1 ELSE 0 END AS home_has_european,
            CASE WHEN eu.home_team_id = cf.away_team_id OR eu.away_team_id = cf.away_team_id THEN 1 ELSE 0 END AS away_has_european
        FROM current_fixture cf
        JOIN fixtures eu ON (eu.home_team_id = cf.home_team_id OR eu.away_team_id = cf.home_team_id
                              OR eu.home_team_id = cf.away_team_id OR eu.away_team_id = cf.away_team_id)
         AND eu.match_datetime_utc > cf.match_datetime_utc
         AND eu.match_datetime_utc <= cf.match_datetime_utc + (%s || ' days')::interval
         AND eu.league_code = ANY(%s)
    )
    SELECT 
        COALESCE(MAX(home_has_european), 0) AS home_has_big_match,
        COALESCE(MAX(away_has_european), 0) AS away_has_big_match
    FROM upcoming_european
    GROUP BY current_fixture_id
    """

    with conn.cursor() as cur:
        cur.execute(query, (fixture_id, days_ahead, EUROPEAN_CUP_CODES))
        row = cur.fetchone()
        if row:
            return int(row[0]), int(row[1])
    return 0, 0


def compute_motivation_tier(
    fixture_id: int, conn: Any
) -> tuple[int | None, int | None]:
    """
    Compute motivation tier (0-3) based on league position stakes.

    Tiers:
        3 = Title race (top 3, within 6 pts of 1st)
        2 = European spots (positions 4-6)
        1 = Relegation battle (bottom 3)
        0 = Mid-table (nothing to play for)

    Returns: (home_tier, away_tier)
    """
    query = """
    WITH current_fixture AS (
        SELECT 
            fixture_id,
            home_team_id,
            away_team_id,
            league_code,
            season,
            match_datetime_utc
        FROM fixtures
        WHERE fixture_id = %s
    ),
    standings AS (
        SELECT 
            team_id,
            points,
            rank() OVER (ORDER BY points DESC) AS position,
            COUNT(*) OVER () AS total_teams,
            FIRST_VALUE(points) OVER (ORDER BY points DESC) AS leader_points
        FROM team_league_standings tls
        JOIN current_fixture cf ON tls.league_code = cf.league_code
        WHERE tls.season = cf.season
    )
    SELECT 
        home.position AS home_pos,
        home.points AS home_pts,
        home.leader_points AS leader_pts,
        home.total_teams AS total_teams,
        away.position AS away_pos,
        away.points AS away_pts
    FROM current_fixture cf
    JOIN standings home ON home.team_id = cf.home_team_id
    JOIN standings away ON away.team_id = cf.away_team_id
    """

    def get_tier(pos: int, pts: int, leader_pts: int, total_teams: int) -> int:
        """Calculate motivation tier."""
        # Title race: top 3 within 6 points of leader
        if pos <= 3 and (leader_pts - pts) <= 6:
            return 3
        # European spots: positions 4-6
        if 4 <= pos <= 6:
            return 2
        # Relegation battle: bottom 3
        if pos >= total_teams - 2:
            return 1
        # Mid-table
        return 0

    with conn.cursor() as cur:
        cur.execute(query, (fixture_id,))
        row = cur.fetchone()
        if row and all(v is not None for v in row):
            home_pos, home_pts, leader_pts, total_teams, away_pos, away_pts = row
            home_tier = get_tier(
                int(home_pos), int(home_pts), int(leader_pts), int(total_teams)
            )
            away_tier = get_tier(
                int(away_pos), int(away_pts), int(leader_pts), int(total_teams)
            )
            return home_tier, away_tier
    return None, None


def compute_missing_players_count(fixture_id: int, conn: Any) -> tuple[int, int]:
    """
    Count missing players for each team from player_availability.
    """
    query = """
    SELECT 
        COUNT(*) FILTER (WHERE team_id = (SELECT home_team_id FROM fixtures WHERE fixture_id = %s) AND status IN ('missing', 'doubtful')) AS home_missing,
        COUNT(*) FILTER (WHERE team_id = (SELECT away_team_id FROM fixtures WHERE fixture_id = %s) AND status IN ('missing', 'doubtful')) AS away_missing
    FROM player_availability
    WHERE fixture_id = %s
    """

    with conn.cursor() as cur:
        cur.execute(query, (fixture_id, fixture_id, fixture_id))
        row = cur.fetchone()
        if row:
            return int(row[0]), int(row[1])
    return 0, 0


def compute_star_xg_lost(fixture_id: int, conn: Any) -> tuple[float, float]:
    """
    Compute the historical xG contribution of missing/doubtful players.
    
    Logic:
    1. Find missing/doubtful players for this fixture.
    2. Get their average xG from their last 10 appearances prior to this match date.
    3. Sum by team.
    """
    query = """
    WITH current_fixture AS (
        SELECT home_team_id, away_team_id, match_datetime_utc 
        FROM fixtures WHERE fixture_id = %s
    ),
    missing_players AS (
        SELECT pa.player_id, pa.team_id, cf.match_datetime_utc
        FROM player_availability pa
        CROSS JOIN current_fixture cf
        WHERE pa.fixture_id = %s AND pa.status IN ('missing', 'doubtful')
    ),
    player_impact AS (
        SELECT 
            mp.team_id,
            mp.player_id,
            AVG(fps.expected_goals) as avg_xg
        FROM missing_players mp
        JOIN fixture_player_stats fps ON fps.player_id = mp.player_id
        JOIN fixtures f ON f.fixture_id = fps.fixture_id
        WHERE f.match_datetime_utc < mp.match_datetime_utc
        GROUP BY mp.team_id, mp.player_id
    )
    SELECT 
        COALESCE(SUM(avg_xg) FILTER (WHERE team_id = (SELECT home_team_id FROM current_fixture)), 0) as home_xg_lost,
        COALESCE(SUM(avg_xg) FILTER (WHERE team_id = (SELECT away_team_id FROM current_fixture)), 0) as away_xg_lost
    FROM player_impact
    """

    with conn.cursor() as cur:
        cur.execute(query, (fixture_id, fixture_id))
        row = cur.fetchone()
        if row:
            return float(row[0]), float(row[1])
    return 0.0, 0.0


def compute_missing_player_impact(fixture_id: int, conn: Any) -> float:
    """
    Compute net situational impact of missing players.
    
    Impact = (Away xG Lost) - (Home xG Lost)
    Positive value = Home Advantage (Home improved relative to Away)
    Negative value = Away Advantage
    """
    h_lost, a_lost = compute_star_xg_lost(fixture_id, conn)
    return a_lost - h_lost


# =============================================================================
# MAIN FEATURE BUILDER
# =============================================================================


def build_all_features(fixture_id: int, conn: Any) -> dict[str, Any]:
    """Build all situational features for a single fixture."""

    # Core features (work with existing data)
    rest_delta = compute_rest_delta(fixture_id, conn)
    congestion_flag = compute_congestion_flag(fixture_id, conn)
    position_gap = compute_position_gap(fixture_id, conn)
    home_remaining, away_remaining, season_pct, home_played, away_played = compute_games_remaining(
        fixture_id, conn
    )
    home_tier, away_tier = compute_motivation_tier(fixture_id, conn)

    # European cup features (requires European fixtures)
    home_big_match, away_big_match = compute_upcoming_big_match(fixture_id, conn)

    # Injury features (normalized player_availability)
    home_missing, away_missing = compute_missing_players_count(fixture_id, conn)
    home_xg_lost, away_xg_lost = compute_star_xg_lost(fixture_id, conn)
    injury_impact = away_xg_lost - home_xg_lost

    return {
        "fixture_id": fixture_id,
        "rest_delta": rest_delta,
        "congestion_flag": congestion_flag,
        "position_gap": position_gap,
        "games_remaining_home": home_remaining,
        "games_remaining_away": away_remaining,
        "games_played_home": home_played,
        "games_played_away": away_played,
        "season_pct_complete": season_pct,
        "has_upcoming_big_match_home": home_big_match,
        "has_upcoming_big_match_away": away_big_match,
        "motivation_tier_home": home_tier,
        "motivation_tier_away": away_tier,
        "missing_players_home": home_missing,
        "missing_players_away": away_missing,
        "home_xg_lost": home_xg_lost,
        "away_xg_lost": away_xg_lost,
        "injury_impact": injury_impact,
        "is_valid_for_training": home_played >= 6 and away_played >= 6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build situational features V2")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/v1/situational_features_v2.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of fixtures to process",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="ft",
        help="Fixture status filter (ft, scheduled, all)",
    )
    args = parser.parse_args()

    print(f"Building situational features V2...")
    print(f"Status filter: {args.status}")

    conn = connect_db()

    try:
        # Get fixtures
        query = """
        SELECT fixture_id FROM fixtures
        WHERE status = %s
        ORDER BY match_datetime_utc ASC
        """
        params = [args.status]

        if args.status == "all":
            query = "SELECT fixture_id FROM fixtures ORDER BY match_datetime_utc ASC"
            params = []

        with conn.cursor() as cur:
            cur.execute(query, params)
            fixture_ids = [row[0] for row in cur.fetchall()]

        if args.limit:
            fixture_ids = fixture_ids[: args.limit]

        print(f"Processing {len(fixture_ids)} fixtures...")

        # Build features
        features = []
        for i, fixture_id in enumerate(fixture_ids):
            feat = build_all_features(fixture_id, conn)
            features.append(feat)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(fixture_ids)}")

        # Save to CSV
        df = pd.DataFrame(features)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)

        print(f"\nSaved {len(df)} rows to {args.output}")
        print(f"\nFeature statistics:")
        print(df.describe())

        # Show data quality
        print(f"\n=== DATA QUALITY ===")
        print(
            f"rest_delta null: {df['rest_delta'].isna().sum()} ({df['rest_delta'].isna().mean() * 100:.1f}%)"
        )
        print(
            f"position_gap null: {df['position_gap'].isna().sum()} ({df['position_gap'].isna().mean() * 100:.1f}%)"
        )
        print(f"missing_players_home > 0: {(df['missing_players_home'] > 0).sum()}")
        print(
            f"has_upcoming_big_match_home > 0: {(df['has_upcoming_big_match_home'] > 0).sum()}"
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
