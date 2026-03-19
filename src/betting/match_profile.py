"""
match_profile.py - Main data puller for match profiles.

Pulls all relevant data for a fixture into a structured MatchProfile that can be
used for analysis, display, or feeding into downstream models.

Usage:
    python src/betting/match_profile.py --fixture 12345
    python src/betting/match_profile.py --today
    python src/betting/match_profile.py --fixture 12345 --output json
    python src/betting/match_profile.py --fixture 12345 --output markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class MissingPlayer:
    """A player who is missing or doubtful for the fixture."""
    name: str
    position: str
    market_value: float
    reason: str


@dataclass(frozen=True)
class TeamProfile:
    """Complete profile for a team in a fixture."""
    team_id: int
    name: str
    lambda_value: float
    style_cluster: str
    rolling_xg: float
    rolling_xg_against: float
    rolling_corners: float
    rolling_possession: float
    form_points_last5: int
    form_sequence: str
    league_position: int
    missing_players: list[MissingPlayer]
    missing_market_value: float


@dataclass(frozen=True)
class MatchContext:
    """Match-level context (H2H, derby, streaks)."""
    h2h_home_wins: int
    h2h_draws: int
    h2h_away_wins: int
    h2h_matches_count: int
    is_derby: bool
    rivalry_name: str | None
    home_streak: str | None
    away_streak: str | None


@dataclass(frozen=True)
class OddsContext:
    """Odds snapshot for a specific market."""
    market_code: str
    line: float | None
    odds_home: float | None
    odds_draw: float | None
    odds_away: float | None
    odds_over: float | None
    odds_under: float | None
    snapshot_type: str
    snapshot_time: datetime


@dataclass(frozen=True)
class MatchProfile:
    """Complete profile for a fixture with all relevant data."""
    fixture_id: int
    home_team: TeamProfile
    away_team: TeamProfile
    match_context: MatchContext
    odds: list[OddsContext]
    predictions: dict[str, float]
    created_at: datetime


# =============================================================================
# Helper Functions
# =============================================================================


def _safe_float(value: Any) -> float | None:
    """Safely convert to float, returning None for invalid values."""
    if value is None:
        return None
    try:
        out = float(value)
        return out if pd.notna(out) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Safely convert to int, returning None for invalid values."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    """Parse a datetime value, returning None if invalid."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def _as_json_dict(value: Any) -> dict[str, Any]:
    """Convert a JSON string or dict to a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


# =============================================================================
# Data Fetching Functions
# =============================================================================


def _fetch_fixture_info(conn, fixture_id: int) -> dict[str, Any] | None:
    """Fetch basic fixture information."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                f.fixture_id,
                f.league_code,
                f.home_team_id,
                f.away_team_id,
                f.match_datetime_utc,
                f.status,
                ht.team_name as home_team_name,
                at.team_name as away_team_name
            FROM fixtures f
            JOIN teams ht ON ht.team_id = f.home_team_id
            JOIN teams at ON at.team_id = f.away_team_id
            WHERE f.fixture_id = %s
            """,
            (fixture_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cur.description]
        return dict(zip(cols, row))


def _fetch_lambda_values(conn, fixture_id: int) -> dict[str, float]:
    """Fetch lambda values from predictions table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT market_code, metadata_json->>'lambda' as lambda_value
            FROM predictions
            WHERE fixture_id = %s
              AND model_name = 'lambda_xgb'
              AND model_version = 'v1'
              AND market_code IN ('lambda_home', 'lambda_away')
            """,
            (fixture_id,),
        )
        return {
            row[0].replace("lambda_", ""): _safe_float(row[1]) or 1.0
            for row in cur.fetchall()
        }


def _fetch_style_cluster(fixture_id: int, team_id: int) -> str:
    """Fetch style cluster from parquet file."""
    cluster_labels_path = ROOT_DIR / "model_artifacts" / "style_clusters" / "team_cluster_labels.parquet"
    if not cluster_labels_path.exists():
        return "Unknown"

    try:
        df = pd.read_parquet(cluster_labels_path)
        match = df[(df["fixture_id"] == fixture_id) & (df["team_id"] == team_id)]
        if not match.empty:
            return str(match.iloc[0].get("style_cluster", "Unknown"))
    except Exception:
        pass
    return "Unknown"


def _fetch_rolling_stats(conn, fixture_id: int, team_id: int) -> dict[str, float | None]:
    """Fetch rolling stats from team_premium_snapshots."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                rolling_xg,
                rolling_xg_against,
                rolling_corners,
                rolling_possession,
                sample_size
            FROM team_premium_snapshots
            WHERE fixture_id = %s AND team_id = %s
            """,
            (fixture_id, team_id),
        )
        row = cur.fetchone()
        if row is None:
            return {
                "rolling_xg": None,
                "rolling_xg_against": None,
                "rolling_corners": None,
                "rolling_possession": None,
                "sample_size": 0,
            }
        return {
            "rolling_xg": _safe_float(row[0]),
            "rolling_xg_against": _safe_float(row[1]),
            "rolling_corners": _safe_float(row[2]),
            "rolling_possession": _safe_float(row[3]),
            "sample_size": _safe_int(row[4]) or 0,
        }


def _fetch_form_and_position(conn, fixture_id: int) -> dict[str, Any]:
    """Fetch form and position from match_external_context."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                context_type,
                home_form_sequence,
                away_form_sequence,
                home_form_points_last5,
                away_form_points_last5,
                home_position,
                away_position
            FROM match_external_context
            WHERE fixture_id = %s
              AND context_type = 'pre_match_form'
            ORDER BY snapshot_time_utc DESC
            LIMIT 1
            """,
            (fixture_id,),
        )
        row = cur.fetchone()
        if row is None:
            return {
                "home_form_sequence": "",
                "away_form_sequence": "",
                "home_form_points_last5": 0,
                "away_form_points_last5": 0,
                "home_position": 0,
                "away_position": 0,
            }
        return {
            "home_form_sequence": str(row[1] or ""),
            "away_form_sequence": str(row[2] or ""),
            "home_form_points_last5": _safe_int(row[3]) or 0,
            "away_form_points_last5": _safe_int(row[4]) or 0,
            "home_position": _safe_int(row[5]) or 0,
            "away_position": _safe_int(row[6]) or 0,
        }


def _fetch_missing_players(conn, fixture_id: int, team_id: int) -> list[MissingPlayer]:
    """Fetch missing/doubtful players for a team."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                p.name,
                p.position,
                p.market_value_euro,
                pa.reason,
                pa.status
            FROM player_availability pa
            JOIN players p ON p.player_id = pa.player_id
            WHERE pa.fixture_id = %s
              AND pa.team_id = %s
              AND pa.status IN ('missing', 'doubtful')
            """,
            (fixture_id, team_id),
        )
        players = []
        for row in cur.fetchall():
            players.append(
                MissingPlayer(
                    name=str(row[0] or "Unknown"),
                    position=str(row[1] or "Unknown"),
                    market_value=float(row[2] or 0),
                    reason=str(row[3] or "Unknown"),
                )
            )
        return players


def _calculate_missing_market_value(missing_players: list[MissingPlayer]) -> float:
    """Calculate total market value of missing players."""
    return sum(p.market_value for p in missing_players)


def _fetch_h2h_context(conn, fixture_id: int) -> dict[str, Any]:
    """Fetch H2H context from match_external_context."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                h2h_home_wins_last_n,
                h2h_draws_last_n,
                h2h_away_wins_last_n,
                h2h_matches_count
            FROM match_external_context
            WHERE fixture_id = %s
              AND context_type = 'h2h_results'
            ORDER BY snapshot_time_utc DESC
            LIMIT 1
            """,
            (fixture_id,),
        )
        row = cur.fetchone()
        if row is None:
            return {
                "h2h_home_wins": 0,
                "h2h_draws": 0,
                "h2h_away_wins": 0,
                "h2h_matches_count": 0,
            }
        return {
            "h2h_home_wins": _safe_int(row[0]) or 0,
            "h2h_draws": _safe_int(row[1]) or 0,
            "h2h_away_wins": _safe_int(row[2]) or 0,
            "h2h_matches_count": _safe_int(row[3]) or 0,
        }


def _fetch_rivalry(conn, home_team_id: int, away_team_id: int) -> dict[str, Any]:
    """Check if this match is a derby/rivalry."""
    # Ensure consistent ordering (lower ID first)
    team_a = min(home_team_id, away_team_id)
    team_b = max(home_team_id, away_team_id)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT rivalry_name
            FROM team_rivalries
            WHERE team_id_a = %s AND team_id_b = %s
            """,
            (team_a, team_b),
        )
        row = cur.fetchone()
        if row is None:
            return {"is_derby": False, "rivalry_name": None}
        return {"is_derby": True, "rivalry_name": str(row[0])}


def _fetch_streaks(conn, fixture_id: int) -> dict[str, Any]:
    """Fetch streak information from match_external_context."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                home_streak_win,
                away_streak_win,
                home_streak_unbeaten,
                away_streak_unbeaten
            FROM match_external_context
            WHERE fixture_id = %s
              AND context_type = 'team_streaks'
            ORDER BY snapshot_time_utc DESC
            LIMIT 1
            """,
            (fixture_id,),
        )
        row = cur.fetchone()
        if row is None:
            return {
                "home_streak": None,
                "away_streak": None,
            }

        home_streak = None
        away_streak = None

        if row[0] is not None and int(row[0]) > 0:
            home_streak = f"W{row[0]}"
        elif row[2] is not None and int(row[2]) > 0:
            home_streak = f"U{row[2]}"

        if row[1] is not None and int(row[1]) > 0:
            away_streak = f"W{row[1]}"
        elif row[3] is not None and int(row[3]) > 0:
            away_streak = f"U{row[3]}"

        return {
            "home_streak": home_streak,
            "away_streak": away_streak,
        }


def _fetch_odds(conn, fixture_id: int) -> list[OddsContext]:
    """Fetch odds from fixture_odds_markets."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                market_code,
                line_num,
                odds_json,
                snapshot_type,
                snapshot_time_utc
            FROM fixture_odds_markets
            WHERE fixture_id = %s
            ORDER BY snapshot_time_utc DESC
            """,
            (fixture_id,),
        )
        odds_list = []
        for row in cur.fetchall():
            market_code = str(row[0])
            line = _safe_float(row[1])
            odds_json = _as_json_dict(row[2])
            snapshot_type = str(row[3] or "unknown")
            snapshot_time = _parse_datetime(row[4])

            # Extract prices from odds_json
            prices_latest = _as_json_dict(odds_json.get("prices_latest", {}))

            odds_context = OddsContext(
                market_code=market_code,
                line=line,
                odds_home=_safe_float(prices_latest.get("home")),
                odds_draw=_safe_float(prices_latest.get("draw")),
                odds_away=_safe_float(prices_latest.get("away")),
                odds_over=_safe_float(prices_latest.get("over")),
                odds_under=_safe_float(prices_latest.get("under")),
                snapshot_type=snapshot_type,
                snapshot_time=snapshot_time or datetime.now(timezone.utc),
            )
            odds_list.append(odds_context)

        return odds_list


def _fetch_predictions(conn, fixture_id: int) -> dict[str, float]:
    """Fetch predictions for a fixture, using best model per market family.

    Model preference based on backtest comparison:
    - Anytime markets (h_1up, a_1up, h_2up, a_2up) -> V2 (better calibrated)
    - AH2 markets -> GBM (slightly better)
    - Corners markets -> V2 (slightly better)
    - Default -> V2 (canonical model)
    """
    V2_PREFERRED = {
        "h_1up", "a_1up", "h_2up", "a_2up",
        "c75", "c85", "c95", "c105",
        "hc25", "hc35", "hc45", "hc55",
        "ac25", "ac35", "ac45", "ac55",
    }
    GBM_PREFERRED_PREFIXES = ("ah2_",)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_name, market_code, p_model, metadata_json
            FROM predictions
            WHERE fixture_id = %s
            """,
            (fixture_id,),
        )

        all_preds: dict[str, dict[str, float]] = {}
        lambda_values: dict[str, float] = {}

        for row in cur.fetchall():
            model_name = str(row[0])
            market_code = str(row[1])
            p_model = _safe_float(row[2])
            metadata = _as_json_dict(row[3])

            if "lambda" in market_code.lower():
                lambda_val = _safe_float(metadata.get("lambda"))
                if lambda_val is not None:
                    lambda_values[market_code] = lambda_val
            elif p_model is not None:
                if market_code not in all_preds:
                    all_preds[market_code] = {}
                all_preds[market_code][model_name] = p_model

        predictions: dict[str, float] = {}
        predictions.update(lambda_values)

        for market_code, models in all_preds.items():
            if market_code in V2_PREFERRED:
                if "market_outcome_v2" in models:
                    predictions[market_code] = models["market_outcome_v2"]
                elif "market_outcome_gbm" in models:
                    predictions[market_code] = models["market_outcome_gbm"]
                else:
                    predictions[market_code] = next(iter(models.values()))
            elif market_code.startswith(GBM_PREFERRED_PREFIXES):
                if "market_outcome_gbm" in models:
                    predictions[market_code] = models["market_outcome_gbm"]
                elif "market_outcome_v2" in models:
                    predictions[market_code] = models["market_outcome_v2"]
                else:
                    predictions[market_code] = next(iter(models.values()))
            else:
                if "market_outcome_v2" in models:
                    predictions[market_code] = models["market_outcome_v2"]
                elif "market_outcome_gbm" in models:
                    predictions[market_code] = models["market_outcome_gbm"]
                else:
                    predictions[market_code] = next(iter(models.values()))

        return predictions


# =============================================================================
# Main Functions
# =============================================================================


def build_match_profile(fixture_id: int) -> MatchProfile:
    """
    Pull all data for a fixture and return structured profile.
    """
    conn = connect_db()
    try:
        # Get basic fixture info
        fixture_info = _fetch_fixture_info(conn, fixture_id)
        if fixture_info is None:
            raise ValueError(f"Fixture {fixture_id} not found")

        home_team_id = int(fixture_info["home_team_id"])
        away_team_id = int(fixture_info["away_team_id"])
        home_team_name = str(fixture_info["home_team_name"])
        away_team_name = str(fixture_info["away_team_name"])

        # Fetch lambda values
        lambda_values = _fetch_lambda_values(conn, fixture_id)
        home_lambda = lambda_values.get("home", 1.0)
        away_lambda = lambda_values.get("away", 1.0)

        # Fetch style clusters
        home_style = _fetch_style_cluster(fixture_id, home_team_id)
        away_style = _fetch_style_cluster(fixture_id, away_team_id)

        # Fetch rolling stats
        home_rolling = _fetch_rolling_stats(conn, fixture_id, home_team_id)
        away_rolling = _fetch_rolling_stats(conn, fixture_id, away_team_id)

        # Fetch form and position
        form_data = _fetch_form_and_position(conn, fixture_id)

        # Fetch missing players
        home_missing = _fetch_missing_players(conn, fixture_id, home_team_id)
        away_missing = _fetch_missing_players(conn, fixture_id, away_team_id)

        # Build TeamProfiles
        home_team = TeamProfile(
            team_id=home_team_id,
            name=home_team_name,
            lambda_value=home_lambda,
            style_cluster=home_style,
            rolling_xg=home_rolling.get("rolling_xg") or 0.0,
            rolling_xg_against=home_rolling.get("rolling_xg_against") or 0.0,
            rolling_corners=home_rolling.get("rolling_corners") or 0.0,
            rolling_possession=home_rolling.get("rolling_possession") or 0.0,
            form_points_last5=form_data.get("home_form_points_last5") or 0,
            form_sequence=form_data.get("home_form_sequence") or "",
            league_position=form_data.get("home_position") or 0,
            missing_players=home_missing,
            missing_market_value=_calculate_missing_market_value(home_missing),
        )

        away_team = TeamProfile(
            team_id=away_team_id,
            name=away_team_name,
            lambda_value=away_lambda,
            style_cluster=away_style,
            rolling_xg=away_rolling.get("rolling_xg") or 0.0,
            rolling_xg_against=away_rolling.get("rolling_xg_against") or 0.0,
            rolling_corners=away_rolling.get("rolling_corners") or 0.0,
            rolling_possession=away_rolling.get("rolling_possession") or 0.0,
            form_points_last5=form_data.get("away_form_points_last5") or 0,
            form_sequence=form_data.get("away_form_sequence") or "",
            league_position=form_data.get("away_position") or 0,
            missing_players=away_missing,
            missing_market_value=_calculate_missing_market_value(away_missing),
        )

        # Fetch match context
        h2h_data = _fetch_h2h_context(conn, fixture_id)
        rivalry_data = _fetch_rivalry(conn, home_team_id, away_team_id)
        streak_data = _fetch_streaks(conn, fixture_id)

        match_context = MatchContext(
            h2h_home_wins=h2h_data.get("h2h_home_wins") or 0,
            h2h_draws=h2h_data.get("h2h_draws") or 0,
            h2h_away_wins=h2h_data.get("h2h_away_wins") or 0,
            h2h_matches_count=h2h_data.get("h2h_matches_count") or 0,
            is_derby=rivalry_data.get("is_derby") or False,
            rivalry_name=rivalry_data.get("rivalry_name"),
            home_streak=streak_data.get("home_streak"),
            away_streak=streak_data.get("away_streak"),
        )

        # Fetch odds
        odds = _fetch_odds(conn, fixture_id)

        # Fetch predictions
        predictions = _fetch_predictions(conn, fixture_id)

        return MatchProfile(
            fixture_id=fixture_id,
            home_team=home_team,
            away_team=away_team,
            match_context=match_context,
            odds=odds,
            predictions=predictions,
            created_at=datetime.now(timezone.utc),
        )
    finally:
        conn.close()


def build_match_profiles_batch(fixture_ids: list[int]) -> list[MatchProfile]:
    """
    Build profiles for multiple fixtures efficiently.
    """
    profiles = []
    for fixture_id in fixture_ids:
        try:
            profile = build_match_profile(fixture_id)
            profiles.append(profile)
        except Exception as e:
            print(f"Warning: Failed to build profile for fixture {fixture_id}: {e}")
    return profiles


def get_todays_fixtures(league: str | None = None) -> list[int]:
    """
    Get fixture_ids for today's scheduled matches.
    """
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            if league:
                cur.execute(
                    """
                    SELECT fixture_id
                    FROM fixtures
                    WHERE status = 'scheduled'
                      AND match_datetime_utc >= CURRENT_DATE
                      AND match_datetime_utc < CURRENT_DATE + INTERVAL '1 day'
                      AND league_code = %s
                    ORDER BY match_datetime_utc ASC
                    """,
                    (league,),
                )
            else:
                cur.execute(
                    """
                    SELECT fixture_id
                    FROM fixtures
                    WHERE status = 'scheduled'
                      AND match_datetime_utc >= CURRENT_DATE
                      AND match_datetime_utc < CURRENT_DATE + INTERVAL '1 day'
                    ORDER BY match_datetime_utc ASC
                    """,
                )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


# =============================================================================
# Output Formatting
# =============================================================================


def _profile_to_dict(profile: MatchProfile) -> dict[str, Any]:
    """Convert MatchProfile to a JSON-serializable dict."""
    def convert_value(v: Any) -> Any:
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, (TeamProfile, MatchContext, OddsContext, MissingPlayer)):
            return asdict(v)
        if isinstance(v, list):
            return [convert_value(item) for item in v]
        if isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        return v

    data = asdict(profile)
    return {k: convert_value(v) for k, v in data.items()}


def _format_summary(profile: MatchProfile) -> str:
    """Format a human-readable summary of the profile."""
    lines = [
        f"{'=' * 60}",
        f"MATCH PROFILE: Fixture {profile.fixture_id}",
        f"{'=' * 60}",
        "",
        "--- HOME TEAM ---",
        f"  {profile.home_team.name} (ID: {profile.home_team.team_id})",
        f"  Lambda: {profile.home_team.lambda_value:.3f}",
        f"  Style: {profile.home_team.style_cluster}",
        f"  Rolling xG: {profile.home_team.rolling_xg:.2f} (vs {profile.home_team.rolling_xg_against:.2f})",
        f"  Rolling Corners: {profile.home_team.rolling_corners:.1f}",
        f"  Possession: {profile.home_team.rolling_possession:.1f}%",
        f"  Form (last 5): {profile.home_team.form_sequence} ({profile.home_team.form_points_last5} pts)",
        f"  League Position: {profile.home_team.league_position}",
        f"  Missing Players: {len(profile.home_team.missing_players)} (Value: EUR {profile.home_team.missing_market_value:,.0f})",
        "",
        "--- AWAY TEAM ---",
        f"  {profile.away_team.name} (ID: {profile.away_team.team_id})",
        f"  Lambda: {profile.away_team.lambda_value:.3f}",
        f"  Style: {profile.away_team.style_cluster}",
        f"  Rolling xG: {profile.away_team.rolling_xg:.2f} (vs {profile.away_team.rolling_xg_against:.2f})",
        f"  Rolling Corners: {profile.away_team.rolling_corners:.1f}",
        f"  Possession: {profile.away_team.rolling_possession:.1f}%",
        f"  Form (last 5): {profile.away_team.form_sequence} ({profile.away_team.form_points_last5} pts)",
        f"  League Position: {profile.away_team.league_position}",
        f"  Missing Players: {len(profile.away_team.missing_players)} (Value: EUR {profile.away_team.missing_market_value:,.0f})",
        "",
        "--- MATCH CONTEXT ---",
        f"  H2H: {profile.match_context.h2h_home_wins}H-{profile.match_context.h2h_draws}D-{profile.match_context.h2h_away_wins}A ({profile.match_context.h2h_matches_count} matches)",
        f"  Derby: {'Yes - ' + profile.match_context.rivalry_name if profile.match_context.is_derby else 'No'}",
        f"  Home Streak: {profile.match_context.home_streak or 'N/A'}",
        f"  Away Streak: {profile.match_context.away_streak or 'N/A'}",
        "",
        "--- ODDS ---",
    ]

    if profile.odds:
        for odds in profile.odds[:10]:  # Show first 10 odds
            line_str = f" [{odds.line}]" if odds.line else ""
            prices = []
            if odds.odds_home:
                prices.append(f"H={odds.odds_home:.2f}")
            if odds.odds_draw:
                prices.append(f"D={odds.odds_draw:.2f}")
            if odds.odds_away:
                prices.append(f"A={odds.odds_away:.2f}")
            if odds.odds_over:
                prices.append(f"O={odds.odds_over:.2f}")
            if odds.odds_under:
                prices.append(f"U={odds.odds_under:.2f}")
            lines.append(f"  {odds.market_code}{line_str}: {', '.join(prices)} ({odds.snapshot_type})")
    else:
        lines.append("  No odds available")

    lines.extend([
        "",
        "--- PREDICTIONS ---",
    ])

    if profile.predictions:
        for market, value in sorted(profile.predictions.items()):
            lines.append(f"  {market}: {value:.4f}")
    else:
        lines.append("  No predictions available")

    lines.append("")
    lines.append(f"Created at: {profile.created_at.isoformat()}")

    return "\n".join(lines)


def _format_markdown(profile: MatchProfile) -> str:
    """Format profile as markdown."""
    lines = [
        f"# Match Profile: Fixture {profile.fixture_id}",
        "",
        "## Teams",
        "",
        "### Home Team",
        f"- **Name:** {profile.home_team.name}",
        f"- **Lambda:** {profile.home_team.lambda_value:.3f}",
        f"- **Style:** {profile.home_team.style_cluster}",
        f"- **Rolling xG:** {profile.home_team.rolling_xg:.2f} (conceded: {profile.home_team.rolling_xg_against:.2f})",
        f"- **Rolling Corners:** {profile.home_team.rolling_corners:.1f}",
        f"- **Possession:** {profile.home_team.rolling_possession:.1f}%",
        f"- **Form:** {profile.home_team.form_sequence} ({profile.home_team.form_points_last5} pts)",
        f"- **Position:** {profile.home_team.league_position}",
        f"- **Missing Players:** {len(profile.home_team.missing_players)} (EUR {profile.home_team.missing_market_value:,.0f})",
        "",
        "### Away Team",
        f"- **Name:** {profile.away_team.name}",
        f"- **Lambda:** {profile.away_team.lambda_value:.3f}",
        f"- **Style:** {profile.away_team.style_cluster}",
        f"- **Rolling xG:** {profile.away_team.rolling_xg:.2f} (conceded: {profile.away_team.rolling_xg_against:.2f})",
        f"- **Rolling Corners:** {profile.away_team.rolling_corners:.1f}",
        f"- **Possession:** {profile.away_team.rolling_possession:.1f}%",
        f"- **Form:** {profile.away_team.form_sequence} ({profile.away_team.form_points_last5} pts)",
        f"- **Position:** {profile.away_team.league_position}",
        f"- **Missing Players:** {len(profile.away_team.missing_players)} (EUR {profile.away_team.missing_market_value:,.0f})",
        "",
        "## Match Context",
        f"- **H2H:** {profile.match_context.h2h_home_wins}H-{profile.match_context.h2h_draws}D-{profile.match_context.h2h_away_wins}A ({profile.match_context.h2h_matches_count} matches)",
        f"- **Derby:** {'Yes - ' + profile.match_context.rivalry_name if profile.match_context.is_derby else 'No'}",
        f"- **Home Streak:** {profile.match_context.home_streak or 'N/A'}",
        f"- **Away Streak:** {profile.match_context.away_streak or 'N/A'}",
        "",
        "## Odds",
        "",
    ]

    if profile.odds:
        lines.append("| Market | Line | Home | Draw | Away | Over | Under | Type |")
        lines.append("|--------|------|------|------|------|------|-------|------|")
        for odds in profile.odds:
            lines.append(
                f"| {odds.market_code} | {odds.line or '-'} | "
                f"{odds.odds_home or '-'} | {odds.odds_draw or '-'} | {odds.odds_away or '-'} | "
                f"{odds.odds_over or '-'} | {odds.odds_under or '-'} | {odds.snapshot_type} |"
            )
    else:
        lines.append("No odds available.")

    lines.extend([
        "",
        "## Predictions",
        "",
    ])

    if profile.predictions:
        lines.append("| Market | Value |")
        lines.append("|--------|-------|")
        for market, value in sorted(profile.predictions.items()):
            lines.append(f"| {market} | {value:.4f} |")
    else:
        lines.append("No predictions available.")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build match profiles for fixtures"
    )
    parser.add_argument(
        "--fixture",
        type=int,
        help="Single fixture ID to profile",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Profile all today's fixtures",
    )
    parser.add_argument(
        "--league",
        type=str,
        help="Filter by league code (for --today)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.fixture and not args.today:
        print("Error: Must specify --fixture or --today")
        return 1

    if args.fixture:
        fixture_ids = [args.fixture]
    else:
        fixture_ids = get_todays_fixtures(league=args.league)
        if not fixture_ids:
            print("No fixtures found for today")
            return 0
        print(f"Found {len(fixture_ids)} fixtures for today")

    profiles = build_match_profiles_batch(fixture_ids)

    if not profiles:
        print("No profiles built")
        return 1

    for profile in profiles:
        if args.output == "json":
            print(json.dumps(_profile_to_dict(profile), indent=2))
        elif args.output == "markdown":
            print(_format_markdown(profile))
        else:
            print(_format_summary(profile))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
