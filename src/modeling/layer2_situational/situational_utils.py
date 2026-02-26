"""
Layer 2 Situational Feature Contract and Shared API.

FEATURE CONTRACT:
- Standings/State ("before kickoff" values):
    - home_points, away_points: Total points accumulated by home/away teams before kickoff.
    - home_played, away_played: Total matches played by home/away teams before kickoff.
    - home_position, away_position: League position (1-indexed) before kickoff.
- Gaps:
    - home_points_gap: home_points - away_points.
    - away_points_gap: away_points - home_points.
    - position_gap: away_position - home_position (positive means home is higher ranked).
    - points_gap: Signed difference (home_points - away_points).
- Schedule/Congestion:
    - congestion_flag: 1 if either team played 3+ games in last 14 days, else 0.
    - home_upcoming_tier, away_upcoming_tier: Upcoming high-tier match (CL/EL=3, ECL=2, else 0) within 4 days.
- Form:
    - home_form_streak, away_form_streak: Points from last 3 games before kickoff.

SEMANTICS:
- points_gap: Signed home_points - away_points. Positive means home team has more points.
- season: Registry-based season_start_year mapping from scrapers/config/league_registry.json.
          Unknown leagues default to start_month=1. Season key is the season-start-year string.
- point-in-time: All features reflect state strictly BEFORE the current fixture kickoff.
- ordering: Fixtures are sorted by match_datetime_utc ascending.

SHARED API:
- add_season_key(df: pd.DataFrame) -> pd.DataFrame
- compute_point_in_time_state(df: pd.DataFrame) -> pd.DataFrame
- build_team_schedule(all_fixtures_df: pd.DataFrame) -> pd.DataFrame
- compute_congestion_for_targets(base_team_dates: pd.DataFrame, all_team_sched: pd.DataFrame, window_days: int = 14, threshold_games: int = 3) -> pd.DataFrame
- build_euro_team_dates(euro_fixtures_df: pd.DataFrame) -> pd.DataFrame
- compute_upcoming_tier_for_targets(base_team_dates: pd.DataFrame, euro_team_dates: pd.DataFrame, horizon_days: int = 4) -> pd.DataFrame
"""

from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false

import json
from collections import deque
from pathlib import Path

import pandas as pd


_LEAGUE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[3] / "scrapers" / "config" / "league_registry.json"
)


def _as_datetime_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _load_season_start_months() -> dict[str, int]:
    with _LEAGUE_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)

    mapping: dict[str, int] = {}
    for row in registry:
        league_code = row.get("league_code")
        if isinstance(league_code, str):
            mapping[league_code] = int(row.get("season_start_month", 1))
    return mapping


def _load_relegation_spots_map() -> dict[str, int]:
    with _LEAGUE_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)

    mapping: dict[str, int] = {}
    for row in registry:
        league_code = row.get("league_code")
        if isinstance(league_code, str):
            relegation_spots = row.get("relegation_spots")
            if relegation_spots is None:
                continue
            mapping[league_code] = int(relegation_spots)
    return mapping


def infer_relegation_spots(team_count: int) -> int:
    if team_count <= 12:
        return 2
    return 3


def latest_lambda_pairs_sql() -> str:
    """
    Returns SQL that picks one deterministic lambda_home/lambda_away pair per fixture:
    the latest model_version that has both markets present for the requested model_name.
    """
    return """
        WITH lambda_rows AS (
            SELECT
                fixture_id,
                market_code,
                model_name,
                model_version,
                created_at,
                prediction_id,
                (metadata_json->>'lambda')::double precision AS lambda_value
            FROM predictions
            WHERE model_name = %s
              AND market_code IN ('lambda_home', 'lambda_away')
        ),
        pair_versions AS (
            SELECT
                fixture_id,
                model_version,
                MAX(created_at) AS latest_created_at,
                MAX(prediction_id) AS latest_prediction_id
            FROM lambda_rows
            GROUP BY fixture_id, model_version
            HAVING COUNT(DISTINCT market_code) = 2
        ),
        chosen_version AS (
            SELECT
                fixture_id,
                model_version,
                ROW_NUMBER() OVER (
                    PARTITION BY fixture_id
                    ORDER BY latest_created_at DESC, latest_prediction_id DESC, model_version DESC
                ) AS rn
            FROM pair_versions
        )
        SELECT
            ph.fixture_id,
            ph.lambda_value AS lambda_home,
            pa.lambda_value AS lambda_away,
            cv.model_version AS lambda_model_version
        FROM chosen_version cv
        JOIN lambda_rows ph
          ON ph.fixture_id = cv.fixture_id
         AND ph.model_version = cv.model_version
         AND ph.market_code = 'lambda_home'
        JOIN lambda_rows pa
          ON pa.fixture_id = cv.fixture_id
         AND pa.model_version = cv.model_version
         AND pa.market_code = 'lambda_away'
        WHERE cv.rn = 1
    """


def _row_datetime(value: object) -> pd.Timestamp:
    return _as_datetime_utc(pd.Series([value])).iloc[0]


def add_season_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'season' column based on kickoff date and league registry.
    """
    out = df.copy()
    kickoff = _as_datetime_utc(out["match_datetime_utc"])
    season_start_months = _load_season_start_months()

    if "league_code" in out.columns:
        league_start = out["league_code"].map(season_start_months)
    else:
        league_start = pd.Series(index=out.index, dtype="float64")
    start_month = league_start.fillna(1).astype("int64")

    season_year = kickoff.dt.year.where(
        kickoff.dt.month >= start_month, kickoff.dt.year - 1
    )
    out["season"] = season_year.astype("Int64").astype(str)
    return out


def compute_point_in_time_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes points and games played for each team at the time of each fixture.
    """
    out = df.copy()
    out["match_datetime_utc"] = _as_datetime_utc(out["match_datetime_utc"])
    if "season" not in out.columns:
        out = add_season_key(out)
    out = out.sort_values(
        "match_datetime_utc", ascending=True, kind="mergesort"
    ).reset_index(drop=True)

    standings_state: dict[tuple[str, str, int], dict[str, object]] = {}

    home_points: list[int] = []
    away_points: list[int] = []
    home_played: list[int] = []
    away_played: list[int] = []
    home_position: list[int] = []
    away_position: list[int] = []
    home_form_streak: list[int] = []
    away_form_streak: list[int] = []
    home_points_to_top: list[int] = []
    away_points_to_top: list[int] = []
    home_points_to_relegation: list[int] = []
    away_points_to_relegation: list[int] = []
    league_team_count: list[int] = []
    league_relegation_spots: list[int] = []

    relegation_spots_by_league = _load_relegation_spots_map()

    home_goals_col = "home_goals" if "home_goals" in out.columns else None
    away_goals_col = "away_goals" if "away_goals" in out.columns else None

    def ensure_team_state(
        state: dict[tuple[str, str, int], dict[str, object]],
        team_key: tuple[str, str, int],
    ) -> None:
        if team_key in state:
            return
        state[team_key] = {
            "points": 0,
            "played": 0,
            "recent_points": deque(maxlen=3),
        }

    for _, kickoff_group in out.groupby("match_datetime_utc", sort=False, dropna=False):
        group_rows = list(kickoff_group.itertuples(index=False))

        for row in group_rows:
            league_code = str(getattr(row, "league_code"))
            season = str(getattr(row, "season"))
            home_team_id = int(getattr(row, "home_team_id"))
            away_team_id = int(getattr(row, "away_team_id"))
            ensure_team_state(standings_state, (league_code, season, home_team_id))
            ensure_team_state(standings_state, (league_code, season, away_team_id))

        snapshot_state: dict[tuple[str, str, int], dict[str, object]] = {
            key: {
                "points": int(value["points"]),
                "played": int(value["played"]),
                "recent_points": deque(value["recent_points"], maxlen=3),
            }
            for key, value in standings_state.items()
        }

        for row in group_rows:
            league_code = str(getattr(row, "league_code"))
            season = str(getattr(row, "season"))
            home_team_id = int(getattr(row, "home_team_id"))
            away_team_id = int(getattr(row, "away_team_id"))

            home_key = (league_code, season, home_team_id)
            away_key = (league_code, season, away_team_id)
            h_state = snapshot_state[home_key]
            a_state = snapshot_state[away_key]

            h_points_before = int(h_state["points"])
            a_points_before = int(a_state["points"])
            h_played_before = int(h_state["played"])
            a_played_before = int(a_state["played"])

            league_points: dict[int, int] = {}
            for (lg, ssn, team_id), team_state in snapshot_state.items():
                if lg == league_code and ssn == season:
                    league_points[team_id] = int(team_state["points"])

            ranked = sorted(
                league_points.items(), key=lambda pair: pair[1], reverse=True
            )
            observed_team_count = len(ranked)
            inferred_team_count = max(
                4,
                observed_team_count,
                h_played_before + 1,
                a_played_before + 1,
            )
            relegation_spots = relegation_spots_by_league.get(
                league_code, infer_relegation_spots(inferred_team_count)
            )
            relegation_line = max(1, inferred_team_count - relegation_spots + 1)
            top_points = max(league_points.values()) if league_points else 0

            h_rank_before = next(
                (
                    idx + 1
                    for idx, (team_id, _) in enumerate(ranked)
                    if team_id == home_team_id
                ),
                1,
            )
            a_rank_before = next(
                (
                    idx + 1
                    for idx, (team_id, _) in enumerate(ranked)
                    if team_id == away_team_id
                ),
                1,
            )
            relegation_points_pool = sorted(league_points.values(), reverse=True)
            points_at_relegation_line = 0
            if relegation_points_pool:
                rel_idx = min(
                    max(0, relegation_line - 1), len(relegation_points_pool) - 1
                )
                points_at_relegation_line = int(relegation_points_pool[rel_idx])

            home_points.append(h_points_before)
            away_points.append(a_points_before)
            home_played.append(h_played_before)
            away_played.append(a_played_before)
            home_position.append(h_rank_before)
            away_position.append(a_rank_before)
            home_form_streak.append(sum(h_state["recent_points"]))
            away_form_streak.append(sum(a_state["recent_points"]))
            home_points_to_top.append(int(top_points - h_points_before))
            away_points_to_top.append(int(top_points - a_points_before))
            home_points_to_relegation.append(
                int(h_points_before - points_at_relegation_line)
            )
            away_points_to_relegation.append(
                int(a_points_before - points_at_relegation_line)
            )
            league_team_count.append(inferred_team_count)
            league_relegation_spots.append(relegation_spots)

        for row in group_rows:
            status = str(getattr(row, "status", "")).lower()
            if status != "ft" or home_goals_col is None or away_goals_col is None:
                continue

            home_goals = getattr(row, home_goals_col)
            away_goals = getattr(row, away_goals_col)
            if pd.isna(home_goals) or pd.isna(away_goals):
                continue

            league_code = str(getattr(row, "league_code"))
            season = str(getattr(row, "season"))
            home_team_id = int(getattr(row, "home_team_id"))
            away_team_id = int(getattr(row, "away_team_id"))

            home_key = (league_code, season, home_team_id)
            away_key = (league_code, season, away_team_id)
            h_state = standings_state[home_key]
            a_state = standings_state[away_key]

            h_pts = 0
            a_pts = 0
            if home_goals > away_goals:
                h_pts = 3
            elif away_goals > home_goals:
                a_pts = 3
            else:
                h_pts = 1
                a_pts = 1

            h_state["points"] = int(h_state["points"]) + h_pts
            h_state["played"] = int(h_state["played"]) + 1
            h_recent = h_state["recent_points"]
            h_recent.append(h_pts)

            a_state["points"] = int(a_state["points"]) + a_pts
            a_state["played"] = int(a_state["played"]) + 1
            a_recent = a_state["recent_points"]
            a_recent.append(a_pts)

    out["home_points"] = home_points
    out["away_points"] = away_points
    out["home_played"] = home_played
    out["away_played"] = away_played
    out["home_position"] = home_position
    out["away_position"] = away_position
    out["home_form_streak"] = home_form_streak
    out["away_form_streak"] = away_form_streak
    out["home_points_to_top"] = home_points_to_top
    out["away_points_to_top"] = away_points_to_top
    out["home_points_to_relegation"] = home_points_to_relegation
    out["away_points_to_relegation"] = away_points_to_relegation
    out["league_team_count"] = league_team_count
    out["league_relegation_spots"] = league_relegation_spots
    out["home_points_gap"] = out["home_points"] - out["away_points"]
    out["away_points_gap"] = -out["home_points_gap"]
    out["position_gap"] = out["away_position"] - out["home_position"]
    out["points_gap"] = out["home_points_gap"]
    return out


def build_team_schedule(all_fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens fixtures into a per-team, per-date schedule.
    """
    base = all_fixtures_df[
        ["fixture_id", "home_team_id", "away_team_id", "match_datetime_utc"]
    ].copy()
    base["match_datetime_utc"] = _as_datetime_utc(base["match_datetime_utc"])

    home = base[["fixture_id", "home_team_id", "match_datetime_utc"]].rename(
        columns={"home_team_id": "team_id"}
    )
    away = base[["fixture_id", "away_team_id", "match_datetime_utc"]].rename(
        columns={"away_team_id": "team_id"}
    )
    return pd.concat([home, away], ignore_index=True)


def compute_congestion_for_targets(
    base_team_dates: pd.DataFrame,
    all_team_sched: pd.DataFrame,
    window_days: int = 14,
    threshold_games: int = 3,
) -> pd.DataFrame:
    """
    Computes match congestion (games in window) for target team-dates.
    """
    sched = all_team_sched[["team_id", "match_datetime_utc"]].copy()
    sched["match_datetime_utc"] = _as_datetime_utc(sched["match_datetime_utc"])

    team_times: dict[int, pd.Series] = {}
    for team_id, grp in sched.groupby("team_id", sort=False):
        team_times[int(team_id)] = (
            grp["match_datetime_utc"].sort_values().reset_index(drop=True)
        )

    def count_recent(team_id: int, kickoff: pd.Timestamp) -> int:
        if pd.isna(kickoff):
            return 0
        times = team_times.get(team_id)
        if times is None or times.empty:
            return 0
        window_start = kickoff - pd.Timedelta(days=window_days)
        left = times.searchsorted(window_start, side="left")
        right = times.searchsorted(kickoff, side="left")
        return int(max(0, right - left))

    home_recent_vals: list[int] = []
    away_recent_vals: list[int] = []
    for row in base_team_dates.itertuples(index=False):
        kickoff = _row_datetime(getattr(row, "match_datetime_utc"))
        home_team_id = int(getattr(row, "home_team_id"))
        away_team_id = int(getattr(row, "away_team_id"))
        home_recent_vals.append(count_recent(home_team_id, kickoff))
        away_recent_vals.append(count_recent(away_team_id, kickoff))

    result = pd.DataFrame(
        {"home_recent": home_recent_vals, "away_recent": away_recent_vals},
        index=base_team_dates.index,
    )
    result["congestion_flag"] = (
        (result["home_recent"] >= threshold_games)
        | (result["away_recent"] >= threshold_games)
    ).astype(int)
    return result


def build_euro_team_dates(euro_fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts team-dates for European/Cup fixtures.
    """
    base = euro_fixtures_df[
        [
            "fixture_id",
            "home_team_id",
            "away_team_id",
            "match_datetime_utc",
            "league_code",
        ]
    ].copy()
    base["match_datetime_utc"] = _as_datetime_utc(base["match_datetime_utc"])
    base["tier"] = (
        base["league_code"].map({"CL": 3, "EL": 3, "ECL": 2}).fillna(0).astype(int)
    )

    home = base[["fixture_id", "home_team_id", "match_datetime_utc", "tier"]].rename(
        columns={"home_team_id": "team_id"}
    )
    away = base[["fixture_id", "away_team_id", "match_datetime_utc", "tier"]].rename(
        columns={"away_team_id": "team_id"}
    )
    return pd.concat([home, away], ignore_index=True)


def compute_upcoming_tier_for_targets(
    base_team_dates: pd.DataFrame,
    euro_team_dates: pd.DataFrame,
    horizon_days: int = 4,
) -> pd.DataFrame:
    """
    Checks if a team has a high-tier match within the horizon.
    """
    euro = euro_team_dates[["team_id", "match_datetime_utc", "tier"]].copy()
    euro["match_datetime_utc"] = _as_datetime_utc(euro["match_datetime_utc"])
    euro["tier"] = euro["tier"].fillna(0).astype(int)

    team_dates: dict[int, pd.Series] = {}
    team_tiers: dict[int, pd.Series] = {}
    for team_id, grp in euro.groupby("team_id", sort=False):
        sorted_grp = grp.sort_values("match_datetime_utc", ascending=True)
        team_key = int(team_id)
        team_dates[team_key] = sorted_grp["match_datetime_utc"].reset_index(drop=True)
        team_tiers[team_key] = sorted_grp["tier"].reset_index(drop=True)

    def max_upcoming_tier(team_id: int, kickoff: pd.Timestamp) -> int:
        if pd.isna(kickoff):
            return 0
        dates = team_dates.get(team_id)
        tiers = team_tiers.get(team_id)
        if dates is None or tiers is None or dates.empty:
            return 0
        horizon = kickoff + pd.Timedelta(days=horizon_days)
        left = dates.searchsorted(kickoff, side="right")
        right = dates.searchsorted(horizon, side="right")
        if right <= left:
            return 0
        return int(tiers.iloc[left:right].max())

    home_upcoming: list[int] = []
    away_upcoming: list[int] = []
    for row in base_team_dates.itertuples(index=False):
        kickoff = _row_datetime(getattr(row, "match_datetime_utc"))
        home_team_id = int(getattr(row, "home_team_id"))
        away_team_id = int(getattr(row, "away_team_id"))
        home_upcoming.append(max_upcoming_tier(home_team_id, kickoff))
        away_upcoming.append(max_upcoming_tier(away_team_id, kickoff))

    return pd.DataFrame(
        {
            "home_upcoming_tier": home_upcoming,
            "away_upcoming_tier": away_upcoming,
        },
        index=base_team_dates.index,
    )
