from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import numpy as np
import pandas as pd

from src.db.db_utils import connect_db


ATTACK_POSITIONS = {"F", "FW", "ATT", "FORWARD", "STRIKER"}
MIDFIELD_POSITIONS = {"M", "MID", "MIDFIELDER", "AM", "CAM", "LM", "RM", "LW", "RW"}
DEFENSE_POSITIONS = {"D", "DEF", "DEFENDER", "CB", "LB", "RB", "LWB", "RWB"}
GOALKEEPER_POSITIONS = {"G", "GK", "GOALKEEPER"}

ANYTIME_PLAYER_CONTEXT_COLUMNS: tuple[str, ...] = (
    "home_missing_market_value_total",
    "away_missing_market_value_total",
    "home_missing_market_value_attack",
    "away_missing_market_value_attack",
    "home_missing_market_value_midfield",
    "away_missing_market_value_midfield",
    "home_missing_market_value_defense",
    "away_missing_market_value_defense",
    "home_missing_market_value_goalkeeper",
    "away_missing_market_value_goalkeeper",
    "home_missing_market_value_top1",
    "away_missing_market_value_top1",
    "home_missing_market_value_top2",
    "away_missing_market_value_top2",
    "home_missing_market_value_share",
    "away_missing_market_value_share",
    "home_listed_player_count",
    "away_listed_player_count",
    "home_starter_known_count",
    "away_starter_known_count",
    "home_bench_known_count",
    "away_bench_known_count",
    "home_missing_known_count",
    "away_missing_known_count",
    "home_bench_attack_market_value",
    "away_bench_attack_market_value",
    "home_attack_form_sum_xg",
    "away_attack_form_sum_xg",
    "home_attack_form_sum_xga",
    "away_attack_form_sum_xga",
    "home_top1_attack_xg_share",
    "away_top1_attack_xg_share",
    "home_top2_attack_xg_share",
    "away_top2_attack_xg_share",
    "home_top2_attack_xga_share",
    "away_top2_attack_xga_share",
    "home_top2_attack_xg_sum",
    "away_top2_attack_xg_sum",
    "home_bench_attack_form_xga_sum",
    "away_bench_attack_form_xga_sum",
    "home_availability_refresh_hours_dbreuse",
    "away_availability_refresh_hours_dbreuse",
)

STANDINGS_CONTEXT_COLUMNS: tuple[str, ...] = (
    "home_points_per_match",
    "away_points_per_match",
    "home_goal_diff_per_match",
    "away_goal_diff_per_match",
    "home_goals_for_per_match",
    "away_goals_for_per_match",
    "home_goals_against_per_match",
    "away_goals_against_per_match",
    "home_wins_per_match",
    "away_wins_per_match",
    "home_draws_per_match",
    "away_draws_per_match",
    "home_losses_per_match",
    "away_losses_per_match",
    "standings_points_gap",
    "standings_goal_diff_gap",
    "standings_goals_for_gap",
    "standings_goals_against_gap",
    "standings_win_rate_gap",
    "standings_draw_rate_gap",
    "standings_loss_rate_gap",
)

EXTERNAL_TEAM_MATCH_CONTEXT_COLUMNS: tuple[str, ...] = (
    "home_external_overall_rank",
    "away_external_overall_rank",
    "home_external_side_rank",
    "away_external_side_rank",
    "home_external_overall_points_per_match",
    "away_external_overall_points_per_match",
    "home_external_side_points_per_match",
    "away_external_side_points_per_match",
    "home_external_overall_goal_diff_per_match",
    "away_external_overall_goal_diff_per_match",
    "home_external_side_goal_diff_per_match",
    "away_external_side_goal_diff_per_match",
    "home_external_form_points_last5",
    "away_external_form_points_last5",
    "home_external_pregame_avg_rating",
    "away_external_pregame_avg_rating",
    "home_external_pregame_position",
    "away_external_pregame_position",
    "home_external_pregame_value",
    "away_external_pregame_value",
    "home_external_league_goals_scored_per_match",
    "away_external_league_goals_scored_per_match",
    "home_external_league_goals_conceded_per_match",
    "away_external_league_goals_conceded_per_match",
    "home_external_league_big_chances_per_match",
    "away_external_league_big_chances_per_match",
    "home_external_league_corners_per_match",
    "away_external_league_corners_per_match",
    "home_external_league_avg_ball_possession",
    "away_external_league_avg_ball_possession",
    "home_external_league_duels_won_percentage",
    "away_external_league_duels_won_percentage",
    "home_external_perf_points_avg",
    "away_external_perf_points_avg",
    "home_external_perf_goal_diff_avg",
    "away_external_perf_goal_diff_avg",
    "external_overall_rank_gap",
    "external_side_rank_gap",
    "external_overall_points_gap",
    "external_side_points_gap",
    "external_overall_goal_diff_gap",
    "external_side_goal_diff_gap",
    "external_form_points_gap",
    "external_pregame_rating_gap",
    "external_pregame_position_gap",
    "external_pregame_value_gap",
    "external_league_goals_scored_gap",
    "external_league_goals_conceded_gap",
    "external_league_big_chances_gap",
    "external_league_corners_gap",
    "external_league_avg_ball_possession_gap",
    "external_league_duels_won_percentage_gap",
    "external_perf_points_avg_gap",
    "external_perf_goal_diff_avg_gap",
    "home_external_match_form_points_last5",
    "away_external_match_form_points_last5",
    "home_external_match_avg_rating",
    "away_external_match_avg_rating",
    "home_external_match_position",
    "away_external_match_position",
    "home_external_match_value",
    "away_external_match_value",
    "home_external_streak_win",
    "away_external_streak_win",
    "home_external_streak_unbeaten",
    "away_external_streak_unbeaten",
    "external_match_form_points_gap",
    "external_match_avg_rating_gap",
    "external_match_position_gap",
    "external_match_value_gap",
    "external_streak_win_gap",
    "external_streak_unbeaten_gap",
    "external_h2h_home_wins_last_n",
    "external_h2h_draws_last_n",
    "external_h2h_away_wins_last_n",
    "external_h2h_matches_count",
    "external_h2h_home_win_rate",
    "external_h2h_draw_rate",
    "external_h2h_away_win_rate",
)


@dataclass(frozen=True)
class DbReuseFeatureConfig:
    availability_time_column: str | None
    players_table_available: bool
    fixture_player_stats_available: bool
    team_league_standings_available: bool


def _table_exists(cur, table_name: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
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


def resolve_db_reuse_feature_config() -> DbReuseFeatureConfig:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            availability_time_column = next(
                (
                    column
                    for column in (
                        "last_refreshed_at",
                        "event_recorded_at",
                        "first_recorded_at",
                        "recorded_at",
                    )
                    if _column_exists(cur, "player_availability", column)
                ),
                None,
            )
            return DbReuseFeatureConfig(
                availability_time_column=availability_time_column,
                players_table_available=_table_exists(cur, "players"),
                fixture_player_stats_available=_table_exists(cur, "fixture_player_stats"),
                team_league_standings_available=_table_exists(cur, "team_league_standings"),
            )
    finally:
        conn.close()


def _should_skip_db_enrichment() -> bool:
    return not any(
        os.getenv(key)
        for key in ("DATABASE_URL", "DEV_DATABASE_URL", "PROD_DATABASE_URL")
    )


def _normalize_position_bucket(value: object) -> str:
    token = str(value or "").strip().upper()
    if token in GOALKEEPER_POSITIONS:
        return "goalkeeper"
    if token in ATTACK_POSITIONS:
        return "attack"
    if token in DEFENSE_POSITIONS:
        return "defense"
    if token in MIDFIELD_POSITIONS:
        return "midfield"
    if token.startswith("G"):
        return "goalkeeper"
    if token.startswith("D"):
        return "defense"
    if token.startswith("F"):
        return "attack"
    if token.startswith("M"):
        return "midfield"
    return "unknown"


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_ratio(numerator: object, denominator: object) -> object:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    if isinstance(den, pd.Series):
        den = den.replace(0.0, np.nan)
    elif pd.notna(den) and float(den) == 0.0:
        den = np.nan
    return num / den


def _last_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[-1]


def _load_base_fixture_frame(conn, source_df: pd.DataFrame, fixture_ids: list[int]) -> pd.DataFrame:
    if not fixture_ids:
        return pd.DataFrame()
    frame = pd.read_sql(
        """
        SELECT
            fixture_id,
            league_code,
            season,
            home_team_id,
            away_team_id,
            match_datetime_utc
        FROM fixtures
        WHERE fixture_id = ANY(%s)
        """,
        conn,
        params=(fixture_ids,),
    )
    if frame.empty:
        return frame
    frame["match_datetime_utc"] = pd.to_datetime(frame["match_datetime_utc"], utc=True, errors="coerce")
    prediction_times = None
    if "prediction_time_utc" in source_df.columns:
        prediction_times = source_df[["fixture_id", "prediction_time_utc"]].copy()
        prediction_times["fixture_id"] = pd.to_numeric(
            prediction_times["fixture_id"], errors="coerce"
        ).astype("Int64")
        prediction_times["prediction_time_utc"] = pd.to_datetime(
            prediction_times["prediction_time_utc"], utc=True, errors="coerce"
        )
        prediction_times = prediction_times.dropna(subset=["fixture_id"]).sort_values(
            ["fixture_id", "prediction_time_utc"], kind="mergesort"
        )
        prediction_times = prediction_times.groupby("fixture_id", as_index=False).agg(
            {"prediction_time_utc": _last_non_null}
        )
        prediction_times["fixture_id"] = prediction_times["fixture_id"].astype(int)
        frame = frame.merge(prediction_times, on="fixture_id", how="left")
    else:
        frame["prediction_time_utc"] = pd.NaT
    frame["context_cutoff_time_utc"] = frame["prediction_time_utc"].where(
        frame["prediction_time_utc"].notna(),
        frame["match_datetime_utc"],
    )
    return frame


def _load_availability_context(
    conn,
    fixture_ids: list[int],
    *,
    availability_time_column: str | None,
) -> pd.DataFrame:
    if not fixture_ids or availability_time_column is None:
        return pd.DataFrame()
    query = f"""
        SELECT
            pa.fixture_id,
            pa.player_id,
            pa.team_id,
            pa.status,
            pa.{availability_time_column} AS availability_time_utc,
            p.market_value_euro,
            p.position
        FROM player_availability pa
        LEFT JOIN players p
          ON p.player_id = pa.player_id
        WHERE pa.fixture_id = ANY(%s)
          AND pa.status IN ('starter', 'bench', 'missing', 'doubtful')
    """
    frame = pd.read_sql(query, conn, params=(fixture_ids,))
    if frame.empty:
        return frame
    frame["availability_time_utc"] = pd.to_datetime(
        frame["availability_time_utc"], utc=True, errors="coerce"
    )
    frame["market_value_euro"] = _safe_numeric(frame["market_value_euro"])
    frame["position_bucket"] = frame["position"].map(_normalize_position_bucket)
    return frame


def _aggregate_missing_market_values(group: pd.DataFrame) -> pd.Series:
    missing = group[group["status"].isin(["missing", "doubtful"])].copy()
    listed_total = float(group["market_value_euro"].fillna(0.0).sum())
    missing_total = float(missing["market_value_euro"].fillna(0.0).sum())
    ordered_missing = missing.sort_values("market_value_euro", ascending=False)
    return pd.Series(
        {
            "listed_player_count": int(len(group)),
            "starter_known_count": int((group["status"] == "starter").sum()),
            "bench_known_count": int((group["status"] == "bench").sum()),
            "missing_known_count": int(group["status"].isin(["missing", "doubtful"]).sum()),
            "missing_market_value_total": missing_total,
            "missing_market_value_attack": float(
                missing.loc[missing["position_bucket"] == "attack", "market_value_euro"].fillna(0.0).sum()
            ),
            "missing_market_value_midfield": float(
                missing.loc[missing["position_bucket"] == "midfield", "market_value_euro"].fillna(0.0).sum()
            ),
            "missing_market_value_defense": float(
                missing.loc[missing["position_bucket"] == "defense", "market_value_euro"].fillna(0.0).sum()
            ),
            "missing_market_value_goalkeeper": float(
                missing.loc[missing["position_bucket"] == "goalkeeper", "market_value_euro"].fillna(0.0).sum()
            ),
            "missing_market_value_top1": float(
                ordered_missing["market_value_euro"].fillna(0.0).head(1).sum()
            ),
            "missing_market_value_top2": float(
                ordered_missing["market_value_euro"].fillna(0.0).head(2).sum()
            ),
            "missing_market_value_share": float(missing_total / listed_total) if listed_total > 0.0 else np.nan,
            "bench_attack_market_value": float(
                group.loc[
                    (group["status"] == "bench") & (group["position_bucket"] == "attack"),
                    "market_value_euro",
                ]
                .fillna(0.0)
                .sum()
            ),
            "latest_availability_refresh_utc": group["availability_time_utc"].max(),
        }
    )


def _prepare_player_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    out = history.copy()
    out["match_datetime_utc"] = pd.to_datetime(out["match_datetime_utc"], utc=True, errors="coerce")
    out["expected_goals"] = _safe_numeric(out["expected_goals"]).fillna(0.0)
    out["expected_assists"] = _safe_numeric(out["expected_assists"]).fillna(0.0)
    out["minutes_played"] = _safe_numeric(out["minutes_played"]).fillna(0.0)
    out["attack_xga"] = out["expected_goals"] + out["expected_assists"]
    out = out.sort_values(["player_id", "match_datetime_utc", "fixture_id"], kind="mergesort").reset_index(drop=True)
    by_player = out.groupby("player_id", sort=False)
    out["prior_avg_xg10"] = by_player["expected_goals"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    out["prior_avg_xga10"] = by_player["attack_xga"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    return out[["player_id", "match_datetime_utc", "prior_avg_xg10", "prior_avg_xga10"]]


def _load_player_attack_context(
    conn,
    base_fixture_frame: pd.DataFrame,
    availability_frame: pd.DataFrame,
) -> pd.DataFrame:
    if base_fixture_frame.empty or availability_frame.empty:
        return pd.DataFrame()
    player_ids = sorted(
        {int(pid) for pid in availability_frame["player_id"].dropna().astype(int).tolist()}
    )
    if not player_ids:
        return pd.DataFrame()
    history = pd.read_sql(
        """
        SELECT
            fps.player_id,
            fps.fixture_id,
            f.match_datetime_utc,
            fps.expected_goals,
            fps.expected_assists,
            fps.minutes_played
        FROM fixture_player_stats fps
        JOIN fixtures f
          ON f.fixture_id = fps.fixture_id
        WHERE fps.player_id = ANY(%s)
          AND f.match_datetime_utc IS NOT NULL
          AND f.status = 'ft'
        """,
        conn,
        params=(player_ids,),
    )
    if history.empty:
        return pd.DataFrame()
    history_prepared = _prepare_player_history(history)
    if history_prepared.empty:
        return pd.DataFrame()

    merge_columns = ["fixture_id"] + [
        column
        for column in ("match_datetime_utc", "home_team_id", "away_team_id")
        if column not in availability_frame.columns
    ]
    if len(merge_columns) > 1:
        avail = availability_frame.merge(
            base_fixture_frame[merge_columns],
            on="fixture_id",
            how="left",
        )
    else:
        avail = availability_frame.copy()
    avail = avail.sort_values(["player_id", "match_datetime_utc", "fixture_id"], kind="mergesort").reset_index(drop=True)
    history_prepared = history_prepared.sort_values(
        ["player_id", "match_datetime_utc"], kind="mergesort"
    ).reset_index(drop=True)

    history_by_player: dict[int, pd.DataFrame] = {
        int(player_id): group.reset_index(drop=True)
        for player_id, group in history_prepared.groupby("player_id", sort=False)
    }
    merged_parts: list[pd.DataFrame] = []
    for player_id, player_avail in avail.groupby("player_id", sort=False):
        player_frame = player_avail.copy().reset_index(drop=True)
        if pd.isna(player_id):
            player_frame["prior_avg_xg10"] = np.nan
            player_frame["prior_avg_xga10"] = np.nan
            merged_parts.append(player_frame)
            continue
        history_frame = history_by_player.get(int(player_id))
        if history_frame is None or history_frame.empty:
            player_frame["prior_avg_xg10"] = np.nan
            player_frame["prior_avg_xga10"] = np.nan
            merged_parts.append(player_frame)
            continue
        history_times = pd.to_datetime(history_frame["match_datetime_utc"], utc=True, errors="coerce")
        player_times = pd.to_datetime(player_frame["match_datetime_utc"], utc=True, errors="coerce")
        positions = np.searchsorted(history_times.to_numpy(), player_times.to_numpy(), side="left") - 1
        prior_xg = np.full(len(player_frame), np.nan, dtype=float)
        prior_xga = np.full(len(player_frame), np.nan, dtype=float)
        valid_mask = positions >= 0
        if valid_mask.any():
            valid_positions = positions[valid_mask]
            prior_xg[valid_mask] = history_frame["prior_avg_xg10"].to_numpy(dtype=float)[valid_positions]
            prior_xga[valid_mask] = history_frame["prior_avg_xga10"].to_numpy(dtype=float)[valid_positions]
        player_frame["prior_avg_xg10"] = prior_xg
        player_frame["prior_avg_xga10"] = prior_xga
        merged_parts.append(player_frame)
    merged = pd.concat(merged_parts, axis=0, ignore_index=True) if merged_parts else avail.copy()
    merged["prior_avg_xg10"] = _safe_numeric(merged["prior_avg_xg10"]).fillna(0.0)
    merged["prior_avg_xga10"] = _safe_numeric(merged["prior_avg_xga10"]).fillna(0.0)

    def _attack_summary(group: pd.DataFrame) -> pd.Series:
        ordered_xg = group.sort_values("prior_avg_xg10", ascending=False)
        ordered_xga = group.sort_values("prior_avg_xga10", ascending=False)
        total_xg = float(group["prior_avg_xg10"].sum())
        total_xga = float(group["prior_avg_xga10"].sum())
        return pd.Series(
            {
                "attack_form_sum_xg": total_xg,
                "attack_form_sum_xga": total_xga,
                "top1_attack_xg_share": float(ordered_xg["prior_avg_xg10"].head(1).sum() / total_xg)
                if total_xg > 0.0
                else np.nan,
                "top2_attack_xg_share": float(ordered_xg["prior_avg_xg10"].head(2).sum() / total_xg)
                if total_xg > 0.0
                else np.nan,
                "top2_attack_xga_share": float(ordered_xga["prior_avg_xga10"].head(2).sum() / total_xga)
                if total_xga > 0.0
                else np.nan,
                "top2_attack_xg_sum": float(ordered_xg["prior_avg_xg10"].head(2).sum()),
                "bench_attack_form_xga_sum": float(
                    group.loc[group["status"] == "bench", "prior_avg_xga10"].sum()
                ),
            }
        )

    return (
        merged.groupby(["fixture_id", "team_id"], dropna=False, sort=False)
        .apply(_attack_summary, include_groups=False)
        .reset_index()
    )


def _pivot_team_features(
    base_fixture_frame: pd.DataFrame,
    per_team_frame: pd.DataFrame,
    *,
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if base_fixture_frame.empty or per_team_frame.empty:
        return pd.DataFrame(index=base_fixture_frame.index)
    out = base_fixture_frame[["fixture_id", "home_team_id", "away_team_id", "match_datetime_utc"]].copy()
    metrics = [col for col in per_team_frame.columns if col not in {"fixture_id", "team_id"}]
    home = per_team_frame.rename(columns={metric: f"home_{metric}" for metric in metrics})
    away = per_team_frame.rename(columns={metric: f"away_{metric}" for metric in metrics})
    out = out.merge(
        home.rename(columns={"team_id": "home_team_id"}),
        on=["fixture_id", "home_team_id"],
        how="left",
    )
    out = out.merge(
        away.rename(columns={"team_id": "away_team_id"}),
        on=["fixture_id", "away_team_id"],
        how="left",
    )
    if rename_map:
        out = out.rename(columns=rename_map)
    drop_cols = [col for col in ("home_team_id", "away_team_id") if col in out.columns]
    return out.drop(columns=drop_cols)


def _load_standings_context(conn, base_fixture_frame: pd.DataFrame) -> pd.DataFrame:
    if base_fixture_frame.empty:
        return pd.DataFrame()
    league_codes = sorted({str(code) for code in base_fixture_frame["league_code"].dropna().astype(str).tolist()})
    if not league_codes:
        return pd.DataFrame()
    standings = pd.read_sql(
        """
        SELECT
            team_id,
            league_code,
            season,
            matches_played,
            wins,
            draws,
            losses,
            goals_for,
            goals_against,
            points,
            computed_at
        FROM team_league_standings
        WHERE league_code = ANY(%s)
        """,
        conn,
        params=(league_codes,),
    )
    if standings.empty:
        return pd.DataFrame()
    standings["matches_played"] = _safe_numeric(standings["matches_played"]).replace(0.0, np.nan)
    standings["points_per_match"] = _safe_numeric(standings["points"]) / standings["matches_played"]
    standings["goal_diff_per_match"] = (
        _safe_numeric(standings["goals_for"]) - _safe_numeric(standings["goals_against"])
    ) / standings["matches_played"]
    standings["goals_for_per_match"] = _safe_numeric(standings["goals_for"]) / standings["matches_played"]
    standings["goals_against_per_match"] = _safe_numeric(standings["goals_against"]) / standings["matches_played"]
    standings["wins_per_match"] = _safe_numeric(standings["wins"]) / standings["matches_played"]
    standings["draws_per_match"] = _safe_numeric(standings["draws"]) / standings["matches_played"]
    standings["losses_per_match"] = _safe_numeric(standings["losses"]) / standings["matches_played"]
    keep = [
        "team_id",
        "league_code",
        "season",
        "points_per_match",
        "goal_diff_per_match",
        "goals_for_per_match",
        "goals_against_per_match",
        "wins_per_match",
        "draws_per_match",
        "losses_per_match",
    ]
    standings = standings[keep].drop_duplicates(["team_id", "league_code", "season"])
    out = base_fixture_frame.copy()
    home = standings.rename(
        columns={
            "team_id": "home_team_id",
            "points_per_match": "home_points_per_match",
            "goal_diff_per_match": "home_goal_diff_per_match",
            "goals_for_per_match": "home_goals_for_per_match",
            "goals_against_per_match": "home_goals_against_per_match",
            "wins_per_match": "home_wins_per_match",
            "draws_per_match": "home_draws_per_match",
            "losses_per_match": "home_losses_per_match",
        }
    )
    away = standings.rename(
        columns={
            "team_id": "away_team_id",
            "points_per_match": "away_points_per_match",
            "goal_diff_per_match": "away_goal_diff_per_match",
            "goals_for_per_match": "away_goals_for_per_match",
            "goals_against_per_match": "away_goals_against_per_match",
            "wins_per_match": "away_wins_per_match",
            "draws_per_match": "away_draws_per_match",
            "losses_per_match": "away_losses_per_match",
        }
    )
    out = out.merge(
        home,
        on=["home_team_id", "league_code", "season"],
        how="left",
    )
    out = out.merge(
        away,
        on=["away_team_id", "league_code", "season"],
        how="left",
    )
    out["standings_points_gap"] = out["home_points_per_match"] - out["away_points_per_match"]
    out["standings_goal_diff_gap"] = out["home_goal_diff_per_match"] - out["away_goal_diff_per_match"]
    out["standings_goals_for_gap"] = out["home_goals_for_per_match"] - out["away_goals_for_per_match"]
    out["standings_goals_against_gap"] = out["home_goals_against_per_match"] - out["away_goals_against_per_match"]
    out["standings_win_rate_gap"] = out["home_wins_per_match"] - out["away_wins_per_match"]
    out["standings_draw_rate_gap"] = out["home_draws_per_match"] - out["away_draws_per_match"]
    out["standings_loss_rate_gap"] = out["home_losses_per_match"] - out["away_losses_per_match"]
    return out.drop(
        columns=[
            "home_team_id",
            "away_team_id",
            "league_code",
            "season",
            "match_datetime_utc",
            "prediction_time_utc",
            "context_cutoff_time_utc",
        ],
        errors="ignore",
    )


def _load_external_team_context(conn, base_fixture_frame: pd.DataFrame) -> pd.DataFrame:
    if base_fixture_frame.empty:
        return pd.DataFrame()
    team_ids = sorted(
        {
            int(team_id)
            for column in ("home_team_id", "away_team_id")
            for team_id in pd.to_numeric(base_fixture_frame[column], errors="coerce").dropna().astype(int).tolist()
        }
    )
    if not team_ids:
        return pd.DataFrame()
    context = pd.read_sql(
        """
        SELECT
            context_type,
            team_id,
            league_code,
            snapshot_time_utc,
            overall_rank,
            overall_points,
            overall_played,
            overall_goal_diff,
            home_rank,
            home_points,
            home_played,
            home_goal_diff,
            away_rank,
            away_points,
            away_played,
            away_goal_diff,
            form_points_last5,
            pregame_avg_rating,
            pregame_position,
            pregame_value,
            performance_graph_points_avg,
            performance_graph_goal_diff_avg,
            league_stats_matches,
            league_stats_goals_scored,
            league_stats_goals_conceded,
            league_stats_big_chances,
            league_stats_corners,
            league_stats_average_ball_possession,
            league_stats_duels_won_percentage
        FROM team_external_context
        WHERE provider = 'sofascore'
          AND team_id = ANY(%s)
          AND context_type IN (
              'standings_total',
              'standings_home',
              'standings_away',
              'team_overview',
              'league_stats',
              'performance_graph'
          )
        """,
        conn,
        params=(team_ids,),
    )
    if context.empty:
        return pd.DataFrame()
    context["snapshot_time_utc"] = pd.to_datetime(context["snapshot_time_utc"], utc=True, errors="coerce")

    def _collapse_team_side(side: str) -> pd.DataFrame:
        team_col = f"{side}_team_id"
        fixture_side = base_fixture_frame[
            ["fixture_id", "league_code", "context_cutoff_time_utc", team_col]
        ].rename(columns={team_col: "team_id"})
        merged = fixture_side.merge(context, on="team_id", how="left")
        merged = merged[
            merged["snapshot_time_utc"].notna()
            & (merged["snapshot_time_utc"] <= merged["context_cutoff_time_utc"])
            & (
                merged["league_code_y"].isna()
                | (merged["league_code_x"].astype(str) == merged["league_code_y"].astype(str))
            )
        ].copy()
        if merged.empty:
            return pd.DataFrame(columns=["fixture_id"])
        merged = merged.sort_values(["fixture_id", "context_type", "snapshot_time_utc"], kind="mergesort")
        latest = merged.drop_duplicates(["fixture_id", "context_type"], keep="last")
        value_columns = [
            col
            for col in latest.columns
            if col
            not in {
                "fixture_id",
                "league_code_x",
                "league_code_y",
                "context_cutoff_time_utc",
                "team_id",
                "context_type",
                "snapshot_time_utc",
            }
        ]
        collapsed = latest.groupby("fixture_id", as_index=False)[value_columns].agg(_last_non_null)
        return collapsed.rename(columns={col: f"{side}_{col}" for col in value_columns})

    home = _collapse_team_side("home")
    away = _collapse_team_side("away")
    out = base_fixture_frame[["fixture_id"]].drop_duplicates().copy()
    out = out.merge(home, on="fixture_id", how="left")
    out = out.merge(away, on="fixture_id", how="left")

    out["home_external_overall_rank"] = _safe_numeric(out.get("home_overall_rank"))
    out["away_external_overall_rank"] = _safe_numeric(out.get("away_overall_rank"))
    out["home_external_side_rank"] = _safe_numeric(out.get("home_home_rank"))
    out["away_external_side_rank"] = _safe_numeric(out.get("away_away_rank"))
    out["home_external_overall_points_per_match"] = _safe_ratio(
        out.get("home_overall_points"), out.get("home_overall_played")
    )
    out["away_external_overall_points_per_match"] = _safe_ratio(
        out.get("away_overall_points"), out.get("away_overall_played")
    )
    out["home_external_side_points_per_match"] = _safe_ratio(
        out.get("home_home_points"), out.get("home_home_played")
    )
    out["away_external_side_points_per_match"] = _safe_ratio(
        out.get("away_away_points"), out.get("away_away_played")
    )
    out["home_external_overall_goal_diff_per_match"] = _safe_ratio(
        out.get("home_overall_goal_diff"), out.get("home_overall_played")
    )
    out["away_external_overall_goal_diff_per_match"] = _safe_ratio(
        out.get("away_overall_goal_diff"), out.get("away_overall_played")
    )
    out["home_external_side_goal_diff_per_match"] = _safe_ratio(
        out.get("home_home_goal_diff"), out.get("home_home_played")
    )
    out["away_external_side_goal_diff_per_match"] = _safe_ratio(
        out.get("away_away_goal_diff"), out.get("away_away_played")
    )
    out["home_external_form_points_last5"] = _safe_numeric(out.get("home_form_points_last5"))
    out["away_external_form_points_last5"] = _safe_numeric(out.get("away_form_points_last5"))
    out["home_external_pregame_avg_rating"] = _safe_numeric(out.get("home_pregame_avg_rating"))
    out["away_external_pregame_avg_rating"] = _safe_numeric(out.get("away_pregame_avg_rating"))
    out["home_external_pregame_position"] = _safe_numeric(out.get("home_pregame_position"))
    out["away_external_pregame_position"] = _safe_numeric(out.get("away_pregame_position"))
    out["home_external_pregame_value"] = _safe_numeric(out.get("home_pregame_value"))
    out["away_external_pregame_value"] = _safe_numeric(out.get("away_pregame_value"))
    out["home_external_league_goals_scored_per_match"] = _safe_ratio(
        out.get("home_league_stats_goals_scored"), out.get("home_league_stats_matches")
    )
    out["away_external_league_goals_scored_per_match"] = _safe_ratio(
        out.get("away_league_stats_goals_scored"), out.get("away_league_stats_matches")
    )
    out["home_external_league_goals_conceded_per_match"] = _safe_ratio(
        out.get("home_league_stats_goals_conceded"), out.get("home_league_stats_matches")
    )
    out["away_external_league_goals_conceded_per_match"] = _safe_ratio(
        out.get("away_league_stats_goals_conceded"), out.get("away_league_stats_matches")
    )
    out["home_external_league_big_chances_per_match"] = _safe_ratio(
        out.get("home_league_stats_big_chances"), out.get("home_league_stats_matches")
    )
    out["away_external_league_big_chances_per_match"] = _safe_ratio(
        out.get("away_league_stats_big_chances"), out.get("away_league_stats_matches")
    )
    out["home_external_league_corners_per_match"] = _safe_ratio(
        out.get("home_league_stats_corners"), out.get("home_league_stats_matches")
    )
    out["away_external_league_corners_per_match"] = _safe_ratio(
        out.get("away_league_stats_corners"), out.get("away_league_stats_matches")
    )
    out["home_external_league_avg_ball_possession"] = _safe_numeric(
        out.get("home_league_stats_average_ball_possession")
    )
    out["away_external_league_avg_ball_possession"] = _safe_numeric(
        out.get("away_league_stats_average_ball_possession")
    )
    out["home_external_league_duels_won_percentage"] = _safe_numeric(
        out.get("home_league_stats_duels_won_percentage")
    )
    out["away_external_league_duels_won_percentage"] = _safe_numeric(
        out.get("away_league_stats_duels_won_percentage")
    )
    out["home_external_perf_points_avg"] = _safe_numeric(out.get("home_performance_graph_points_avg"))
    out["away_external_perf_points_avg"] = _safe_numeric(out.get("away_performance_graph_points_avg"))
    out["home_external_perf_goal_diff_avg"] = _safe_numeric(out.get("home_performance_graph_goal_diff_avg"))
    out["away_external_perf_goal_diff_avg"] = _safe_numeric(out.get("away_performance_graph_goal_diff_avg"))
    out["external_overall_rank_gap"] = out["away_external_overall_rank"] - out["home_external_overall_rank"]
    out["external_side_rank_gap"] = out["away_external_side_rank"] - out["home_external_side_rank"]
    out["external_overall_points_gap"] = (
        out["home_external_overall_points_per_match"] - out["away_external_overall_points_per_match"]
    )
    out["external_side_points_gap"] = (
        out["home_external_side_points_per_match"] - out["away_external_side_points_per_match"]
    )
    out["external_overall_goal_diff_gap"] = (
        out["home_external_overall_goal_diff_per_match"] - out["away_external_overall_goal_diff_per_match"]
    )
    out["external_side_goal_diff_gap"] = (
        out["home_external_side_goal_diff_per_match"] - out["away_external_side_goal_diff_per_match"]
    )
    out["external_form_points_gap"] = (
        out["home_external_form_points_last5"] - out["away_external_form_points_last5"]
    )
    out["external_pregame_rating_gap"] = (
        out["home_external_pregame_avg_rating"] - out["away_external_pregame_avg_rating"]
    )
    out["external_pregame_position_gap"] = (
        out["away_external_pregame_position"] - out["home_external_pregame_position"]
    )
    out["external_pregame_value_gap"] = (
        out["home_external_pregame_value"] - out["away_external_pregame_value"]
    )
    out["external_league_goals_scored_gap"] = (
        out["home_external_league_goals_scored_per_match"] - out["away_external_league_goals_scored_per_match"]
    )
    out["external_league_goals_conceded_gap"] = (
        out["home_external_league_goals_conceded_per_match"] - out["away_external_league_goals_conceded_per_match"]
    )
    out["external_league_big_chances_gap"] = (
        out["home_external_league_big_chances_per_match"] - out["away_external_league_big_chances_per_match"]
    )
    out["external_league_corners_gap"] = (
        out["home_external_league_corners_per_match"] - out["away_external_league_corners_per_match"]
    )
    out["external_league_avg_ball_possession_gap"] = (
        out["home_external_league_avg_ball_possession"] - out["away_external_league_avg_ball_possession"]
    )
    out["external_league_duels_won_percentage_gap"] = (
        out["home_external_league_duels_won_percentage"] - out["away_external_league_duels_won_percentage"]
    )
    out["external_perf_points_avg_gap"] = (
        out["home_external_perf_points_avg"] - out["away_external_perf_points_avg"]
    )
    out["external_perf_goal_diff_avg_gap"] = (
        out["home_external_perf_goal_diff_avg"] - out["away_external_perf_goal_diff_avg"]
    )
    keep = ["fixture_id", *EXTERNAL_TEAM_MATCH_CONTEXT_COLUMNS]
    keep = [column for column in keep if column in out.columns]
    return out[keep]


def _load_external_match_context(conn, base_fixture_frame: pd.DataFrame) -> pd.DataFrame:
    if base_fixture_frame.empty:
        return pd.DataFrame()
    fixture_ids = sorted(
        {int(fixture_id) for fixture_id in pd.to_numeric(base_fixture_frame["fixture_id"], errors="coerce").dropna().astype(int).tolist()}
    )
    if not fixture_ids:
        return pd.DataFrame()
    context = pd.read_sql(
        """
        SELECT
            fixture_id,
            context_type,
            snapshot_time_utc,
            home_form_points_last5,
            away_form_points_last5,
            home_avg_rating,
            away_avg_rating,
            home_position,
            away_position,
            home_value,
            away_value,
            home_streak_win,
            away_streak_win,
            home_streak_unbeaten,
            away_streak_unbeaten,
            h2h_home_wins_last_n,
            h2h_draws_last_n,
            h2h_away_wins_last_n,
            h2h_matches_count
        FROM match_external_context
        WHERE provider = 'sofascore'
          AND fixture_id = ANY(%s)
          AND context_type IN ('pre_match_form', 'team_streaks', 'h2h_results')
        """,
        conn,
        params=(fixture_ids,),
    )
    if context.empty:
        return pd.DataFrame()
    context["snapshot_time_utc"] = pd.to_datetime(context["snapshot_time_utc"], utc=True, errors="coerce")
    merged = base_fixture_frame[["fixture_id", "context_cutoff_time_utc"]].merge(
        context,
        on="fixture_id",
        how="left",
    )
    merged = merged[
        merged["snapshot_time_utc"].notna()
        & (merged["snapshot_time_utc"] <= merged["context_cutoff_time_utc"])
    ].copy()
    if merged.empty:
        return pd.DataFrame()
    merged = merged.sort_values(["fixture_id", "context_type", "snapshot_time_utc"], kind="mergesort")
    latest = merged.drop_duplicates(["fixture_id", "context_type"], keep="last")
    value_columns = [
        col
        for col in latest.columns
        if col not in {"fixture_id", "context_cutoff_time_utc", "context_type", "snapshot_time_utc"}
    ]
    out = latest.groupby("fixture_id", as_index=False)[value_columns].agg(_last_non_null)
    out["home_external_match_form_points_last5"] = _safe_numeric(out.get("home_form_points_last5"))
    out["away_external_match_form_points_last5"] = _safe_numeric(out.get("away_form_points_last5"))
    out["home_external_match_avg_rating"] = _safe_numeric(out.get("home_avg_rating"))
    out["away_external_match_avg_rating"] = _safe_numeric(out.get("away_avg_rating"))
    out["home_external_match_position"] = _safe_numeric(out.get("home_position"))
    out["away_external_match_position"] = _safe_numeric(out.get("away_position"))
    out["home_external_match_value"] = _safe_numeric(out.get("home_value"))
    out["away_external_match_value"] = _safe_numeric(out.get("away_value"))
    out["home_external_streak_win"] = _safe_numeric(out.get("home_streak_win"))
    out["away_external_streak_win"] = _safe_numeric(out.get("away_streak_win"))
    out["home_external_streak_unbeaten"] = _safe_numeric(out.get("home_streak_unbeaten"))
    out["away_external_streak_unbeaten"] = _safe_numeric(out.get("away_streak_unbeaten"))
    out["external_match_form_points_gap"] = (
        out["home_external_match_form_points_last5"] - out["away_external_match_form_points_last5"]
    )
    out["external_match_avg_rating_gap"] = (
        out["home_external_match_avg_rating"] - out["away_external_match_avg_rating"]
    )
    out["external_match_position_gap"] = (
        out["away_external_match_position"] - out["home_external_match_position"]
    )
    out["external_match_value_gap"] = (
        out["home_external_match_value"] - out["away_external_match_value"]
    )
    out["external_streak_win_gap"] = out["home_external_streak_win"] - out["away_external_streak_win"]
    out["external_streak_unbeaten_gap"] = (
        out["home_external_streak_unbeaten"] - out["away_external_streak_unbeaten"]
    )
    out["external_h2h_home_wins_last_n"] = _safe_numeric(out.get("h2h_home_wins_last_n"))
    out["external_h2h_draws_last_n"] = _safe_numeric(out.get("h2h_draws_last_n"))
    out["external_h2h_away_wins_last_n"] = _safe_numeric(out.get("h2h_away_wins_last_n"))
    out["external_h2h_matches_count"] = _safe_numeric(out.get("h2h_matches_count"))
    h2h_den = out["external_h2h_matches_count"].replace(0.0, np.nan)
    out["external_h2h_home_win_rate"] = out["external_h2h_home_wins_last_n"] / h2h_den
    out["external_h2h_draw_rate"] = out["external_h2h_draws_last_n"] / h2h_den
    out["external_h2h_away_win_rate"] = out["external_h2h_away_wins_last_n"] / h2h_den
    keep = ["fixture_id", *EXTERNAL_TEAM_MATCH_CONTEXT_COLUMNS]
    keep = [column for column in keep if column in out.columns]
    return out[keep]


def add_db_reuse_context_features(
    df: pd.DataFrame,
    *,
    include_anytime_player_context: bool = False,
    include_standings_context: bool = False,
    include_external_team_match_context: bool = False,
) -> pd.DataFrame:
    has_anytime_context = all(column in df.columns for column in ANYTIME_PLAYER_CONTEXT_COLUMNS)
    has_standings_context = all(column in df.columns for column in STANDINGS_CONTEXT_COLUMNS)
    has_external_context = all(column in df.columns for column in EXTERNAL_TEAM_MATCH_CONTEXT_COLUMNS)
    if (not include_anytime_player_context or has_anytime_context) and (
        not include_standings_context or has_standings_context
    ) and (
        not include_external_team_match_context or has_external_context
    ):
        return df
    if df.empty or "fixture_id" not in df.columns or _should_skip_db_enrichment():
        return df
    fixture_ids = [int(x) for x in pd.to_numeric(df["fixture_id"], errors="coerce").dropna().astype(int).tolist()]
    if not fixture_ids:
        return df
    try:
        config = resolve_db_reuse_feature_config()
        conn = connect_db()
    except Exception:
        return df
    try:
        base_fixture_frame = _load_base_fixture_frame(conn, df, fixture_ids)
        if base_fixture_frame.empty:
            return df
        out = df.copy()
        if include_anytime_player_context:
            availability = _load_availability_context(
                conn,
                fixture_ids,
                availability_time_column=config.availability_time_column,
            )
            if not availability.empty:
                availability = availability.merge(
                    base_fixture_frame[["fixture_id", "match_datetime_utc"]],
                    on="fixture_id",
                    how="left",
                )
                availability = availability[
                    availability["availability_time_utc"].isna()
                    | (availability["availability_time_utc"] <= availability["match_datetime_utc"])
                ].copy()
                per_team_value = (
                    availability.groupby(["fixture_id", "team_id"], dropna=False, sort=False)
                    .apply(_aggregate_missing_market_values, include_groups=False)
                    .reset_index()
                )
                attack_context = (
                    _load_player_attack_context(conn, base_fixture_frame, availability)
                    if config.fixture_player_stats_available
                    else pd.DataFrame()
                )
                if attack_context.empty or "fixture_id" not in attack_context.columns:
                    combined = per_team_value
                else:
                    combined = per_team_value.merge(
                        attack_context,
                        on=["fixture_id", "team_id"],
                        how="outer",
                    )
                player_context = _pivot_team_features(base_fixture_frame, combined)
                if not player_context.empty:
                    player_context["home_availability_refresh_hours_dbreuse"] = np.where(
                        player_context["home_latest_availability_refresh_utc"].notna(),
                        (
                            pd.to_datetime(player_context["match_datetime_utc"], utc=True, errors="coerce")
                            - pd.to_datetime(
                                player_context["home_latest_availability_refresh_utc"], utc=True, errors="coerce"
                            )
                        )
                        .dt.total_seconds()
                        / 3600.0,
                        np.nan,
                    )
                    player_context["away_availability_refresh_hours_dbreuse"] = np.where(
                        player_context["away_latest_availability_refresh_utc"].notna(),
                        (
                            pd.to_datetime(player_context["match_datetime_utc"], utc=True, errors="coerce")
                            - pd.to_datetime(
                                player_context["away_latest_availability_refresh_utc"], utc=True, errors="coerce"
                            )
                        )
                        .dt.total_seconds()
                        / 3600.0,
                        np.nan,
                    )
                    drop_cols = [
                        col
                        for col in (
                            "match_datetime_utc",
                            "home_latest_availability_refresh_utc",
                            "away_latest_availability_refresh_utc",
                        )
                        if col in player_context.columns
                    ]
                    out = out.merge(
                        player_context.drop(columns=drop_cols),
                        on="fixture_id",
                        how="left",
                    )
        if include_standings_context and config.team_league_standings_available:
            standings_context = _load_standings_context(conn, base_fixture_frame)
            if not standings_context.empty:
                out = out.merge(standings_context, on="fixture_id", how="left")
        if include_external_team_match_context:
            external_team_context = _load_external_team_context(conn, base_fixture_frame)
            if not external_team_context.empty:
                out = out.merge(external_team_context, on="fixture_id", how="left")
            external_match_context = _load_external_match_context(conn, base_fixture_frame)
            if not external_match_context.empty:
                out = out.merge(external_match_context, on="fixture_id", how="left")
        return out
    finally:
        conn.close()


def build_feature_coverage_snapshot(df: pd.DataFrame, feature_names: Iterable[str]) -> list[dict[str, object]]:
    total_rows = int(len(df))
    snapshot: list[dict[str, object]] = []
    for feature_name in feature_names:
        present = feature_name in df.columns
        non_null = int(df[feature_name].notna().sum()) if present else 0
        snapshot.append(
            {
                "feature": str(feature_name),
                "present": bool(present),
                "non_null_rows": non_null,
                "coverage_ratio": (float(non_null) / float(total_rows)) if total_rows > 0 else 0.0,
            }
        )
    return snapshot
