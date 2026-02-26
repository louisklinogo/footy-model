from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.layer2_situational import situational_utils
from src.modeling.layer2_situational.train_situational_residual import (
    add_odds_model_gap,
    load_feature_data,
)
from src.modeling.layer2_situational.validate_player_impact_assumptions import (
    _build_team_side_frame,
    _fit_player_model,
    run_football_logic_checks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep player-impact feature variants (xG-lost + key-absence thresholds)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation"),
        help="Output directory for sweep artifacts.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="player_impact_variant_sweep",
        help="Output file stem.",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=4,
        help="Minimum games filter used by Layer 2 residual training.",
    )
    parser.add_argument(
        "--xg-thresholds",
        type=str,
        default="0.08,0.10,0.12",
        help="Comma-separated xG share thresholds for key-absence flag.",
    )
    parser.add_argument(
        "--minutes-thresholds",
        type=str,
        default="0.07,0.09,0.11",
        help="Comma-separated minutes share thresholds for key-absence flag.",
    )
    return parser.parse_args()


def _parse_float_csv(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one numeric threshold.")
    return values


def _load_base_frame() -> pd.DataFrame:
    df = load_feature_data()
    df = add_odds_model_gap(df)
    return df


def _load_missing_events_and_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = connect_db()
    try:
        missing = pd.read_sql(
            """
            SELECT
                pa.fixture_id,
                pa.player_id,
                pa.team_id,
                f.home_team_id,
                f.away_team_id,
                f.league_code,
                f.match_datetime_utc
            FROM player_availability pa
            JOIN fixtures f ON f.fixture_id = pa.fixture_id
            WHERE f.status = 'ft'
              AND pa.status IN ('missing', 'doubtful')
            """,
            conn,
        )
        history = pd.read_sql(
            """
            SELECT
                fps.fixture_id,
                fps.player_id,
                fps.team_id,
                f.league_code,
                f.match_datetime_utc,
                COALESCE(fps.expected_goals, 0)::double precision AS expected_goals,
                COALESCE(fps.minutes_played, 0)::double precision AS minutes_played,
                CASE WHEN fps.substituted_in = FALSE THEN 1 ELSE 0 END AS is_start
            FROM fixture_player_stats fps
            JOIN fixtures f ON f.fixture_id = fps.fixture_id
            WHERE f.status = 'ft'
            """,
            conn,
        )
        return missing, history
    finally:
        conn.close()


def _prepare_history_snapshots(history: pd.DataFrame) -> pd.DataFrame:
    hist = history.copy()
    hist["match_datetime_utc"] = pd.to_datetime(
        hist["match_datetime_utc"], utc=True, errors="coerce"
    )
    hist = situational_utils.add_season_key(hist)
    hist = hist.sort_values(
        ["player_id", "match_datetime_utc", "fixture_id"], kind="mergesort"
    ).reset_index(drop=True)

    hist["expected_goals"] = hist["expected_goals"].astype(float)
    hist["minutes_played"] = hist["minutes_played"].astype(float)
    hist["is_start"] = hist["is_start"].astype(float)
    hist["xg90_match"] = np.where(
        hist["minutes_played"] > 0,
        hist["expected_goals"] * 90.0 / hist["minutes_played"],
        np.nan,
    )

    by_player = hist.groupby("player_id", sort=False)
    hist["apps10_prev"] = by_player["expected_goals"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).count()
    )
    hist["avg_xg10_prev"] = by_player["expected_goals"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    hist["starts10_prev"] = by_player["is_start"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum()
    )
    hist["minutes10_prev"] = by_player["minutes_played"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum()
    )
    hist["start_rate10_prev"] = hist["starts10_prev"] / hist["apps10_prev"].clip(lower=1)
    hist["minute_rate10_prev"] = hist["minutes10_prev"] / (
        hist["apps10_prev"].clip(lower=1) * 90.0
    )
    hist["xg90_ewm_prev"] = by_player["xg90_match"].transform(
        lambda s: s.shift(1).ewm(alpha=0.30, adjust=False, min_periods=3).mean()
    )

    by_player_season = ["player_id", "season"]
    hist["season_apps_prev"] = hist.groupby(by_player_season).cumcount()
    hist["player_xg_cum_prev"] = hist.groupby(by_player_season)["expected_goals"].transform(
        lambda s: s.cumsum().shift(1)
    )
    hist["player_minutes_cum_prev"] = hist.groupby(by_player_season)[
        "minutes_played"
    ].transform(lambda s: s.cumsum().shift(1))
    hist["season_xg90_prev"] = np.where(
        hist["player_minutes_cum_prev"] > 0,
        hist["player_xg_cum_prev"] * 90.0 / hist["player_minutes_cum_prev"],
        np.nan,
    )

    team_fixture = (
        hist.groupby(
            ["team_id", "season", "fixture_id", "match_datetime_utc"], as_index=False
        )
        .agg(
            team_xg_fixture=("expected_goals", "sum"),
            team_minutes_fixture=("minutes_played", "sum"),
        )
        .sort_values(["team_id", "season", "match_datetime_utc", "fixture_id"])
    )
    team_fixture["team_xg_cum_prev"] = team_fixture.groupby(["team_id", "season"])[
        "team_xg_fixture"
    ].transform(lambda s: s.cumsum().shift(1))
    team_fixture["team_minutes_cum_prev"] = team_fixture.groupby(["team_id", "season"])[
        "team_minutes_fixture"
    ].transform(lambda s: s.cumsum().shift(1))

    hist = hist.merge(
        team_fixture[
            [
                "team_id",
                "season",
                "fixture_id",
                "team_xg_cum_prev",
                "team_minutes_cum_prev",
            ]
        ],
        on=["team_id", "season", "fixture_id"],
        how="left",
    )
    hist["xg_share_season_prev"] = np.where(
        hist["team_xg_cum_prev"] > 0,
        hist["player_xg_cum_prev"] / hist["team_xg_cum_prev"],
        np.nan,
    )
    hist["minutes_share_season_prev"] = np.where(
        hist["team_minutes_cum_prev"] > 0,
        hist["player_minutes_cum_prev"] / hist["team_minutes_cum_prev"],
        np.nan,
    )

    keep_cols = [
        "player_id",
        "match_datetime_utc",
        "avg_xg10_prev",
        "apps10_prev",
        "start_rate10_prev",
        "minute_rate10_prev",
        "xg90_ewm_prev",
        "season_xg90_prev",
        "season_apps_prev",
        "xg_share_season_prev",
        "minutes_share_season_prev",
    ]
    return hist[keep_cols].copy()


def _attach_missing_profiles(missing: pd.DataFrame, player_snap: pd.DataFrame) -> pd.DataFrame:
    miss = missing.copy()
    miss["match_datetime_utc"] = pd.to_datetime(
        miss["match_datetime_utc"], utc=True, errors="coerce"
    )
    miss = situational_utils.add_season_key(miss)

    right_cols = [
        "match_datetime_utc",
        "avg_xg10_prev",
        "apps10_prev",
        "start_rate10_prev",
        "minute_rate10_prev",
        "xg90_ewm_prev",
        "season_xg90_prev",
        "season_apps_prev",
        "xg_share_season_prev",
        "minutes_share_season_prev",
    ]
    right_by_player: dict[int, pd.DataFrame] = {}
    for player_id, grp in player_snap.groupby("player_id", sort=False):
        right_by_player[int(player_id)] = grp[right_cols].sort_values(
            "match_datetime_utc"
        )

    parts: list[pd.DataFrame] = []
    for player_id, grp in miss.groupby("player_id", sort=False):
        left_grp = grp.sort_values("match_datetime_utc")
        right_grp = right_by_player.get(int(player_id))
        if right_grp is None or right_grp.empty:
            merged_grp = left_grp.copy()
        else:
            merged_grp = pd.merge_asof(
                left_grp,
                right_grp,
                on="match_datetime_utc",
                direction="backward",
                allow_exact_matches=False,
            )
        parts.append(merged_grp)

    enriched = pd.concat(parts, ignore_index=True)
    for col in (
        "avg_xg10_prev",
        "apps10_prev",
        "start_rate10_prev",
        "minute_rate10_prev",
        "xg90_ewm_prev",
        "season_xg90_prev",
        "season_apps_prev",
        "xg_share_season_prev",
        "minutes_share_season_prev",
    ):
        if col not in enriched.columns:
            enriched[col] = np.nan
    return enriched


def _variant_contribution(enriched: pd.DataFrame, variant_name: str) -> pd.Series:
    apps10 = enriched["apps10_prev"].fillna(0.0)
    season_apps = enriched["season_apps_prev"].fillna(0.0)
    minute_rate = enriched["minute_rate10_prev"].fillna(0.0).clip(lower=0.0, upper=1.0)
    start_rate = enriched["start_rate10_prev"].fillna(0.0).clip(lower=0.0, upper=1.0)

    if variant_name == "baseline_avg_xg10":
        value = enriched["avg_xg10_prev"].fillna(0.0)
        return np.where(apps10 >= 3, value, 0.0)

    if variant_name == "season_xg90":
        value = enriched["season_xg90_prev"].fillna(0.0) * minute_rate
        return np.where(season_apps >= 5, value, 0.0)

    if variant_name == "recency_xg90":
        value = enriched["xg90_ewm_prev"].fillna(0.0) * minute_rate
        return np.where(apps10 >= 5, value, 0.0)

    if variant_name == "starts_minutes_weighted":
        role_weight = (0.5 * start_rate) + (0.5 * minute_rate)
        value = enriched["avg_xg10_prev"].fillna(0.0) * role_weight
        return np.where(apps10 >= 3, value, 0.0)

    raise ValueError(f"Unknown variant: {variant_name}")


def _build_player_features(
    base_df: pd.DataFrame,
    enriched: pd.DataFrame,
    variant_name: str,
    xg_share_threshold: float,
    minutes_share_threshold: float,
) -> pd.DataFrame:
    work = enriched.copy()
    work["player_contrib"] = _variant_contribution(work, variant_name).astype(float)
    work["key_flag"] = (
        (work["season_apps_prev"].fillna(0) >= 5)
        & (work["xg_share_season_prev"].fillna(0.0) >= xg_share_threshold)
        & (work["minutes_share_season_prev"].fillna(0.0) >= minutes_share_threshold)
    ).astype(int)

    team = (
        work.groupby(["fixture_id", "team_id"], as_index=False)
        .agg(
            team_xg_lost=("player_contrib", "sum"),
            has_key_absence=("key_flag", "max"),
        )
        .copy()
    )

    fixture_map = base_df[["fixture_id", "home_team_id", "away_team_id"]].drop_duplicates()
    out = fixture_map.copy()
    out = out.merge(
        team.rename(
            columns={
                "team_id": "home_team_id",
                "team_xg_lost": "home_xg_lost",
                "has_key_absence": "home_key_absent",
            }
        ),
        on=["fixture_id", "home_team_id"],
        how="left",
    )
    out = out.merge(
        team.rename(
            columns={
                "team_id": "away_team_id",
                "team_xg_lost": "away_xg_lost",
                "has_key_absence": "away_key_absent",
            }
        ),
        on=["fixture_id", "away_team_id"],
        how="left",
    )
    out["home_xg_lost"] = out["home_xg_lost"].fillna(0.0).astype(float)
    out["away_xg_lost"] = out["away_xg_lost"].fillna(0.0).astype(float)
    out["home_key_absent"] = out["home_key_absent"].fillna(0).astype(int)
    out["away_key_absent"] = out["away_key_absent"].fillna(0).astype(int)
    out["injury_impact"] = out["home_xg_lost"] - out["away_xg_lost"]
    return out[
        [
            "fixture_id",
            "home_xg_lost",
            "away_xg_lost",
            "home_key_absent",
            "away_key_absent",
            "injury_impact",
        ]
    ].copy()


def _prepare_train_test(df: pd.DataFrame, min_games: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.sort_values("match_datetime_utc").reset_index(drop=True).copy()
    split_time = work["match_datetime_utc"].dropna().quantile(0.8)
    train_df = work[work["match_datetime_utc"] <= split_time].copy()
    test_df = work[work["match_datetime_utc"] > split_time].copy()

    train_df = train_df[
        (train_df["home_played"] >= min_games)
        & (train_df["away_played"] >= min_games)
        & train_df["lambda_home"].notna()
        & train_df["lambda_away"].notna()
    ].copy()
    test_df = test_df[
        (test_df["home_played"] >= min_games)
        & (test_df["away_played"] >= min_games)
        & test_df["lambda_home"].notna()
        & test_df["lambda_away"].notna()
    ].copy()

    train_df["home_residual"] = train_df["home_goals"] - train_df["lambda_home"]
    train_df["away_residual"] = train_df["away_goals"] - train_df["lambda_away"]
    test_df["home_residual"] = test_df["home_goals"] - test_df["lambda_home"]
    test_df["away_residual"] = test_df["away_goals"] - test_df["lambda_away"]
    return train_df, test_df


def _evaluate_predictive(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, Any]:
    runs = {
        "key_absent_only": _fit_player_model(
            train_df, test_df, ["home_key_absent", "away_key_absent"]
        ),
        "player_block_all": _fit_player_model(
            train_df,
            test_df,
            [
                "home_xg_lost",
                "away_xg_lost",
                "home_key_absent",
                "away_key_absent",
                "injury_impact",
            ],
        ),
    }
    best_triggered_name = max(
        runs, key=lambda name: runs[name].get("triggered_combined_lift", -9999.0)
    )
    best_global_name = max(runs, key=lambda name: runs[name].get("global_combined_lift", -9999.0))
    best_triggered = runs[best_triggered_name]
    best_global = runs[best_global_name]

    status = "warn"
    if (
        best_triggered["triggered_combined_lift"] >= 0.002
        and best_global["global_combined_lift"] >= 0.0
    ):
        status = "pass"
    elif best_triggered["triggered_combined_lift"] <= 0.0:
        status = "fail"

    return {
        "status": status,
        "runs": runs,
        "best_by_triggered_lift": {"name": best_triggered_name, **best_triggered},
        "best_by_global_lift": {"name": best_global_name, **best_global},
    }


def main() -> None:
    args = parse_args()
    xg_thresholds = _parse_float_csv(args.xg_thresholds)
    minutes_thresholds = _parse_float_csv(args.minutes_thresholds)

    print("Loading baseline FT frame...")
    base_df = _load_base_frame()

    print("Loading missing events and player history...")
    missing, history = _load_missing_events_and_history()
    print(f"  missing rows: {len(missing)}")
    print(f"  player history rows: {len(history)}")

    print("Preparing player history snapshots...")
    player_snap = _prepare_history_snapshots(history)
    print("Attaching player snapshots to missing events...")
    enriched = _attach_missing_profiles(missing, player_snap)

    variants = [
        "baseline_avg_xg10",
        "season_xg90",
        "recency_xg90",
        "starts_minutes_weighted",
    ]

    rows: list[dict[str, Any]] = []
    for variant in variants:
        for xg_thr in xg_thresholds:
            for min_thr in minutes_thresholds:
                feature_df = _build_player_features(
                    base_df,
                    enriched,
                    variant_name=variant,
                    xg_share_threshold=xg_thr,
                    minutes_share_threshold=min_thr,
                )
                eval_df = base_df.drop(
                    columns=[
                        "home_xg_lost",
                        "away_xg_lost",
                        "home_key_absent",
                        "away_key_absent",
                        "injury_impact",
                    ],
                    errors="ignore",
                ).merge(feature_df, on="fixture_id", how="left")
                eval_df["home_xg_lost"] = eval_df["home_xg_lost"].fillna(0.0)
                eval_df["away_xg_lost"] = eval_df["away_xg_lost"].fillna(0.0)
                eval_df["home_key_absent"] = eval_df["home_key_absent"].fillna(0).astype(int)
                eval_df["away_key_absent"] = eval_df["away_key_absent"].fillna(0).astype(int)
                eval_df["injury_impact"] = eval_df["injury_impact"].fillna(0.0)

                side_frame = _build_team_side_frame(eval_df)
                football_logic = run_football_logic_checks(side_frame)
                train_df, test_df = _prepare_train_test(eval_df, min_games=args.min_games)
                predictive = _evaluate_predictive(train_df, test_df)

                best_triggered = predictive["best_by_triggered_lift"]
                best_global = predictive["best_by_global_lift"]
                logic_pass = bool(
                    football_logic["corr_xg_lost_vs_residual"] < 0
                    and football_logic["mean_residual_delta_key1_minus_key0"] < 0
                )
                predictive_pass = predictive["status"] == "pass"
                rows.append(
                    {
                        "variant": variant,
                        "xg_share_threshold": xg_thr,
                        "minutes_share_threshold": min_thr,
                        "football_logic_status": football_logic["status"],
                        "predictive_status": predictive["status"],
                        "logic_pass": logic_pass,
                        "predictive_pass": predictive_pass,
                        "corr_xg_lost_vs_residual": football_logic["corr_xg_lost_vs_residual"],
                        "key_absent_residual_delta": football_logic[
                            "mean_residual_delta_key1_minus_key0"
                        ],
                        "triggered_share": football_logic["triggered_share"],
                        "best_triggered_model": best_triggered["name"],
                        "best_triggered_lift": best_triggered["triggered_combined_lift"],
                        "best_triggered_global_lift": best_triggered["global_combined_lift"],
                        "best_global_model": best_global["name"],
                        "best_global_lift": best_global["global_combined_lift"],
                        "train_rows": int(len(train_df)),
                        "test_rows": int(len(test_df)),
                    }
                )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        [
            "predictive_pass",
            "logic_pass",
            "best_triggered_lift",
            "best_triggered_global_lift",
        ],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    best_row = result_df.iloc[0].to_dict() if not result_df.empty else None
    pass_rows = result_df[
        (result_df["predictive_pass"] == True) & (result_df["logic_pass"] == True)
    ]
    decision = (
        "promote_best_variant" if len(pass_rows) > 0 else "keep_player_block_conditional"
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "FT-only player-impact variant sweep (xg_lost definitions + key_absent thresholds)",
        "grid": {
            "variants": variants,
            "xg_share_thresholds": xg_thresholds,
            "minutes_share_thresholds": minutes_thresholds,
        },
        "n_runs": int(len(result_df)),
        "n_pass_runs": int(len(pass_rows)),
        "decision": decision,
        "best_run": best_row,
        "top10": result_df.head(10).to_dict(orient="records"),
        "all_runs": result_df.to_dict(orient="records"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    md_path = args.output_dir / f"{args.output_stem}.md"
    csv_path = args.output_dir / f"{args.output_stem}.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result_df.to_csv(csv_path, index=False, encoding="utf-8")

    lines: list[str] = []
    lines.append("# Player-Impact Variant Sweep")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- decision: `{decision}`")
    lines.append(f"- runs: `{len(result_df)}`")
    lines.append(f"- passing runs (logic+predictive): `{len(pass_rows)}`")
    lines.append("")
    if best_row:
        lines.append("## Best Run")
        lines.append(f"- variant: `{best_row['variant']}`")
        lines.append(
            f"- thresholds: xg_share>={best_row['xg_share_threshold']:.2f}, "
            f"minutes_share>={best_row['minutes_share_threshold']:.2f}"
        )
        lines.append(
            f"- football: corr={best_row['corr_xg_lost_vs_residual']:.4f}, "
            f"key_delta={best_row['key_absent_residual_delta']:.4f}, "
            f"status={best_row['football_logic_status']}"
        )
        lines.append(
            f"- predictive: triggered_lift={best_row['best_triggered_lift']:.2%}, "
            f"global_lift={best_row['best_triggered_global_lift']:.2%}, "
            f"status={best_row['predictive_status']}"
        )
        lines.append("")

    lines.append("## Top 10")
    for i, row in result_df.head(10).iterrows():
        lines.append(
            f"- {i+1}. {row['variant']} | xg_thr={row['xg_share_threshold']:.2f}, "
            f"min_thr={row['minutes_share_threshold']:.2f} | "
            f"logic={row['football_logic_status']} | pred={row['predictive_status']} | "
            f"trig={row['best_triggered_lift']:.2%} | global={row['best_triggered_global_lift']:.2%}"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
