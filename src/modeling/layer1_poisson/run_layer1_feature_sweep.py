from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.features.build_features import build_point_in_time_features
from src.modeling.layer1_poisson.train_lambda import (  # noqa: E402
    compute_time_decay_weights,
    calibrate_dixon_coles,
)
from src.pricing.poisson import PoissonPricer  # noqa: E402


# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false


@dataclass(frozen=True)
class SweepRow:
    config_name: str
    xi: float
    use_shots_inside_box: bool
    train_n: int
    cal_n: int
    test_n: int
    feature_count: int
    home_rmse: float
    away_rmse: float
    mean_rmse: float
    home_mae: float
    away_mae: float
    home_bias: float
    away_bias: float
    rho: float
    test_draw_actual: float
    test_draw_pred: float
    draw_abs_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer 1 feature sweep: time-decay xi and shots_inside_box feature block."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max finished fixtures to pull for the experiment",
    )
    parser.add_argument(
        "--target-window-mins",
        type=int,
        default=60,
        help="Decision window before kickoff used in feature build",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window size for shots_inside_box team features",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=6,
        help="Minimum games-played filter for both teams",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=800,
        help="Max xgboost rounds per model",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=80,
        help="Early stopping rounds for xgboost",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer1_sweep"),
        help="Output directory for sweep report",
    )
    return parser.parse_args()


def fetch_target_fixture_ids(limit: int | None = None) -> list[int]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            limit_sql = "LIMIT %s" if limit else ""
            params: tuple[object, ...] = (limit,) if limit else ()
            cur.execute(
                f"""
                SELECT DISTINCT f.fixture_id, f.match_datetime_utc
                FROM fixtures f
                JOIN fixture_results r ON r.fixture_id = f.fixture_id
                JOIN team_premium_snapshots s ON s.fixture_id = f.fixture_id
                WHERE f.status = 'ft'
                  AND f.match_datetime_utc IS NOT NULL
                  AND r.home_goals IS NOT NULL
                  AND r.away_goals IS NOT NULL
                ORDER BY f.match_datetime_utc DESC
                {limit_sql}
                """,
                params,
            )
            return [int(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_fixture_core(target_fixture_ids: list[int]) -> pd.DataFrame:
    conn = connect_db()
    try:
        query = """
            SELECT fixture_id, match_datetime_utc, home_team_id, away_team_id
            FROM fixtures
            WHERE fixture_id = ANY(%s)
        """
        df = pd.read_sql(query, conn, params=(target_fixture_ids,))
    finally:
        conn.close()
    df["match_datetime_utc"] = pd.to_datetime(
        df["match_datetime_utc"], utc=True, errors="coerce"
    )
    return df


def build_shots_inside_box_rollups(
    target_fixture_ids: list[int],
    rolling_window: int,
) -> pd.DataFrame:
    conn = connect_db()
    try:
        stats = pd.read_sql(
            """
            SELECT
                f.fixture_id,
                f.match_datetime_utc,
                f.home_team_id,
                f.away_team_id,
                p.h_shots_inside_box,
                p.a_shots_inside_box
            FROM fixtures f
            JOIN fixture_results r ON r.fixture_id = f.fixture_id
            JOIN fixture_stats_premium p ON p.fixture_id = f.fixture_id
            WHERE f.status = 'ft'
              AND f.match_datetime_utc IS NOT NULL
              AND r.home_goals IS NOT NULL
              AND r.away_goals IS NOT NULL
            ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC
            """,
            conn,
        )
    finally:
        conn.close()

    stats["match_datetime_utc"] = pd.to_datetime(
        stats["match_datetime_utc"], utc=True, errors="coerce"
    )
    stats["h_shots_inside_box"] = pd.to_numeric(
        stats["h_shots_inside_box"], errors="coerce"
    )
    stats["a_shots_inside_box"] = pd.to_numeric(
        stats["a_shots_inside_box"], errors="coerce"
    )

    home_long = stats[
        [
            "fixture_id",
            "match_datetime_utc",
            "home_team_id",
            "h_shots_inside_box",
            "a_shots_inside_box",
        ]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "h_shots_inside_box": "shots_for",
            "a_shots_inside_box": "shots_against",
        }
    )
    home_long["is_home"] = True

    away_long = stats[
        [
            "fixture_id",
            "match_datetime_utc",
            "away_team_id",
            "a_shots_inside_box",
            "h_shots_inside_box",
        ]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "a_shots_inside_box": "shots_for",
            "h_shots_inside_box": "shots_against",
        }
    )
    away_long["is_home"] = False

    long_df = pd.concat([home_long, away_long], ignore_index=True)
    long_df = long_df.sort_values(
        ["team_id", "match_datetime_utc", "fixture_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    def _add_roll(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(rolling_window, min_periods=1).mean()

    long_df["shots_inside_box_for_roll"] = (
        long_df.groupby("team_id", group_keys=False)["shots_for"].apply(_add_roll)
    )
    long_df["shots_inside_box_against_roll"] = (
        long_df.groupby("team_id", group_keys=False)["shots_against"].apply(_add_roll)
    )

    home_roll = long_df[long_df["is_home"]][
        ["fixture_id", "shots_inside_box_for_roll", "shots_inside_box_against_roll"]
    ].rename(
        columns={
            "shots_inside_box_for_roll": "h_shots_inside_box_roll",
            "shots_inside_box_against_roll": "h_shots_inside_box_against_roll",
        }
    )
    away_roll = long_df[~long_df["is_home"]][
        ["fixture_id", "shots_inside_box_for_roll", "shots_inside_box_against_roll"]
    ].rename(
        columns={
            "shots_inside_box_for_roll": "a_shots_inside_box_roll",
            "shots_inside_box_against_roll": "a_shots_inside_box_against_roll",
        }
    )

    out = pd.DataFrame({"fixture_id": target_fixture_ids})
    out = out.merge(home_roll, on="fixture_id", how="left")
    out = out.merge(away_roll, on="fixture_id", how="left")
    return out


def build_model_frame(
    fixture_ids: list[int],
    target_window_mins: int,
    min_games: int,
    rolling_window: int,
) -> pd.DataFrame:
    df = build_point_in_time_features(
        fixture_ids=fixture_ids,
        target_window_mins=target_window_mins,
    )
    if df.empty:
        return df

    core = fetch_fixture_core(fixture_ids)
    shots = build_shots_inside_box_rollups(
        target_fixture_ids=fixture_ids,
        rolling_window=rolling_window,
    )

    df = df.merge(core, on="fixture_id", how="left")
    df = df.merge(shots, on="fixture_id", how="left")
    df = df.sort_values(["kickoff_time", "fixture_id"], kind="mergesort").reset_index(
        drop=True
    )

    df = df.dropna(subset=["h_xg", "a_xg", "league_avg_xg"]).copy()
    df = df[(df["h_sample_size"] >= min_games) & (df["a_sample_size"] >= min_games)].copy()
    return df


def get_feature_cols(df: pd.DataFrame, use_shots_inside_box: bool) -> list[str]:
    shots_cols = {
        "h_shots_inside_box_roll",
        "a_shots_inside_box_roll",
        "h_shots_inside_box_against_roll",
        "a_shots_inside_box_against_roll",
    }
    cols = [
        c
        for c in df.columns
        if (c.startswith("h_") or c.startswith("a_") or c.startswith("league_"))
        and c
        not in {
            "league_code",
            "fidelity_score",
            "h_sample_size",
            "a_sample_size",
        }
    ]
    if not use_shots_inside_box:
        cols = [c for c in cols if c not in shots_cols]
    return cols


def train_pair_model(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    feature_cols: list[str],
    xi: float,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[xgb.Booster, xgb.Booster]:
    params = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 15,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "random_state": 42,
    }

    if xi <= 0:
        train_weights = np.ones(len(train_df), dtype=float)
    else:
        train_weights = compute_time_decay_weights(
            train_df["kickoff_time"],
            as_of=pd.to_datetime(train_df["kickoff_time"]).max(),
            xi=xi,
        )

    x_train = train_df[feature_cols].fillna(0.0)
    x_cal = cal_df[feature_cols].fillna(0.0)

    dtrain_h = xgb.DMatrix(x_train, label=train_df["home_goals"], weight=train_weights)
    dcal_h = xgb.DMatrix(x_cal, label=cal_df["home_goals"])
    model_h = xgb.train(
        params=params,
        dtrain=dtrain_h,
        num_boost_round=num_boost_round,
        evals=[(dtrain_h, "train"), (dcal_h, "cal")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    dtrain_a = xgb.DMatrix(x_train, label=train_df["away_goals"], weight=train_weights)
    dcal_a = xgb.DMatrix(x_cal, label=cal_df["away_goals"])
    model_a = xgb.train(
        params=params,
        dtrain=dtrain_a,
        num_boost_round=num_boost_round,
        evals=[(dtrain_a, "train"), (dcal_a, "cal")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    return model_h, model_a


def evaluate_config(
    name: str,
    xi: float,
    use_shots_inside_box: bool,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> SweepRow:
    feature_cols = get_feature_cols(train_df, use_shots_inside_box=use_shots_inside_box)
    model_h, model_a = train_pair_model(
        train_df=train_df,
        cal_df=cal_df,
        feature_cols=feature_cols,
        xi=xi,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    x_cal = xgb.DMatrix(cal_df[feature_cols].fillna(0.0))
    cal_pred_h = model_h.predict(x_cal)
    cal_pred_a = model_a.predict(x_cal)
    cal_actual_draws = (cal_df["home_goals"] == cal_df["away_goals"]).astype(int).to_numpy()
    rho = calibrate_dixon_coles(cal_pred_h, cal_pred_a, cal_actual_draws)

    x_test = xgb.DMatrix(test_df[feature_cols].fillna(0.0))
    pred_h = model_h.predict(x_test)
    pred_a = model_a.predict(x_test)
    true_h = test_df["home_goals"].to_numpy(dtype=float)
    true_a = test_df["away_goals"].to_numpy(dtype=float)

    home_err = true_h - pred_h
    away_err = true_a - pred_a
    home_rmse = float(np.sqrt(np.mean(np.square(home_err))))
    away_rmse = float(np.sqrt(np.mean(np.square(away_err))))
    home_mae = float(np.mean(np.abs(home_err)))
    away_mae = float(np.mean(np.abs(away_err)))
    home_bias = float(np.mean(home_err))
    away_bias = float(np.mean(away_err))

    pricer = PoissonPricer()
    test_draw_probs: list[float] = []
    for lh, la in zip(pred_h, pred_a):
        matrix = pricer.generate_matrix(float(lh), float(la), rho=float(rho))
        test_draw_probs.append(float(np.diag(matrix).sum()))
    test_draw_actual = float(np.mean((test_df["home_goals"] == test_df["away_goals"]).astype(int)))
    test_draw_pred = float(np.mean(test_draw_probs))

    return SweepRow(
        config_name=name,
        xi=xi,
        use_shots_inside_box=use_shots_inside_box,
        train_n=int(len(train_df)),
        cal_n=int(len(cal_df)),
        test_n=int(len(test_df)),
        feature_count=int(len(feature_cols)),
        home_rmse=home_rmse,
        away_rmse=away_rmse,
        mean_rmse=float(np.mean([home_rmse, away_rmse])),
        home_mae=home_mae,
        away_mae=away_mae,
        home_bias=home_bias,
        away_bias=away_bias,
        rho=float(rho),
        test_draw_actual=test_draw_actual,
        test_draw_pred=test_draw_pred,
        draw_abs_error=float(abs(test_draw_actual - test_draw_pred)),
    )


def main() -> None:
    args = parse_args()

    fixture_ids = fetch_target_fixture_ids(limit=args.limit)
    if not fixture_ids:
        raise RuntimeError("No finished fixtures found for sweep.")

    print(f"Building Layer 1 frame for {len(fixture_ids)} fixtures...")
    df = build_model_frame(
        fixture_ids=fixture_ids,
        target_window_mins=args.target_window_mins,
        min_games=args.min_games,
        rolling_window=args.rolling_window,
    )
    if df.empty:
        raise RuntimeError("Layer 1 frame is empty after filters.")

    n = len(df)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.80)
    train_df = df.iloc[:train_end].copy()
    cal_df = df.iloc[train_end:cal_end].copy()
    test_df = df.iloc[cal_end:].copy()
    if train_df.empty or cal_df.empty or test_df.empty:
        raise RuntimeError("Invalid split: one of train/cal/test is empty.")

    configs = [
        ("no_decay_core", 0.0, False),
        ("no_decay_plus_shots", 0.0, True),
        ("xi_0.001_core", 0.001, False),
        ("xi_0.001_plus_shots", 0.001, True),
        ("xi_0.002_core", 0.002, False),
        ("xi_0.002_plus_shots", 0.002, True),
        ("xi_0.003_core", 0.003, False),
        ("xi_0.003_plus_shots", 0.003, True),
    ]

    rows: list[SweepRow] = []
    for name, xi, use_shots in configs:
        print(f"Running config: {name} (xi={xi}, use_shots_inside_box={use_shots})")
        row = evaluate_config(
            name=name,
            xi=xi,
            use_shots_inside_box=use_shots,
            train_df=train_df,
            cal_df=cal_df,
            test_df=test_df,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r.mean_rmse)
    best = rows_sorted[0]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "args": {
            "limit": args.limit,
            "target_window_mins": args.target_window_mins,
            "rolling_window": args.rolling_window,
            "min_games": args.min_games,
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
        "split": {
            "train_n": int(len(train_df)),
            "cal_n": int(len(cal_df)),
            "test_n": int(len(test_df)),
            "train_start": str(train_df["kickoff_time"].min()),
            "train_end": str(train_df["kickoff_time"].max()),
            "cal_start": str(cal_df["kickoff_time"].min()),
            "cal_end": str(cal_df["kickoff_time"].max()),
            "test_start": str(test_df["kickoff_time"].min()),
            "test_end": str(test_df["kickoff_time"].max()),
        },
        "best_config": asdict(best),
        "rows": [asdict(r) for r in rows_sorted],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "layer1_feature_sweep.json"
    md_path = args.output_dir / "layer1_feature_sweep.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Layer 1 Feature Sweep")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append("")
    lines.append(
        f"- split train/cal/test: {len(train_df)} / {len(cal_df)} / {len(test_df)}"
    )
    lines.append(f"- best config: `{best.config_name}`")
    lines.append("")
    lines.append("| Config | xi | +shots_inside_box | Features | Home RMSE | Away RMSE | Mean RMSE | Home Bias | Away Bias | Draw Abs Error |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows_sorted:
        lines.append(
            f"| {r.config_name} | {r.xi:.3f} | {str(r.use_shots_inside_box).lower()} | "
            f"{r.feature_count} | {r.home_rmse:.4f} | {r.away_rmse:.4f} | {r.mean_rmse:.4f} | "
            f"{r.home_bias:.4f} | {r.away_bias:.4f} | {r.draw_abs_error:.4f} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(
        "Best config summary: "
        f"{best.config_name} | mean_rmse={best.mean_rmse:.4f} "
        f"(home={best.home_rmse:.4f}, away={best.away_rmse:.4f})"
    )


if __name__ == "__main__":
    main()
