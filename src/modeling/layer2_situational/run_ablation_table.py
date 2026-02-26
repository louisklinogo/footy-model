from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from src.modeling.layer2_situational.train_situational_residual import (
    FEATURE_COLS,
    ODDS_FEATURE_COLS,
    add_odds_model_gap,
    load_feature_data,
)


GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    min_samples_leaf=30,
    learning_rate=0.05,
    random_state=42,
)


def _apply_filters(df, min_games: int = 4):
    df = df[
        (df["home_played"] >= min_games) & (df["away_played"] >= min_games)
    ].copy()
    df = df[df["lambda_home"].notna() & df["lambda_away"].notna()].copy()
    return df


def _train_and_score(train_df, test_df, features):
    if not features:
        return None, None

    X_train = train_df[features].fillna(0).values
    X_test = test_df[features].fillna(0).values
    y_home_train = train_df["home_residual"].values
    y_away_train = train_df["away_residual"].values
    y_home_test = test_df["home_residual"].values
    y_away_test = test_df["away_residual"].values

    home_model = GradientBoostingRegressor(**GBM_PARAMS)
    home_model.fit(X_train, y_home_train)
    away_model = GradientBoostingRegressor(**GBM_PARAMS)
    away_model.fit(X_train, y_away_train)

    home_rmse = mean_squared_error(y_home_test, home_model.predict(X_test)) ** 0.5
    away_rmse = mean_squared_error(y_away_test, away_model.predict(X_test)) ** 0.5
    return home_rmse, away_rmse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation table for Layer 2 situational model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/ablation"),
        help="Output directory for ablation artifacts",
    )
    args = parser.parse_args()

    df = load_feature_data()
    df = add_odds_model_gap(df)
    df = df.sort_values("match_datetime_utc").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = _apply_filters(df.iloc[:split_idx].copy())
    test_df = _apply_filters(df.iloc[split_idx:].copy())

    if train_df.empty or test_df.empty:
        raise RuntimeError("Insufficient data to run ablation table.")

    train_df["home_residual"] = train_df["home_goals"] - train_df["lambda_home"]
    train_df["away_residual"] = train_df["away_goals"] - train_df["lambda_away"]
    test_df["home_residual"] = test_df["home_goals"] - test_df["lambda_home"]
    test_df["away_residual"] = test_df["away_goals"] - test_df["lambda_away"]

    baseline_home = mean_squared_error(
        test_df["home_residual"].values, np.zeros(len(test_df))
    ) ** 0.5
    baseline_away = mean_squared_error(
        test_df["away_residual"].values, np.zeros(len(test_df))
    ) ** 0.5

    standings_feats = [
        "position_gap",
        "points_gap",
        "home_lame_duck",
        "away_lame_duck",
        "home_playing_top4",
        "away_playing_top4",
        "derby_position_gap",
    ]
    schedule_feats = [
        "rest_delta",
        "congestion_flag",
        "home_upcoming_tier",
        "away_upcoming_tier",
    ]
    player_feats = [
        "home_xg_lost",
        "away_xg_lost",
        "home_key_absent",
        "away_key_absent",
        "injury_impact",
    ]

    stages = [
        ("baseline", []),
        ("standings_gaps", standings_feats),
        ("standings_gaps+schedule", standings_feats + schedule_feats),
        ("standings_gaps+schedule+player", standings_feats + schedule_feats + player_feats),
        ("standings_gaps+schedule+player+odds", standings_feats + schedule_feats + player_feats + ODDS_FEATURE_COLS),
    ]

    results = []
    for name, feats in stages:
        if not feats:
            home_rmse = baseline_home
            away_rmse = baseline_away
        else:
            home_rmse, away_rmse = _train_and_score(train_df, test_df, feats)
            if home_rmse is None or away_rmse is None:
                home_rmse = baseline_home
                away_rmse = baseline_away

        home_lift = (baseline_home - home_rmse) / baseline_home if baseline_home else 0.0
        away_lift = (baseline_away - away_rmse) / baseline_away if baseline_away else 0.0
        results.append(
            {
                "stage": name,
                "features": feats,
                "home_rmse": home_rmse,
                "away_rmse": away_rmse,
                "home_lift": home_lift,
                "away_lift": away_lift,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ablation_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_lines = [
        "| Stage | Home RMSE | Away RMSE | Home Lift | Away Lift |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in results:
        md_lines.append(
            f"| {row['stage']} | {row['home_rmse']:.4f} | {row['away_rmse']:.4f} | {row['home_lift']:.2%} | {row['away_lift']:.2%} |"
        )
    md_path = args.output_dir / "ablation_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
