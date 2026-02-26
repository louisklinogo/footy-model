from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.modeling.layer2_situational import situational_utils
from src.modeling.layer2_situational.train_situational_residual import (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate player-impact feature assumptions end-to-end."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation"),
        help="Output directory for validation artifacts.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="player_impact_assumption_validation",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=4,
        help="Minimum games-played filter (same as Layer 2 train).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lambda_xgb",
        help="Layer 1 lambda model_name for residual references.",
    )
    return parser.parse_args()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def _lift_vs_zero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    baseline = _rmse(y_true, np.zeros_like(y_true))
    if baseline <= 0:
        return 0.0
    return float((baseline - _rmse(y_true, y_pred)) / baseline)


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 20:
        return 0.0
    val = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    if pd.isna(val):
        return 0.0
    return float(val)


def _column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
        LIMIT 1
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def _availability_timing_column(cur) -> str:
    if _column_exists(cur, "player_availability", "event_recorded_at"):
        return "event_recorded_at"
    if _column_exists(cur, "player_availability", "first_recorded_at"):
        return "first_recorded_at"
    return "recorded_at"


def _player_impact_train_style_sql() -> str:
    return """
    WITH base_f AS (
        SELECT fixture_id, home_team_id, away_team_id, match_datetime_utc
        FROM fixtures
        WHERE status = 'ft'
    ),
    mps AS (
        SELECT pa.fixture_id, pa.player_id, pa.team_id, b.match_datetime_utc
        FROM player_availability pa
        JOIN base_f b ON b.fixture_id = pa.fixture_id
        WHERE pa.status IN ('missing', 'doubtful')
    ),
    pi AS (
        SELECT
            mps.fixture_id,
            mps.team_id,
            SUM(q.avg_xg) AS team_xg_lost,
            MAX(CASE WHEN CAST(q.starts AS FLOAT) / GREATEST(q.apps, 1) >= 0.7 AND q.apps >= 3 THEN 1 ELSE 0 END) AS has_key_absence
        FROM mps
        LEFT JOIN LATERAL (
            SELECT
                AVG(fps.expected_goals) AS avg_xg,
                COUNT(*) AS apps,
                COUNT(*) FILTER (WHERE fps.substituted_in = False) AS starts
            FROM (
                SELECT fps2.expected_goals, fps2.substituted_in
                FROM fixture_player_stats fps2
                JOIN fixtures f2 ON f2.fixture_id = fps2.fixture_id
                WHERE fps2.player_id = mps.player_id
                  AND f2.match_datetime_utc < mps.match_datetime_utc
                ORDER BY f2.match_datetime_utc DESC
                LIMIT 10
            ) fps
        ) q ON TRUE
        GROUP BY mps.fixture_id, mps.team_id
    )
    SELECT
        bf.fixture_id,
        COALESCE(SUM(pi.team_xg_lost) FILTER (WHERE pi.team_id = bf.home_team_id), 0) AS home_xg_lost,
        COALESCE(SUM(pi.team_xg_lost) FILTER (WHERE pi.team_id = bf.away_team_id), 0) AS away_xg_lost,
        COALESCE(MAX(pi.has_key_absence) FILTER (WHERE pi.team_id = bf.home_team_id), 0) AS home_key_absent,
        COALESCE(MAX(pi.has_key_absence) FILTER (WHERE pi.team_id = bf.away_team_id), 0) AS away_key_absent
    FROM base_f bf
    LEFT JOIN pi ON pi.fixture_id = bf.fixture_id
    GROUP BY bf.fixture_id
    """


def _player_impact_predict_style_sql() -> str:
    return """
    WITH mps AS (
        SELECT pa.fixture_id, pa.player_id, pa.team_id, f.match_datetime_utc
        FROM player_availability pa
        JOIN fixtures f ON f.fixture_id = pa.fixture_id
        WHERE pa.status IN ('missing', 'doubtful')
          AND pa.fixture_id = ANY(%s)
    ),
    pi AS (
        SELECT
            mps.fixture_id,
            mps.team_id,
            SUM(q.avg_xg) AS team_xg_lost,
            MAX(CASE WHEN CAST(q.starts AS FLOAT) / GREATEST(q.apps, 1) >= 0.7 AND q.apps >= 3 THEN 1 ELSE 0 END) AS has_key_absence
        FROM mps
        LEFT JOIN LATERAL (
            SELECT
                AVG(fps.expected_goals) AS avg_xg,
                COUNT(*) AS apps,
                COUNT(*) FILTER (WHERE fps.substituted_in = False) AS starts
            FROM (
                SELECT fps2.expected_goals, fps2.substituted_in
                FROM fixture_player_stats fps2
                JOIN fixtures f2 ON f2.fixture_id = fps2.fixture_id
                WHERE fps2.player_id = mps.player_id
                  AND f2.match_datetime_utc < mps.match_datetime_utc
                ORDER BY f2.match_datetime_utc DESC
                LIMIT 10
            ) fps
        ) q ON TRUE
        GROUP BY mps.fixture_id, mps.team_id
    )
    SELECT
        tf.fixture_id,
        COALESCE(SUM(pi.team_xg_lost) FILTER (WHERE pi.team_id = tf.home_team_id), 0) AS home_xg_lost,
        COALESCE(SUM(pi.team_xg_lost) FILTER (WHERE pi.team_id = tf.away_team_id), 0) AS away_xg_lost,
        COALESCE(MAX(pi.has_key_absence) FILTER (WHERE pi.team_id = tf.home_team_id), 0) AS home_key_absent,
        COALESCE(MAX(pi.has_key_absence) FILTER (WHERE pi.team_id = tf.away_team_id), 0) AS away_key_absent
    FROM fixtures tf
    LEFT JOIN pi ON pi.fixture_id = tf.fixture_id
    WHERE tf.fixture_id = ANY(%s)
    GROUP BY tf.fixture_id
    """


def run_correctness_checks(model_name: str) -> dict[str, Any]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            timing_col = _availability_timing_column(cur)

        ft_ids = pd.read_sql(
            """
            SELECT fixture_id
            FROM fixtures
            WHERE status = 'ft'
            ORDER BY fixture_id
            """,
            conn,
        )["fixture_id"].astype(int).tolist()

        train_style = pd.read_sql(_player_impact_train_style_sql(), conn)
        predict_style = pd.read_sql(
            _player_impact_predict_style_sql(),
            conn,
            params=(ft_ids, ft_ids),
        )

        merged = train_style.merge(
            predict_style,
            on="fixture_id",
            how="inner",
            suffixes=("_train", "_predict"),
        )
        float_cols = ["home_xg_lost", "away_xg_lost"]
        int_cols = ["home_key_absent", "away_key_absent"]

        mismatch = {}
        for col in float_cols:
            delta = (
                merged[f"{col}_train"].astype(float) - merged[f"{col}_predict"].astype(float)
            ).abs()
            mismatch[col] = {
                "mismatch_count_abs_gt_1e-9": int((delta > 1e-9).sum()),
                "max_abs_diff": float(delta.max()) if len(delta) else 0.0,
            }
        for col in int_cols:
            neq = merged[f"{col}_train"].astype(int) != merged[f"{col}_predict"].astype(int)
            mismatch[col] = {
                "mismatch_count": int(neq.sum()),
                "max_abs_diff": int(
                    (merged[f"{col}_train"].astype(int) - merged[f"{col}_predict"].astype(int))
                    .abs()
                    .max()
                )
                if len(merged)
                else 0,
            }

        timing = pd.read_sql(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE pa.{timing_col} IS NOT NULL) AS known_timing_rows,
                COUNT(*) FILTER (WHERE pa.{timing_col} IS NULL) AS unknown_timing_rows,
                COUNT(*) FILTER (WHERE pa.{timing_col} IS NOT NULL AND pa.{timing_col} > f.match_datetime_utc) AS late_rows
            FROM player_availability pa
            JOIN fixtures f ON f.fixture_id = pa.fixture_id
            WHERE f.status = 'ft'
            """,
            conn,
        ).iloc[0]

        parity_fail = any(
            v.get("mismatch_count_abs_gt_1e-9", 0) > 0 or v.get("mismatch_count", 0) > 0
            for v in mismatch.values()
        )
        timing_fail = int(timing["late_rows"]) > 0
        status = "pass" if (not parity_fail and not timing_fail) else "fail"

        return {
            "status": status,
            "timing_column_used": timing_col,
            "rows_compared": int(len(merged)),
            "parity_mismatch": mismatch,
            "timing": {
                "known_timing_rows": int(timing["known_timing_rows"] or 0),
                "unknown_timing_rows": int(timing["unknown_timing_rows"] or 0),
                "late_rows": int(timing["late_rows"] or 0),
            },
            "notes": [
                "Train-style and predict-style formulas were recomputed directly from SQL and compared fixture-by-fixture.",
                f"Residual references use latest pair from predictions model_name='{model_name}'.",
            ],
        }
    finally:
        conn.close()


def _build_team_side_frame(df: pd.DataFrame) -> pd.DataFrame:
    home = df[
        [
            "fixture_id",
            "league_code",
            "match_datetime_utc",
            "home_xg_lost",
            "home_key_absent",
            "home_goals",
            "lambda_home",
        ]
    ].copy()
    home["side"] = "home"
    home = home.rename(
        columns={
            "home_xg_lost": "xg_lost",
            "home_key_absent": "key_absent",
            "home_goals": "goals",
            "lambda_home": "lambda",
        }
    )

    away = df[
        [
            "fixture_id",
            "league_code",
            "match_datetime_utc",
            "away_xg_lost",
            "away_key_absent",
            "away_goals",
            "lambda_away",
        ]
    ].copy()
    away["side"] = "away"
    away = away.rename(
        columns={
            "away_xg_lost": "xg_lost",
            "away_key_absent": "key_absent",
            "away_goals": "goals",
            "lambda_away": "lambda",
        }
    )

    frame = pd.concat([home, away], ignore_index=True)
    frame["xg_lost"] = frame["xg_lost"].fillna(0.0).astype(float)
    frame["key_absent"] = frame["key_absent"].fillna(0).astype(int)
    frame["lambda"] = frame["lambda"].astype(float)
    frame["goals"] = frame["goals"].astype(float)
    frame["residual"] = frame["goals"] - frame["lambda"]
    frame["triggered"] = (frame["xg_lost"] > 0) | (frame["key_absent"] == 1)
    return frame


def run_football_logic_checks(frame: pd.DataFrame) -> dict[str, Any]:
    corr = _safe_corr(frame["xg_lost"], frame["residual"])

    key0 = frame[frame["key_absent"] == 0]["residual"]
    key1 = frame[frame["key_absent"] == 1]["residual"]
    key_delta = float(key1.mean() - key0.mean()) if len(key1) and len(key0) else 0.0

    bins = [-1e-12, 0.0, 0.1, 0.25, 0.5, 1.0, np.inf]
    labels = ["0", "(0,0.1]", "(0.1,0.25]", "(0.25,0.5]", "(0.5,1.0]", "(1.0,+)"]
    binned = frame.copy()
    binned["xg_lost_bucket"] = pd.cut(
        binned["xg_lost"], bins=bins, labels=labels, include_lowest=True, right=True
    )
    bucket_table = (
        binned.groupby("xg_lost_bucket", observed=False)
        .agg(
            n=("residual", "size"),
            mean_residual=("residual", "mean"),
            mean_goals=("goals", "mean"),
            mean_lambda=("lambda", "mean"),
        )
        .reset_index()
    )

    status = "pass" if (corr < 0 and key_delta < 0) else "warn"
    return {
        "status": status,
        "corr_xg_lost_vs_residual": float(corr),
        "mean_residual_key_absent_0": float(key0.mean()) if len(key0) else None,
        "mean_residual_key_absent_1": float(key1.mean()) if len(key1) else None,
        "mean_residual_delta_key1_minus_key0": float(key_delta),
        "xg_lost_bucket_summary": bucket_table.to_dict(orient="records"),
        "triggered_share": float(frame["triggered"].mean()),
    }


def _fit_player_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    x_train = train_df[features].fillna(0.0).to_numpy(dtype=float)
    x_test = test_df[features].fillna(0.0).to_numpy(dtype=float)

    y_home_train = train_df["home_residual"].to_numpy(dtype=float)
    y_away_train = train_df["away_residual"].to_numpy(dtype=float)
    y_home_test = test_df["home_residual"].to_numpy(dtype=float)
    y_away_test = test_df["away_residual"].to_numpy(dtype=float)

    w_home = 1.0 / np.sqrt(np.maximum(train_df["lambda_home"].to_numpy(dtype=float), 0.1))
    w_away = 1.0 / np.sqrt(np.maximum(train_df["lambda_away"].to_numpy(dtype=float), 0.1))

    model_h = GradientBoostingRegressor(**GBM_PARAMS)
    model_h.fit(x_train, y_home_train, sample_weight=w_home)
    pred_home = model_h.predict(x_test)
    model_a = GradientBoostingRegressor(**GBM_PARAMS)
    model_a.fit(x_train, y_away_train, sample_weight=w_away)
    pred_away = model_a.predict(x_test)

    home_rmse = _rmse(y_home_test, pred_home)
    away_rmse = _rmse(y_away_test, pred_away)
    home_lift = _lift_vs_zero(y_home_test, pred_home)
    away_lift = _lift_vs_zero(y_away_test, pred_away)

    trigger_mask = (
        (test_df["home_xg_lost"] > 0)
        | (test_df["away_xg_lost"] > 0)
        | (test_df["home_key_absent"] == 1)
        | (test_df["away_key_absent"] == 1)
    ).to_numpy(dtype=bool)

    if trigger_mask.any():
        y_h_t = y_home_test[trigger_mask]
        y_a_t = y_away_test[trigger_mask]
        p_h_t = pred_home[trigger_mask]
        p_a_t = pred_away[trigger_mask]
        trig_home_lift = _lift_vs_zero(y_h_t, p_h_t)
        trig_away_lift = _lift_vs_zero(y_a_t, p_a_t)
    else:
        trig_home_lift = 0.0
        trig_away_lift = 0.0

    return {
        "features": features,
        "home_rmse": float(home_rmse),
        "away_rmse": float(away_rmse),
        "home_lift": float(home_lift),
        "away_lift": float(away_lift),
        "triggered_home_lift": float(trig_home_lift),
        "triggered_away_lift": float(trig_away_lift),
        "triggered_combined_lift": float(np.mean([trig_home_lift, trig_away_lift])),
        "global_combined_lift": float(np.mean([home_lift, away_lift])),
        "triggered_test_fixtures": int(trigger_mask.sum()),
        "test_fixtures": int(len(test_df)),
    }


def run_predictive_checks(df: pd.DataFrame, min_games: int) -> dict[str, Any]:
    df = df.sort_values("match_datetime_utc").reset_index(drop=True)
    split_time = df["match_datetime_utc"].dropna().quantile(0.8)
    train_df = df[df["match_datetime_utc"] <= split_time].copy()
    test_df = df[df["match_datetime_utc"] > split_time].copy()

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

    feature_sets = {
        "key_absent_only": ["home_key_absent", "away_key_absent"],
        "xg_lost_only": ["home_xg_lost", "away_xg_lost", "injury_impact"],
        "player_block_all": [
            "home_xg_lost",
            "away_xg_lost",
            "home_key_absent",
            "away_key_absent",
            "injury_impact",
        ],
        "player_block_no_injury_impact": [
            "home_xg_lost",
            "away_xg_lost",
            "home_key_absent",
            "away_key_absent",
        ],
    }

    runs = {}
    for name, features in feature_sets.items():
        runs[name] = _fit_player_model(train_df, test_df, features)

    best_triggered = max(
        runs.items(), key=lambda kv: kv[1].get("triggered_combined_lift", -9999.0)
    )
    best_global = max(
        runs.items(), key=lambda kv: kv[1].get("global_combined_lift", -9999.0)
    )

    status = "warn"
    if (
        best_triggered[1]["triggered_combined_lift"] >= 0.002
        and best_global[1]["global_combined_lift"] >= 0
    ):
        status = "pass"
    elif best_triggered[1]["triggered_combined_lift"] <= 0:
        status = "fail"

    return {
        "status": status,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "runs": runs,
        "best_by_triggered_lift": {"name": best_triggered[0], **best_triggered[1]},
        "best_by_global_lift": {"name": best_global[0], **best_global[1]},
    }


def run_operational_checks() -> dict[str, Any]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            timing_col = _availability_timing_column(cur)

        xg_stats = pd.read_sql(
            """
            SELECT
                COUNT(*) AS rows_total,
                COUNT(*) FILTER (WHERE expected_goals IS NULL) AS rows_xg_null
            FROM fixture_player_stats
            """,
            conn,
        ).iloc[0]

        hist_cov = pd.read_sql(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE EXISTS (
                        SELECT 1
                        FROM player_availability pa
                        WHERE pa.fixture_id = f.fixture_id
                    )
                ) AS fixtures_with_availability,
                COUNT(*) AS fixtures_total
            FROM fixtures f
            WHERE f.status = 'ft'
            """,
            conn,
        ).iloc[0]

        upcoming_cov = pd.read_sql(
            f"""
            WITH upcoming AS (
                SELECT fixture_id
                FROM fixtures
                WHERE status = 'scheduled'
                  AND match_datetime_utc >= NOW()
                  AND match_datetime_utc < NOW() + interval '14 days'
            )
            SELECT
                COUNT(*) FILTER (
                    WHERE EXISTS (
                        SELECT 1
                        FROM player_availability pa
                        WHERE pa.fixture_id = u.fixture_id
                          AND pa.{timing_col} IS NOT NULL
                    )
                ) AS fixtures_with_availability,
                COUNT(*) AS fixtures_total
            FROM upcoming u
            """,
            conn,
        ).iloc[0]

        hist_pct = (
            float(hist_cov["fixtures_with_availability"]) / float(hist_cov["fixtures_total"])
            if int(hist_cov["fixtures_total"]) > 0
            else 0.0
        )
        upcoming_pct = (
            float(upcoming_cov["fixtures_with_availability"]) / float(upcoming_cov["fixtures_total"])
            if int(upcoming_cov["fixtures_total"]) > 0
            else 0.0
        )
        xg_null_pct = (
            float(xg_stats["rows_xg_null"]) / float(xg_stats["rows_total"])
            if int(xg_stats["rows_total"]) > 0
            else 0.0
        )

        status = "pass"
        reasons = []
        if upcoming_pct < 0.5:
            status = "fail"
            reasons.append("upcoming_availability_coverage_below_50pct")
        if xg_null_pct > 0.5:
            if status != "fail":
                status = "warn"
            reasons.append("fixture_player_stats_expected_goals_missingness_high")

        return {
            "status": status,
            "reasons": reasons,
            "fixture_player_stats_expected_goals_null_pct": float(xg_null_pct),
            "historical_ft_coverage_pct": float(hist_pct),
            "historical_ft_fixtures_with_availability": int(
                hist_cov["fixtures_with_availability"]
            ),
            "historical_ft_fixtures_total": int(hist_cov["fixtures_total"]),
            "upcoming_14d_coverage_pct": float(upcoming_pct),
            "upcoming_14d_fixtures_with_availability": int(
                upcoming_cov["fixtures_with_availability"]
            ),
            "upcoming_14d_fixtures_total": int(upcoming_cov["fixtures_total"]),
        }
    finally:
        conn.close()


def _status_score(status: str) -> int:
    return {"pass": 0, "warn": 1, "fail": 2}.get(status, 2)


def main() -> None:
    args = parse_args()

    correctness = run_correctness_checks(model_name=args.model_name)

    df = load_feature_data()
    df = add_odds_model_gap(df)
    side_frame = _build_team_side_frame(df)
    football_logic = run_football_logic_checks(side_frame)
    predictive = run_predictive_checks(df, min_games=args.min_games)
    operational = run_operational_checks()

    gate_status = {
        "correctness": correctness["status"],
        "football_logic": football_logic["status"],
        "predictive_value": predictive["status"],
        "operational_readiness": operational["status"],
    }
    worst = max(gate_status.values(), key=_status_score)

    recommendation = []
    if correctness["status"] != "pass":
        recommendation.append(
            "Do not trust the player-impact features until train/predict parity and timing checks are fixed."
        )
    if operational["status"] == "fail":
        recommendation.append(
            "Keep player-impact features as conditional for serving until upcoming availability coverage improves."
        )
    if predictive["status"] == "fail":
        recommendation.append(
            "Do not production-weight player-impact block globally; keep rule-based usage only."
        )
    if not recommendation:
        recommendation.append(
            "Player-impact implementation is consistent; keep as conditional and continue segment-level monitoring."
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Layer 2 player-impact assumptions (home_xg_lost, away_xg_lost, home_key_absent, away_key_absent, injury_impact)",
        "gate_status": gate_status,
        "overall_status": worst,
        "correctness": correctness,
        "football_logic": football_logic,
        "predictive_value": predictive,
        "operational_readiness": operational,
        "recommendation": recommendation,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    md_path = args.output_dir / f"{args.output_stem}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Player-Impact Assumption Validation")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append("")
    lines.append("## Gate Status")
    lines.append(f"- correctness: `{gate_status['correctness']}`")
    lines.append(f"- football_logic: `{gate_status['football_logic']}`")
    lines.append(f"- predictive_value: `{gate_status['predictive_value']}`")
    lines.append(f"- operational_readiness: `{gate_status['operational_readiness']}`")
    lines.append(f"- overall_status: `{payload['overall_status']}`")
    lines.append("")
    lines.append("## Correctness")
    lines.append(f"- rows compared (train vs predict formula): {correctness['rows_compared']}")
    lines.append(
        f"- timing late rows: {correctness['timing']['late_rows']} "
        f"(column: {correctness['timing_column_used']})"
    )
    for feature, stats in correctness["parity_mismatch"].items():
        lines.append(f"- parity {feature}: {stats}")
    lines.append("")
    lines.append("## Football Logic")
    lines.append(
        f"- corr(xg_lost, residual): {football_logic['corr_xg_lost_vs_residual']:.4f}"
    )
    lines.append(
        f"- residual delta (key_absent=1 minus key_absent=0): "
        f"{football_logic['mean_residual_delta_key1_minus_key0']:.4f}"
    )
    lines.append(f"- triggered share: {football_logic['triggered_share']:.2%}")
    lines.append("")
    lines.append("## Predictive Value")
    lines.append(
        f"- best by triggered lift: {predictive['best_by_triggered_lift']['name']} "
        f"(triggered_combined_lift={predictive['best_by_triggered_lift']['triggered_combined_lift']:.2%}, "
        f"global_combined_lift={predictive['best_by_triggered_lift']['global_combined_lift']:.2%})"
    )
    lines.append(
        f"- best by global lift: {predictive['best_by_global_lift']['name']} "
        f"(triggered_combined_lift={predictive['best_by_global_lift']['triggered_combined_lift']:.2%}, "
        f"global_combined_lift={predictive['best_by_global_lift']['global_combined_lift']:.2%})"
    )
    lines.append("")
    lines.append("## Operational Readiness")
    lines.append(
        f"- upcoming coverage (14d): {operational['upcoming_14d_fixtures_with_availability']}/"
        f"{operational['upcoming_14d_fixtures_total']} "
        f"({operational['upcoming_14d_coverage_pct']:.2%})"
    )
    lines.append(
        f"- historical FT coverage: {operational['historical_ft_fixtures_with_availability']}/"
        f"{operational['historical_ft_fixtures_total']} "
        f"({operational['historical_ft_coverage_pct']:.2%})"
    )
    lines.append(
        f"- fixture_player_stats expected_goals null pct: "
        f"{operational['fixture_player_stats_expected_goals_null_pct']:.2%}"
    )
    lines.append("")
    lines.append("## Recommendation")
    for rec in recommendation:
        lines.append(f"- {rec}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
