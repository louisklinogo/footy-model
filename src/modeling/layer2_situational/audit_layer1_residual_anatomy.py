from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


@dataclass(frozen=True)
class SegmentSummary:
    segment: str
    value: str
    n: int
    home_bias_mean: float
    away_bias_mean: float
    home_bias_ci_low: float
    home_bias_ci_high: float
    away_bias_ci_low: float
    away_bias_ci_high: float
    home_rmse: float
    away_rmse: float
    home_mae: float
    away_mae: float
    decision_grade: bool
    informative_grade: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Layer 1 residual anatomy for Layer 2 reconciliation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation"),
        help="Directory for residual anatomy artifacts",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lambda_xgb",
        help="Layer 1 lambda model_name in predictions table",
    )
    parser.add_argument(
        "--decision-min-n",
        type=int,
        default=120,
        help="Minimum segment size for decision-grade evidence",
    )
    parser.add_argument(
        "--informative-min-n",
        type=int,
        default=50,
        help="Minimum segment size for informative (non-decision) evidence",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=365,
        help="Recent window used for drift-focused summary",
    )
    args = parser.parse_args()
    if args.decision_min_n <= 0:
        raise ValueError("--decision-min-n must be > 0")
    if args.informative_min_n <= 0:
        raise ValueError("--informative-min-n must be > 0")
    if args.recent_days <= 0:
        raise ValueError("--recent-days must be > 0")
    return args


def fetch_base_fixtures() -> pd.DataFrame:
    query = """
    SELECT
        f.fixture_id,
        f.league_code,
        f.match_datetime_utc,
        fr.home_goals,
        fr.away_goals
    FROM fixtures f
    JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
    WHERE f.status = 'ft'
      AND f.match_datetime_utc IS NOT NULL
      AND fr.home_goals IS NOT NULL
      AND fr.away_goals IS NOT NULL
    ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC
    """
    conn = connect_db()
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    df["match_datetime_utc"] = pd.to_datetime(df["match_datetime_utc"], utc=True, errors="coerce")
    return df


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
    return bool(row and row[0])


def fetch_lambda_rows(model_name: str) -> pd.DataFrame:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            has_feature_asof = _column_exists(cur, "predictions", "feature_asof_utc")
            has_first_created = _column_exists(cur, "predictions", "first_created_at")

        lineage_parts: list[str] = []
        if has_feature_asof:
            lineage_parts.append("feature_asof_utc")
        if has_first_created:
            lineage_parts.append("first_created_at")
        lineage_parts.append("created_at")
        lineage_expr = f"COALESCE({', '.join(lineage_parts)})"

        source_parts: list[str] = []
        if has_feature_asof:
            source_parts.append("WHEN feature_asof_utc IS NOT NULL THEN 'feature_asof_utc'")
        if has_first_created:
            source_parts.append("WHEN first_created_at IS NOT NULL THEN 'first_created_at'")
        source_case = "CASE " + " ".join(source_parts) + " ELSE 'created_at' END"

        query = f"""
        SELECT
            fixture_id,
            model_version,
            market_code,
            {lineage_expr} AS lineage_at,
            {source_case} AS lineage_source,
            prediction_id,
            (metadata_json->>'lambda')::double precision AS lambda_value
        FROM predictions
        WHERE model_name = %s
          AND market_code IN ('lambda_home', 'lambda_away')
          AND (metadata_json->>'lambda') IS NOT NULL
        """
        df = pd.read_sql(query, conn, params=(model_name,))
    finally:
        conn.close()
    df["lineage_at"] = pd.to_datetime(df["lineage_at"], utc=True, errors="coerce")
    return df


def build_lambda_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "model_version",
                "pair_lineage_at",
                "pair_prediction_id",
                "lambda_home",
                "lambda_away",
            ]
        )

    source_rank = {
        "feature_asof_utc": 3,
        "first_created_at": 2,
        "created_at": 1,
    }
    rows = rows.copy()
    rows["lineage_source_rank"] = rows["lineage_source"].map(source_rank).fillna(1).astype(int)

    grouped = (
        rows.groupby(["fixture_id", "model_version"], dropna=False)
        .agg(
            pair_lineage_at=("lineage_at", "max"),
            pair_lineage_source_rank=("lineage_source_rank", "max"),
            pair_prediction_id=("prediction_id", "max"),
            market_count=("market_code", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["market_count"] == 2].copy()
    if grouped.empty:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "model_version",
                "pair_lineage_at",
                "pair_lineage_source",
                "pair_prediction_id",
                "lambda_home",
                "lambda_away",
            ]
        )

    pivot = (
        rows.pivot_table(
            index=["fixture_id", "model_version"],
            columns="market_code",
            values="lambda_value",
            aggfunc="last",
        )
        .reset_index()
        .rename(columns={"lambda_home": "lambda_home", "lambda_away": "lambda_away"})
    )

    pairs = grouped.merge(pivot, on=["fixture_id", "model_version"], how="inner")
    reverse_rank = {v: k for k, v in source_rank.items()}
    pairs["pair_lineage_source"] = pairs["pair_lineage_source_rank"].map(reverse_rank).fillna("created_at")
    pairs = pairs.drop(columns=["market_count"])
    pairs = pairs.dropna(subset=["lambda_home", "lambda_away", "pair_lineage_at"])
    return pairs


def choose_latest_pairs(
    fixtures: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = fixtures[["fixture_id", "match_datetime_utc"]].merge(
        pairs, on="fixture_id", how="left"
    )
    merged = merged.dropna(subset=["pair_lineage_at"]).copy()

    if merged.empty:
        empty = pd.DataFrame(
            columns=[
                "fixture_id",
                "lambda_home",
                "lambda_away",
                "lambda_model_version",
                "lambda_lineage_at",
                "lambda_lineage_source",
            ]
        )
        return empty, empty

    merged = merged.sort_values(
        ["fixture_id", "pair_lineage_at", "pair_prediction_id", "model_version"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    latest_any = merged.drop_duplicates(subset=["fixture_id"], keep="first").copy()

    pre = merged[merged["pair_lineage_at"] <= merged["match_datetime_utc"]].copy()
    pre = pre.drop_duplicates(subset=["fixture_id"], keep="first")

    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        out = df[
            [
                "fixture_id",
                "lambda_home",
                "lambda_away",
                "model_version",
                "pair_lineage_at",
                "pair_lineage_source",
            ]
        ].copy()
        out = out.rename(
            columns={
                "model_version": "lambda_model_version",
                "pair_lineage_at": "lambda_lineage_at",
                "pair_lineage_source": "lambda_lineage_source",
            }
        )
        return out

    return _finalize(latest_any), _finalize(pre)


def _bias_ci(residuals: np.ndarray) -> tuple[float, float]:
    if residuals.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(residuals))
    if residuals.size < 2:
        return mean, mean
    std = float(np.std(residuals, ddof=1))
    se = std / float(np.sqrt(residuals.size))
    ci = 1.96 * se
    return mean - ci, mean + ci


def compute_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    n = int(len(df))
    if n == 0:
        return {
            "n": 0,
            "home_bias_mean": 0.0,
            "away_bias_mean": 0.0,
            "home_bias_ci_low": 0.0,
            "home_bias_ci_high": 0.0,
            "away_bias_ci_low": 0.0,
            "away_bias_ci_high": 0.0,
            "home_rmse": 0.0,
            "away_rmse": 0.0,
            "home_mae": 0.0,
            "away_mae": 0.0,
        }

    h = df["home_residual"].to_numpy(dtype=float)
    a = df["away_residual"].to_numpy(dtype=float)
    h_ci_low, h_ci_high = _bias_ci(h)
    a_ci_low, a_ci_high = _bias_ci(a)

    return {
        "n": n,
        "home_bias_mean": float(np.mean(h)),
        "away_bias_mean": float(np.mean(a)),
        "home_bias_ci_low": float(h_ci_low),
        "home_bias_ci_high": float(h_ci_high),
        "away_bias_ci_low": float(a_ci_low),
        "away_bias_ci_high": float(a_ci_high),
        "home_rmse": float(np.sqrt(np.mean(np.square(h)))),
        "away_rmse": float(np.sqrt(np.mean(np.square(a)))),
        "home_mae": float(np.mean(np.abs(h))),
        "away_mae": float(np.mean(np.abs(a))),
    }


def summarize_segment(
    df: pd.DataFrame,
    segment_col: str,
    decision_min_n: int,
    informative_min_n: int,
) -> list[SegmentSummary]:
    rows: list[SegmentSummary] = []
    grouped = df.groupby(segment_col, dropna=False)
    for raw_value, g in grouped:
        value = str(raw_value) if pd.notna(raw_value) else "NA"
        metrics = compute_metrics(g)
        n = int(metrics["n"])
        rows.append(
            SegmentSummary(
                segment=segment_col,
                value=value,
                n=n,
                home_bias_mean=float(metrics["home_bias_mean"]),
                away_bias_mean=float(metrics["away_bias_mean"]),
                home_bias_ci_low=float(metrics["home_bias_ci_low"]),
                home_bias_ci_high=float(metrics["home_bias_ci_high"]),
                away_bias_ci_low=float(metrics["away_bias_ci_low"]),
                away_bias_ci_high=float(metrics["away_bias_ci_high"]),
                home_rmse=float(metrics["home_rmse"]),
                away_rmse=float(metrics["away_rmse"]),
                home_mae=float(metrics["home_mae"]),
                away_mae=float(metrics["away_mae"]),
                decision_grade=n >= decision_min_n,
                informative_grade=n >= informative_min_n,
            )
        )
    return rows


def add_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lambda_total"] = out["lambda_home"] + out["lambda_away"]
    out["lambda_edge"] = out["lambda_home"] - out["lambda_away"]

    out["lambda_total_bucket"] = pd.cut(
        out["lambda_total"],
        bins=[-np.inf, 2.2, 2.8, np.inf],
        labels=["low_total", "mid_total", "high_total"],
    ).astype(str)
    out["lambda_edge_bucket"] = pd.cut(
        out["lambda_edge"],
        bins=[-np.inf, -0.35, 0.35, np.inf],
        labels=["away_favored", "balanced", "home_favored"],
    ).astype(str)
    out["kickoff_month"] = out["match_datetime_utc"].dt.to_period("M").astype(str)
    return out


def top_problem_segments(
    segments: list[SegmentSummary],
    min_n: int,
    top_k: int = 12,
) -> list[dict[str, object]]:
    eligible = [s for s in segments if s.n >= min_n]
    scored = sorted(
        eligible,
        key=lambda s: (
            abs(s.home_bias_mean) + abs(s.away_bias_mean),
            s.home_rmse + s.away_rmse,
        ),
        reverse=True,
    )
    return [asdict(s) for s in scored[:top_k]]


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC)

    fixtures = fetch_base_fixtures()
    lambda_rows = fetch_lambda_rows(args.model_name)
    lambda_pairs = build_lambda_pairs(lambda_rows)
    latest_any, latest_pre = choose_latest_pairs(fixtures, lambda_pairs)

    latest_df = fixtures.merge(latest_any, on="fixture_id", how="inner")
    pre_df = fixtures.merge(latest_pre, on="fixture_id", how="inner")

    if latest_df.empty:
        raise RuntimeError("No latest Layer 1 lambda pairs were found for completed fixtures.")
    if pre_df.empty:
        raise RuntimeError("No pre-kickoff Layer 1 lambda pairs were found for completed fixtures.")

    for frame in (latest_df, pre_df):
        frame["home_residual"] = frame["home_goals"] - frame["lambda_home"]
        frame["away_residual"] = frame["away_goals"] - frame["lambda_away"]
        frame["lambda_lineage_at"] = pd.to_datetime(
            frame["lambda_lineage_at"], utc=True, errors="coerce"
        )
        frame["match_datetime_utc"] = pd.to_datetime(frame["match_datetime_utc"], utc=True, errors="coerce")

    late_latest = latest_df["lambda_lineage_at"] > latest_df["match_datetime_utc"]
    late_count = int(late_latest.sum())

    pre_lag_minutes = (
        (pre_df["match_datetime_utc"] - pre_df["lambda_lineage_at"]).dt.total_seconds() / 60.0
    )
    pre_lag_minutes = pre_lag_minutes.replace([np.inf, -np.inf], np.nan).dropna()
    lineage_source_counts = (
        latest_df["lambda_lineage_source"].value_counts(dropna=False).to_dict()
        if "lambda_lineage_source" in latest_df.columns
        else {}
    )

    latest_metrics = compute_metrics(latest_df)
    pre_metrics = compute_metrics(pre_df)

    recent_cutoff = datetime.now(UTC) - timedelta(days=args.recent_days)
    pre_recent = pre_df[pre_df["match_datetime_utc"] >= recent_cutoff].copy()
    pre_hist = pre_df[pre_df["match_datetime_utc"] < recent_cutoff].copy()
    recent_metrics = compute_metrics(pre_recent)
    historical_metrics = compute_metrics(pre_hist)

    pre_regime = add_regime_columns(pre_df)
    segment_columns = ["league_code", "lambda_total_bucket", "lambda_edge_bucket", "kickoff_month"]

    segment_summaries: dict[str, list[dict[str, object]]] = {}
    all_segment_rows: list[SegmentSummary] = []
    for col in segment_columns:
        rows = summarize_segment(
            df=pre_regime,
            segment_col=col,
            decision_min_n=args.decision_min_n,
            informative_min_n=args.informative_min_n,
        )
        segment_summaries[col] = [asdict(r) for r in rows]
        all_segment_rows.extend(rows)

    findings = {
        "started_at": started_at.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(),
        "defaults": {
            "scope": "all_leagues",
            "window": "full_history_plus_recent_window",
            "recent_days": args.recent_days,
            "decision_min_n": args.decision_min_n,
            "informative_min_n": args.informative_min_n,
            "model_name": args.model_name,
        },
        "coverage": {
            "fixtures_total": int(len(fixtures)),
            "fixtures_with_latest_pair": int(len(latest_df)),
            "fixtures_with_pre_kickoff_pair": int(len(pre_df)),
            "fixtures_missing_pre_kickoff_pair": int(len(fixtures) - len(pre_df)),
            "latest_pair_after_kickoff_count": late_count,
            "latest_pair_after_kickoff_pct": (
                float(late_count / len(latest_df)) if len(latest_df) else 0.0
            ),
            "pre_pair_lag_minutes": {
                "p50": float(pre_lag_minutes.quantile(0.50)) if not pre_lag_minutes.empty else 0.0,
                "p90": float(pre_lag_minutes.quantile(0.90)) if not pre_lag_minutes.empty else 0.0,
                "p99": float(pre_lag_minutes.quantile(0.99)) if not pre_lag_minutes.empty else 0.0,
            },
            "latest_pair_lineage_source_counts": {
                str(k): int(v) for k, v in lineage_source_counts.items()
            },
        },
        "overall_metrics": {
            "latest_anytime_pair": latest_metrics,
            "latest_pre_kickoff_pair": pre_metrics,
            "recent_pre_kickoff_pair": recent_metrics,
            "historical_pre_kickoff_pair": historical_metrics,
        },
        "segment_summaries": segment_summaries,
        "top_problem_segments": top_problem_segments(
            segments=all_segment_rows,
            min_n=args.decision_min_n,
            top_k=15,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "layer1_residual_anatomy.json"
    md_path = args.output_dir / "layer1_residual_anatomy.md"

    json_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Layer 1 Residual Anatomy Audit")
    lines.append("")
    lines.append(f"Generated: {findings['generated_at']}")
    lines.append("")
    lines.append("## Defaults Used")
    lines.append(f"- scope: all leagues")
    lines.append(f"- recent window: {args.recent_days} days")
    lines.append(f"- decision-grade segment threshold: n >= {args.decision_min_n}")
    lines.append(f"- informative segment threshold: n >= {args.informative_min_n}")
    lines.append("")
    lines.append("## Coverage and Timing Integrity")
    cov = findings["coverage"]
    lines.append(f"- fixtures total: {cov['fixtures_total']}")
    lines.append(f"- fixtures with latest lambda pair: {cov['fixtures_with_latest_pair']}")
    lines.append(f"- fixtures with pre-kickoff lambda pair: {cov['fixtures_with_pre_kickoff_pair']}")
    lines.append(f"- fixtures missing pre-kickoff lambda pair: {cov['fixtures_missing_pre_kickoff_pair']}")
    lines.append(
        "- latest chosen lambda pair occurs after kickoff: "
        f"{cov['latest_pair_after_kickoff_count']} ({cov['latest_pair_after_kickoff_pct']:.2%})"
    )
    lines.append(
        "- pre-kickoff lambda lag minutes (p50/p90/p99): "
        f"{cov['pre_pair_lag_minutes']['p50']:.1f} / {cov['pre_pair_lag_minutes']['p90']:.1f} / {cov['pre_pair_lag_minutes']['p99']:.1f}"
    )
    lines.append("")
    lines.append("## Overall Residual Metrics")
    lines.append("| View | n | Home Bias | Away Bias | Home RMSE | Away RMSE | Home MAE | Away MAE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in (
        "latest_anytime_pair",
        "latest_pre_kickoff_pair",
        "recent_pre_kickoff_pair",
        "historical_pre_kickoff_pair",
    ):
        m = findings["overall_metrics"][key]
        lines.append(
            f"| {key} | {m['n']} | {m['home_bias_mean']:.4f} | {m['away_bias_mean']:.4f} | "
            f"{m['home_rmse']:.4f} | {m['away_rmse']:.4f} | {m['home_mae']:.4f} | {m['away_mae']:.4f} |"
        )
    lines.append("")
    lines.append("## Top Problem Segments (Decision-Grade n)")
    lines.append("| Segment | Value | n | Home Bias | Away Bias | Home RMSE | Away RMSE |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in findings["top_problem_segments"]:
        lines.append(
            f"| {row['segment']} | {row['value']} | {row['n']} | "
            f"{row['home_bias_mean']:.4f} | {row['away_bias_mean']:.4f} | "
            f"{row['home_rmse']:.4f} | {row['away_rmse']:.4f} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- JSON: `{json_path}`")
    lines.append(f"- Markdown: `{md_path}`")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
