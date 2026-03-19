"""
Prediction Feedback Analysis

Analyzes prediction accuracy by grouping predictions into probability buckets
and comparing predicted probability vs actual hit rate (calibration analysis).
Identifies systematic error patterns by league, market, and team.

Usage:
    python src/analysis/prediction_feedback.py

Outputs:
    - Calibration by probability bucket
    - Error patterns by league
    - "When model said X%, what happened" report
    - Saved to artifacts/analysis/prediction_feedback_report.txt
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

# Probability bucket definitions (in percentage points)
PROBABILITY_BUCKETS = [
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.00, "90-100%"),
]

# Default lookback window for weekly report
DEFAULT_LOOKBACK_DAYS = 7


def fetch_scored_predictions(days: int = 30) -> pd.DataFrame:
    """
    Fetch scored predictions with fixture and team context.

    Returns DataFrame with:
    - prediction details (p_model, p_final, market_code)
    - scoring metrics (actual, brier, edge, odds_used)
    - fixture context (league_code, match_datetime, home/away teams)
    """
    conn = connect_db()

    query = """
    SELECT
        p.prediction_id,
        p.fixture_id,
        p.market_code,
        p.model_name,
        p.model_version,
        COALESCE(p.p_final, p.p_model) AS p_pred,
        p.p_model,
        p.p_final,
        p.created_at AS prediction_created_at,
        ps.actual,
        ps.brier,
        ps.log_loss,
        ps.hit,
        ps.odds_used,
        ps.edge,
        ps.roi_unit,
        ps.scored_at,
        f.league_code,
        f.match_datetime_utc,
        f.status AS fixture_status,
        ht.team_name AS home_team,
        at.team_name AS away_team,
        ht.team_id AS home_team_id,
        at.team_id AS away_team_id
    FROM predictions p
    JOIN prediction_scores ps ON ps.prediction_id = p.prediction_id
    JOIN fixtures f ON f.fixture_id = p.fixture_id
    LEFT JOIN teams ht ON ht.team_id = f.home_team_id
    LEFT JOIN teams at ON at.team_id = f.away_team_id
    WHERE f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
      AND ps.actual IS NOT NULL
    ORDER BY f.match_datetime_utc DESC
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query, (days,))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)
    return df


def assign_probability_bucket(p: float) -> str:
    """Assign a probability to a bucket label."""
    for low, high, label in PROBABILITY_BUCKETS:
        if low <= p < high:
            return label
    # Edge case: exactly 1.0
    if p >= 0.90:
        return "90-100%"
    return "50-60%"


def compute_calibration_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute calibration statistics by probability bucket.

    For each bucket:
    - Count of predictions
    - Mean predicted probability (confidence)
    - Actual hit rate (accuracy)
    - Calibration gap (accuracy - confidence)
    - Brier score
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["prob_bucket"] = df["p_pred"].apply(assign_probability_bucket)

    results = []
    for bucket_label in [b[2] for b in PROBABILITY_BUCKETS]:
        bucket_df = df[df["prob_bucket"] == bucket_label]
        if bucket_df.empty:
            continue

        n = len(bucket_df)
        mean_predicted = bucket_df["p_pred"].mean()
        actual_hit_rate = bucket_df["actual"].mean()
        calibration_gap = actual_hit_rate - mean_predicted
        mean_brier = bucket_df["brier"].mean()
        hits = int(bucket_df["actual"].sum())
        misses = n - hits

        results.append({
            "bucket": bucket_label,
            "n": n,
            "mean_predicted_pct": round(mean_predicted * 100, 1),
            "actual_hit_rate_pct": round(actual_hit_rate * 100, 1),
            "calibration_gap_pct": round(calibration_gap * 100, 2),
            "mean_brier": round(mean_brier, 4) if pd.notna(mean_brier) else None,
            "hits": hits,
            "misses": misses,
        })

    return pd.DataFrame(results)


def compute_error_patterns_by_league(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute error patterns grouped by league.

    Identifies leagues where the model systematically over/under-predicts.
    """
    if df.empty:
        return pd.DataFrame()

    results = []
    for league_code in df["league_code"].unique():
        if pd.isna(league_code):
            continue
        league_df = df[df["league_code"] == league_code]
        if len(league_df) < 10:  # Minimum sample size
            continue

        n = len(league_df)
        mean_predicted = league_df["p_pred"].mean()
        actual_rate = league_df["actual"].mean()
        bias = actual_rate - mean_predicted  # Positive = model under-predicts
        mean_brier = league_df["brier"].mean()
        positive_edge = league_df[league_df["edge"] > 0]["edge"].mean() if "edge" in league_df.columns else None

        results.append({
            "league_code": league_code,
            "n": n,
            "mean_predicted_pct": round(mean_predicted * 100, 1),
            "actual_rate_pct": round(actual_rate * 100, 1),
            "bias_pct": round(bias * 100, 2),
            "mean_brier": round(mean_brier, 4) if pd.notna(mean_brier) else None,
            "mean_positive_edge": round(positive_edge, 4) if pd.notna(positive_edge) else None,
            "bias_direction": "under-predicts" if bias > 0.05 else "over-predicts" if bias < -0.05 else "neutral",
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("n", ascending=False)
    return result_df


def compute_error_patterns_by_market(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute error patterns grouped by market.

    Identifies markets where the model systematically over/under-predicts.
    """
    if df.empty:
        return pd.DataFrame()

    results = []
    for market_code in df["market_code"].unique():
        if pd.isna(market_code):
            continue
        market_df = df[df["market_code"] == market_code]
        if len(market_df) < 10:
            continue

        n = len(market_df)
        mean_predicted = market_df["p_pred"].mean()
        actual_rate = market_df["actual"].mean()
        bias = actual_rate - mean_predicted
        mean_brier = market_df["brier"].mean()
        mean_edge = market_df["edge"].mean() if "edge" in market_df.columns else None

        results.append({
            "market_code": market_code,
            "n": n,
            "mean_predicted_pct": round(mean_predicted * 100, 1),
            "actual_rate_pct": round(actual_rate * 100, 1),
            "bias_pct": round(bias * 100, 2),
            "mean_brier": round(mean_brier, 4) if pd.notna(mean_brier) else None,
            "mean_edge": round(mean_edge, 4) if pd.notna(mean_edge) else None,
            "bias_direction": "under-predicts" if bias > 0.05 else "over-predicts" if bias < -0.05 else "neutral",
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("n", ascending=False)
    return result_df


def compute_error_patterns_by_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute error patterns by team (home or away).

    Identifies teams where predictions involving them have systematic bias.
    """
    if df.empty:
        return pd.DataFrame()

    # Collect team performance
    team_stats: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        market_code = row.get("market_code", "")

        # Determine if prediction is for home or away team
        # Common patterns: "home_win", "away_win", "btts_yes", "o2.5", etc.
        is_home_market = "home" in market_code.lower() if market_code else False
        is_away_market = "away" in market_code.lower() if market_code else False

        if is_home_market and home_team:
            team = str(home_team)
            if team not in team_stats:
                team_stats[team] = {"n": 0, "predicted_sum": 0.0, "actual_sum": 0.0, "brier_sum": 0.0}
            team_stats[team]["n"] += 1
            team_stats[team]["predicted_sum"] += row["p_pred"]
            team_stats[team]["actual_sum"] += row["actual"]
            if pd.notna(row.get("brier")):
                team_stats[team]["brier_sum"] += row["brier"]

        elif is_away_market and away_team:
            team = str(away_team)
            if team not in team_stats:
                team_stats[team] = {"n": 0, "predicted_sum": 0.0, "actual_sum": 0.0, "brier_sum": 0.0}
            team_stats[team]["n"] += 1
            team_stats[team]["predicted_sum"] += row["p_pred"]
            team_stats[team]["actual_sum"] += row["actual"]
            if pd.notna(row.get("brier")):
                team_stats[team]["brier_sum"] += row["brier"]

    # Convert to DataFrame
    results = []
    for team, stats in team_stats.items():
        if stats["n"] < 5:  # Minimum sample size
            continue

        n = stats["n"]
        mean_predicted = stats["predicted_sum"] / n
        actual_rate = stats["actual_sum"] / n
        bias = actual_rate - mean_predicted
        mean_brier = stats["brier_sum"] / n if n > 0 else None

        results.append({
            "team": team,
            "n": n,
            "mean_predicted_pct": round(mean_predicted * 100, 1),
            "actual_rate_pct": round(actual_rate * 100, 1),
            "bias_pct": round(bias * 100, 2),
            "mean_brier": round(mean_brier, 4) if mean_brier else None,
            "bias_direction": "under-predicts" if bias > 0.05 else "over-predicts" if bias < -0.05 else "neutral",
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        # Sort by absolute bias to show most biased teams
        result_df["abs_bias"] = result_df["bias_pct"].abs()
        result_df = result_df.sort_values("abs_bias", ascending=False)
        result_df = result_df.drop(columns=["abs_bias"])
    return result_df


def generate_probability_bucket_report(df: pd.DataFrame) -> str:
    """
    Generate "When model said X%, what happened" detailed report.

    For each probability bucket, shows:
    - What the model predicted
    - What actually happened
    - Example matches
    """
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("PROBABILITY BUCKET ANALYSIS: \"When the model said X%, what happened?\"")
    lines.append("=" * 80)
    lines.append("")

    if df.empty:
        lines.append("No data available for analysis.")
        return "\n".join(lines)

    df = df.copy()
    df["prob_bucket"] = df["p_pred"].apply(assign_probability_bucket)

    for bucket_label in [b[2] for b in PROBABILITY_BUCKETS]:
        bucket_df = df[df["prob_bucket"] == bucket_label]
        if bucket_df.empty:
            continue

        n = len(bucket_df)
        hits = int(bucket_df["actual"].sum())
        misses = n - hits
        hit_rate = bucket_df["actual"].mean() * 100
        mean_predicted = bucket_df["p_pred"].mean() * 100

        # Extract bucket midpoint for display
        bucket_mid = float(bucket_label.replace("%", "").split("-")[0]) + 5

        lines.append(f"")
        lines.append(f"--- {bucket_label} (midpoint: {bucket_mid:.0f}%) ---")
        lines.append(f"  Predictions: {n}")
        lines.append(f"  Model said: ~{mean_predicted:.1f}% probability")
        lines.append(f"  What happened: {hit_rate:.1f}% hit rate ({hits} hits, {misses} misses)")
        lines.append(f"  Calibration gap: {hit_rate - mean_predicted:+.1f} percentage points")

        if hit_rate > mean_predicted + 2:
            lines.append(f"  => Model is UNDER-CONFIDENT (predictions too low)")
        elif hit_rate < mean_predicted - 2:
            lines.append(f"  => Model is OVER-CONFIDENT (predictions too high)")
        else:
            lines.append(f"  => Model is WELL-CALIBRATED")

        # Show sample matches
        lines.append(f"  Sample matches:")
        sample = bucket_df.head(5)
        for _, row in sample.iterrows():
            home = row.get("home_team", "?")
            away = row.get("away_team", "?")
            market = row.get("market_code", "?")
            p = row.get("p_pred", 0) * 100
            actual = "HIT" if row.get("actual") else "MISS"
            lines.append(f"    - {home} vs {away} | {market} | {p:.0f}% -> {actual}")

    return "\n".join(lines)


def generate_full_report(
    calibration_df: pd.DataFrame,
    league_df: pd.DataFrame,
    market_df: pd.DataFrame,
    team_df: pd.DataFrame,
    bucket_report: str,
    total_predictions: int,
    lookback_days: int,
) -> str:
    """Generate the complete weekly feedback report."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("PREDICTION FEEDBACK WEEKLY REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Lookback period: {lookback_days} days")
    lines.append(f"Total predictions analyzed: {total_predictions}")
    lines.append("")

    # Overall summary
    if not calibration_df.empty:
        total_n = calibration_df["n"].sum()
        total_hits = calibration_df["hits"].sum()
        overall_hit_rate = (total_hits / total_n * 100) if total_n > 0 else 0
        lines.append("OVERALL SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total predictions: {total_n}")
        lines.append(f"Total hits: {total_hits}")
        lines.append(f"Overall hit rate: {overall_hit_rate:.1f}%")
        lines.append("")

    # Calibration by probability bucket
    lines.append("=" * 80)
    lines.append("CALIBRATION BY PROBABILITY BUCKET")
    lines.append("=" * 80)
    lines.append("")

    if calibration_df.empty:
        lines.append("No calibration data available.")
    else:
        lines.append(f"{'Bucket':<12} {'N':>8} {'Pred%':>8} {'Actual%':>8} {'Gap':>8} {'Brier':>8}")
        lines.append("-" * 52)
        for _, row in calibration_df.iterrows():
            gap = row.get("calibration_gap_pct", 0)
            gap_str = f"{gap:+.1f}" if pd.notna(gap) else "N/A"
            brier_str = f"{row['mean_brier']:.4f}" if pd.notna(row.get("mean_brier")) else "N/A"
            lines.append(
                f"{row['bucket']:<12} {row['n']:>8} {row['mean_predicted_pct']:>8.1f} "
                f"{row['actual_hit_rate_pct']:>8.1f} {gap_str:>8} {brier_str:>8}"
            )
        lines.append("")
        lines.append("Gap = Actual% - Predicted% (positive = model under-confident)")

    lines.append("")

    # Detailed bucket report
    lines.append(bucket_report)
    lines.append("")

    # Error patterns by league
    lines.append("=" * 80)
    lines.append("ERROR PATTERNS BY LEAGUE")
    lines.append("=" * 80)
    lines.append("")

    if league_df.empty:
        lines.append("No league data available.")
    else:
        lines.append(f"{'League':<12} {'N':>6} {'Pred%':>8} {'Actual%':>8} {'Bias':>8} {'Direction':<15}")
        lines.append("-" * 65)
        for _, row in league_df.head(20).iterrows():
            bias = row.get("bias_pct", 0)
            bias_str = f"{bias:+.1f}" if pd.notna(bias) else "N/A"
            lines.append(
                f"{row['league_code']:<12} {row['n']:>6} {row['mean_predicted_pct']:>8.1f} "
                f"{row['actual_rate_pct']:>8.1f} {bias_str:>8} {row['bias_direction']:<15}"
            )
        lines.append("")
        lines.append("Bias = Actual% - Predicted% (positive = model under-predicts for this league)")

    lines.append("")

    # Error patterns by market
    lines.append("=" * 80)
    lines.append("ERROR PATTERNS BY MARKET")
    lines.append("=" * 80)
    lines.append("")

    if market_df.empty:
        lines.append("No market data available.")
    else:
        lines.append(f"{'Market':<20} {'N':>6} {'Pred%':>8} {'Actual%':>8} {'Bias':>8} {'Direction':<15}")
        lines.append("-" * 70)
        for _, row in market_df.iterrows():
            bias = row.get("bias_pct", 0)
            bias_str = f"{bias:+.1f}" if pd.notna(bias) else "N/A"
            lines.append(
                f"{row['market_code']:<20} {row['n']:>6} {row['mean_predicted_pct']:>8.1f} "
                f"{row['actual_rate_pct']:>8.1f} {bias_str:>8} {row['bias_direction']:<15}"
            )
        lines.append("")
        lines.append("Bias = Actual% - Predicted% (positive = model under-predicts for this market)")

    lines.append("")

    # Error patterns by team (top biased)
    lines.append("=" * 80)
    lines.append("ERROR PATTERNS BY TEAM (Top 15 Most Biased)")
    lines.append("=" * 80)
    lines.append("")

    if team_df.empty:
        lines.append("No team data available.")
    else:
        lines.append(f"{'Team':<25} {'N':>5} {'Pred%':>8} {'Actual%':>8} {'Bias':>8} {'Direction':<15}")
        lines.append("-" * 75)
        for _, row in team_df.head(15).iterrows():
            bias = row.get("bias_pct", 0)
            bias_str = f"{bias:+.1f}" if pd.notna(bias) else "N/A"
            lines.append(
                f"{row['team'][:25]:<25} {row['n']:>5} {row['mean_predicted_pct']:>8.1f} "
                f"{row['actual_rate_pct']:>8.1f} {bias_str:>8} {row['bias_direction']:<15}"
            )
        lines.append("")
        lines.append("Bias = Actual% - Predicted% (positive = model under-predicts for this team)")

    lines.append("")

    # Key insights
    lines.append("=" * 80)
    lines.append("KEY INSIGHTS")
    lines.append("=" * 80)
    lines.append("")

    insights: list[str] = []

    # Calibration insights
    if not calibration_df.empty:
        abs_gaps = calibration_df["calibration_gap_pct"].abs()
        max_idx = abs_gaps.idxmax()
        max_gap_row = calibration_df.loc[max_idx]
        if abs(max_gap_row["calibration_gap_pct"]) > 5:
            direction = "under-confident" if max_gap_row["calibration_gap_pct"] > 0 else "over-confident"
            insights.append(
                f"- Largest calibration gap in {max_gap_row['bucket']}: "
                f"model is {direction} by {abs(max_gap_row['calibration_gap_pct']):.1f}pp"
            )

    # League insights
    if not league_df.empty:
        biased_leagues = league_df[league_df["bias_direction"] != "neutral"]
        if len(biased_leagues) > 0:
            top_biased = biased_leagues.head(3)
            for _, row in top_biased.iterrows():
                insights.append(
                    f"- {row['league_code']}: model {row['bias_direction']} by {abs(row['bias_pct']):.1f}pp "
                    f"(n={row['n']})"
                )

    # Market insights
    if not market_df.empty:
        biased_markets = market_df[market_df["bias_direction"] != "neutral"]
        if len(biased_markets) > 0:
            for _, row in biased_markets.head(3).iterrows():
                insights.append(
                    f"- {row['market_code']}: model {row['bias_direction']} by {abs(row['bias_pct']):.1f}pp"
                )

    if insights:
        for insight in insights:
            lines.append(insight)
    else:
        lines.append("- Model appears well-calibrated across all segments")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> None:
    """Main entry point for prediction feedback analysis."""
    print("=" * 60)
    print("PREDICTION FEEDBACK ANALYSIS")
    print("=" * 60)
    print()

    # Fetch data
    print(f"Fetching scored predictions from last {DEFAULT_LOOKBACK_DAYS} days...")
    df = fetch_scored_predictions(days=DEFAULT_LOOKBACK_DAYS)

    if df.empty:
        print("ERROR: No scored predictions found in the database.")
        print("Ensure predictions have been made and scored.")
        return

    print(f"Loaded {len(df)} scored predictions")
    print()

    # Compute analyses
    print("Computing calibration by probability bucket...")
    calibration_df = compute_calibration_by_bucket(df)

    print("Computing error patterns by league...")
    league_df = compute_error_patterns_by_league(df)

    print("Computing error patterns by market...")
    market_df = compute_error_patterns_by_market(df)

    print("Computing error patterns by team...")
    team_df = compute_error_patterns_by_team(df)

    print("Generating probability bucket report...")
    bucket_report = generate_probability_bucket_report(df)

    # Generate full report
    print("Generating full report...")
    report = generate_full_report(
        calibration_df=calibration_df,
        league_df=league_df,
        market_df=market_df,
        team_df=team_df,
        bucket_report=bucket_report,
        total_predictions=len(df),
        lookback_days=DEFAULT_LOOKBACK_DAYS,
    )

    # Print report
    print()
    print(report)

    # Save to file
    output_dir = ROOT_DIR / "artifacts" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "prediction_feedback_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print()
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
