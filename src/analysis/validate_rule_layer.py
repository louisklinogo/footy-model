"""
Rule Layer Validation Script

Validates the hand-tuned percentages in the rule layer against actual match data.

Analyzes whether the rule adjustments (key_absent: -8%, congestion: -4%, rest_disadvantage: -5%)
are supported by residual analysis.

Usage:
    python src/analysis/validate_rule_layer.py
    python src/analysis/validate_rule_layer.py --min-games 4
    python src/analysis/validate_rule_layer.py --output-dir artifacts/analysis

Output:
    - Mean residual with/without each condition
    - Effect size in goals
    - Sample size
    - Statistical significance (t-test)
    - Comparison to hand-tuned percentages
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


# Hand-tuned percentages from rule_layer.py
HAND_TUNED_PERCENTAGES = {
    "key_absent": -0.08,  # -8%
    "congestion": -0.04,  # -4%
    "rest_disadvantage": -0.05,  # -5%
}

# Congestion threshold from rule_layer.py
CONGESTION_THRESHOLD_GAMES_14D = 3

# Rest disadvantage threshold from rule_layer.py
REST_DISADVANTAGE_THRESHOLD_DAYS = -2.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate rule layer percentages against actual match data."
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=4,
        help="Minimum games played filter (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis"),
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="rule_layer_validation",
        help="Output file stem.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of fixtures (for testing).",
    )
    return parser.parse_args()


def fetch_validation_data(limit: int | None = None, min_games: int = 4) -> pd.DataFrame:
    """
    Fetch fixtures with:
    - Actual goals (home_goals, away_goals)
    - Lambda predictions (from predictions table)
    - Player availability (key_absent flags)
    - Congestion flags
    - Rest days

    Returns a DataFrame with one row per fixture.
    """
    conn = connect_db()

    limit_sql = ""
    params: list[Any] = []
    if limit:
        limit_sql = "LIMIT %s"
        params.append(limit)

    query = f"""
        WITH lambda_values AS (
            SELECT
                fixture_id,
                MAX(CASE WHEN market_code = 'lambda_home'
                    THEN (metadata_json->>'lambda')::double precision END) AS lambda_home,
                MAX(CASE WHEN market_code = 'lambda_away'
                    THEN (metadata_json->>'lambda')::double precision END) AS lambda_away
            FROM predictions
            WHERE model_name = 'lambda_xgb'
              AND model_version = 'v1'
              AND (metadata_json->>'lambda') IS NOT NULL
            GROUP BY fixture_id
        ),
        player_impact AS (
            -- Compute key_absent flags from player_availability
            SELECT
                bf.fixture_id,
                COALESCE(MAX(pi.has_key_absence) FILTER (WHERE pi.team_id = bf.home_team_id), 0) AS home_key_absent,
                COALESCE(MAX(pi.has_key_absence) FILTER (WHERE pi.team_id = bf.away_team_id), 0) AS away_key_absent
            FROM (
                SELECT fixture_id, home_team_id, away_team_id, match_datetime_utc
                FROM fixtures
                WHERE status = 'ft'
            ) bf
            LEFT JOIN (
                SELECT
                    pa.fixture_id,
                    pa.team_id,
                    MAX(CASE WHEN CAST(q.starts AS FLOAT) / GREATEST(q.apps, 1) >= 0.7 AND q.apps >= 3 THEN 1 ELSE 0 END) AS has_key_absence
                FROM player_availability pa
                JOIN fixtures f ON f.fixture_id = pa.fixture_id
                LEFT JOIN LATERAL (
                    SELECT
                        COUNT(*) AS apps,
                        COUNT(*) FILTER (WHERE fps.substituted_in = False) AS starts
                    FROM (
                        SELECT fps2.substituted_in
                        FROM fixture_player_stats fps2
                        JOIN fixtures f2 ON f2.fixture_id = fps2.fixture_id
                        WHERE fps2.player_id = pa.player_id
                          AND f2.match_datetime_utc < f.match_datetime_utc
                        ORDER BY f2.match_datetime_utc DESC
                        LIMIT 10
                    ) fps
                ) q ON TRUE
                WHERE pa.status IN ('missing', 'doubtful')
                GROUP BY pa.fixture_id, pa.team_id
            ) pi ON pi.fixture_id = bf.fixture_id
            GROUP BY bf.fixture_id
        ),
        team_snapshots AS (
            SELECT
                fixture_id,
                MAX(CASE WHEN is_home THEN rolling_rest_days END) AS home_rest_days,
                MAX(CASE WHEN NOT is_home THEN rolling_rest_days END) AS away_rest_days
            FROM team_premium_snapshots
            GROUP BY fixture_id
        ),
        congestion_data AS (
            -- Compute congestion: count fixtures in last 14 days for each team
            SELECT
                fixture_id,
                home_recent_14d,
                away_recent_14d
            FROM (
                SELECT
                    f.fixture_id,
                    (
                        SELECT COUNT(*) FROM fixtures f2
                        WHERE (f2.home_team_id = f.home_team_id OR f2.away_team_id = f.home_team_id)
                          AND f2.status = 'ft'
                          AND f2.match_datetime_utc < f.match_datetime_utc
                          AND f2.match_datetime_utc >= f.match_datetime_utc - INTERVAL '14 days'
                    ) AS home_recent_14d,
                    (
                        SELECT COUNT(*) FROM fixtures f2
                        WHERE (f2.home_team_id = f.away_team_id OR f2.away_team_id = f.away_team_id)
                          AND f2.status = 'ft'
                          AND f2.match_datetime_utc < f.match_datetime_utc
                          AND f2.match_datetime_utc >= f.match_datetime_utc - INTERVAL '14 days'
                    ) AS away_recent_14d
                FROM fixtures f
                WHERE f.status = 'ft'
            ) subq
        )
        SELECT
            f.fixture_id,
            f.league_code,
            f.match_datetime_utc,
            fr.home_goals,
            fr.away_goals,
            l.lambda_home,
            l.lambda_away,
            pi.home_key_absent,
            pi.away_key_absent,
            ts.home_rest_days,
            ts.away_rest_days,
            cd.home_recent_14d,
            cd.away_recent_14d
        FROM fixtures f
        JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
        JOIN lambda_values l ON l.fixture_id = f.fixture_id
        LEFT JOIN player_impact pi ON pi.fixture_id = f.fixture_id
        LEFT JOIN team_snapshots ts ON ts.fixture_id = f.fixture_id
        LEFT JOIN congestion_data cd ON cd.fixture_id = f.fixture_id
        WHERE f.status = 'ft'
          AND fr.home_goals IS NOT NULL
          AND fr.away_goals IS NOT NULL
          AND l.lambda_home IS NOT NULL
          AND l.lambda_away IS NOT NULL
        ORDER BY f.match_datetime_utc DESC
        {limit_sql}
    """

    df = pd.read_sql(query, conn, params=tuple(params) if params else None)
    conn.close()

    print(f"Loaded {len(df)} fixtures with lambda values and match results")

    # Fill NAs
    df["home_key_absent"] = df["home_key_absent"].fillna(0).astype(int)
    df["away_key_absent"] = df["away_key_absent"].fillna(0).astype(int)
    df["home_rest_days"] = df["home_rest_days"].fillna(7.0)  # Default to 7 days rest
    df["away_rest_days"] = df["away_rest_days"].fillna(7.0)
    df["home_recent_14d"] = df["home_recent_14d"].fillna(0).astype(int)
    df["away_recent_14d"] = df["away_recent_14d"].fillna(0).astype(int)

    return df


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate residuals: residual = actual_goals - lambda
    """
    df = df.copy()
    df["home_residual"] = df["home_goals"] - df["lambda_home"]
    df["away_residual"] = df["away_goals"] - df["lambda_away"]
    return df


def compute_condition_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute condition flags for each rule:

    - key_absent: 1 if key player is absent
    - congestion: 1 if recent_14d >= CONGESTION_THRESHOLD_GAMES_14D
    - rest_disadvantage: 1 if rest_delta <= REST_DISADVANTAGE_THRESHOLD_DAYS
    """
    df = df.copy()

    # Key absent flag (home or away)
    df["home_key_absent_flag"] = (df["home_key_absent"] == 1).astype(int)
    df["away_key_absent_flag"] = (df["away_key_absent"] == 1).astype(int)

    # Congestion flag (home or away)
    df["home_congestion_flag"] = (df["home_recent_14d"] >= CONGESTION_THRESHOLD_GAMES_14D).astype(int)
    df["away_congestion_flag"] = (df["away_recent_14d"] >= CONGESTION_THRESHOLD_GAMES_14D).astype(int)

    # Rest delta (home - away)
    df["rest_delta"] = df["home_rest_days"] - df["away_rest_days"]

    # Rest disadvantage flag
    # Home has disadvantage if rest_delta <= -2.5 (away has more rest)
    # Away has disadvantage if rest_delta >= 2.5 (home has more rest)
    df["home_rest_disadvantage_flag"] = (df["rest_delta"] <= REST_DISADVANTAGE_THRESHOLD_DAYS).astype(int)
    df["away_rest_disadvantage_flag"] = (df["rest_delta"] >= -REST_DISADVANTAGE_THRESHOLD_DAYS).astype(int)

    return df


def analyze_rule_effect(
    df: pd.DataFrame,
    condition_col: str,
    residual_col: str,
    rule_name: str,
    hand_tuned_pct: float,
) -> dict[str, Any]:
    """
    Analyze the effect of a rule condition on residuals.

    Returns:
        - Mean residual with condition
        - Mean residual without condition
        - Effect size (difference)
        - Sample sizes
        - t-test results
        - Comparison to hand-tuned percentage
    """
    # Split into with/without condition
    with_condition = df[df[condition_col] == 1][residual_col].dropna()
    without_condition = df[df[condition_col] == 0][residual_col].dropna()

    n_with = len(with_condition)
    n_without = len(without_condition)

    if n_with < 10 or n_without < 10:
        return {
            "rule_name": rule_name,
            "condition_col": condition_col,
            "residual_col": residual_col,
            "n_with_condition": int(n_with),
            "n_without_condition": int(n_without),
            "mean_residual_with": None,
            "mean_residual_without": None,
            "effect_size": None,
            "effect_size_pct": None,
            "hand_tuned_pct": float(hand_tuned_pct),
            "t_statistic": None,
            "p_value": None,
            "significant_005": None,
            "direction_correct": None,
            "magnitude_supported": None,
            "supported": None,
            "note": "Insufficient sample size (need at least 10 in each group).",
        }

    mean_with = float(with_condition.mean())
    mean_without = float(without_condition.mean())
    effect_size = mean_with - mean_without

    # Effect size as percentage of mean lambda
    # Column names are lambda_home, lambda_away
    # residual_col is home_residual or away_residual
    if "home" in residual_col:
        lambda_col = "lambda_home"
    else:
        lambda_col = "lambda_away"
    mean_lambda = df[lambda_col].mean()
    effect_size_pct = effect_size / mean_lambda if mean_lambda and mean_lambda > 0 else None

    # Welch's t-test (unequal variance)
    t_stat, p_value = stats.ttest_ind(with_condition, without_condition, equal_var=False)

    # Is the direction correct?
    # Hand-tuned says reduce lambda by X%, so we expect negative residual when condition is true
    # Expected: mean_residual_with < mean_residual_without (effect_size < 0)
    expected_direction = hand_tuned_pct < 0  # Expect negative effect
    observed_direction = effect_size < 0
    direction_correct = expected_direction == observed_direction

    # Is the magnitude reasonable?
    # Hand-tuned says -8%, so we expect roughly -8% effect on lambda
    # But this is tricky: the effect size is in goals, not percentage
    # We'll compare the effect size percentage to the hand-tuned percentage
    magnitude_supported = False
    if effect_size_pct is not None:
        # Allow 50% tolerance: if hand-tuned is -8%, we accept -4% to -12%
        if hand_tuned_pct < 0:
            magnitude_supported = effect_size_pct <= hand_tuned_pct * 0.5
        else:
            magnitude_supported = effect_size_pct >= hand_tuned_pct * 0.5

    supported = direction_correct and (p_value < 0.05) and magnitude_supported

    return {
        "rule_name": rule_name,
        "condition_col": condition_col,
        "residual_col": residual_col,
        "n_with_condition": int(n_with),
        "n_without_condition": int(n_without),
        "mean_residual_with": round(float(mean_with), 4),
        "mean_residual_without": round(float(mean_without), 4),
        "effect_size": round(float(effect_size), 4),
        "effect_size_pct": round(float(effect_size_pct), 4) if effect_size_pct is not None else None,
        "hand_tuned_pct": float(hand_tuned_pct),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_005": bool(p_value < 0.05),
        "direction_correct": bool(direction_correct),
        "magnitude_supported": bool(magnitude_supported),
        "supported": bool(supported),
    }


def analyze_all_rules(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze all three rules: key_absent, congestion, rest_disadvantage.
    """
    results: dict[str, Any] = {
        "rules": {},
        "summary": {},
    }

    # Key absent - Home
    key_absent_home = analyze_rule_effect(
        df=df,
        condition_col="home_key_absent_flag",
        residual_col="home_residual",
        rule_name="key_absent_home",
        hand_tuned_pct=HAND_TUNED_PERCENTAGES["key_absent"],
    )
    results["rules"]["key_absent_home"] = key_absent_home

    # Key absent - Away
    key_absent_away = analyze_rule_effect(
        df=df,
        condition_col="away_key_absent_flag",
        residual_col="away_residual",
        rule_name="key_absent_away",
        hand_tuned_pct=HAND_TUNED_PERCENTAGES["key_absent"],
    )
    results["rules"]["key_absent_away"] = key_absent_away

    # Congestion - Home
    congestion_home = analyze_rule_effect(
        df=df,
        condition_col="home_congestion_flag",
        residual_col="home_residual",
        rule_name="congestion_home",
        hand_tuned_pct=HAND_TUNED_PERCENTAGES["congestion"],
    )
    results["rules"]["congestion_home"] = congestion_home

    # Congestion - Away
    congestion_away = analyze_rule_effect(
        df=df,
        condition_col="away_congestion_flag",
        residual_col="away_residual",
        rule_name="congestion_away",
        hand_tuned_pct=HAND_TUNED_PERCENTAGES["congestion"],
    )
    results["rules"]["congestion_away"] = congestion_away

    # Rest disadvantage - Home
    rest_disadvantage_home = analyze_rule_effect(
        df=df,
        condition_col="home_rest_disadvantage_flag",
        residual_col="home_residual",
        rule_name="rest_disadvantage_home",
        hand_tuned_pct=HAND_TUNED_PERCENTAGES["rest_disadvantage"],
    )
    results["rules"]["rest_disadvantage_home"] = rest_disadvantage_home

    # Rest disadvantage - Away
    rest_disadvantage_away = analyze_rule_effect(
        df=df,
        condition_col="away_rest_disadvantage_flag",
        residual_col="away_residual",
        rule_name="rest_disadvantage_away",
        hand_tuned_pct=HAND_TUNED_PERCENTAGES["rest_disadvantage"],
    )
    results["rules"]["rest_disadvantage_away"] = rest_disadvantage_away

    # Summary: aggregate home+away for each rule family
    for rule_family in ["key_absent", "congestion", "rest_disadvantage"]:
        home_result = results["rules"][f"{rule_family}_home"]
        away_result = results["rules"][f"{rule_family}_away"]

        # Aggregate effect sizes
        effects = []
        for r in [home_result, away_result]:
            if r.get("effect_size") is not None:
                effects.append(r["effect_size"])

        if effects:
            avg_effect = np.mean(effects)
        else:
            avg_effect = None

        # Count supported
        supported_count = sum(
            1 for r in [home_result, away_result]
            if r.get("supported") is True
        )

        results["summary"][rule_family] = {
            "hand_tuned_pct": float(HAND_TUNED_PERCENTAGES[rule_family]),
            "avg_effect_size": round(float(avg_effect), 4) if avg_effect is not None else None,
            "home_supported": bool(home_result.get("supported")) if home_result.get("supported") is not None else None,
            "away_supported": bool(away_result.get("supported")) if away_result.get("supported") is not None else None,
            "supported_count": int(supported_count),
            "verdict": "SUPPORTED" if supported_count >= 2 else "NOT SUPPORTED",
        }

    return results


def generate_report(df: pd.DataFrame, results: dict[str, Any]) -> str:
    """Generate a human-readable summary report."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("RULE LAYER VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Data summary
    lines.append("DATA SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total fixtures analyzed: {len(df)}")
    lines.append(f"Fixtures with lambda values: {df['lambda_home'].notna().sum()}")
    lines.append(f"")
    lines.append(f"Lambda Home - Mean: {df['lambda_home'].mean():.3f}, Std: {df['lambda_home'].std():.3f}")
    lines.append(f"Lambda Away - Mean: {df['lambda_away'].mean():.3f}, Std: {df['lambda_away'].std():.3f}")
    lines.append(f"")
    lines.append(f"Home Residual - Mean: {df['home_residual'].mean():.4f}, Std: {df['home_residual'].std():.4f}")
    lines.append(f"Away Residual - Mean: {df['away_residual'].mean():.4f}, Std: {df['away_residual'].std():.4f}")
    lines.append("")

    # Condition frequencies
    lines.append("CONDITION FREQUENCIES")
    lines.append("-" * 40)
    lines.append(f"Home key_absent: {df['home_key_absent_flag'].sum()} ({df['home_key_absent_flag'].mean():.1%})")
    lines.append(f"Away key_absent: {df['away_key_absent_flag'].sum()} ({df['away_key_absent_flag'].mean():.1%})")
    lines.append(f"Home congestion: {df['home_congestion_flag'].sum()} ({df['home_congestion_flag'].mean():.1%})")
    lines.append(f"Away congestion: {df['away_congestion_flag'].sum()} ({df['away_congestion_flag'].mean():.1%})")
    lines.append(f"Home rest_disadvantage: {df['home_rest_disadvantage_flag'].sum()} ({df['home_rest_disadvantage_flag'].mean():.1%})")
    lines.append(f"Away rest_disadvantage: {df['away_rest_disadvantage_flag'].sum()} ({df['away_rest_disadvantage_flag'].mean():.1%})")
    lines.append("")

    # Rule analysis
    lines.append("RULE ANALYSIS")
    lines.append("-" * 40)

    for rule_name, rule_data in results["rules"].items():
        lines.append(f"")
        lines.append(f"### {rule_name} ###")
        lines.append(f"  Condition: {rule_data.get('condition_col', 'N/A')}")
        lines.append(f"  Residual: {rule_data.get('residual_col', 'N/A')}")
        lines.append(f"  N with condition: {rule_data.get('n_with_condition', 0)}")
        lines.append(f"  N without condition: {rule_data.get('n_without_condition', 0)}")

        if rule_data.get("mean_residual_with") is not None:
            lines.append(f"  Mean residual WITH condition: {rule_data['mean_residual_with']:.4f}")
            lines.append(f"  Mean residual WITHOUT condition: {rule_data['mean_residual_without']:.4f}")
            lines.append(f"  Effect size (goals): {rule_data['effect_size']:.4f}")
            if rule_data.get("effect_size_pct") is not None:
                lines.append(f"  Effect size (% of lambda): {rule_data['effect_size_pct']:.2%}")
            lines.append(f"  Hand-tuned percentage: {rule_data['hand_tuned_pct']:.2%}")
            lines.append(f"  t-statistic: {rule_data['t_statistic']:.4f}")
            lines.append(f"  p-value: {rule_data['p_value']:.6f}")
            sig = "***" if rule_data["p_value"] < 0.001 else "**" if rule_data["p_value"] < 0.01 else "*" if rule_data["p_value"] < 0.05 else ""
            lines.append(f"  Significant at 0.05: {rule_data['significant_005']} {sig}")
            lines.append(f"  Direction correct: {rule_data.get('direction_correct')}")
            lines.append(f"  Magnitude supported: {rule_data.get('magnitude_supported')}")
            lines.append(f"  SUPPORTED: {rule_data.get('supported')}")
        else:
            lines.append(f"  {rule_data.get('note', 'Insufficient data')}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("SUMMARY VERDICTS")
    lines.append("-" * 40)

    for rule_family, summary in results["summary"].items():
        lines.append(f"")
        lines.append(f"### {rule_family.upper()} ###")
        lines.append(f"  Hand-tuned: {summary['hand_tuned_pct']:.2%}")
        if summary.get("avg_effect_size") is not None:
            lines.append(f"  Average effect size: {summary['avg_effect_size']:.4f} goals")
        lines.append(f"  Home supported: {summary['home_supported']}")
        lines.append(f"  Away supported: {summary['away_supported']}")
        lines.append(f"  VERDICT: {summary['verdict']}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("CONCLUSION")
    lines.append("-" * 40)

    total_supported = sum(1 for s in results["summary"].values() if s["verdict"] == "SUPPORTED")
    total_rules = len(results["summary"])

    if total_supported == total_rules:
        lines.append("All hand-tuned percentages are SUPPORTED by the data.")
    elif total_supported == 0:
        lines.append("None of the hand-tuned percentages are supported by the data.")
        lines.append("Consider revising the rule layer configuration.")
    else:
        lines.append(f"Mixed results: {total_supported}/{total_rules} rules are supported.")
        lines.append("Review individual rule analysis for details.")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    print("Loading validation data from database...")
    df = fetch_validation_data(limit=args.limit, min_games=args.min_games)

    if df.empty:
        print("ERROR: No data found.")
        return

    print("Computing residuals...")
    df = compute_residuals(df)

    print("Computing condition flags...")
    df = compute_condition_flags(df)

    print("Analyzing rule effects...")
    results = analyze_all_rules(df)

    # Generate report
    report = generate_report(df, results)
    print("\n" + report)

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = args.output_dir / f"{args.output_stem}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_fixtures": len(df),
        "hand_tuned_percentages": HAND_TUNED_PERCENTAGES,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON saved to: {json_path}")

    # Save report
    report_path = args.output_dir / f"{args.output_stem}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_path}")

    # Save CSV with detailed data
    csv_path = args.output_dir / f"{args.output_stem}_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")


if __name__ == "__main__":
    main()
