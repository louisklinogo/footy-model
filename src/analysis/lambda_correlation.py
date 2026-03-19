"""
Lambda Correlation Analysis

Analyzes the relationship between Poisson lambda values (home/away attack strength)
and match statistics (corners, fouls, cards, shots, possession, crosses).

Usage:
    python src/analysis/lambda_correlation.py

Outputs:
    - Pearson correlations between lambda and match statistics
    - Regression models: corners = f(lambda_home, lambda_away)
    - Summary report with coefficients and R-squared values
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


# Match statistics to analyze
STAT_COLUMNS = [
    "corners",
    "fouls",
    "yellow_cards",
    "red_cards",
    "shots",  # Total shots (we'll derive from existing columns)
    "possession",
    "crosses",
]


def fetch_lambda_and_match_stats(limit: int | None = None) -> pd.DataFrame:
    """
    Fetch lambda values from predictions table and match statistics from fixture_stats_premium.

    Returns a DataFrame with one row per fixture containing:
    - lambda_home, lambda_away: Poisson attack strength estimates
    - Match statistics for home and away teams
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
        )
        SELECT
            l.fixture_id,
            l.lambda_home,
            l.lambda_away,
            -- Home team stats
            s.h_corners,
            s.h_fouls,
            s.h_yellow_cards,
            s.h_red_cards,
            s.h_sot AS h_shots_on_target,
            s.h_possession,
            s.h_crosses,
            s.h_shots_inside_box,
            s.h_blocked_shots,
            -- Away team stats
            s.a_corners,
            s.a_fouls,
            s.a_yellow_cards,
            s.a_red_cards,
            s.a_sot AS a_shots_on_target,
            s.a_possession,
            s.a_crosses,
            s.a_shots_inside_box,
            s.a_blocked_shots,
            -- Match metadata
            f.league_code,
            f.match_datetime_utc
        FROM lambda_values l
        JOIN fixtures f ON f.fixture_id = l.fixture_id
        LEFT JOIN fixture_stats_premium s ON s.fixture_id = l.fixture_id
        WHERE f.status = 'ft'
          AND l.lambda_home IS NOT NULL
          AND l.lambda_away IS NOT NULL
        ORDER BY f.match_datetime_utc DESC
        {limit_sql}
    """

    df = pd.read_sql(query, conn, params=tuple(params) if params else None)
    conn.close()

    return df


def compute_derived_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived statistics:
    - Total corners, fouls, cards per match
    - Total shots (approximation from available columns)
    """
    df = df.copy()

    # Total match statistics
    df["total_corners"] = df["h_corners"].fillna(0) + df["a_corners"].fillna(0)
    df["total_fouls"] = df["h_fouls"].fillna(0) + df["a_fouls"].fillna(0)
    df["total_yellow_cards"] = df["h_yellow_cards"].fillna(0) + df["a_yellow_cards"].fillna(0)
    df["total_red_cards"] = df["h_red_cards"].fillna(0) + df["a_red_cards"].fillna(0)
    df["total_crosses"] = df["h_crosses"].fillna(0) + df["a_crosses"].fillna(0)

    # Shots approximation: SOT + blocked + shots inside box (rough total)
    df["h_total_shots"] = (
        df["h_shots_on_target"].fillna(0) +
        df["h_blocked_shots"].fillna(0) +
        df["h_shots_inside_box"].fillna(0)
    )
    df["a_total_shots"] = (
        df["a_shots_on_target"].fillna(0) +
        df["a_blocked_shots"].fillna(0) +
        df["a_shots_inside_box"].fillna(0)
    )
    df["total_shots"] = df["h_total_shots"] + df["a_total_shots"]

    # Possession differential (home - away)
    df["possession_diff"] = df["h_possession"] - df["a_possession"]

    return df


def compute_correlations(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute Pearson correlations between lambda values and match statistics.

    Returns a nested dict: {stat_name: {correlation_type: value}}
    """
    results: dict[str, dict[str, float]] = {}

    # Stats to correlate with lambda
    correlations_to_compute = [
        ("h_corners", "lambda_home", "Home Corners vs Lambda Home"),
        ("a_corners", "lambda_away", "Away Corners vs Lambda Away"),
        ("total_corners", "lambda_home", "Total Corners vs Lambda Home"),
        ("total_corners", "lambda_away", "Total Corners vs Lambda Away"),
        ("h_fouls", "lambda_home", "Home Fouls vs Lambda Home"),
        ("a_fouls", "lambda_away", "Away Fouls vs Lambda Away"),
        ("total_fouls", "lambda_home", "Total Fouls vs Lambda Home"),
        ("total_fouls", "lambda_away", "Total Fouls vs Lambda Away"),
        ("h_yellow_cards", "lambda_home", "Home Yellow Cards vs Lambda Home"),
        ("a_yellow_cards", "lambda_away", "Away Yellow Cards vs Lambda Away"),
        ("total_yellow_cards", "lambda_home", "Total Yellow Cards vs Lambda Home"),
        ("total_yellow_cards", "lambda_away", "Total Yellow Cards vs Lambda Away"),
        ("h_total_shots", "lambda_home", "Home Shots vs Lambda Home"),
        ("a_total_shots", "lambda_away", "Away Shots vs Lambda Away"),
        ("total_shots", "lambda_home", "Total Shots vs Lambda Home"),
        ("total_shots", "lambda_away", "Total Shots vs Lambda Away"),
        ("h_possession", "lambda_home", "Home Possession vs Lambda Home"),
        ("a_possession", "lambda_away", "Away Possession vs Lambda Away"),
        ("h_crosses", "lambda_home", "Home Crosses vs Lambda Home"),
        ("a_crosses", "lambda_away", "Away Crosses vs Lambda Away"),
        ("total_crosses", "lambda_home", "Total Crosses vs Lambda Home"),
        ("total_crosses", "lambda_away", "Total Crosses vs Lambda Away"),
    ]

    for stat_col, lambda_col, label in correlations_to_compute:
        if stat_col not in df.columns or lambda_col not in df.columns:
            continue

        # Drop NA pairs
        valid_mask = df[stat_col].notna() & df[lambda_col].notna()
        valid_data = df.loc[valid_mask, [stat_col, lambda_col]]

        if len(valid_data) < 30:  # Minimum sample size
            continue

        corr, p_value = scipy_stats.pearsonr(valid_data[stat_col], valid_data[lambda_col])

        results[label] = {
            "stat_column": stat_col,
            "lambda_column": lambda_col,
            "correlation": round(corr, 4),
            "p_value": round(p_value, 6),
            "n_samples": len(valid_data),
            "significant_005": p_value < 0.05,
        }

    return results


def fit_corner_regression(df: pd.DataFrame) -> dict[str, Any]:
    """
    Fit regression models for corners as a function of lambda_home and lambda_away.

    Returns regression coefficients and model statistics.
    """
    results: dict[str, Any] = {}

    # Prepare data - need both lambda values and corner counts
    required_cols = ["lambda_home", "lambda_away", "h_corners", "a_corners", "total_corners"]
    valid_mask = df[required_cols].notna().all(axis=1)
    valid_df = df.loc[valid_mask].copy()

    if len(valid_df) < 100:
        return {"error": f"Insufficient data: only {len(valid_df)} valid samples"}

    X = valid_df[["lambda_home", "lambda_away"]].values

    # Models to fit
    targets = [
        ("h_corners", "Home Corners"),
        ("a_corners", "Away Corners"),
        ("total_corners", "Total Corners"),
    ]

    for target_col, target_name in targets:
        y = valid_df[target_col].values

        # Linear Regression (OLS)
        ols_model = LinearRegression()
        ols_model.fit(X, y)
        y_pred_ols = ols_model.predict(X)

        # R-squared
        ss_res = np.sum((y - y_pred_ols) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_ols = 1 - (ss_res / ss_tot)

        # RMSE
        rmse_ols = np.sqrt(np.mean((y - y_pred_ols) ** 2))

        # Standard errors (approximate)
        n = len(y)
        p = 2  # number of predictors
        mse = ss_res / (n - p)
        X_with_intercept = np.column_stack([np.ones(n), X])
        var_covar = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se = np.sqrt(np.diag(var_covar))

        results[target_name] = {
            "model_type": "OLS Linear Regression",
            "target": target_col,
            "n_samples": n,
            "intercept": round(ols_model.intercept_, 4),
            "intercept_se": round(se[0], 4),
            "coef_lambda_home": round(ols_model.coef_[0], 4),
            "coef_lambda_home_se": round(se[1], 4),
            "coef_lambda_away": round(ols_model.coef_[1], 4),
            "coef_lambda_away_se": round(se[2], 4),
            "r_squared": round(r2_ols, 4),
            "rmse": round(rmse_ols, 4),
            "formula": f"{target_col} = {ols_model.intercept_:.3f} + {ols_model.coef_[0]:.3f} * lambda_home + {ols_model.coef_[1]:.3f} * lambda_away",
        }

        # Ridge Regression (regularized)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_scaled, y)
        y_pred_ridge = ridge_model.predict(X_scaled)

        ss_res_ridge = np.sum((y - y_pred_ridge) ** 2)
        r2_ridge = 1 - (ss_res_ridge / ss_tot)

        results[f"{target_name} (Ridge)"] = {
            "model_type": "Ridge Regression (scaled)",
            "target": target_col,
            "n_samples": n,
            "coef_lambda_home": round(ridge_model.coef_[0], 4),
            "coef_lambda_away": round(ridge_model.coef_[1], 4),
            "r_squared": round(r2_ridge, 4),
        }

    return results


def generate_summary_report(
    df: pd.DataFrame,
    correlations: dict[str, dict[str, float]],
    regressions: dict[str, Any],
) -> str:
    """Generate a human-readable summary report."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("LAMBDA CORRELATION ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Data summary
    lines.append("DATA SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total fixtures analyzed: {len(df)}")
    lines.append(f"Fixtures with lambda values: {df['lambda_home'].notna().sum()}")
    lines.append(f"Fixtures with match stats: {df['h_corners'].notna().sum()}")
    lines.append(f"")
    lines.append(f"Lambda Home - Mean: {df['lambda_home'].mean():.3f}, Std: {df['lambda_home'].std():.3f}")
    lines.append(f"Lambda Away - Mean: {df['lambda_away'].mean():.3f}, Std: {df['lambda_away'].std():.3f}")
    lines.append("")

    # Correlations
    lines.append("CORRELATIONS (Pearson r)")
    lines.append("-" * 40)
    lines.append(f"{'Statistic':<35} {'r':>8} {'p-value':>12} {'Sig.':>6} {'n':>8}")
    lines.append("-" * 70)

    # Sort by absolute correlation
    sorted_corr = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]["correlation"]),
        reverse=True
    )

    for label, data in sorted_corr:
        sig = "***" if data["p_value"] < 0.001 else "**" if data["p_value"] < 0.01 else "*" if data["p_value"] < 0.05 else ""
        lines.append(f"{label:<35} {data['correlation']:>8.4f} {data['p_value']:>12.6f} {sig:>6} {data['n_samples']:>8}")

    lines.append("")
    lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    lines.append("")

    # Regression models
    lines.append("REGRESSION MODELS: corners = f(lambda_home, lambda_away)")
    lines.append("-" * 40)

    for target_name, data in regressions.items():
        if "error" in data:
            lines.append(f"{target_name}: {data['error']}")
            continue

        lines.append(f"")
        lines.append(f"### {target_name} ###")
        lines.append(f"  Model: {data['model_type']}")
        lines.append(f"  N samples: {data['n_samples']}")
        lines.append(f"  R-squared: {data['r_squared']:.4f}")

        if "formula" in data:
            lines.append(f"  Formula: {data['formula']}")

        if "intercept" in data:
            lines.append(f"  Intercept: {data['intercept']:.4f} (SE: {data['intercept_se']:.4f})")
            lines.append(f"  Coef lambda_home: {data['coef_lambda_home']:.4f} (SE: {data['coef_lambda_home_se']:.4f})")
            lines.append(f"  Coef lambda_away: {data['coef_lambda_away']:.4f} (SE: {data['coef_lambda_away_se']:.4f})")
        else:
            lines.append(f"  Coef lambda_home (scaled): {data['coef_lambda_home']:.4f}")
            lines.append(f"  Coef lambda_away (scaled): {data['coef_lambda_away']:.4f}")

        if "rmse" in data:
            lines.append(f"  RMSE: {data['rmse']:.4f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("Key Insights:")
    lines.append("-" * 40)

    # Auto-generate insights
    insights: list[str] = []

    # Find strongest correlations
    strong_home = [k for k, v in correlations.items()
                   if "Home" in k and v["significant_005"] and abs(v["correlation"]) >= 0.3]
    strong_away = [k for k, v in correlations.items()
                   if "Away" in k and v["significant_005"] and abs(v["correlation"]) >= 0.3]

    if strong_home:
        insights.append(f"- Lambda_home shows significant correlation with home team statistics")
    if strong_away:
        insights.append(f"- Lambda_away shows significant correlation with away team statistics")

    # Corner regression insights
    if "Total Corners" in regressions:
        tc_data = regressions["Total Corners"]
        if tc_data.get("r_squared", 0) > 0.1:
            insights.append(
                f"- Corners are moderately predictable from lambda values (R²={tc_data['r_squared']:.2f})"
            )
        home_coef = tc_data.get("coef_lambda_home", 0)
        away_coef = tc_data.get("coef_lambda_away", 0)
        if home_coef > away_coef:
            insights.append("- Lambda_home has stronger effect on corners than lambda_away")
        else:
            insights.append("- Lambda_away has stronger effect on corners than lambda_home")

    for insight in insights:
        lines.append(insight)

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> None:
    """Main entry point for lambda correlation analysis."""
    print("Loading lambda values and match statistics from database...")
    df = fetch_lambda_and_match_stats()

    if df.empty:
        print("ERROR: No data found. Ensure predictions have been generated and match stats ingested.")
        return

    print(f"Loaded {len(df)} fixtures with lambda values")

    # Compute derived statistics
    print("Computing derived statistics...")
    df = compute_derived_stats(df)

    # Compute correlations
    print("Computing correlations...")
    correlations = compute_correlations(df)

    # Fit regression models
    print("Fitting regression models...")
    regressions = fit_corner_regression(df)

    # Generate and print report
    print("\n")
    report = generate_summary_report(df, correlations, regressions)
    print(report)

    # Optionally save to file
    output_dir = ROOT_DIR / "artifacts" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "lambda_correlation_report.txt"
    with open(output_file, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
