"""
Layer 2 feature health audit.

Builds the same training dataframe as train_situational_residual.py and reports:
- Missingness / coverage
- Zero-variance features
- Binary flag prevalence
- Odds gap distribution stats
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.layer2_situational.train_situational_residual import (
    FEATURE_COLS,
    ODDS_FEATURE_COLS,
    add_odds_model_gap,
    load_feature_data,
)


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_feature_health(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    stats_rows: list[dict] = []
    zero_variance: list[str] = []
    binary_flags: list[dict] = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        series = df[col]
        missing_pct = float(series.isna().mean())
        non_missing = series.dropna()

        row = {
            "feature": col,
            "missing_pct": missing_pct,
            "non_missing_pct": 1.0 - missing_pct,
            "count": int(non_missing.shape[0]),
            "mean": _safe_float(non_missing.mean()) if not non_missing.empty else None,
            "std": _safe_float(non_missing.std()) if not non_missing.empty else None,
            "min": _safe_float(non_missing.min()) if not non_missing.empty else None,
            "p25": _safe_float(non_missing.quantile(0.25)) if not non_missing.empty else None,
            "p50": _safe_float(non_missing.quantile(0.50)) if not non_missing.empty else None,
            "p75": _safe_float(non_missing.quantile(0.75)) if not non_missing.empty else None,
            "max": _safe_float(non_missing.max()) if not non_missing.empty else None,
        }
        stats_rows.append(row)

        if not non_missing.empty and non_missing.nunique(dropna=True) <= 1:
            zero_variance.append(col)

        if not non_missing.empty:
            unique_vals = set(non_missing.unique().tolist())
            if unique_vals.issubset({0, 1}):
                prevalence = float(non_missing.mean())
                binary_flags.append(
                    {
                        "feature": col,
                        "prevalence": prevalence,
                        "count": int(non_missing.shape[0]),
                    }
                )

    odds_gap_stats: dict[str, dict] = {}
    for col in ODDS_FEATURE_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        desc = series.describe(percentiles=[0.25, 0.5, 0.75])
        odds_gap_stats[col] = {k: _safe_float(v) for k, v in desc.to_dict().items()}

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "row_count": int(len(df)),
        "feature_stats": stats_rows,
        "zero_variance_features": sorted(zero_variance),
        "binary_flags": binary_flags,
        "odds_gap_stats": odds_gap_stats,
    }


def feature_stats_dataframe(health_report: dict) -> pd.DataFrame:
    return pd.DataFrame(health_report.get("feature_stats", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Layer 2 feature health")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_feature_health"),
        help="Output directory for audit artifacts",
    )
    args = parser.parse_args()

    print("Building Layer 2 feature dataframe...")
    df = load_feature_data()
    df = add_odds_model_gap(df)

    all_features = FEATURE_COLS + ODDS_FEATURE_COLS
    report = compute_feature_health(df, all_features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "feature_health.json"
    csv_path = args.output_dir / "feature_health.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    feature_stats_dataframe(report).to_csv(csv_path, index=False)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
