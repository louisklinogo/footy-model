from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.layer2_situational.train_situational_residual import (  # noqa: E402
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


FEATURE_HYPOTHESIS: dict[str, str] = {
    "home_rolling_xg": "Home attacking form should explain upward residual pressure for home goals.",
    "home_rolling_xg_against": "Home defensive weakness should lift away residuals.",
    "away_rolling_xg": "Away attacking form should lift away scoring residuals.",
    "away_rolling_xg_against": "Away defensive weakness should lift home residuals.",
    "xg_diff": "Net chance-quality edge should explain directional residual correction.",
    "home_rolling_corners": "Corner pressure proxies sustained home attacking territory.",
    "away_rolling_corners": "Away corner pressure proxies away attacking territory.",
    "rest_delta": "Rest advantage should shift residuals toward better-prepared side.",
    "congestion_flag": "Fixture congestion should capture fatigue underperformance.",
    "home_upcoming_tier": "Upcoming high-tier match may depress current fixture intensity.",
    "away_upcoming_tier": "Upcoming high-tier match may depress away-side intensity.",
    "position_gap": "Standings gap may capture structural team-quality mismatch.",
    "points_gap": "Points gap may capture short-run quality differential.",
    "home_lame_duck": "Low-motivation home contexts may distort baseline expectations.",
    "away_lame_duck": "Low-motivation away contexts may distort baseline expectations.",
    "is_derby": "Derbies may increase variance and weaken baseline calibration.",
    "home_form_streak": "Short-run form momentum can shift residual outcomes.",
    "away_form_streak": "Short-run form momentum can shift residual outcomes.",
    "home_xg_lost": "Missing home player quality should reduce home residual scoring.",
    "away_xg_lost": "Missing away player quality should reduce away residual scoring.",
    "home_key_absent": "Key home absences should suppress home residuals.",
    "away_key_absent": "Key away absences should suppress away residuals.",
    "injury_impact": "Net injury burden should shift residual edge.",
    "home_xg_over_scored": "Home finishing overperformance mean-reverts in residuals.",
    "home_xg_over_conceded": "Home defensive over-concession mean-reverts in residuals.",
    "away_xg_over_scored": "Away finishing overperformance mean-reverts in residuals.",
    "away_xg_over_conceded": "Away defensive over-concession mean-reverts in residuals.",
    "home_playing_top4": "Facing elite opposition may induce baseline miss.",
    "away_playing_top4": "Facing elite opposition may induce baseline miss.",
    "derby_position_gap": "Derby context may interact with ranking gap nonlinearly.",
    "odds_model_gap_home": "Market-vs-model discrepancy should correct home-side miss.",
    "odds_model_gap_draw": "Market draw pricing discrepancy may signal balance miss.",
    "odds_model_gap_away": "Market-vs-model discrepancy should correct away-side miss.",
    "odds_opening_gap_home": "Opening market signal may capture earlier information set.",
    "odds_opening_gap_draw": "Opening draw discrepancy may capture regime miss.",
    "odds_opening_gap_away": "Opening away discrepancy may capture regime miss.",
}


FEATURE_SOURCE: dict[str, str] = {
    **{f: "team_premium_snapshots" for f in [
        "home_rolling_xg",
        "home_rolling_xg_against",
        "away_rolling_xg",
        "away_rolling_xg_against",
        "xg_diff",
        "home_rolling_corners",
        "away_rolling_corners",
        "rest_delta",
        "congestion_flag",
        "home_upcoming_tier",
        "away_upcoming_tier",
        "home_form_streak",
        "away_form_streak",
        "home_xg_over_scored",
        "home_xg_over_conceded",
        "away_xg_over_scored",
        "away_xg_over_conceded",
    ]},
    **{f: "fixtures+results+standings" for f in [
        "position_gap",
        "points_gap",
        "home_lame_duck",
        "away_lame_duck",
        "home_playing_top4",
        "away_playing_top4",
    ]},
    **{f: "team_rivalries" for f in ["is_derby", "derby_position_gap"]},
    **{f: "player_availability+fixture_player_stats" for f in [
        "home_xg_lost",
        "away_xg_lost",
        "home_key_absent",
        "away_key_absent",
        "injury_impact",
    ]},
    **{f: "fixture_odds_markets+predictions(lambda_xgb)" for f in ODDS_FEATURE_COLS},
}


SOURCE_RISK = {
    "team_premium_snapshots": "low",
    "fixtures+results+standings": "low",
    "team_rivalries": "critical",
    "player_availability+fixture_player_stats": "critical",
    "fixture_odds_markets+predictions(lambda_xgb)": "critical",
}


RISK_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass(frozen=True)
class FeatureLedgerRow:
    feature: str
    source: str
    source_risk: str
    hypothesis: str
    train_coverage_pct: float
    test_coverage_pct: float
    corr_home_residual: float
    corr_away_residual: float
    single_lift_home: float
    single_lift_away: float
    drop_delta_home_rmse: float
    drop_delta_away_rmse: float
    provisional_decision: str
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-by-feature Layer 2 audit ledger")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation"),
        help="Output directory for feature ledger outputs",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=4,
        help="Minimum played-games filter for home and away",
    )
    parser.add_argument(
        "--source-audit-json",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation/layer2_data_sources_audit.json"),
        help="Optional source-audit JSON used to override hardcoded source risk levels",
    )
    return parser.parse_args()


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 20:
        return 0.0
    val = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    if pd.isna(val):
        return 0.0
    return float(val)


def _fit_rmse(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[float, float]:
    x_train = train_df[features].fillna(0.0).to_numpy(dtype=float)
    x_test = test_df[features].fillna(0.0).to_numpy(dtype=float)
    y_h_train = train_df["home_residual"].to_numpy(dtype=float)
    y_a_train = train_df["away_residual"].to_numpy(dtype=float)
    y_h_test = test_df["home_residual"].to_numpy(dtype=float)
    y_a_test = test_df["away_residual"].to_numpy(dtype=float)

    model_h = GradientBoostingRegressor(**GBM_PARAMS)
    model_h.fit(x_train, y_h_train)
    pred_h = model_h.predict(x_test)
    rmse_h = float(mean_squared_error(y_h_test, pred_h) ** 0.5)

    model_a = GradientBoostingRegressor(**GBM_PARAMS)
    model_a.fit(x_train, y_a_train)
    pred_a = model_a.predict(x_test)
    rmse_a = float(mean_squared_error(y_a_test, pred_a) ** 0.5)

    return rmse_h, rmse_a


def _decision(
    *,
    source_risk: str,
    train_cov: float,
    single_home: float,
    single_away: float,
    drop_home: float,
    drop_away: float,
) -> tuple[str, str]:
    if train_cov < 0.10:
        return (
            "defer_sparse",
            "Coverage below 10% in training slice; not decision-grade for production weighting.",
        )

    if source_risk == "critical":
        return (
            "fix_lineage_before_judgment",
            "Upstream source timing/semantics are critical-risk; defer final keep/drop until lineage repair.",
        )

    if drop_home > 0 and drop_away > 0:
        return (
            "keep_candidate",
            "Removing this feature worsens both home and away RMSE from full-model baseline.",
        )
    if drop_home < 0 and drop_away < 0 and single_home <= 0 and single_away <= 0:
        return (
            "drop_candidate",
            "Feature hurts both sides in drop-column test and has no standalone lift signal.",
        )
    return (
        "conditional_candidate",
        "Mixed contribution across sides; keep only if aligns with stable segment evidence.",
    )


def _load_source_risk_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    out: dict[str, str] = {}
    for row in rows:
        source_name = row.get("source_name")
        risk_level = row.get("risk_level")
        if isinstance(source_name, str) and isinstance(risk_level, str):
            out[source_name] = risk_level
    return out


def _max_risk(risks: list[str], fallback: str) -> str:
    valid = [r for r in risks if r in RISK_ORDER]
    if not valid:
        return fallback
    return max(valid, key=lambda x: RISK_ORDER[x])


def _resolve_source_risk(source: str, source_risk_overrides: dict[str, str]) -> str:
    if source in source_risk_overrides:
        return source_risk_overrides[source]

    if source == "fixture_odds_markets+predictions(lambda_xgb)":
        return _max_risk(
            [
                source_risk_overrides.get("fixture_odds_markets(sofascore_1x2)", ""),
                source_risk_overrides.get("predictions(lambda_xgb)", ""),
            ],
            SOURCE_RISK.get(source, "high"),
        )

    if source == "player_availability+fixture_player_stats":
        return _max_risk(
            [
                source_risk_overrides.get("player_availability", ""),
                source_risk_overrides.get("fixture_player_stats", ""),
            ],
            SOURCE_RISK.get(source, "high"),
        )

    if source == "team_rivalries":
        return source_risk_overrides.get("team_rivalries", SOURCE_RISK.get(source, "high"))

    return SOURCE_RISK.get(source, "high")


def main() -> None:
    args = parse_args()
    generated_at = datetime.now(UTC).isoformat()
    source_risk_overrides = _load_source_risk_overrides(args.source_audit_json)

    df = load_feature_data()
    df = add_odds_model_gap(df)
    df = df.sort_values("match_datetime_utc").reset_index(drop=True)

    split_time = df["match_datetime_utc"].dropna().quantile(0.8)
    train_df = df[df["match_datetime_utc"] <= split_time].copy()
    test_df = df[df["match_datetime_utc"] > split_time].copy()

    train_df = train_df[
        (train_df["home_played"] >= args.min_games)
        & (train_df["away_played"] >= args.min_games)
        & train_df["lambda_home"].notna()
        & train_df["lambda_away"].notna()
    ].copy()
    test_df = test_df[
        (test_df["home_played"] >= args.min_games)
        & (test_df["away_played"] >= args.min_games)
        & test_df["lambda_home"].notna()
        & test_df["lambda_away"].notna()
    ].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError("Insufficient train/test rows after standard Layer 2 filters.")

    train_df["home_residual"] = train_df["home_goals"] - train_df["lambda_home"]
    train_df["away_residual"] = train_df["away_goals"] - train_df["lambda_away"]
    test_df["home_residual"] = test_df["home_goals"] - test_df["lambda_home"]
    test_df["away_residual"] = test_df["away_goals"] - test_df["lambda_away"]

    all_features = FEATURE_COLS + ODDS_FEATURE_COLS
    baseline_home_rmse = float(
        mean_squared_error(
            test_df["home_residual"].to_numpy(dtype=float),
            np.zeros(len(test_df), dtype=float),
        )
        ** 0.5
    )
    baseline_away_rmse = float(
        mean_squared_error(
            test_df["away_residual"].to_numpy(dtype=float),
            np.zeros(len(test_df), dtype=float),
        )
        ** 0.5
    )

    full_home_rmse, full_away_rmse = _fit_rmse(train_df, test_df, all_features)

    rows: list[FeatureLedgerRow] = []
    for feat in all_features:
        train_cov = float(train_df[feat].notna().mean())
        test_cov = float(test_df[feat].notna().mean())
        corr_h = _safe_corr(train_df[feat], train_df["home_residual"])
        corr_a = _safe_corr(train_df[feat], train_df["away_residual"])

        single_home_rmse, single_away_rmse = _fit_rmse(train_df, test_df, [feat])
        single_lift_home = (
            (baseline_home_rmse - single_home_rmse) / baseline_home_rmse
            if baseline_home_rmse > 0
            else 0.0
        )
        single_lift_away = (
            (baseline_away_rmse - single_away_rmse) / baseline_away_rmse
            if baseline_away_rmse > 0
            else 0.0
        )

        drop_feats = [f for f in all_features if f != feat]
        drop_home_rmse, drop_away_rmse = _fit_rmse(train_df, test_df, drop_feats)
        drop_delta_home = drop_home_rmse - full_home_rmse
        drop_delta_away = drop_away_rmse - full_away_rmse

        source = FEATURE_SOURCE.get(feat, "unknown")
        source_risk = _resolve_source_risk(source, source_risk_overrides)
        decision, rationale = _decision(
            source_risk=source_risk,
            train_cov=train_cov,
            single_home=single_lift_home,
            single_away=single_lift_away,
            drop_home=drop_delta_home,
            drop_away=drop_delta_away,
        )

        rows.append(
            FeatureLedgerRow(
                feature=feat,
                source=source,
                source_risk=source_risk,
                hypothesis=FEATURE_HYPOTHESIS.get(feat, "Hypothesis not documented yet."),
                train_coverage_pct=train_cov * 100.0,
                test_coverage_pct=test_cov * 100.0,
                corr_home_residual=corr_h,
                corr_away_residual=corr_a,
                single_lift_home=float(single_lift_home),
                single_lift_away=float(single_lift_away),
                drop_delta_home_rmse=float(drop_delta_home),
                drop_delta_away_rmse=float(drop_delta_away),
                provisional_decision=decision,
                rationale=rationale,
            )
        )

    payload = {
        "generated_at": generated_at,
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "split_time": str(split_time),
        "baseline_home_rmse": baseline_home_rmse,
        "baseline_away_rmse": baseline_away_rmse,
        "full_home_rmse": full_home_rmse,
        "full_away_rmse": full_away_rmse,
        "rows": [asdict(r) for r in rows],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "layer2_feature_ledger.json"
    md_path = args.output_dir / "layer2_feature_ledger.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Layer 2 Feature Ledger")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append(f"- train_n: {len(train_df)}")
    lines.append(f"- test_n: {len(test_df)}")
    lines.append(f"- split_time: {split_time}")
    lines.append(
        f"- baseline_rmse home/away: {baseline_home_rmse:.4f} / {baseline_away_rmse:.4f}"
    )
    lines.append(f"- full_model_rmse home/away: {full_home_rmse:.4f} / {full_away_rmse:.4f}")
    lines.append("")
    lines.append("| Feature | Source | Train Cov % | Test Cov % | Single Lift H | Single Lift A | Drop dRMSE H | Drop dRMSE A | Decision |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in rows:
        lines.append(
            f"| {r.feature} | {r.source} | {r.train_coverage_pct:.1f} | {r.test_coverage_pct:.1f} | "
            f"{r.single_lift_home:.4f} | {r.single_lift_away:.4f} | "
            f"{r.drop_delta_home_rmse:.4f} | {r.drop_delta_away_rmse:.4f} | {r.provisional_decision} |"
        )
    lines.append("")
    lines.append("## Decision Key")
    lines.append("- `fix_lineage_before_judgment`: source timing semantics are critical-risk; decision deferred.")
    lines.append("- `keep_candidate`: drop-column worsens both sides.")
    lines.append("- `drop_candidate`: hurts both sides and no standalone lift.")
    lines.append("- `conditional_candidate`: mixed behavior.")
    lines.append("- `defer_sparse`: insufficient feature coverage.")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
