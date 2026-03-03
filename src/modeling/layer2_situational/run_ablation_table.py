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

    # --- New tactical feature groups ---
    tactical_feats = [
        "home_rolling_possession",
        "away_rolling_possession",
        "possession_delta",
    ]
    h1h2_feats = [
        "home_rolling_xg_p1",
        "away_rolling_xg_p1",
        "xg_p1_delta",
        "home_rolling_xg_h2_delta",
        "away_rolling_xg_h2_delta",
        "h2_surge_delta",
        "home_rolling_sot_p1",
        "away_rolling_sot_p1",
        "home_rolling_sot_h2_delta",
        "away_rolling_sot_h2_delta",
    ]
    formation_feats = [
        "home_defenders",
        "home_midfielders",
        "home_forwards",
        "away_defenders",
        "away_midfielders",
        "away_forwards",
        "style_delta",
    ]
    defensive_feats = [
        "home_rolling_tackles_pct",
        "away_rolling_tackles_pct",
        "home_rolling_errors_lead_to_shot",
        "away_rolling_errors_lead_to_shot",
    ]
    
    style_cluster_feats = [c for c in df.columns if (
        c.startswith("home_style_cluster_") or 
        c.startswith("away_style_cluster_") or 
        c.startswith("style_matchup_")
    ) and not c.endswith("_raw") and not c.endswith("_DROP")]

    existing_full = standings_feats + schedule_feats + player_feats + ODDS_FEATURE_COLS
    all_candidates = existing_full + tactical_feats + h1h2_feats + formation_feats + defensive_feats + style_cluster_feats

    # Filter to only columns actually in the dataframe
    def _available(feats):
        return sorted(list(set([f for f in feats if f in train_df.columns])))

    # ================================================================
    # PASS 1: Additive ablation
    # ================================================================
    print("\n=== PASS 1: Additive Ablation ===")
    additive_stages = [
        ("baseline", []),
        ("existing_full", _available(existing_full)),
        ("existing_full+tactical", _available(existing_full + tactical_feats)),
        ("existing_full+tactical+h1h2", _available(existing_full + tactical_feats + h1h2_feats)),
        ("existing_full+tactical+h1h2+formations", _available(existing_full + tactical_feats + h1h2_feats + formation_feats)),
        ("existing_full+tactical+h1h2+formations+defensive", _available(existing_full + tactical_feats + h1h2_feats + formation_feats + defensive_feats)),
        ("all_candidates (with style)", _available(all_candidates)),
    ]

    additive_results = []
    for name, feats in additive_stages:
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
        additive_results.append({
            "pass": "additive",
            "stage": name,
            "n_features": len(feats),
            "home_rmse": home_rmse,
            "away_rmse": away_rmse,
            "home_lift": home_lift,
            "away_lift": away_lift,
        })
        print(f"  {name}: home_rmse={home_rmse:.4f} away_rmse={away_rmse:.4f} | lift h={home_lift:.2%} a={away_lift:.2%}")

    # ================================================================
    # PASS 2: Drop-one ablation (from all_candidates)
    # ================================================================
    print("\n=== PASS 2: Drop-one Ablation ===")
    full_avail = _available(all_candidates)
    full_home, full_away = _train_and_score(train_df, test_df, full_avail)
    if full_home is None:
        full_home, full_away = baseline_home, baseline_away

    drop_one_groups = [
        ("drop_tactical", tactical_feats),
        ("drop_h1h2", h1h2_feats),
        ("drop_formations", formation_feats),
        ("drop_defensive", defensive_feats),
        ("drop_style_clusters", style_cluster_feats),
    ]
    drop_one_results = [{"pass": "drop_one", "stage": "all_candidates", "n_features": len(full_avail),
                         "home_rmse": full_home, "away_rmse": full_away,
                         "home_lift": (baseline_home - full_home) / baseline_home if baseline_home else 0.0,
                         "away_lift": (baseline_away - full_away) / baseline_away if baseline_away else 0.0,
                         "verdict": "anchor"}]

    for drop_name, drop_feats in drop_one_groups:
        remaining = [f for f in full_avail if f not in set(drop_feats)]
        h_rmse, a_rmse = _train_and_score(train_df, test_df, remaining)
        if h_rmse is None:
            h_rmse, a_rmse = baseline_home, baseline_away
        # If dropping hurts (RMSE goes up), the group is valuable
        h_lift = (baseline_home - h_rmse) / baseline_home if baseline_home else 0.0
        a_lift = (baseline_away - a_rmse) / baseline_away if baseline_away else 0.0
        hurts_when_dropped = (h_rmse > full_home + 1e-6) or (a_rmse > full_away + 1e-6)
        verdict = "VALUABLE" if hurts_when_dropped else "REDUNDANT"
        drop_one_results.append({
            "pass": "drop_one",
            "stage": drop_name,
            "n_features": len(remaining),
            "home_rmse": h_rmse,
            "away_rmse": a_rmse,
            "home_lift": h_lift,
            "away_lift": a_lift,
            "verdict": verdict,
        })
        print(f"  {drop_name}: home_rmse={h_rmse:.4f} away_rmse={a_rmse:.4f} | {verdict}")

    # ================================================================
    # PASS 3: H1/H2 coverage sub-test (premium-only rows)
    # ================================================================
    print("\n=== PASS 3: H1/H2 Coverage Sub-test (premium-only rows) ===")
    h1h2_coverage_results = []
    h1h2_avail = _available(h1h2_feats)
    if h1h2_avail:
        # Premium-only: rows where at least one H1/H2 feature is non-null/non-zero
        premium_mask = train_df[h1h2_avail].notna().any(axis=1)
        premium_mask_test = test_df[h1h2_avail].notna().any(axis=1)
        prem_train = train_df[premium_mask].copy()
        prem_test = test_df[premium_mask_test].copy()
        print(f"  Premium coverage: train={len(prem_train)} ({premium_mask.mean():.1%}), test={len(prem_test)} ({premium_mask_test.mean():.1%})")
        if len(prem_train) >= 200 and len(prem_test) >= 50:
            base_feats_avail = _available(existing_full)
            h_base, a_base = _train_and_score(prem_train, prem_test, base_feats_avail)
            h_plus, a_plus = _train_and_score(prem_train, prem_test, base_feats_avail + h1h2_avail)
            if h_base and h_plus:
                h1h2_coverage_results = [
                    {"pass": "h1h2_coverage", "stage": "base_premium_only",
                     "n": len(prem_test), "home_rmse": h_base, "away_rmse": a_base},
                    {"pass": "h1h2_coverage", "stage": "base+h1h2_premium_only",
                     "n": len(prem_test), "home_rmse": h_plus, "away_rmse": a_plus,
                     "verdict": "SIGNAL" if h_plus < h_base - 1e-6 else "NOISE"},
                ]
                print(f"  base={h_base:.4f}/{a_base:.4f}  +h1h2={h_plus:.4f}/{a_plus:.4f}  verdict={h1h2_coverage_results[-1]['verdict']}")
        else:
            print("  Insufficient premium rows for sub-test. Skipping.")
    else:
        print("  H1/H2 features not in dataframe. Skipping.")

    # ================================================================
    # Determine which groups pass gate (for promotion)
    # ================================================================
    # A group passes if:
    # (a) additive: adding it does not worsen RMSE vs previous stage
    # (b) drop-one: dropping it is VALUABLE (hurts RMSE)
    # Both criteria must pass.
    additive_stage_map = {r["stage"]: r for r in additive_results}
    drop_one_verdicts = {r["stage"]: r["verdict"] for r in drop_one_results if "drop_" in r["stage"]}

    def _additive_verdict(group_name, after_stage, before_stage):
        after = additive_stage_map.get(after_stage)
        before = additive_stage_map.get(before_stage)
        if not after or not before:
            return "UNKNOWN"
        comb_after = (after["home_rmse"] + after["away_rmse"]) / 2
        comb_before = (before["home_rmse"] + before["away_rmse"]) / 2
        return "PASS" if comb_after <= comb_before + 1e-6 else "FAIL"

    promotions = {
        "tactical": {
            "additive": _additive_verdict("tactical", "existing_full+tactical", "existing_full"),
            "drop_one": drop_one_verdicts.get("drop_tactical", "UNKNOWN"),
        },
        "h1h2": {
            "additive": _additive_verdict("h1h2", "existing_full+tactical+h1h2", "existing_full+tactical"),
            "drop_one": drop_one_verdicts.get("drop_h1h2", "UNKNOWN"),
            "coverage_sub_test": h1h2_coverage_results[-1]["verdict"] if h1h2_coverage_results else "NOT_RUN",
        },
        "formations": {
            "additive": _additive_verdict("formations", "existing_full+tactical+h1h2+formations", "existing_full+tactical+h1h2"),
            "drop_one": drop_one_verdicts.get("drop_formations", "UNKNOWN"),
        },
        "defensive": {
            "additive": _additive_verdict("defensive", "existing_full+tactical+h1h2+formations+defensive", "existing_full+tactical+h1h2+formations"),
            "drop_one": drop_one_verdicts.get("drop_defensive", "UNKNOWN"),
        },
        "style_clusters": {
            "additive": _additive_verdict("style_clusters", "all_candidates (with style)", "existing_full+tactical+h1h2+formations+defensive"),
            "drop_one": drop_one_verdicts.get("drop_style_clusters", "UNKNOWN"),
        },
    }
    for group, verdict in promotions.items():
        if group == "h1h2":
            overall = "PROMOTE" if verdict["additive"] == "PASS" and verdict["drop_one"] == "VALUABLE" and verdict.get("coverage", "WEAK") == "SIGNAL" else "REJECT"
        else:
            overall = "PROMOTE" if verdict["additive"] == "PASS" and verdict["drop_one"] == "VALUABLE" else "REJECT"
        verdict["overall"] = overall
        print(f"  {group}: additive={verdict['additive']} drop_one={verdict['drop_one']} -> {overall}")

    all_results = additive_results + drop_one_results + h1h2_coverage_results

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ablation_results.json"
    json_path.write_text(json.dumps({"results": all_results, "promotions": promotions}, indent=2), encoding="utf-8")

    md_lines = [
        "# Ablation Results",
        "",
        "## Pass 1: Additive",
        "| Stage | Features | Home RMSE | Away RMSE | Home Lift | Away Lift |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in additive_results:
        md_lines.append(
            f"| {row['stage']} | {row['n_features']} | {row['home_rmse']:.4f} | {row['away_rmse']:.4f} | {row['home_lift']:.2%} | {row['away_lift']:.2%} |"
        )
    md_lines += [
        "",
        "## Pass 2: Drop-one",
        "| Stage | Features | Home RMSE | Away RMSE | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in drop_one_results:
        md_lines.append(
            f"| {row['stage']} | {row['n_features']} | {row['home_rmse']:.4f} | {row['away_rmse']:.4f} | {row.get('verdict', '-')} |"
        )
    md_lines += [
        "",
        "## Promotion Decisions",
        "| Group | Additive | Drop-one | Coverage | Decision |",
        "| --- | --- | --- | --- | --- |",
    ]
    for group, verdict in promotions.items():
        md_lines.append(
            f"| {group} | {verdict['additive']} | {verdict['drop_one']} | {verdict.get('coverage_sub_test', 'N/A')} | **{verdict['overall']}** |"
        )
    md_path = args.output_dir / "ablation_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
