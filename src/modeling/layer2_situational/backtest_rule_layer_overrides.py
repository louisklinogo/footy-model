from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.layer2_situational.deployment_policy import (
    load_layer2_deployment_policy,
    resolve_league_policy,
)
from src.modeling.layer2_situational.rule_layer import (
    apply_rule_adjustment,
    load_rule_layer_config,
)
from src.modeling.layer2_situational.train_situational_residual import (
    add_odds_model_gap,
    load_feature_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest deterministic rule-layer overrides on Layer 2 holdout."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model_artifacts/situational_model"),
        help="Layer 2 model artifact directory.",
    )
    parser.add_argument(
        "--rule-layer-config",
        type=Path,
        default=None,
        help="Optional rule-layer config JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation"),
        help="Output directory for backtest artifacts.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="layer2_rule_layer_segmented_backtest",
        help="Output file stem.",
    )
    return parser.parse_args()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _lift_vs_baseline(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    baseline = _rmse(y_true, np.zeros_like(y_true))
    if baseline <= 0:
        return 0.0
    return float((baseline - _rmse(y_true, y_pred)) / baseline)


def _directional_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size == 0:
        return None
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return float(np.mean(true_sign == pred_sign))


def _apply_filters(frame: pd.DataFrame, min_games: int = 4) -> pd.DataFrame:
    out = frame[(frame["home_played"] >= min_games) & (frame["away_played"] >= min_games)]
    out = out[out["lambda_home"].notna() & out["lambda_away"].notna()]
    return out.copy()


def _market_scaled_alpha(base_alpha: float, odds_gap: float | None, residual_raw: float) -> float:
    effective = float(base_alpha)
    if effective <= 0.0:
        return 0.0
    if odds_gap is None or np.isnan(odds_gap) or float(odds_gap) == 0.0:
        return effective
    agrees = (residual_raw >= 0.0) == (float(odds_gap) >= 0.0)
    return effective if agrees else effective * 0.4


def _prepare_holdout_frame() -> pd.DataFrame:
    df = load_feature_data()
    df = add_odds_model_gap(df)
    df = df.sort_values("match_datetime_utc").reset_index(drop=True)
    split_time = df["match_datetime_utc"].dropna().quantile(0.8)
    test_df = df[df["match_datetime_utc"] > split_time].copy()
    test_df = _apply_filters(test_df)
    test_df["home_residual"] = test_df["home_goals"] - test_df["lambda_home"]
    test_df["away_residual"] = test_df["away_goals"] - test_df["lambda_away"]
    return test_df.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    model_blob = joblib.load(args.model_dir / "situational_model.pkl")
    if not isinstance(model_blob, dict):
        raise RuntimeError("situational_model.pkl has unexpected shape.")

    features = list(model_blob["features"])
    home_model = model_blob["home_model"]
    away_model = model_blob["away_model"]
    policy = load_layer2_deployment_policy(args.model_dir)
    rule_cfg_path = args.rule_layer_config or (args.model_dir / "rule_layer_config.json")
    rule_cfg = load_rule_layer_config(rule_cfg_path)

    test_df = _prepare_holdout_frame()
    if test_df.empty:
        raise RuntimeError("No holdout rows available after filters.")

    x_test = test_df[features].fillna(0).to_numpy()
    pred_home_raw = home_model.predict(x_test)
    pred_away_raw = away_model.predict(x_test)

    y_home_true = test_df["home_residual"].to_numpy(dtype=float)
    y_away_true = test_df["away_residual"].to_numpy(dtype=float)
    y_home_layer2 = np.zeros(len(test_df), dtype=float)
    y_away_layer2 = np.zeros(len(test_df), dtype=float)
    y_home_layer2_rule = np.zeros(len(test_df), dtype=float)
    y_away_layer2_rule = np.zeros(len(test_df), dtype=float)

    home_rule_applied = np.zeros(len(test_df), dtype=bool)
    away_rule_applied = np.zeros(len(test_df), dtype=bool)
    rule_counter_home: Counter[str] = Counter()
    rule_counter_away: Counter[str] = Counter()

    for i, row in enumerate(test_df.itertuples(index=False)):
        fixture_id = int(getattr(row, "fixture_id"))
        _ = fixture_id
        league_policy = resolve_league_policy(policy, getattr(row, "league_code"))
        layer2_enabled = bool(league_policy.get("enabled", False))
        alpha_base = float(league_policy.get("alpha", 0.0)) if layer2_enabled else 0.0

        h_raw = float(pred_home_raw[i])
        a_raw = float(pred_away_raw[i])
        lh = float(getattr(row, "lambda_home"))
        la = float(getattr(row, "lambda_away"))

        alpha_home = _market_scaled_alpha(
            alpha_base,
            getattr(row, "odds_model_gap_home", np.nan),
            h_raw,
        )
        alpha_away = _market_scaled_alpha(
            alpha_base,
            getattr(row, "odds_model_gap_away", np.nan),
            a_raw,
        )
        h_applied = alpha_home * h_raw
        a_applied = alpha_away * a_raw

        lambda_home_layer2 = max(0.01, lh + h_applied)
        lambda_away_layer2 = max(0.01, la + a_applied)

        y_home_layer2[i] = lambda_home_layer2 - lh
        y_away_layer2[i] = lambda_away_layer2 - la

        lambda_home_final = lambda_home_layer2
        lambda_away_final = lambda_away_layer2
        is_low_conf = bool(getattr(row, "home_played", 0) < 4 or getattr(row, "away_played", 0) < 4)
        if layer2_enabled and not is_low_conf:
            home_rule = apply_rule_adjustment(
                lambda_home_layer2, row=row, side="home", config=rule_cfg
            )
            away_rule = apply_rule_adjustment(
                lambda_away_layer2, row=row, side="away", config=rule_cfg
            )
            lambda_home_final = float(home_rule["lambda_after"])
            lambda_away_final = float(away_rule["lambda_after"])
            if bool(home_rule["applied"]):
                home_rule_applied[i] = True
                rule_counter_home.update(list(home_rule["rules"]))
            if bool(away_rule["applied"]):
                away_rule_applied[i] = True
                rule_counter_away.update(list(away_rule["rules"]))

        y_home_layer2_rule[i] = lambda_home_final - lh
        y_away_layer2_rule[i] = lambda_away_final - la

    fixture_trigger_mask = home_rule_applied | away_rule_applied
    home_trigger_idx = np.where(home_rule_applied)[0]
    away_trigger_idx = np.where(away_rule_applied)[0]

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(args.model_dir.resolve()),
        "rule_layer_config_path": str(rule_cfg_path),
        "n_test_fixtures": int(len(test_df)),
        "n_triggered_fixtures": int(fixture_trigger_mask.sum()),
        "n_triggered_home_events": int(home_rule_applied.sum()),
        "n_triggered_away_events": int(away_rule_applied.sum()),
        "rule_counts_home": dict(sorted(rule_counter_home.items())),
        "rule_counts_away": dict(sorted(rule_counter_away.items())),
    }

    global_metrics = {
        "layer1_only": {
            "home_rmse": _rmse(y_home_true, np.zeros_like(y_home_true)),
            "away_rmse": _rmse(y_away_true, np.zeros_like(y_away_true)),
        },
        "layer1_plus_layer2": {
            "home_rmse": _rmse(y_home_true, y_home_layer2),
            "away_rmse": _rmse(y_away_true, y_away_layer2),
            "home_lift": _lift_vs_baseline(y_home_true, y_home_layer2),
            "away_lift": _lift_vs_baseline(y_away_true, y_away_layer2),
        },
        "layer1_plus_layer2_plus_rules": {
            "home_rmse": _rmse(y_home_true, y_home_layer2_rule),
            "away_rmse": _rmse(y_away_true, y_away_layer2_rule),
            "home_lift": _lift_vs_baseline(y_home_true, y_home_layer2_rule),
            "away_lift": _lift_vs_baseline(y_away_true, y_away_layer2_rule),
        },
    }

    triggered_metrics: dict[str, Any] = {
        "fixtures_triggered": int(fixture_trigger_mask.sum()),
        "home_events": int(home_rule_applied.sum()),
        "away_events": int(away_rule_applied.sum()),
    }
    if fixture_trigger_mask.any():
        y_h_t = y_home_true[fixture_trigger_mask]
        y_a_t = y_away_true[fixture_trigger_mask]
        y_h_l2 = y_home_layer2[fixture_trigger_mask]
        y_a_l2 = y_away_layer2[fixture_trigger_mask]
        y_h_rl = y_home_layer2_rule[fixture_trigger_mask]
        y_a_rl = y_away_layer2_rule[fixture_trigger_mask]
        triggered_metrics.update(
            {
                "layer1_only": {
                    "home_rmse": _rmse(y_h_t, np.zeros_like(y_h_t)),
                    "away_rmse": _rmse(y_a_t, np.zeros_like(y_a_t)),
                },
                "layer1_plus_layer2": {
                    "home_rmse": _rmse(y_h_t, y_h_l2),
                    "away_rmse": _rmse(y_a_t, y_a_l2),
                    "home_lift": _lift_vs_baseline(y_h_t, y_h_l2),
                    "away_lift": _lift_vs_baseline(y_a_t, y_a_l2),
                },
                "layer1_plus_layer2_plus_rules": {
                    "home_rmse": _rmse(y_h_t, y_h_rl),
                    "away_rmse": _rmse(y_a_t, y_a_rl),
                    "home_lift": _lift_vs_baseline(y_h_t, y_h_rl),
                    "away_lift": _lift_vs_baseline(y_a_t, y_a_rl),
                },
            }
        )
    else:
        triggered_metrics["note"] = "No triggered fixtures; triggered-segment metrics unavailable."

    directional = {
        "home_trigger_events": {
            "n": int(home_trigger_idx.size),
            "layer1_plus_layer2_hit_rate": _directional_hit_rate(
                y_home_true[home_trigger_idx], y_home_layer2[home_trigger_idx]
            )
            if home_trigger_idx.size > 0
            else None,
            "layer1_plus_layer2_plus_rules_hit_rate": _directional_hit_rate(
                y_home_true[home_trigger_idx], y_home_layer2_rule[home_trigger_idx]
            )
            if home_trigger_idx.size > 0
            else None,
        },
        "away_trigger_events": {
            "n": int(away_trigger_idx.size),
            "layer1_plus_layer2_hit_rate": _directional_hit_rate(
                y_away_true[away_trigger_idx], y_away_layer2[away_trigger_idx]
            )
            if away_trigger_idx.size > 0
            else None,
            "layer1_plus_layer2_plus_rules_hit_rate": _directional_hit_rate(
                y_away_true[away_trigger_idx], y_away_layer2_rule[away_trigger_idx]
            )
            if away_trigger_idx.size > 0
            else None,
        },
    }

    payload = {
        "summary": summary,
        "global_metrics": global_metrics,
        "triggered_metrics": triggered_metrics,
        "directional_metrics": directional,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    md_path = args.output_dir / f"{args.output_stem}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Layer 2 Rule-Layer Segmented Backtest")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- test fixtures: {summary['n_test_fixtures']}")
    lines.append(f"- triggered fixtures: {summary['n_triggered_fixtures']}")
    lines.append(f"- home trigger events: {summary['n_triggered_home_events']}")
    lines.append(f"- away trigger events: {summary['n_triggered_away_events']}")
    lines.append("")
    lines.append("## Global RMSE")
    lines.append("| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| layer1_only | {global_metrics['layer1_only']['home_rmse']:.4f} | {global_metrics['layer1_only']['away_rmse']:.4f} | - | - |"
    )
    lines.append(
        f"| layer1_plus_layer2 | {global_metrics['layer1_plus_layer2']['home_rmse']:.4f} | {global_metrics['layer1_plus_layer2']['away_rmse']:.4f} | {global_metrics['layer1_plus_layer2']['home_lift']:.2%} | {global_metrics['layer1_plus_layer2']['away_lift']:.2%} |"
    )
    lines.append(
        f"| layer1_plus_layer2_plus_rules | {global_metrics['layer1_plus_layer2_plus_rules']['home_rmse']:.4f} | {global_metrics['layer1_plus_layer2_plus_rules']['away_rmse']:.4f} | {global_metrics['layer1_plus_layer2_plus_rules']['home_lift']:.2%} | {global_metrics['layer1_plus_layer2_plus_rules']['away_lift']:.2%} |"
    )
    lines.append("")
    lines.append("## Triggered Segment")
    if fixture_trigger_mask.any():
        lines.append("| Variant | Home RMSE | Away RMSE | Home Lift | Away Lift |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        lines.append(
            f"| layer1_only | {triggered_metrics['layer1_only']['home_rmse']:.4f} | {triggered_metrics['layer1_only']['away_rmse']:.4f} | - | - |"
        )
        lines.append(
            f"| layer1_plus_layer2 | {triggered_metrics['layer1_plus_layer2']['home_rmse']:.4f} | {triggered_metrics['layer1_plus_layer2']['away_rmse']:.4f} | {triggered_metrics['layer1_plus_layer2']['home_lift']:.2%} | {triggered_metrics['layer1_plus_layer2']['away_lift']:.2%} |"
        )
        lines.append(
            f"| layer1_plus_layer2_plus_rules | {triggered_metrics['layer1_plus_layer2_plus_rules']['home_rmse']:.4f} | {triggered_metrics['layer1_plus_layer2_plus_rules']['away_rmse']:.4f} | {triggered_metrics['layer1_plus_layer2_plus_rules']['home_lift']:.2%} | {triggered_metrics['layer1_plus_layer2_plus_rules']['away_lift']:.2%} |"
        )
    else:
        lines.append("- No triggered fixtures.")
    lines.append("")
    lines.append("## Directional Hit Rate (Triggered Side Events)")
    lines.append(
        f"- home events: n={directional['home_trigger_events']['n']}, "
        f"layer2={directional['home_trigger_events']['layer1_plus_layer2_hit_rate']}, "
        f"layer2+rules={directional['home_trigger_events']['layer1_plus_layer2_plus_rules_hit_rate']}"
    )
    lines.append(
        f"- away events: n={directional['away_trigger_events']['n']}, "
        f"layer2={directional['away_trigger_events']['layer1_plus_layer2_hit_rate']}, "
        f"layer2+rules={directional['away_trigger_events']['layer1_plus_layer2_plus_rules_hit_rate']}"
    )
    lines.append("")
    lines.append("## Rule Counts (Home)")
    for k, v in sorted(rule_counter_home.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Rule Counts (Away)")
    for k, v in sorted(rule_counter_away.items()):
        lines.append(f"- {k}: {v}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
