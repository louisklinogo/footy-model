from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


def _load_pickle(path: Path) -> dict[str, Any]:
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected model payload type in {path}")
    return obj


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _enabled_leagues(policy: dict[str, Any]) -> list[str]:
    leagues = policy.get("leagues", {})
    if not isinstance(leagues, dict):
        return []
    return sorted(
        str(league)
        for league, cfg in leagues.items()
        if isinstance(cfg, dict) and bool(cfg.get("enabled", False))
    )


def _variant_metrics(model_dir: Path) -> dict[str, Any]:
    model_blob = _load_pickle(model_dir / "situational_model.pkl")
    meta = _load_json(model_dir / "situational_model.meta.json")
    policy = _load_json(model_dir / "layer2_deployment_policy.json")
    enabled = _enabled_leagues(policy)
    cv_summary = meta.get("cv_summary", {}) if isinstance(meta, dict) else {}
    return {
        "model_dir": str(model_dir.resolve()),
        "feature_count": int(len(model_blob.get("features", []))),
        "home_test_rmse": float(model_blob["home_test_rmse"]),
        "away_test_rmse": float(model_blob["away_test_rmse"]),
        "home_baseline_rmse": float(model_blob["home_baseline_rmse"]),
        "away_baseline_rmse": float(model_blob["away_baseline_rmse"]),
        "home_lift": float(
            (model_blob["home_baseline_rmse"] - model_blob["home_test_rmse"])
            / model_blob["home_baseline_rmse"]
        )
        if float(model_blob["home_baseline_rmse"]) > 0
        else 0.0,
        "away_lift": float(
            (model_blob["away_baseline_rmse"] - model_blob["away_test_rmse"])
            / model_blob["away_baseline_rmse"]
        )
        if float(model_blob["away_baseline_rmse"]) > 0
        else 0.0,
        "enabled_leagues": enabled,
        "cv_home_rmse_mean": cv_summary.get("home_rmse_mean"),
        "cv_home_rmse_std": cv_summary.get("home_rmse_std"),
        "cv_away_rmse_mean": cv_summary.get("away_rmse_mean"),
        "cv_away_rmse_std": cv_summary.get("away_rmse_std"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Layer 2 global model artifact directories."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("model_artifacts/situational_model"),
        help="Baseline model artifact directory.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        required=True,
        help="Candidate model artifact directory.",
    )
    parser.add_argument(
        "--label-baseline",
        type=str,
        default="baseline_full_features",
        help="Label for baseline variant.",
    )
    parser.add_argument(
        "--label-candidate",
        type=str,
        default="candidate_pruned_drop9",
        help="Label for candidate variant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation"),
        help="Output directory for comparison artifacts.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="layer2_pruned_drop9_vs_full_comparison",
        help="Output filename stem (without extension).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = _variant_metrics(args.baseline_dir)
    candidate = _variant_metrics(args.candidate_dir)

    delta = {
        "home_test_rmse": float(candidate["home_test_rmse"] - baseline["home_test_rmse"]),
        "away_test_rmse": float(candidate["away_test_rmse"] - baseline["away_test_rmse"]),
        "home_lift": float(candidate["home_lift"] - baseline["home_lift"]),
        "away_lift": float(candidate["away_lift"] - baseline["away_lift"]),
        "enabled_leagues_removed": sorted(
            set(baseline["enabled_leagues"]) - set(candidate["enabled_leagues"])
        ),
        "enabled_leagues_added": sorted(
            set(candidate["enabled_leagues"]) - set(baseline["enabled_leagues"])
        ),
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        args.label_baseline: baseline,
        args.label_candidate: candidate,
        "delta_candidate_minus_baseline": delta,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    md_path = args.output_dir / f"{args.output_stem}.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Layer 2 Global Variant Comparison")
    lines.append("")
    lines.append(f"Generated: {output['generated_at']}")
    lines.append("")
    lines.append(
        "| Variant | Features | Home RMSE | Away RMSE | Home Lift | Away Lift | Enabled Leagues |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    lines.append(
        f"| {args.label_baseline} | {baseline['feature_count']} | {baseline['home_test_rmse']:.4f} | {baseline['away_test_rmse']:.4f} | {baseline['home_lift']:.2%} | {baseline['away_lift']:.2%} | {', '.join(baseline['enabled_leagues']) or '-'} |"
    )
    lines.append(
        f"| {args.label_candidate} | {candidate['feature_count']} | {candidate['home_test_rmse']:.4f} | {candidate['away_test_rmse']:.4f} | {candidate['home_lift']:.2%} | {candidate['away_lift']:.2%} | {', '.join(candidate['enabled_leagues']) or '-'} |"
    )
    lines.append("")
    lines.append("## Delta (candidate - baseline)")
    lines.append(f"- home_test_rmse: {delta['home_test_rmse']:+.6f}")
    lines.append(f"- away_test_rmse: {delta['away_test_rmse']:+.6f}")
    lines.append(f"- home_lift: {delta['home_lift']:+.2%}")
    lines.append(f"- away_lift: {delta['away_lift']:+.2%}")
    lines.append(
        f"- enabled leagues removed: {', '.join(delta['enabled_leagues_removed']) or '-'}"
    )
    lines.append(
        f"- enabled leagues added: {', '.join(delta['enabled_leagues_added']) or '-'}"
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
