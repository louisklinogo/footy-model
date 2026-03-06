from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.v2.io.baseline_registry import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_SCOPE_PATH,
    BaselineMetric,
    load_baseline_metrics,
    load_scope_markets,
)


DEFAULT_EVALUATION_REPORT = ROOT_DIR / "model_artifacts" / "v2" / "evaluation" / "evaluation_report.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "model_artifacts" / "v2" / "evaluation" / "promotion_registry.json"


def _read_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load evaluation report: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Evaluation report must be a JSON object: {path}")
    return payload


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def evaluate_market(
    *,
    market: str,
    holdout_row: dict[str, Any] | None,
    walkforward_row: dict[str, Any] | None,
    baseline: BaselineMetric | None,
    min_support: int,
    min_folds: int,
    max_ece: float | None,
    auc_tolerance: float,
    brier_tolerance: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if baseline is None:
        reasons.append("missing_baseline")
    if holdout_row is None:
        reasons.append("missing_holdout")
    if walkforward_row is None:
        reasons.append("missing_walkforward")

    holdout_auc = _to_float((holdout_row or {}).get("auc"))
    holdout_brier = _to_float((holdout_row or {}).get("brier"))
    holdout_ece = _to_float((holdout_row or {}).get("ece"))
    walk_n_total = int((walkforward_row or {}).get("n_total") or 0)
    walk_folds = int((walkforward_row or {}).get("folds_used") or 0)

    if baseline is not None and holdout_auc is not None and baseline.auc is not None:
        if holdout_auc + float(auc_tolerance) < float(baseline.auc):
            reasons.append("auc_failed")
    if baseline is not None and holdout_brier is not None and baseline.brier is not None:
        if holdout_brier > float(baseline.brier) + float(brier_tolerance):
            reasons.append("brier_failed")
    if max_ece is not None and holdout_ece is not None and holdout_ece > float(max_ece):
        reasons.append("ece_failed")
    if walk_n_total < int(min_support):
        reasons.append("support_failed")
    if walk_folds < int(min_folds):
        reasons.append("folds_failed")

    passed = not reasons
    return {
        "market": market,
        "status": "passed" if passed else "failed",
        "reasons": reasons,
        "family": (holdout_row or walkforward_row or {}).get("family"),
        "baseline": None
        if baseline is None
        else {
            "auc": baseline.auc,
            "brier": baseline.brier,
            "pr_auc": baseline.pr_auc,
            "accuracy": baseline.accuracy,
            "log_loss": baseline.log_loss,
            "ece": baseline.ece,
            "n": baseline.n,
        },
        "holdout": holdout_row,
        "walkforward": walkforward_row,
    }


def build_promotion_registry(
    *,
    scope_path: Path,
    baseline_path: Path,
    evaluation_report_path: Path,
    min_support: int,
    min_folds: int,
    max_ece: float | None,
    auc_tolerance: float,
    brier_tolerance: float,
) -> dict[str, Any]:
    scope_markets = load_scope_markets(scope_path)
    baselines = load_baseline_metrics(baseline_path)
    report = _read_report(evaluation_report_path)
    holdout_by_market = report.get("holdout_by_market") if isinstance(report.get("holdout_by_market"), dict) else {}
    walkforward_by_market = report.get("walkforward_by_market") if isinstance(report.get("walkforward_by_market"), dict) else {}

    rows = [
        evaluate_market(
            market=market,
            holdout_row=holdout_by_market.get(market),
            walkforward_row=walkforward_by_market.get(market),
            baseline=baselines.get(market),
            min_support=min_support,
            min_folds=min_folds,
            max_ece=max_ece,
            auc_tolerance=auc_tolerance,
            brier_tolerance=brier_tolerance,
        )
        for market in scope_markets
    ]
    passed = [row["market"] for row in rows if row["status"] == "passed"]
    failed = [row["market"] for row in rows if row["status"] != "passed"]
    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "scope_path": str(scope_path),
        "baseline_path": str(baseline_path),
        "evaluation_report_path": str(evaluation_report_path),
        "rules": {
            "min_support": int(min_support),
            "min_folds": int(min_folds),
            "max_ece": max_ece,
            "auc_tolerance": float(auc_tolerance),
            "brier_tolerance": float(brier_tolerance),
        },
        "summary": {
            "markets_total": int(len(rows)),
            "markets_passed": int(len(passed)),
            "markets_failed": int(len(failed)),
            "passed_markets": passed,
            "failed_markets": failed,
        },
        "markets": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v2 candidate promotion readiness against baseline registry and walk-forward evidence."
    )
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE_PATH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--evaluation-report", type=Path, default=DEFAULT_EVALUATION_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--min-support", type=int, default=200)
    parser.add_argument("--min-folds", type=int, default=3)
    parser.add_argument("--max-ece", type=float, default=0.05)
    parser.add_argument("--auc-tolerance", type=float, default=0.0)
    parser.add_argument("--brier-tolerance", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = build_promotion_registry(
        scope_path=args.scope,
        baseline_path=args.baseline,
        evaluation_report_path=args.evaluation_report,
        min_support=int(args.min_support),
        min_folds=int(args.min_folds),
        max_ece=float(args.max_ece) if args.max_ece is not None else None,
        auc_tolerance=float(args.auc_tolerance),
        brier_tolerance=float(args.brier_tolerance),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Saved promotion registry to {args.output}")
    print(f"Markets passed: {registry['summary']['markets_passed']}")
    print(f"Markets failed: {registry['summary']['markets_failed']}")


if __name__ == "__main__":
    main()
