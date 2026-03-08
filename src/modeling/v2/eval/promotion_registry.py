from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

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
from src.modeling.v2.io.promotion_policy import load_promotion_policy


DEFAULT_EVALUATION_REPORT = ROOT_DIR / "model_artifacts" / "v2" / "evaluation" / "evaluation_report.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "model_artifacts" / "v2" / "evaluation" / "promotion_registry.json"
METRIC_COMPARISON_EPSILON = 1e-12


def _read_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load evaluation report: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Evaluation report must be a JSON object: {path}")
    return payload


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _comparable_calibrated_holdout(
    holdout_row: dict[str, Any] | None,
    calibrated_holdout_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if calibrated_holdout_row is None:
        return None
    raw_n = _to_float((holdout_row or {}).get("n"))
    calibrated_n = _to_float((calibrated_holdout_row or {}).get("n"))
    if raw_n is None or calibrated_n is None:
        return None
    if int(raw_n) != int(calibrated_n):
        return None
    return calibrated_holdout_row


def _normalize_market_list(markets: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in markets or []:
        market = str(raw).strip()
        if not market or market in seen:
            continue
        out.append(market)
        seen.add(market)
    return out


def _meaningfully_less_than(left: float, right: float, *, eps: float = METRIC_COMPARISON_EPSILON) -> bool:
    return (float(right) - float(left)) > float(eps)


def _meaningfully_greater_than(left: float, right: float, *, eps: float = METRIC_COMPARISON_EPSILON) -> bool:
    return (float(left) - float(right)) > float(eps)


def evaluate_market(
    *,
    market: str,
    holdout_row: dict[str, Any] | None,
    calibrated_holdout_row: dict[str, Any] | None,
    walkforward_row: dict[str, Any] | None,
    baseline: BaselineMetric | None,
    min_support: int,
    min_folds: int,
    max_ece: float | None,
    auc_tolerance: float,
    brier_tolerance: float,
    log_loss_tolerance: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if baseline is None:
        reasons.append("missing_baseline")
    if holdout_row is None:
        reasons.append("missing_holdout")
    if walkforward_row is None:
        reasons.append("missing_walkforward")

    effective_calibrated_holdout = _comparable_calibrated_holdout(holdout_row, calibrated_holdout_row)
    effective_holdout = effective_calibrated_holdout if effective_calibrated_holdout is not None else holdout_row
    holdout_auc = _to_float((effective_holdout or {}).get("auc"))
    holdout_brier = _to_float((effective_holdout or {}).get("brier"))
    holdout_log_loss = _to_float((effective_holdout or {}).get("log_loss"))
    holdout_ece = _to_float((effective_holdout or {}).get("ece"))
    walk_n_total = int((walkforward_row or {}).get("n_total") or 0)
    walk_folds = int((walkforward_row or {}).get("folds_used") or 0)

    if baseline is not None and holdout_auc is not None and baseline.auc is not None:
        if _meaningfully_less_than(holdout_auc + float(auc_tolerance), float(baseline.auc)):
            reasons.append("auc_failed")
    if baseline is not None and holdout_brier is not None and baseline.brier is not None:
        if _meaningfully_greater_than(holdout_brier, float(baseline.brier) + float(brier_tolerance)):
            reasons.append("brier_failed")
    if baseline is not None and holdout_log_loss is not None and baseline.log_loss is not None:
        if _meaningfully_greater_than(
            holdout_log_loss,
            float(baseline.log_loss) + float(log_loss_tolerance),
        ):
            reasons.append("log_loss_failed")
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
        "holdout_source": "calibrated_holdout" if effective_calibrated_holdout is not None else "raw_holdout",
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
        "effective_holdout": effective_holdout,
        "calibration": calibrated_holdout_row,
        "walkforward": walkforward_row,
    }


def build_promotion_registry(
    *,
    scope_path: Path,
    baseline_path: Path,
    evaluation_report_path: Path,
    calibration_report_path: Path | None,
    required_markets: list[str] | None,
    min_support: int,
    min_folds: int,
    max_ece: float | None,
    auc_tolerance: float,
    brier_tolerance: float,
    log_loss_tolerance: float,
) -> dict[str, Any]:
    scope_markets = load_scope_markets(scope_path)
    required_market_list = _normalize_market_list(required_markets)
    baselines = load_baseline_metrics(baseline_path)
    report = _read_report(evaluation_report_path)
    holdout_by_market = report.get("holdout_by_market") if isinstance(report.get("holdout_by_market"), dict) else {}
    walkforward_by_market = report.get("walkforward_by_market") if isinstance(report.get("walkforward_by_market"), dict) else {}
    calibration_report: dict[str, Any] = {}
    if calibration_report_path is not None and calibration_report_path.exists():
        calibration_report = _read_report(calibration_report_path)
    calibrated_holdout_by_market = (
        calibration_report.get("promotion_holdout_by_market")
        if isinstance(calibration_report.get("promotion_holdout_by_market"), dict)
        else {}
    )

    rows = [
        evaluate_market(
            market=market,
            holdout_row=holdout_by_market.get(market),
            calibrated_holdout_row=calibrated_holdout_by_market.get(market),
            walkforward_row=walkforward_by_market.get(market),
            baseline=baselines.get(market),
            min_support=min_support,
            min_folds=min_folds,
            max_ece=max_ece,
            auc_tolerance=auc_tolerance,
            brier_tolerance=brier_tolerance,
            log_loss_tolerance=log_loss_tolerance,
        )
        for market in scope_markets
    ]
    row_by_market = {row["market"]: row for row in rows}
    passed = [row["market"] for row in rows if row["status"] == "passed"]
    failed = [row["market"] for row in rows if row["status"] != "passed"]
    required_missing = [market for market in required_market_list if market not in row_by_market]
    required_passed = [market for market in required_market_list if row_by_market.get(market, {}).get("status") == "passed"]
    required_failed = [market for market in required_market_list if market in row_by_market and row_by_market[market]["status"] != "passed"]
    decision_basis = "required_markets" if required_market_list else "all_scope_markets"
    decision_failed = required_failed if required_market_list else failed
    decision_passed = required_passed if required_market_list else passed
    decision_ok = not decision_failed and not required_missing
    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "scope_path": str(scope_path),
        "baseline_path": str(baseline_path),
        "evaluation_report_path": str(evaluation_report_path),
        "calibration_report_path": str(calibration_report_path) if calibration_report_path is not None else None,
        "rules": {
            "min_support": int(min_support),
            "min_folds": int(min_folds),
            "max_ece": max_ece,
            "auc_tolerance": float(auc_tolerance),
            "brier_tolerance": float(brier_tolerance),
            "log_loss_tolerance": float(log_loss_tolerance),
            "required_markets": required_market_list,
        },
        "summary": {
            "markets_total": int(len(rows)),
            "markets_passed": int(len(passed)),
            "markets_failed": int(len(failed)),
            "passed_markets": passed,
            "failed_markets": failed,
            "required_markets_total": int(len(required_market_list)),
            "required_markets_passed": int(len(required_passed)),
            "required_markets_failed": int(len(required_failed)),
            "required_markets_missing": required_missing,
            "required_market_list": required_market_list,
        },
        "decision": {
            "status": "passed" if decision_ok else "failed",
            "basis": decision_basis,
            "passed_markets": decision_passed,
            "failed_markets": decision_failed,
            "missing_required_markets": required_missing,
        },
        "markets": rows,
    }


def resolve_required_markets(
    *,
    required_markets: list[str] | None,
    promotion_policy_path: Path | None,
) -> tuple[list[str], dict[str, str | None]]:
    merged = list(required_markets or [])
    policy_name: str | None = None
    policy_path_str: str | None = None
    if promotion_policy_path is not None:
        policy = load_promotion_policy(promotion_policy_path)
        merged.extend(policy.required_markets)
        policy_name = policy.name
        policy_path_str = str(promotion_policy_path)
    return _normalize_market_list(merged), {
        "promotion_policy_path": policy_path_str,
        "promotion_policy_name": policy_name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v2 candidate promotion readiness against baseline registry and walk-forward evidence."
    )
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE_PATH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--evaluation-report", type=Path, default=DEFAULT_EVALUATION_REPORT)
    parser.add_argument("--calibration-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--min-support", type=int, default=200)
    parser.add_argument("--min-folds", type=int, default=3)
    parser.add_argument("--max-ece", type=float, default=0.05)
    parser.add_argument("--auc-tolerance", type=float, default=0.0)
    parser.add_argument("--brier-tolerance", type=float, default=0.0)
    parser.add_argument("--log-loss-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--promotion-policy",
        type=Path,
        default=None,
        help="Optional YAML policy file whose required_markets become the top-level promotion gate.",
    )
    parser.add_argument(
        "--required-market",
        action="append",
        default=[],
        help="Optional market code that must pass for the top-level promotion decision. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_markets, required_market_metadata = resolve_required_markets(
        required_markets=list(args.required_market or []),
        promotion_policy_path=args.promotion_policy,
    )
    registry = build_promotion_registry(
        scope_path=args.scope,
        baseline_path=args.baseline,
        evaluation_report_path=args.evaluation_report,
        calibration_report_path=args.calibration_report,
        required_markets=required_markets,
        min_support=int(args.min_support),
        min_folds=int(args.min_folds),
        max_ece=float(args.max_ece) if args.max_ece is not None else None,
        auc_tolerance=float(args.auc_tolerance),
        brier_tolerance=float(args.brier_tolerance),
        log_loss_tolerance=float(args.log_loss_tolerance),
    )
    registry.setdefault("rules", {}).update(required_market_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Saved promotion registry to {args.output}")
    print(f"Markets passed: {registry['summary']['markets_passed']}")
    print(f"Markets failed: {registry['summary']['markets_failed']}")


if __name__ == "__main__":
    main()
