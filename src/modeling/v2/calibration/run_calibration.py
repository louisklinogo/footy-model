from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.v2.calibration.methods import evaluate_market_calibration

DEFAULT_EVALUATION_REPORT = ROOT_DIR / "model_artifacts" / "v2" / "evaluation" / "evaluation_report.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "evaluation"


def _read_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load evaluation report: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Evaluation report must be a JSON object: {path}")
    return payload


def _load_holdout_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["market", "fixture_id", "y_true", "p_model"])
    frame = pd.read_csv(path)
    for column in ("market", "fixture_id", "y_true", "p_model"):
        if column not in frame.columns:
            raise RuntimeError(f"Missing required holdout prediction column '{column}': {path}")
    return frame


def _promotion_holdout_row(market: str, family: str, market_report: dict[str, Any]) -> dict[str, Any] | None:
    metrics = market_report.get("calibrated_eval_metrics")
    if not isinstance(metrics, dict):
        return None
    return {
        "market": market,
        "family": family,
        **metrics,
        "calibration_method": market_report.get("selected_method"),
        "calibration_status": market_report.get("status"),
        "calibration_fit_rows": int(market_report.get("fit_rows") or 0),
        "calibration_eval_rows": int(market_report.get("eval_rows") or 0),
    }


def build_calibration_report(
    *,
    evaluation_report_path: Path,
    output_dir: Path,
    fit_fraction: float,
    min_fit_rows: int,
    min_eval_rows: int,
    max_auc_drop: float,
) -> dict[str, Any]:
    evaluation_report = _read_report(evaluation_report_path)
    families_payload = evaluation_report.get("families")
    if not isinstance(families_payload, dict):
        raise RuntimeError("Evaluation report missing families payload")

    families_summary: dict[str, Any] = {}
    calibration_by_market: dict[str, Any] = {}
    promotion_holdout_by_market: dict[str, Any] = {}
    for family, payload in families_payload.items():
        if not isinstance(payload, dict):
            continue
        artifact_dir = Path(str(payload.get("artifact_dir") or ""))
        holdout_path = artifact_dir / "holdout_predictions.csv"
        family_frame = _load_holdout_predictions(holdout_path)
        family_markets: dict[str, Any] = {}
        calibrators: dict[str, Any] = {}
        for market, market_frame in family_frame.groupby("market", sort=True):
            report_row, calibrator = evaluate_market_calibration(
                market_frame,
                fit_fraction=fit_fraction,
                min_fit_rows=min_fit_rows,
                min_eval_rows=min_eval_rows,
                max_auc_drop=max_auc_drop,
            )
            report_row.update({"market": str(market), "family": family})
            family_markets[str(market)] = report_row
            calibration_by_market[str(market)] = report_row
            promotion_row = _promotion_holdout_row(str(market), family, report_row)
            if promotion_row is not None:
                promotion_holdout_by_market[str(market)] = promotion_row
            if calibrator is not None:
                calibrators[str(market)] = calibrator

        calibration_report_path = artifact_dir / "calibration_report.json"
        family_report = {
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "family": family,
            "artifact_dir": str(artifact_dir),
            "holdout_predictions_path": str(holdout_path),
            "rules": {
                "fit_fraction": float(fit_fraction),
                "min_fit_rows": int(min_fit_rows),
                "min_eval_rows": int(min_eval_rows),
                "max_auc_drop": float(max_auc_drop),
            },
            "markets": family_markets,
        }
        calibration_report_path.write_text(json.dumps(family_report, indent=2), encoding="utf-8")
        calibrators_path = artifact_dir / "calibrators.joblib"
        if calibrators:
            joblib.dump(calibrators, calibrators_path)
        families_summary[family] = {
            "artifact_dir": str(artifact_dir),
            "holdout_predictions_path": str(holdout_path),
            "calibration_report_path": str(calibration_report_path),
            "calibrators_path": str(calibrators_path) if calibrators else None,
            "attempted_markets": int(len(family_markets)),
            "calibrated_markets": int(sum(1 for row in family_markets.values() if row.get("status") == "calibrated")),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "evaluation_report_path": str(evaluation_report_path),
        "output_dir": str(output_dir),
        "rules": {
            "fit_fraction": float(fit_fraction),
            "min_fit_rows": int(min_fit_rows),
            "min_eval_rows": int(min_eval_rows),
            "max_auc_drop": float(max_auc_drop),
        },
        "families": families_summary,
        "calibration_by_market": calibration_by_market,
        "promotion_holdout_by_market": promotion_holdout_by_market,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2 post-holdout calibration before promotion.")
    parser.add_argument("--evaluation-report", type=Path, default=DEFAULT_EVALUATION_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--min-fit-rows", type=int, default=80)
    parser.add_argument("--min-eval-rows", type=int, default=80)
    parser.add_argument("--max-auc-drop", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_calibration_report(
        evaluation_report_path=args.evaluation_report,
        output_dir=args.output_dir,
        fit_fraction=float(args.fit_fraction),
        min_fit_rows=int(args.min_fit_rows),
        min_eval_rows=int(args.min_eval_rows),
        max_auc_drop=float(args.max_auc_drop),
    )
    out_path = Path(args.output_dir) / "calibration_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved calibration report to {out_path}")
    print(f"Markets calibrated: {len(report['promotion_holdout_by_market'])}")


if __name__ == "__main__":
    main()