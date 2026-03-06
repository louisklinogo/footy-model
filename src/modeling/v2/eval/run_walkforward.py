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

from src.modeling.v2.eval.metrics import aggregate_market_summary, summarize_binary_metric_rows
from src.modeling.v2.io.baseline_registry import DEFAULT_SCOPE_PATH, load_scope_markets


DEFAULT_SCORELINE_DIR = ROOT_DIR / "model_artifacts" / "v2" / "scoreline"
DEFAULT_CORNERS_DIR = ROOT_DIR / "model_artifacts" / "v2" / "corners"
DEFAULT_ANYTIME_DIR = ROOT_DIR / "model_artifacts" / "v2" / "anytime"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "evaluation"


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        return sum(1 for _ in handle)


def _load_family_bundle(name: str, artifact_dir: Path) -> dict[str, Any]:
    holdout_rows = _read_rows(artifact_dir / "metrics_holdout.json")
    walkforward_rows = _read_rows(artifact_dir / "metrics_walkforward_folds.json")
    walkforward_by_league_rows = _read_rows(
        artifact_dir / "metrics_walkforward_folds_by_league.json"
    )
    holdout_by_league_rows = _read_rows(artifact_dir / "metrics_holdout_by_league.json")
    walkforward_summary = summarize_binary_metric_rows(walkforward_rows)
    if not walkforward_summary:
        stored_summary = _read_json(artifact_dir / "metrics_walkforward.json")
        if isinstance(stored_summary, dict):
            walkforward_summary = stored_summary

    holdout_by_market = {
        str(row.get("market")): row
        for row in holdout_rows
        if isinstance(row.get("market"), str)
    }
    holdout_predictions_path = artifact_dir / "holdout_predictions.csv"
    return {
        "family": name,
        "artifact_dir": str(artifact_dir),
        "training_report": _read_json(artifact_dir / "training_report.json"),
        "holdout_rows": holdout_rows,
        "holdout_by_market": holdout_by_market,
        "holdout_by_league_rows": holdout_by_league_rows,
        "walkforward_rows": walkforward_rows,
        "walkforward_summary": walkforward_summary,
        "walkforward_aggregate": aggregate_market_summary(walkforward_summary),
        "walkforward_by_league_rows": walkforward_by_league_rows,
        "holdout_prediction_rows": _count_csv_rows(holdout_predictions_path),
        "holdout_predictions_path": str(holdout_predictions_path) if holdout_predictions_path.exists() else None,
        "scoreline_slice_summary": _read_json(artifact_dir / "scoreline_slice_summary.json") if name == "scoreline" else {},
        "scoreline_slice_summary_by_league": _read_json(artifact_dir / "scoreline_slice_summary_by_league.json") if name == "scoreline" else {},
    }


def build_evaluation_report(
    *,
    scope_path: Path,
    family_dirs: dict[str, Path],
) -> dict[str, Any]:
    scope_markets = load_scope_markets(scope_path)
    families = {
        name: _load_family_bundle(name, artifact_dir)
        for name, artifact_dir in family_dirs.items()
    }

    holdout_by_market: dict[str, dict[str, Any]] = {}
    walkforward_by_market: dict[str, dict[str, Any]] = {}
    combined_walkforward_by_league_rows: list[dict[str, Any]] = []
    combined_holdout_by_league_rows: list[dict[str, Any]] = []
    for family_name, bundle in families.items():
        for market, row in bundle["holdout_by_market"].items():
            holdout_by_market[market] = {"family": family_name, **row}
        for market, row in bundle["walkforward_summary"].items():
            if isinstance(row, dict):
                walkforward_by_market[market] = {"family": family_name, **row}
        combined_walkforward_by_league_rows.extend(bundle["walkforward_by_league_rows"])
        combined_holdout_by_league_rows.extend(bundle["holdout_by_league_rows"])

    family_report: dict[str, dict[str, Any]] = {}
    for name, bundle in families.items():
        entry: dict[str, Any] = {
            "artifact_dir": bundle["artifact_dir"],
            "walkforward_aggregate": bundle["walkforward_aggregate"],
            "holdout_markets": int(len(bundle["holdout_rows"])),
            "walkforward_markets": int(len(bundle["walkforward_summary"])),
            "holdout_prediction_rows": int(bundle.get("holdout_prediction_rows") or 0),
            "training_report": bundle["training_report"],
        }
        if bundle.get("holdout_predictions_path"):
            entry["holdout_predictions_path"] = bundle["holdout_predictions_path"]
        if name == "scoreline":
            entry["scoreline_slice_summary"] = bundle.get("scoreline_slice_summary") or {}
            entry["scoreline_slice_summary_by_league"] = (
                bundle.get("scoreline_slice_summary_by_league") or {}
            )
        family_report[name] = entry

    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "scope_path": str(scope_path),
        "scope_markets": scope_markets,
        "families": family_report,
        "holdout_by_market": holdout_by_market,
        "walkforward_by_market": walkforward_by_market,
        "holdout_by_league": summarize_binary_metric_rows(
            combined_holdout_by_league_rows, group_keys=("market", "league_code")
        ),
        "walkforward_by_league": summarize_binary_metric_rows(
            combined_walkforward_by_league_rows, group_keys=("market", "league_code")
        ),
        "missing_holdout_markets": [
            market for market in scope_markets if market not in holdout_by_market
        ],
        "missing_walkforward_markets": [
            market for market in scope_markets if market not in walkforward_by_market
        ],
        "scoreline_slice_summary": families.get("scoreline", {}).get("scoreline_slice_summary", {}),
        "scoreline_slice_summary_by_league": families.get("scoreline", {}).get(
            "scoreline_slice_summary_by_league", {}
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate v2 walk-forward and holdout artifacts across model families."
    )
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE_PATH)
    parser.add_argument("--scoreline-dir", type=Path, default=DEFAULT_SCORELINE_DIR)
    parser.add_argument("--corners-dir", type=Path, default=DEFAULT_CORNERS_DIR)
    parser.add_argument("--anytime-dir", type=Path, default=DEFAULT_ANYTIME_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_evaluation_report(
        scope_path=args.scope,
        family_dirs={
            "scoreline": args.scoreline_dir,
            "corners": args.corners_dir,
            "anytime": args.anytime_dir,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "walkforward_summary.json").write_text(
        json.dumps(report["walkforward_by_market"], indent=2), encoding="utf-8"
    )
    (args.output_dir / "walkforward_summary_by_league.json").write_text(
        json.dumps(report["walkforward_by_league"], indent=2), encoding="utf-8"
    )
    (args.output_dir / "holdout_summary.json").write_text(
        json.dumps(report["holdout_by_market"], indent=2), encoding="utf-8"
    )
    print(f"Saved evaluation report to {args.output_dir}")
    print(f"Missing holdout markets: {len(report['missing_holdout_markets'])}")
    print(f"Missing walk-forward markets: {len(report['missing_walkforward_markets'])}")


if __name__ == "__main__":
    main()
