from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.v2.eval.promotion_recommendation import write_promotion_recommendation_artifacts
from src.modeling.v2.eval.promotion_registry import resolve_required_markets
from src.modeling.v2.eval.shadow_compare import write_shadow_comparison_artifacts
from src.modeling.v2.io.baseline_registry import (
    load_baseline_metrics,
    load_scope_markets,
    validate_baseline_coverage,
    validate_market_presence,
    validate_markets_in_scope,
)

DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_SCORELINE_DIR = ROOT_DIR / "model_artifacts" / "v2" / "scoreline"
DEFAULT_CORNERS_DIR = ROOT_DIR / "model_artifacts" / "v2" / "corners"
DEFAULT_ANYTIME_DIR = ROOT_DIR / "model_artifacts" / "v2" / "anytime"
DEFAULT_EVAL_DIR = ROOT_DIR / "model_artifacts" / "v2" / "evaluation"
DEFAULT_BASELINE = ROOT_DIR / "model_artifacts" / "v2" / "baselines" / "metrics_baseline_v2.json"


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_promotion_summary(evaluation_dir: Path) -> dict[str, Any] | None:
    registry_path = Path(evaluation_dir) / "promotion_registry.json"
    registry = _read_json_object(registry_path)
    if registry is None:
        return None
    rules = registry.get("rules") if isinstance(registry.get("rules"), dict) else {}
    summary = registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    decision = registry.get("decision") if isinstance(registry.get("decision"), dict) else {}
    return {
        "promotion_registry_path": str(registry_path),
        "decision": decision,
        "summary": {
            "markets_total": summary.get("markets_total"),
            "markets_passed": summary.get("markets_passed"),
            "markets_failed": summary.get("markets_failed"),
            "required_markets_total": summary.get("required_markets_total"),
            "required_markets_passed": summary.get("required_markets_passed"),
            "required_markets_failed": summary.get("required_markets_failed"),
            "required_markets_missing": summary.get("required_markets_missing"),
        },
        "rules": {
            "promotion_policy_name": rules.get("promotion_policy_name"),
            "promotion_policy_path": rules.get("promotion_policy_path"),
            "required_markets": rules.get("required_markets"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current v2 training/evaluation flow.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--scoreline-dir", type=Path, default=DEFAULT_SCORELINE_DIR)
    parser.add_argument("--corners-dir", type=Path, default=DEFAULT_CORNERS_DIR)
    parser.add_argument("--anytime-dir", type=Path, default=DEFAULT_ANYTIME_DIR)
    parser.add_argument("--scoreline-dataset-path", type=Path, default=None)
    parser.add_argument("--corners-dataset-path", type=Path, default=None)
    parser.add_argument("--anytime-dataset-path", type=Path, default=None)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--rebuild-baseline",
        action="store_true",
        help="Opt in to rebuilding the baseline file from current holdout artifacts. By default the supplied baseline is treated as a frozen champion comparator.",
    )
    parser.add_argument("--skip-baseline-rebuild", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-promotion", action="store_true")
    parser.add_argument("--allow-missing-holdout", action="store_true")
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
        help="Optional market code that must pass in the promotion registry decision. May be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_run_plan(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    python_bin = str(args.python_bin)
    plan: list[tuple[str, list[str]]] = []
    rebuild_baseline = bool(getattr(args, "rebuild_baseline", False)) and not bool(
        getattr(args, "skip_baseline_rebuild", False)
    )
    trainer_specs = [
        (
            "train.scoreline",
            ROOT_DIR / "src" / "modeling" / "v2" / "families" / "scoreline" / "train_scoreline.py",
            Path(args.scoreline_dir),
            getattr(args, "scoreline_dataset_path", None),
        ),
        (
            "train.corners",
            ROOT_DIR / "src" / "modeling" / "v2" / "families" / "corners" / "train_corners.py",
            Path(args.corners_dir),
            getattr(args, "corners_dataset_path", None),
        ),
        (
            "train.anytime",
            ROOT_DIR / "src" / "modeling" / "v2" / "families" / "anytime" / "train_anytime.py",
            Path(args.anytime_dir),
            getattr(args, "anytime_dataset_path", None),
        ),
    ]
    if not bool(args.skip_train):
        for name, script_path, output_dir, dataset_path in trainer_specs:
            cmd = [python_bin, str(script_path), "--scope", str(args.scope), "--output-dir", str(output_dir)]
            if dataset_path is not None:
                cmd.extend(["--dataset-path", str(dataset_path)])
            if args.max_rows is not None and int(args.max_rows) > 0:
                cmd.extend(["--max-rows", str(int(args.max_rows))])
            plan.append((name, cmd))
        scoreline_residual_cmd = [
            python_bin,
            str(
                ROOT_DIR
                / "src"
                / "modeling"
                / "v2"
                / "families"
                / "scoreline"
                / "train_scoreline_residuals.py"
            ),
            "--scope",
            str(args.scope),
            "--artifact-dir",
            str(args.scoreline_dir),
        ]
        if getattr(args, "scoreline_dataset_path", None) is not None:
            scoreline_residual_cmd.extend(["--dataset-path", str(args.scoreline_dataset_path)])
        if args.max_rows is not None and int(args.max_rows) > 0:
            scoreline_residual_cmd.extend(["--max-rows", str(int(args.max_rows))])
        plan.append(("train.scoreline_residuals", scoreline_residual_cmd))

    plan.append(
        (
            "eval.walkforward",
            [
                python_bin,
                str(ROOT_DIR / "src" / "modeling" / "v2" / "eval" / "run_walkforward.py"),
                "--scope",
                str(args.scope),
                "--scoreline-dir",
                str(args.scoreline_dir),
                "--corners-dir",
                str(args.corners_dir),
                "--anytime-dir",
                str(args.anytime_dir),
                "--output-dir",
                str(args.evaluation_dir),
            ],
        )
    )
    plan.append(
        (
            "eval.calibration",
            [
                python_bin,
                str(ROOT_DIR / "src" / "modeling" / "v2" / "calibration" / "run_calibration.py"),
                "--evaluation-report",
                str(Path(args.evaluation_dir) / "evaluation_report.json"),
                "--output-dir",
                str(args.evaluation_dir),
            ],
        )
    )

    if rebuild_baseline:
        rebuild_cmd = [
            python_bin,
            str(ROOT_DIR / "src" / "modeling" / "v2" / "io" / "rebuild_baseline_metrics.py"),
            "--scope",
            str(args.scope),
            "--baseline",
            str(args.baseline_path),
            "--scoreline",
            str(Path(args.scoreline_dir) / "metrics_holdout.json"),
            "--corners",
            str(Path(args.corners_dir) / "metrics_holdout.json"),
            "--anytime",
            str(Path(args.anytime_dir) / "metrics_holdout.json"),
        ]
        if bool(args.allow_missing_holdout):
            rebuild_cmd.append("--allow-missing")
        plan.append(("baseline.rebuild", rebuild_cmd))

    if not bool(args.skip_promotion):
        promotion_cmd = [
            python_bin,
            str(ROOT_DIR / "src" / "modeling" / "v2" / "eval" / "promotion_registry.py"),
            "--scope",
            str(args.scope),
            "--baseline",
            str(args.baseline_path),
            "--evaluation-report",
            str(Path(args.evaluation_dir) / "evaluation_report.json"),
            "--calibration-report",
            str(Path(args.evaluation_dir) / "calibration_report.json"),
            "--output",
            str(Path(args.evaluation_dir) / "promotion_registry.json"),
            "--min-support",
            str(int(args.min_support)),
            "--min-folds",
            str(int(args.min_folds)),
            "--max-ece",
            str(float(args.max_ece)),
            "--auc-tolerance",
            str(float(args.auc_tolerance)),
            "--brier-tolerance",
            str(float(args.brier_tolerance)),
            "--log-loss-tolerance",
            str(float(args.log_loss_tolerance)),
        ]
        if getattr(args, "promotion_policy", None) is not None:
            promotion_cmd.extend(["--promotion-policy", str(args.promotion_policy)])
        for market in list(getattr(args, "required_market", []) or []):
            promotion_cmd.extend(["--required-market", str(market)])
        plan.append(("eval.promotion_registry", promotion_cmd))
    return plan


def _run_step(name: str, command: list[str], *, dry_run: bool) -> dict[str, Any]:
    print(f"COMMAND[{name}]: {' '.join(command)}")
    if dry_run:
        return {"name": name, "command": command, "returncode": 0, "stdout": "", "stderr": "", "dry_run": True}
    completed = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True)
    result = {
        "name": name,
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "dry_run": False,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {name}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return result


def validate_promotion_inputs(
    args: argparse.Namespace,
    *,
    validate_baseline: bool,
) -> dict[str, Any]:
    scope_path = Path(args.scope)
    baseline_path = Path(args.baseline_path)
    scope_markets = load_scope_markets(scope_path)
    if not scope_markets:
        raise RuntimeError(f"No markets found in scope file: {scope_path}")

    required_markets, required_market_metadata = resolve_required_markets(
        required_markets=list(getattr(args, "required_market", []) or []),
        promotion_policy_path=getattr(args, "promotion_policy", None),
    )
    validate_markets_in_scope(
        selected_markets=required_markets,
        scope_markets=scope_markets,
        strict=True,
        scope_path=scope_path,
        market_set_name="required",
    )

    baseline_summary: dict[str, Any] = {
        "validated": False,
        "baseline_path": str(baseline_path),
        "scope_markets_checked": 0,
        "required_markets_checked": int(len(required_markets)),
    }
    if validate_baseline:
        baselines = load_baseline_metrics(baseline_path)
        validate_market_presence(
            market_list=required_markets,
            baseline_metrics=baselines,
            strict=True,
            baseline_path=baseline_path,
            market_set_name="required",
        )
        validate_baseline_coverage(
            scope_markets=scope_markets,
            baseline_metrics=baselines,
            strict=True,
            baseline_path=baseline_path,
        )
        baseline_summary["validated"] = True
        baseline_summary["scope_markets_checked"] = int(len(scope_markets))

    return {
        "required_markets": required_markets,
        "required_market_metadata": required_market_metadata,
        "baseline_validation": baseline_summary,
    }


def main() -> None:
    args = parse_args()
    rebuild_baseline = bool(getattr(args, "rebuild_baseline", False)) and not bool(
        getattr(args, "skip_baseline_rebuild", False)
    )
    promotion_inputs: dict[str, Any] | None = None
    if not bool(args.skip_promotion):
        promotion_inputs = validate_promotion_inputs(
            args,
            validate_baseline=not rebuild_baseline,
        )
    plan = build_run_plan(args)
    Path(args.evaluation_dir).mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for name, command in plan:
        results.append(_run_step(name, command, dry_run=bool(args.dry_run)))
        if name == "baseline.rebuild" and not bool(args.skip_promotion):
            promotion_inputs = validate_promotion_inputs(args, validate_baseline=True)
    promotion_summary = build_promotion_summary(Path(args.evaluation_dir))
    report = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dry_run": bool(args.dry_run),
        "rebuild_baseline": rebuild_baseline,
        "scope": str(args.scope),
        "baseline_path": str(args.baseline_path),
        "promotion_preflight": promotion_inputs,
        "evaluation_dir": str(args.evaluation_dir),
        "scoreline_dir": str(args.scoreline_dir),
        "corners_dir": str(args.corners_dir),
        "anytime_dir": str(args.anytime_dir),
        "evaluation_flow_report_path": str(Path(args.evaluation_dir) / "evaluation_flow_report.json"),
        "promotion_summary": promotion_summary,
        "steps": results,
    }
    out_path = Path(args.evaluation_dir) / "evaluation_flow_report.json"
    recommendation_artifacts = write_promotion_recommendation_artifacts(
        flow_report=report,
        evaluation_dir=Path(args.evaluation_dir),
    )
    if recommendation_artifacts is not None:
        recommendation_json_path, recommendation_md_path, recommendation = recommendation_artifacts
        report["promotion_recommendation"] = {
            "status": recommendation.get("recommendation_status"),
            "scope": recommendation.get("recommended_scope"),
            "headline": recommendation.get("headline"),
            "json_path": str(recommendation_json_path),
            "markdown_path": str(recommendation_md_path),
        }
    shadow_compare = write_shadow_comparison_artifacts(
        flow_report=report,
        evaluation_dir=Path(args.evaluation_dir),
    )
    if shadow_compare is not None:
        report["shadow_comparison"] = shadow_compare
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation flow report to {out_path}")


if __name__ == "__main__":
    main()
