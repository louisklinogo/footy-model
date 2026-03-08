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


DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_SCORELINE_DIR = ROOT_DIR / "model_artifacts" / "v2" / "scoreline"
DEFAULT_CORNERS_DIR = ROOT_DIR / "model_artifacts" / "v2" / "corners"
DEFAULT_ANYTIME_DIR = ROOT_DIR / "model_artifacts" / "v2" / "anytime"
DEFAULT_OUT_DIR = ROOT_DIR / "artifacts" / "v2" / "predictions"
DEFAULT_EXPORT_PATH = ROOT_DIR / "storage" / "reports" / "market_predictions_v2.csv"
DEFAULT_MODEL_NAME = "market_outcome_v2"
DEFAULT_MODEL_VERSION = "challenger_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bundled v2 family prediction flow.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--scoreline-dir", type=Path, default=DEFAULT_SCORELINE_DIR)
    parser.add_argument("--corners-dir", type=Path, default=DEFAULT_CORNERS_DIR)
    parser.add_argument("--anytime-dir", type=Path, default=DEFAULT_ANYTIME_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--league", type=str, default=None)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", type=str, default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--run-risk", action="store_true")
    parser.add_argument(
        "--export-out",
        type=Path,
        default=None,
        help="Optional export CSV path. Implies DB upsert so export can query bundled predictions.",
    )
    parser.add_argument("--risk-history-days", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_run_plan(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    python_bin = str(args.python_bin)
    out_dir = Path(args.out_dir)
    model_name = str(args.model_name)
    model_version = str(args.model_version)
    needs_db = bool(args.write_db or args.run_risk or args.export_out)
    plan: list[tuple[str, list[str]]] = []

    family_specs = [
        (
            "predict.scoreline",
            ROOT_DIR / "src" / "modeling" / "v2" / "families" / "scoreline" / "predict_scoreline.py",
            Path(args.scoreline_dir),
        ),
        (
            "predict.corners",
            ROOT_DIR / "src" / "modeling" / "v2" / "families" / "corners" / "predict_corners.py",
            Path(args.corners_dir),
        ),
        (
            "predict.anytime",
            ROOT_DIR / "src" / "modeling" / "v2" / "families" / "anytime" / "predict_anytime.py",
            Path(args.anytime_dir),
        ),
    ]

    for name, script_path, artifact_dir in family_specs:
        cmd = [
            python_bin,
            str(script_path),
            "--scope",
            str(args.scope),
            "--artifact-dir",
            str(artifact_dir),
            "--out-dir",
            str(out_dir),
            "--days",
            str(int(args.days)),
            "--model-name",
            model_name,
            "--model-version",
            model_version,
        ]
        if args.league:
            cmd.extend(["--league", str(args.league)])
        if args.limit is not None and int(args.limit) > 0:
            cmd.extend(["--limit", str(int(args.limit))])
        if needs_db:
            cmd.append("--write-db")
        plan.append((name, cmd))

    if bool(args.run_risk):
        risk_cmd = [
            python_bin,
            str(ROOT_DIR / "src" / "modeling" / "evaluation" / "assess_prediction_risk.py"),
            "--days",
            str(int(args.days)),
            "--history-days",
            str(int(args.risk_history_days)),
            "--model",
            model_name,
            "--version",
            model_version,
        ]
        if args.league:
            risk_cmd.extend(["--league", str(args.league)])
        if args.limit is not None and int(args.limit) > 0:
            risk_cmd.extend(["--limit", str(int(args.limit))])
        plan.append(("risk.assess", risk_cmd))

    if args.export_out is not None:
        export_cmd = [
            python_bin,
            str(ROOT_DIR / "src" / "modeling" / "export" / "export_market_outcomes_fixtures_first.py"),
            "--days",
            str(int(args.days)),
            "--model",
            model_name,
            "--version",
            model_version,
            "--out",
            str(Path(args.export_out)),
        ]
        if args.league:
            export_cmd.extend(["--league", str(args.league)])
        plan.append(("export.csv", export_cmd))

    return plan


def _run_step(name: str, command: list[str], *, dry_run: bool) -> dict[str, Any]:
    print(f"COMMAND[{name}]: {' '.join(command)}")
    if dry_run:
        return {
            "name": name,
            "command": command,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "dry_run": True,
        }
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


def main() -> None:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plan = build_run_plan(args)
    results = [_run_step(name, command, dry_run=bool(args.dry_run)) for name, command in plan]
    report = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dry_run": bool(args.dry_run),
        "model_name": str(args.model_name),
        "model_version": str(args.model_version),
        "scope": str(args.scope),
        "scoreline_dir": str(args.scoreline_dir),
        "corners_dir": str(args.corners_dir),
        "anytime_dir": str(args.anytime_dir),
        "out_dir": str(args.out_dir),
        "export_out": str(args.export_out) if args.export_out is not None else None,
        "steps": results,
    }
    report_path = Path(args.out_dir) / "v2_prediction_flow_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved v2 prediction flow report to {report_path}")


if __name__ == "__main__":
    main()