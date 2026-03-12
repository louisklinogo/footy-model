from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT_DIR / "src" / "modeling" / "v2" / "families" / "corners" / "train_corners.py"
COMPARE_SCRIPT = ROOT_DIR / "src" / "modeling" / "v2" / "eval" / "live_replacement_compare.py"
DEFAULT_CONFIG = Path(__file__).with_name("experiment.local.json")
EXAMPLE_CONFIG = Path(__file__).with_name("experiment.example.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one bounded corners autoresearch experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        if path == DEFAULT_CONFIG:
            raise FileNotFoundError(
                f"Missing {path}. Copy {EXAMPLE_CONFIG.name} to {DEFAULT_CONFIG.name} and edit it first."
            )
        raise FileNotFoundError(f"Missing config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a JSON object.")
    return payload


def resolve_experiment(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = ("candidate_tag", "dataset_path", "contract_path", "path_version")
    missing = [key for key in required if not str(payload.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Experiment config missing required keys: {', '.join(missing)}")
    candidate_tag = str(payload["candidate_tag"]).strip()
    return {
        "candidate_tag": candidate_tag,
        "model_version": str(payload.get("model_version") or candidate_tag),
        "dataset_path": _resolve_repo_path(str(payload["dataset_path"])),
        "contract_path": _resolve_repo_path(str(payload["contract_path"])),
        "scope_path": _resolve_repo_path(str(payload.get("scope_path", "model_v2/market_scope.yaml"))),
        "artifact_dir": _resolve_repo_path(str(payload.get("artifact_dir", f"model_artifacts/v2/{candidate_tag}"))),
        "comparison_dir": _resolve_repo_path(
            str(payload.get("comparison_dir", f"artifacts/v2/family_replacement/live_replacement_{candidate_tag}"))
        ),
        "scoreline_dir": _resolve_repo_path(str(payload.get("scoreline_dir", "model_artifacts/v2/scoreline"))),
        "anytime_dir": _resolve_repo_path(str(payload.get("anytime_dir", "model_artifacts/v2/anytime"))),
        "results_dir": _resolve_repo_path(str(payload.get("results_dir", "autoresearch_corners/results"))),
        "path_version": str(payload["path_version"]),
        "model_type": str(payload.get("model_type", "auto")),
        "max_rows": payload.get("max_rows"),
        "folds": payload.get("folds"),
        "min_fold_test_n": payload.get("min_fold_test_n"),
        "days": int(payload.get("days", 3)),
        "backfill_days": int(payload.get("backfill_days", 60)),
        "league": payload.get("league"),
        "limit": payload.get("limit"),
        "apply_calibrators": bool(payload.get("apply_calibrators", True)),
        "notes": payload.get("notes"),
    }


def build_run_plan(experiment: Mapping[str, Any], python_bin: str) -> list[tuple[str, list[str]]]:
    train_cmd = [
        python_bin,
        str(TRAIN_SCRIPT),
        "--dataset-path",
        str(experiment["dataset_path"]),
        "--contract",
        str(experiment["contract_path"]),
        "--scope",
        str(experiment["scope_path"]),
        "--output-dir",
        str(experiment["artifact_dir"]),
        "--model-version",
        str(experiment["model_version"]),
        "--path-version",
        str(experiment["path_version"]),
        "--model-type",
        str(experiment["model_type"]),
    ]
    for key, flag in (("max_rows", "--max-rows"), ("folds", "--folds"), ("min_fold_test_n", "--min-fold-test-n")):
        value = experiment.get(key)
        if value is not None:
            train_cmd.extend([flag, str(int(value))])

    compare_cmd = [
        python_bin,
        str(COMPARE_SCRIPT),
        "--scoreline-dir",
        str(experiment["scoreline_dir"]),
        "--corners-dir",
        str(experiment["artifact_dir"]),
        "--anytime-dir",
        str(experiment["anytime_dir"]),
        "--output-dir",
        str(experiment["comparison_dir"]),
        "--days",
        str(int(experiment["days"])),
        "--backfill-days",
        str(int(experiment["backfill_days"])),
    ]
    if experiment.get("league"):
        compare_cmd.extend(["--league", str(experiment["league"])])
    if experiment.get("limit") is not None:
        compare_cmd.extend(["--limit", str(int(experiment["limit"]))])
    if not bool(experiment.get("apply_calibrators", True)):
        compare_cmd.append("--no-apply-calibrators")
    return [("train.corners", train_cmd), ("eval.live_replacement", compare_cmd)]


def _validate_runtime_inputs(experiment: Mapping[str, Any], dry_run: bool) -> None:
    for path in (TRAIN_SCRIPT, COMPARE_SCRIPT, experiment["contract_path"], experiment["scope_path"]):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing required path: {path}")
    if dry_run:
        return
    for path in (experiment["dataset_path"], experiment["scoreline_dir"], experiment["anytime_dir"]):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing runtime path: {path}")


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    root_str = str(ROOT_DIR)
    existing = env.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if root_str not in parts:
        parts.insert(0, root_str)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_plan(
    plan: list[tuple[str, list[str]]],
    *,
    run_dir: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, command in plan:
        log_path = run_dir / f"{name.replace('.', '_')}.log"
        result = {"step": name, "command": command, "log_path": str(log_path), "status": "dry_run", "returncode": 0}
        print("COMMAND:", " ".join(command))
        if dry_run:
            results.append(result)
            continue
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("COMMAND: " + " ".join(command) + "\n\n")
            log_file.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT_DIR,
                env=_build_subprocess_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        result["returncode"] = int(completed.returncode)
        result["status"] = "ok" if completed.returncode == 0 else "failed"
        results.append(result)
        if completed.returncode != 0:
            break
    return results


def write_run_summary(
    *,
    config_path: Path,
    experiment: Mapping[str, Any],
    plan: list[tuple[str, list[str]]],
    results: list[dict[str, Any]],
    run_dir: Path,
    dry_run: bool,
) -> Path:
    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "config_path": str(config_path),
        "candidate_tag": experiment["candidate_tag"],
        "artifact_dir": str(experiment["artifact_dir"]),
        "comparison_dir": str(experiment["comparison_dir"]),
        "notes": experiment.get("notes"),
        "plan": [{"step": name, "command": command} for name, command in plan],
        "results": results,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    experiment = resolve_experiment(config)
    _validate_runtime_inputs(experiment, dry_run=bool(args.dry_run))
    plan = build_run_plan(experiment, python_bin=str(args.python_bin))
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(experiment["results_dir"]) / f"{experiment['candidate_tag']}_{timestamp}"
    results = run_plan(plan, run_dir=run_dir, dry_run=bool(args.dry_run))
    summary_path = write_run_summary(
        config_path=Path(args.config),
        experiment=experiment,
        plan=plan,
        results=results,
        run_dir=run_dir,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps({"run_dir": str(run_dir), "summary_path": str(summary_path), "results": results}, indent=2))
    return 0 if all(int(row["returncode"]) == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
