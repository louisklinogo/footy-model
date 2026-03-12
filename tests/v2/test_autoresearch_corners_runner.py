from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

from autoresearch_corners.run_experiment import ROOT_DIR, _build_subprocess_env, build_run_plan, resolve_experiment, run_plan, write_run_summary


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_tag": "corners_candidate_test",
        "dataset_path": "artifacts/v2/datasets/sample/pit_dataset.parquet",
        "contract_path": "model_v2/feature_contracts/corners.yaml",
        "path_version": "totals_surface_calibrated",
        "model_type": "auto",
        "days": 3,
        "backfill_days": 60,
        "apply_calibrators": True,
    }
    payload.update(overrides)
    return payload


def test_resolve_experiment_defaults_model_and_output_paths() -> None:
    experiment = resolve_experiment(_payload())
    assert experiment["model_version"] == "corners_candidate_test"
    assert str(experiment["artifact_dir"]).endswith("model_artifacts\\v2\\corners_candidate_test") or str(experiment["artifact_dir"]).endswith("model_artifacts/v2/corners_candidate_test")
    assert str(experiment["comparison_dir"]).endswith("live_replacement_corners_candidate_test")


def test_build_run_plan_threads_candidate_artifact_into_live_compare() -> None:
    plan = build_run_plan(resolve_experiment(_payload(limit=25, league="E0")), python_bin="python")
    assert [name for name, _ in plan] == ["train.corners", "eval.live_replacement"]
    commands = dict(plan)
    assert "--path-version" in commands["train.corners"]
    assert "totals_surface_calibrated" in commands["train.corners"]
    assert "--corners-dir" in commands["eval.live_replacement"]
    assert any("corners_candidate_test" in arg for arg in commands["eval.live_replacement"])
    assert "--league" in commands["eval.live_replacement"] and "E0" in commands["eval.live_replacement"]
    assert "--limit" in commands["eval.live_replacement"] and "25" in commands["eval.live_replacement"]


def test_build_run_plan_supports_no_apply_calibrators() -> None:
    plan = build_run_plan(resolve_experiment(_payload(apply_calibrators=False)), python_bin="python")
    compare_cmd = dict(plan)["eval.live_replacement"]
    assert "--no-apply-calibrators" in compare_cmd


def test_write_run_summary_persists_json_payload() -> None:
    with tempfile.TemporaryDirectory(prefix="autoresearch_corners_summary_") as td:
        tmp_path = Path(td)
        experiment = resolve_experiment(_payload(results_dir=str(tmp_path / "results")))
        plan = build_run_plan(experiment, python_bin="python")
        run_dir = tmp_path / "results" / "corners_candidate_test_run"
        summary_path = write_run_summary(
            config_path=tmp_path / "experiment.local.json",
            experiment=experiment,
            plan=plan,
            results=[{"step": "train.corners", "returncode": 0, "status": "ok", "command": ["python"], "log_path": "train.log"}],
            run_dir=run_dir,
            dry_run=False,
        )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["candidate_tag"] == "corners_candidate_test"
        assert payload["results"][0]["status"] == "ok"


def test_run_plan_creates_run_dir_before_opening_logs() -> None:
    with tempfile.TemporaryDirectory(prefix="autoresearch_corners_run_plan_") as td:
        tmp_path = Path(td)
        run_dir = tmp_path / "results" / "corners_candidate_test_run"
        results = run_plan(
            [("train.corners", [sys.executable, "-c", "print('ok')"])],
            run_dir=run_dir,
            dry_run=False,
        )
        assert results[0]["status"] == "ok"
        assert (run_dir / "train_corners.log").exists()


def test_build_subprocess_env_prepends_repo_root_to_pythonpath() -> None:
    env = _build_subprocess_env()
    pythonpath = env.get("PYTHONPATH", "")
    assert pythonpath
    assert pythonpath.split(os.pathsep)[0] == str(ROOT_DIR)
