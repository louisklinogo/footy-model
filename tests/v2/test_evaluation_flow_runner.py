from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from src.modeling.v2.run_evaluation_flow import build_promotion_summary, build_run_plan


def _args(**overrides: object) -> argparse.Namespace:
    payload = {
        "python_bin": "python",
        "scope": "scope.yaml",
        "scoreline_dir": "out/scoreline",
        "corners_dir": "out/corners",
        "anytime_dir": "out/anytime",
        "evaluation_dir": "out/evaluation",
        "baseline_path": "out/baseline.json",
        "max_rows": None,
        "skip_train": False,
        "rebuild_baseline": False,
        "skip_baseline_rebuild": False,
        "skip_promotion": False,
        "allow_missing_holdout": False,
        "min_support": 200,
        "min_folds": 3,
        "max_ece": 0.05,
        "auc_tolerance": 0.0,
        "brier_tolerance": 0.0,
        "log_loss_tolerance": 0.0,
        "promotion_policy": None,
        "required_market": [],
        "dry_run": False,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_build_run_plan_defaults_include_all_steps() -> None:
    plan = build_run_plan(_args())
    assert [name for name, _ in plan] == [
        "train.scoreline",
        "train.corners",
        "train.anytime",
        "eval.walkforward",
        "eval.calibration",
        "eval.promotion_registry",
    ]


def test_build_run_plan_respects_skip_flags_and_forwards_options() -> None:
    plan = build_run_plan(
        _args(
            skip_train=True,
            max_rows=123,
            min_support=500,
            min_folds=4,
            max_ece=0.02,
            auc_tolerance=0.01,
            brier_tolerance=0.02,
        )
    )
    names = [name for name, _ in plan]
    assert names == ["eval.walkforward", "eval.calibration", "eval.promotion_registry"]
    promotion_cmd = dict(plan)["eval.promotion_registry"]
    assert "--min-support" in promotion_cmd and "500" in promotion_cmd
    assert "--min-folds" in promotion_cmd and "4" in promotion_cmd
    assert "--max-ece" in promotion_cmd and "0.02" in promotion_cmd
    assert "--auc-tolerance" in promotion_cmd and "0.01" in promotion_cmd
    assert "--brier-tolerance" in promotion_cmd and "0.02" in promotion_cmd
    assert "--calibration-report" in promotion_cmd


def test_build_run_plan_forwards_required_markets_to_promotion_registry() -> None:
    plan = build_run_plan(_args(skip_train=True, required_market=["1x2_h", "dc_1x"]))
    promotion_cmd = dict(plan)["eval.promotion_registry"]
    assert promotion_cmd.count("--required-market") == 2
    assert "1x2_h" in promotion_cmd
    assert "dc_1x" in promotion_cmd


def test_build_run_plan_forwards_promotion_policy_to_promotion_registry() -> None:
    plan = build_run_plan(_args(skip_train=True, promotion_policy="model_v2/promotion_policies/scoreline_core.yaml"))
    promotion_cmd = dict(plan)["eval.promotion_registry"]
    assert "--promotion-policy" in promotion_cmd
    assert "model_v2/promotion_policies/scoreline_core.yaml" in promotion_cmd


def test_build_run_plan_only_rebuilds_baseline_when_opted_in() -> None:
    plan = build_run_plan(_args(skip_train=True, rebuild_baseline=True, allow_missing_holdout=True))
    names = [name for name, _ in plan]
    assert names == ["eval.walkforward", "eval.calibration", "baseline.rebuild", "eval.promotion_registry"]
    rebuild_cmd = dict(plan)["baseline.rebuild"]
    assert "--allow-missing" in rebuild_cmd


def test_build_promotion_summary_reads_compact_decision_view() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_promotion_summary_") as td:
        evaluation_dir = Path(td)
        (evaluation_dir / "promotion_registry.json").write_text(
            json.dumps(
                {
                    "rules": {
                        "promotion_policy_name": "scoreline_core_v1",
                        "promotion_policy_path": "model_v2/promotion_policies/scoreline_core.yaml",
                        "required_markets": ["1x2_h", "1x2_d"],
                    },
                    "summary": {
                        "markets_total": 68,
                        "markets_passed": 59,
                        "markets_failed": 9,
                        "required_markets_total": 2,
                        "required_markets_passed": 2,
                        "required_markets_failed": 0,
                        "required_markets_missing": [],
                    },
                    "decision": {
                        "status": "passed",
                        "basis": "required_markets",
                        "failed_markets": [],
                    },
                }
            ),
            encoding="utf-8",
        )

        summary = build_promotion_summary(evaluation_dir)

        assert summary is not None
        assert summary["decision"]["status"] == "passed"
        assert summary["summary"]["markets_failed"] == 9
        assert summary["rules"]["promotion_policy_name"] == "scoreline_core_v1"
        assert summary["rules"]["required_markets"] == ["1x2_h", "1x2_d"]


def test_build_promotion_summary_returns_none_when_registry_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_promotion_summary_missing_") as td:
        assert build_promotion_summary(Path(td)) is None


def test_build_promotion_summary_preserves_policy_metadata_for_recommendation_layer() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_promotion_summary_policy_meta_") as td:
        evaluation_dir = Path(td)
        (evaluation_dir / "promotion_registry.json").write_text(
            json.dumps(
                {
                    "rules": {
                        "promotion_policy_name": "scoreline_core_v1",
                        "promotion_policy_path": "model_v2/promotion_policies/scoreline_core.yaml",
                        "required_markets": ["1x2_h"],
                    },
                    "summary": {
                        "markets_total": 1,
                        "markets_passed": 1,
                        "markets_failed": 0,
                        "required_markets_total": 1,
                        "required_markets_passed": 1,
                        "required_markets_failed": 0,
                        "required_markets_missing": [],
                    },
                    "decision": {"status": "passed", "basis": "required_markets"},
                }
            ),
            encoding="utf-8",
        )
        summary = build_promotion_summary(evaluation_dir)
        assert summary is not None
        assert summary["rules"]["promotion_policy_path"] == "model_v2/promotion_policies/scoreline_core.yaml"
