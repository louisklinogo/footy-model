from __future__ import annotations

import argparse

from src.modeling.v2.run_evaluation_flow import build_run_plan


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
    assert names == ["eval.walkforward", "eval.promotion_registry"]
    promotion_cmd = dict(plan)["eval.promotion_registry"]
    assert "--min-support" in promotion_cmd and "500" in promotion_cmd
    assert "--min-folds" in promotion_cmd and "4" in promotion_cmd
    assert "--max-ece" in promotion_cmd and "0.02" in promotion_cmd
    assert "--auc-tolerance" in promotion_cmd and "0.01" in promotion_cmd
    assert "--brier-tolerance" in promotion_cmd and "0.02" in promotion_cmd


def test_build_run_plan_only_rebuilds_baseline_when_opted_in() -> None:
    plan = build_run_plan(_args(skip_train=True, rebuild_baseline=True, allow_missing_holdout=True))
    names = [name for name, _ in plan]
    assert names == ["eval.walkforward", "baseline.rebuild", "eval.promotion_registry"]
    rebuild_cmd = dict(plan)["baseline.rebuild"]
    assert "--allow-missing" in rebuild_cmd
