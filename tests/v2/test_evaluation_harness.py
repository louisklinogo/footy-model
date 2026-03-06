from __future__ import annotations

import json
from pathlib import Path
import tempfile

from src.modeling.v2.eval.promotion_registry import build_promotion_registry
from src.modeling.v2.eval.run_walkforward import build_evaluation_report


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_evaluation_report_and_promotion_registry() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_eval_harness_") as td:
        tmp_path = Path(td)
        scope_path = tmp_path / "market_scope.yaml"
        scope_path.write_text("markets:\n  - o15\n  - h_1up\n", encoding="utf-8")

        scoreline_dir = tmp_path / "scoreline"
        anytime_dir = tmp_path / "anytime"
        corners_dir = tmp_path / "corners"
        _write_json(
            scoreline_dir / "metrics_holdout.json",
            [{"market": "o15", "auc": 0.66, "brier": 0.18, "ece": 0.03, "n": 250}],
        )
        _write_json(
            scoreline_dir / "metrics_walkforward.json",
        {"('o15',)": {"auc_mean": 0.63, "brier_mean": 0.19, "folds_used": 4, "n_total": 260}},
        )
        _write_json(
            scoreline_dir / "metrics_walkforward_folds.json",
            [
                {"market": "o15", "fold": 1, "auc": 0.63, "brier": 0.19, "n": 70},
                {"market": "o15", "fold": 2, "auc": 0.64, "brier": 0.18, "n": 60},
                {"market": "o15", "fold": 3, "auc": 0.62, "brier": 0.20, "n": 65},
                {"market": "o15", "fold": 4, "auc": 0.63, "brier": 0.19, "n": 65},
            ],
        )
        _write_json(
            scoreline_dir / "metrics_walkforward_folds_by_league.json",
            [{"market": "o15", "league_code": "EPL", "fold": 1, "auc": 0.63, "brier": 0.19, "n": 120}],
        )
        _write_json(scoreline_dir / "metrics_holdout_by_league.json", [])
        _write_json(scoreline_dir / "training_report.json", {"model": "scoreline"})

        _write_json(
            anytime_dir / "metrics_holdout.json",
            [{"market": "h_1up", "auc": 0.59, "brier": 0.24, "ece": 0.08, "n": 90}],
        )
        _write_json(
            anytime_dir / "metrics_walkforward.json",
            {"h_1up": {"auc_mean": 0.58, "brier_mean": 0.24, "folds_used": 2, "n_total": 90}},
        )
        _write_json(
            anytime_dir / "metrics_walkforward_folds.json",
            [
                {"market": "h_1up", "fold": 1, "auc": 0.58, "brier": 0.24, "n": 45},
                {"market": "h_1up", "fold": 2, "auc": 0.58, "brier": 0.24, "n": 45},
            ],
        )
        _write_json(anytime_dir / "metrics_walkforward_folds_by_league.json", [])
        _write_json(anytime_dir / "metrics_holdout_by_league.json", [])
        _write_json(anytime_dir / "training_report.json", {"model": "anytime"})

        _write_json(corners_dir / "metrics_holdout.json", [])
        _write_json(corners_dir / "metrics_walkforward.json", {})
        _write_json(corners_dir / "metrics_walkforward_folds.json", [])
        _write_json(corners_dir / "metrics_walkforward_folds_by_league.json", [])
        _write_json(corners_dir / "metrics_holdout_by_league.json", [])
        _write_json(corners_dir / "training_report.json", {"model": "corners"})

        report = build_evaluation_report(
            scope_path=scope_path,
            family_dirs={"scoreline": scoreline_dir, "corners": corners_dir, "anytime": anytime_dir},
        )
        assert report["missing_holdout_markets"] == []
        assert report["missing_walkforward_markets"] == []
        assert report["holdout_by_market"]["o15"]["family"] == "scoreline"
        assert report["walkforward_by_market"]["o15"]["family"] == "scoreline"
        assert report["walkforward_by_market"]["h_1up"]["family"] == "anytime"

        baseline_path = tmp_path / "baseline.json"
        _write_json(
            baseline_path,
            {"metrics": [{"market_code": "o15", "auc": 0.60, "brier": 0.20}, {"market_code": "h_1up", "auc": 0.60, "brier": 0.22}]},
        )
        eval_path = tmp_path / "evaluation_report.json"
        _write_json(eval_path, report)

        registry = build_promotion_registry(
            scope_path=scope_path,
            baseline_path=baseline_path,
            evaluation_report_path=eval_path,
            min_support=100,
            min_folds=3,
            max_ece=0.05,
            auc_tolerance=0.0,
            brier_tolerance=0.0,
        )
        by_market = {row["market"]: row for row in registry["markets"]}
        assert by_market["o15"]["status"] == "passed"
        assert by_market["h_1up"]["status"] == "failed"
        assert set(by_market["h_1up"]["reasons"]) >= {"auc_failed", "brier_failed", "ece_failed", "support_failed", "folds_failed"}
