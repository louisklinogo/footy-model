from __future__ import annotations

import json
from pathlib import Path
import tempfile

from src.modeling.v2.eval.promotion_registry import build_promotion_registry
from src.modeling.v2.eval.run_walkforward import build_evaluation_report


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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
            [{"market": "o15", "auc": 0.66, "brier": 0.18, "log_loss": 0.49, "ece": 0.03, "n": 250}],
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
        _write_text(
            scoreline_dir / "holdout_predictions.csv",
            "market,fixture_id,y_true,p_model,league_code\n"
            "o15,11,1,0.71,EPL\n"
            "o15,12,0,0.29,EPL\n",
        )
        _write_json(
            scoreline_dir / "scoreline_slice_summary.json",
            {"exact_0_0": {"n_total": 2, "brier_mean": 0.11}, "draw": {"n_total": 2, "brier_mean": 0.15}},
        )
        _write_json(
            scoreline_dir / "scoreline_slice_summary_by_league.json",
            {"exact_0_0": {"EPL": {"n_total": 2, "brier_mean": 0.11}}},
        )
        _write_json(scoreline_dir / "training_report.json", {"model": "scoreline"})

        _write_json(
            anytime_dir / "metrics_holdout.json",
            [{"market": "h_1up", "auc": 0.59, "brier": 0.24, "log_loss": 0.24, "ece": 0.08, "n": 90}],
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
        _write_text(
            anytime_dir / "holdout_predictions.csv",
            "market,fixture_id,y_true,p_model\n"
            "h_1up,21,1,0.61\n",
        )
        _write_json(anytime_dir / "training_report.json", {"model": "anytime"})

        _write_json(corners_dir / "metrics_holdout.json", [])
        _write_json(corners_dir / "metrics_walkforward.json", {})
        _write_json(corners_dir / "metrics_walkforward_folds.json", [])
        _write_json(corners_dir / "metrics_walkforward_folds_by_league.json", [])
        _write_json(corners_dir / "metrics_holdout_by_league.json", [])
        _write_text(corners_dir / "holdout_predictions.csv", "market,fixture_id,y_true,p_model\n")
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
        assert report["families"]["scoreline"]["holdout_prediction_rows"] == 2
        assert report["families"]["anytime"]["holdout_prediction_rows"] == 1
        assert report["scoreline_slice_summary"]["exact_0_0"]["n_total"] == 2
        assert report["scoreline_slice_summary_by_league"]["exact_0_0"]["EPL"]["n_total"] == 2

        baseline_path = tmp_path / "baseline.json"
        _write_json(
            baseline_path,
            {
                "metrics": [
                    {"market_code": "o15", "auc": 0.60, "brier": 0.20, "log_loss": 0.55},
                    {"market_code": "h_1up", "auc": 0.60, "brier": 0.22, "log_loss": 0.21},
                ]
            },
        )
        eval_path = tmp_path / "evaluation_report.json"
        _write_json(eval_path, report)
        calibration_path = tmp_path / "calibration_report.json"
        _write_json(
            calibration_path,
            {
                "promotion_holdout_by_market": {
                    "o15": {
                        "market": "o15",
                        "family": "scoreline",
                        "auc": 0.65,
                        "brier": 0.19,
                        "log_loss": 0.50,
                        "ece": 0.03,
                        "n": 250,
                        "calibration_method": "sigmoid",
                    }
                }
            },
        )

        registry = build_promotion_registry(
            scope_path=scope_path,
            baseline_path=baseline_path,
            evaluation_report_path=eval_path,
            calibration_report_path=calibration_path,
            required_markets=None,
            min_support=100,
            min_folds=3,
            max_ece=0.05,
            auc_tolerance=0.0,
            brier_tolerance=0.0,
            log_loss_tolerance=0.0,
        )
        by_market = {row["market"]: row for row in registry["markets"]}
        assert by_market["o15"]["status"] == "passed"
        assert by_market["o15"]["holdout_source"] == "calibrated_holdout"
        assert by_market["o15"]["effective_holdout"]["brier"] == 0.19
        assert by_market["h_1up"]["status"] == "failed"
        assert set(by_market["h_1up"]["reasons"]) >= {"auc_failed", "brier_failed", "ece_failed", "log_loss_failed", "support_failed", "folds_failed"}


def test_build_promotion_registry_falls_back_to_raw_holdout_when_calibrated_sample_is_not_comparable() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_eval_harness_raw_fallback_") as td:
        tmp_path = Path(td)
        scope_path = tmp_path / "market_scope.yaml"
        scope_path.write_text("markets:\n  - c105\n", encoding="utf-8")

        baseline_path = tmp_path / "baseline.json"
        _write_json(
            baseline_path,
            {"metrics": [{"market_code": "c105", "auc": 0.58, "brier": 0.23, "log_loss": 0.65, "ece": 0.04, "n": 200}]},
        )
        eval_path = tmp_path / "evaluation_report.json"
        _write_json(
            eval_path,
            {
                "holdout_by_market": {
                    "c105": {"market": "c105", "family": "corners", "auc": 0.59, "brier": 0.22, "log_loss": 0.64, "ece": 0.03, "n": 200}
                },
                "walkforward_by_market": {
                    "c105": {"market": "c105", "family": "corners", "folds_used": 3, "n_total": 250}
                },
            },
        )
        calibration_path = tmp_path / "calibration_report.json"
        _write_json(
            calibration_path,
            {
                "promotion_holdout_by_market": {
                    "c105": {
                        "market": "c105",
                        "family": "corners",
                        "auc": 0.57,
                        "brier": 0.21,
                        "log_loss": 0.62,
                        "ece": 0.06,
                        "n": 100,
                        "calibration_method": "sigmoid",
                    }
                }
            },
        )

        registry = build_promotion_registry(
            scope_path=scope_path,
            baseline_path=baseline_path,
            evaluation_report_path=eval_path,
            calibration_report_path=calibration_path,
            required_markets=None,
            min_support=100,
            min_folds=3,
            max_ece=0.05,
            auc_tolerance=0.0,
            brier_tolerance=0.0,
            log_loss_tolerance=0.0,
        )

        market_row = registry["markets"][0]
        assert market_row["status"] == "passed"
        assert market_row["holdout_source"] == "raw_holdout"
        assert market_row["effective_holdout"]["n"] == 200
        assert market_row["calibration"]["n"] == 100


def test_build_promotion_registry_can_decide_on_required_markets_only() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_eval_harness_required_markets_") as td:
        tmp_path = Path(td)
        scope_path = tmp_path / "market_scope.yaml"
        scope_path.write_text("markets:\n  - 1x2_h\n  - mg_2_4\n", encoding="utf-8")

        baseline_path = tmp_path / "baseline.json"
        _write_json(
            baseline_path,
            {
                "metrics": [
                    {"market_code": "1x2_h", "auc": 0.60, "brier": 0.24, "log_loss": 0.67, "ece": 0.04, "n": 300},
                    {"market_code": "mg_2_4", "auc": 0.55, "brier": 0.24, "log_loss": 0.66, "ece": 0.03, "n": 300},
                ]
            },
        )
        eval_path = tmp_path / "evaluation_report.json"
        _write_json(
            eval_path,
            {
                "holdout_by_market": {
                    "1x2_h": {"market": "1x2_h", "family": "scoreline", "auc": 0.62, "brier": 0.23, "log_loss": 0.65, "ece": 0.03, "n": 300},
                    "mg_2_4": {"market": "mg_2_4", "family": "scoreline", "auc": 0.54, "brier": 0.23, "log_loss": 0.65, "ece": 0.02, "n": 300},
                },
                "walkforward_by_market": {
                    "1x2_h": {"market": "1x2_h", "family": "scoreline", "folds_used": 4, "n_total": 320},
                    "mg_2_4": {"market": "mg_2_4", "family": "scoreline", "folds_used": 4, "n_total": 320},
                },
            },
        )

        registry = build_promotion_registry(
            scope_path=scope_path,
            baseline_path=baseline_path,
            evaluation_report_path=eval_path,
            calibration_report_path=None,
            required_markets=["1x2_h"],
            min_support=100,
            min_folds=3,
            max_ece=0.05,
            auc_tolerance=0.0,
            brier_tolerance=0.0,
            log_loss_tolerance=0.0,
        )

        by_market = {row["market"]: row for row in registry["markets"]}
        assert by_market["1x2_h"]["status"] == "passed"
        assert by_market["mg_2_4"]["status"] == "failed"
        assert registry["decision"]["basis"] == "required_markets"
        assert registry["decision"]["status"] == "passed"
        assert registry["decision"]["failed_markets"] == []
        assert registry["summary"]["required_markets_failed"] == 0


def test_build_promotion_registry_allows_absolute_gate_when_baseline_metrics_are_null() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_eval_harness_null_baseline_") as td:
        tmp_path = Path(td)
        scope_path = tmp_path / "market_scope.yaml"
        scope_path.write_text("markets:\n  - eh3_0_1_draw\n", encoding="utf-8")

        baseline_path = tmp_path / "baseline.json"
        _write_json(
            baseline_path,
            {
                "metrics": [
                    {
                        "market_code": "eh3_0_1_draw",
                        "auc": None,
                        "brier": None,
                        "log_loss": None,
                        "ece": None,
                        "n": None,
                    }
                ]
            },
        )
        eval_path = tmp_path / "evaluation_report.json"
        _write_json(
            eval_path,
            {
                "holdout_by_market": {
                    "eh3_0_1_draw": {
                        "market": "eh3_0_1_draw",
                        "family": "scoreline",
                        "auc": 0.61,
                        "brier": 0.19,
                        "log_loss": 0.55,
                        "ece": 0.03,
                        "n": 260,
                    }
                },
                "walkforward_by_market": {
                    "eh3_0_1_draw": {
                        "market": "eh3_0_1_draw",
                        "family": "scoreline",
                        "folds_used": 4,
                        "n_total": 320,
                    }
                },
            },
        )

        registry = build_promotion_registry(
            scope_path=scope_path,
            baseline_path=baseline_path,
            evaluation_report_path=eval_path,
            calibration_report_path=None,
            required_markets=["eh3_0_1_draw"],
            min_support=100,
            min_folds=3,
            max_ece=0.05,
            auc_tolerance=0.0,
            brier_tolerance=0.0,
            log_loss_tolerance=0.0,
        )

        market_row = registry["markets"][0]
        assert market_row["status"] == "passed"
        assert market_row["reasons"] == []
        assert registry["decision"]["status"] == "passed"


def test_build_promotion_registry_fails_when_required_market_is_missing_from_scope() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_eval_harness_required_missing_") as td:
        tmp_path = Path(td)
        scope_path = tmp_path / "market_scope.yaml"
        scope_path.write_text("markets:\n  - o15\n", encoding="utf-8")

        baseline_path = tmp_path / "baseline.json"
        _write_json(
            baseline_path,
            {"metrics": [{"market_code": "o15", "auc": 0.60, "brier": 0.20, "log_loss": 0.55, "ece": 0.03, "n": 250}]},
        )
        eval_path = tmp_path / "evaluation_report.json"
        _write_json(
            eval_path,
            {
                "holdout_by_market": {
                    "o15": {"market": "o15", "family": "scoreline", "auc": 0.61, "brier": 0.19, "log_loss": 0.54, "ece": 0.02, "n": 250}
                },
                "walkforward_by_market": {
                    "o15": {"market": "o15", "family": "scoreline", "folds_used": 3, "n_total": 260}
                },
            },
        )

        registry = build_promotion_registry(
            scope_path=scope_path,
            baseline_path=baseline_path,
            evaluation_report_path=eval_path,
            calibration_report_path=None,
            required_markets=["1x2_h"],
            min_support=100,
            min_folds=3,
            max_ece=0.05,
            auc_tolerance=0.0,
            brier_tolerance=0.0,
            log_loss_tolerance=0.0,
        )

        assert registry["decision"]["status"] == "failed"
        assert registry["decision"]["missing_required_markets"] == ["1x2_h"]
        assert registry["summary"]["required_markets_missing"] == ["1x2_h"]


def test_build_promotion_registry_ignores_float_noise_scale_metric_regressions() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_eval_harness_float_noise_") as td:
        tmp_path = Path(td)
        scope_path = tmp_path / "market_scope.yaml"
        scope_path.write_text("markets:\n  - o15\n", encoding="utf-8")

        baseline_path = tmp_path / "baseline.json"
        _write_json(
            baseline_path,
            {
                "metrics": [
                    {
                        "market_code": "o15",
                        "auc": 0.60,
                        "brier": 0.20,
                        "log_loss": 0.55,
                        "ece": 0.03,
                        "n": 250,
                    }
                ]
            },
        )
        eval_path = tmp_path / "evaluation_report.json"
        _write_json(
            eval_path,
            {
                "holdout_by_market": {
                    "o15": {
                        "market": "o15",
                        "family": "scoreline",
                        "auc": 0.60,
                        "brier": 0.20000000000000004,
                        "log_loss": 0.5500000000000002,
                        "ece": 0.03,
                        "n": 250,
                    }
                },
                "walkforward_by_market": {
                    "o15": {"market": "o15", "family": "scoreline", "folds_used": 3, "n_total": 260}
                },
            },
        )

        registry = build_promotion_registry(
            scope_path=scope_path,
            baseline_path=baseline_path,
            evaluation_report_path=eval_path,
            calibration_report_path=None,
            required_markets=["o15"],
            min_support=100,
            min_folds=3,
            max_ece=0.05,
            auc_tolerance=0.0,
            brier_tolerance=0.0,
            log_loss_tolerance=0.0,
        )

        market_row = registry["markets"][0]
        assert market_row["status"] == "passed"
        assert market_row["reasons"] == []


def test_build_promotion_registry_still_fails_meaningful_small_metric_regressions() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_eval_harness_meaningful_small_delta_") as td:
        tmp_path = Path(td)
        scope_path = tmp_path / "market_scope.yaml"
        scope_path.write_text("markets:\n  - o15\n", encoding="utf-8")

        baseline_path = tmp_path / "baseline.json"
        _write_json(
            baseline_path,
            {
                "metrics": [
                    {
                        "market_code": "o15",
                        "auc": 0.60,
                        "brier": 0.20,
                        "log_loss": 0.55,
                        "ece": 0.03,
                        "n": 250,
                    }
                ]
            },
        )
        eval_path = tmp_path / "evaluation_report.json"
        _write_json(
            eval_path,
            {
                "holdout_by_market": {
                    "o15": {
                        "market": "o15",
                        "family": "scoreline",
                        "auc": 0.60,
                        "brier": 0.200001,
                        "log_loss": 0.550001,
                        "ece": 0.03,
                        "n": 250,
                    }
                },
                "walkforward_by_market": {
                    "o15": {"market": "o15", "family": "scoreline", "folds_used": 3, "n_total": 260}
                },
            },
        )

        registry = build_promotion_registry(
            scope_path=scope_path,
            baseline_path=baseline_path,
            evaluation_report_path=eval_path,
            calibration_report_path=None,
            required_markets=["o15"],
            min_support=100,
            min_folds=3,
            max_ece=0.05,
            auc_tolerance=0.0,
            brier_tolerance=0.0,
            log_loss_tolerance=0.0,
        )

        market_row = registry["markets"][0]
        assert market_row["status"] == "failed"
        assert set(market_row["reasons"]) >= {"brier_failed", "log_loss_failed"}
