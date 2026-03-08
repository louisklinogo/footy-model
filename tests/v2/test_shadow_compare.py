from __future__ import annotations

import json
from pathlib import Path
import tempfile

from src.modeling.v2.eval.shadow_compare import write_shadow_comparison_artifacts
from src.modeling.v2.io.artifact_identity import build_artifact_metadata, write_artifact_metadata


def _write_metadata(artifact_dir: Path, *, family: str, model_name: str, model_version: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_metadata(
        artifact_dir,
        build_artifact_metadata(
            family=family,
            model_name=model_name,
            model_version=model_version,
            artifact_dir=artifact_dir,
            trained_at_utc="2026-03-06T00:00:00+00:00",
        ),
    )


def test_write_shadow_comparison_artifacts_emits_versioned_baseline_vs_challenger_report() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_shadow_compare_") as td:
        root = Path(td)
        evaluation_dir = root / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        baseline_scoreline = root / "baseline_scoreline"
        challenger_scoreline = root / "challenger_scoreline"
        baseline_corners = root / "baseline_corners"
        challenger_corners = root / "challenger_corners"
        baseline_anytime = root / "baseline_anytime"
        challenger_anytime = root / "challenger_anytime"
        _write_metadata(baseline_scoreline, family="scoreline", model_name="scoreline_v2", model_version="scoreline_champion_v1")
        _write_metadata(challenger_scoreline, family="scoreline", model_name="scoreline_v2", model_version="scoreline_candidate_v2")
        _write_metadata(baseline_corners, family="corners", model_name="corners_v2", model_version="corners_champion_v1")
        _write_metadata(challenger_corners, family="corners", model_name="corners_v2", model_version="corners_candidate_v2")
        _write_metadata(baseline_anytime, family="anytime", model_name="anytime_v2", model_version="anytime_champion_v1")
        _write_metadata(challenger_anytime, family="anytime", model_name="anytime_v2", model_version="anytime_candidate_v2")

        baseline_path = root / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "sources": {
                        "scoreline_holdout": str(baseline_scoreline / "metrics_holdout.json"),
                        "corners_holdout": str(baseline_corners / "metrics_holdout.json"),
                        "anytime_holdout": str(baseline_anytime / "metrics_holdout.json"),
                    }
                }
            ),
            encoding="utf-8",
        )
        (evaluation_dir / "promotion_registry.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "required_markets_total": 1,
                        "required_markets_passed": 1,
                    },
                    "decision": {
                        "status": "passed",
                        "recommended_scope": "promote_scoreline_only",
                    },
                    "markets": [
                        {
                            "market": "1x2_h",
                            "family": "scoreline",
                            "status": "passed",
                            "reasons": ["auc_improved", "log_loss_improved"],
                            "baseline": {"auc": 0.60, "brier": 0.220, "log_loss": 0.640, "ece": 0.030},
                            "effective_holdout": {"auc": 0.64, "brier": 0.210, "log_loss": 0.620, "ece": 0.024},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        artifacts = write_shadow_comparison_artifacts(
            flow_report={
                "baseline_path": str(baseline_path),
                "evaluation_dir": str(evaluation_dir),
                "scoreline_dir": str(challenger_scoreline),
                "corners_dir": str(challenger_corners),
                "anytime_dir": str(challenger_anytime),
                "promotion_recommendation": {
                    "scope": "required_markets_only",
                },
                "promotion_summary": {
                    "promotion_registry_path": str(evaluation_dir / "promotion_registry.json"),
                },
            },
            evaluation_dir=evaluation_dir,
        )

        assert artifacts is not None
        assert artifacts["market_count"] == 1
        shadow_json = json.loads((evaluation_dir / "shadow_comparison.json").read_text(encoding="utf-8"))
        assert shadow_json["candidate_artifacts"]["scoreline"]["model_version"] == "scoreline_candidate_v2"
        assert shadow_json["baseline_artifacts"]["scoreline"]["model_version"] == "scoreline_champion_v1"
        assert shadow_json["markets"][0]["deltas"]["auc"] == 0.040000000000000036
        markdown = (evaluation_dir / "shadow_comparison.md").read_text(encoding="utf-8")
        assert "Recommended scope: required_markets_only" in markdown
        assert "baseline `scoreline_champion_v1` vs challenger `scoreline_candidate_v2`" in markdown
        assert "| 1x2_h | scoreline | passed | 0.0400 | -0.0100 | -0.0200 | -0.0060 |" in markdown


def test_write_shadow_comparison_artifacts_uses_family_defaults_when_metadata_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_shadow_compare_defaults_") as td:
        root = Path(td)
        evaluation_dir = root / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = root / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "sources": {
                        "corners_holdout_path": str((root / "corners") / "metrics_holdout.json"),
                        "anytime_holdout_path": str((root / "anytime") / "metrics_holdout.json"),
                    }
                }
            ),
            encoding="utf-8",
        )
        (evaluation_dir / "promotion_registry.json").write_text(
            json.dumps({"summary": {}, "decision": {}, "markets": []}),
            encoding="utf-8",
        )

        artifacts = write_shadow_comparison_artifacts(
            flow_report={
                "baseline_path": str(baseline_path),
                "evaluation_dir": str(evaluation_dir),
                "corners_dir": str(root / "corners"),
                "anytime_dir": str(root / "anytime"),
            },
            evaluation_dir=evaluation_dir,
        )

        assert artifacts is not None
        shadow_json = json.loads((evaluation_dir / "shadow_comparison.json").read_text(encoding="utf-8"))
        assert shadow_json["candidate_artifacts"]["corners"]["model_version"] == "distribution_head_v1"
        assert shadow_json["candidate_artifacts"]["anytime"]["model_version"] == "markov_head_v1"