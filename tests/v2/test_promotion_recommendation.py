from __future__ import annotations

import json
from pathlib import Path
import tempfile

from src.modeling.v2.eval.promotion_recommendation import (
    build_promotion_recommendation,
    write_promotion_recommendation_artifacts,
)


def _flow_report(tmp_path: Path) -> dict[str, object]:
    return {
        "evaluation_dir": str(tmp_path),
        "evaluation_flow_report_path": str(tmp_path / "evaluation_flow_report.json"),
        "promotion_summary": {
            "promotion_registry_path": str(tmp_path / "promotion_registry.json"),
            "decision": {
                "status": "passed",
                "basis": "required_markets",
                "failed_markets": [],
                "missing_required_markets": [],
            },
            "summary": {
                "markets_total": 68,
                "markets_passed": 59,
                "markets_failed": 9,
                "required_markets_total": 14,
                "required_markets_passed": 14,
                "required_markets_failed": 0,
                "required_markets_missing": [],
            },
            "rules": {
                "promotion_policy_name": "scoreline_core_v1",
                "promotion_policy_path": "model_v2/promotion_policies/scoreline_core.yaml",
                "required_markets": ["1x2_h", "1x2_d"],
            },
        },
    }


def test_build_promotion_recommendation_recommends_core_only_when_tail_markets_fail() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_promotion_reco_core_only_") as td:
        tmp_path = Path(td)
        recommendation = build_promotion_recommendation(
            _flow_report(tmp_path),
            {"summary": {"failed_markets": ["mg_2_4", "ms_other_awaywin"]}},
        )
        assert recommendation is not None
        assert recommendation["recommendation_status"] == "promote"
        assert recommendation["recommended_scope"] == "required_markets_only"
        assert recommendation["follow_up_markets"] == ["mg_2_4", "ms_other_awaywin"]


def test_build_promotion_recommendation_holds_when_top_level_decision_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_promotion_reco_hold_") as td:
        tmp_path = Path(td)
        flow_report = _flow_report(tmp_path)
        flow_report["promotion_summary"]["decision"]["status"] = "failed"  # type: ignore[index]
        flow_report["promotion_summary"]["summary"]["required_markets_failed"] = 1  # type: ignore[index]
        recommendation = build_promotion_recommendation(
            flow_report,
            {"summary": {"failed_markets": ["1x2_h"]}},
        )
        assert recommendation is not None
        assert recommendation["recommendation_status"] == "hold"
        assert recommendation["recommended_scope"] == "none"


def test_write_promotion_recommendation_artifacts_writes_json_and_markdown() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_promotion_reco_write_") as td:
        tmp_path = Path(td)
        flow_report = _flow_report(tmp_path)
        (tmp_path / "promotion_registry.json").write_text(
            json.dumps({"summary": {"failed_markets": ["mg_2_4"]}}),
            encoding="utf-8",
        )
        result = write_promotion_recommendation_artifacts(flow_report=flow_report, evaluation_dir=tmp_path)
        assert result is not None
        json_path, md_path, recommendation = result
        assert json_path.exists()
        assert md_path.exists()
        assert recommendation["recommended_scope"] == "required_markets_only"
        assert "Checklist" in md_path.read_text(encoding="utf-8")