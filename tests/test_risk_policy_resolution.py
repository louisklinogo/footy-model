from __future__ import annotations

import json
from pathlib import Path
import tempfile

from src.modeling.evaluation.assess_prediction_risk import resolve_tradable_markets


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_resolve_tradable_markets_applies_required_gating_and_coverage() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gating_path = tmp_path / "market_gating.json"
        coverage_path = tmp_path / "coverage.json"

        _write_json(
            gating_path,
            {
                "markets": [
                    {"market_code": "o15", "eligible": True},
                    {"market_code": "u35", "eligible": True},
                    {"market_code": "dc_12", "eligible": False},
                    {"market_code": "c105", "eligible": False},
                ]
            },
        )
        _write_json(
            coverage_path,
            {
                "coverage_by_market": [
                    {"market_code": "o15", "coverage_pct": 0.70},
                    {"market_code": "u35", "coverage_pct": 0.55},
                    {"market_code": "dc_12", "coverage_pct": 0.95},
                    {"market_code": "c105", "coverage_pct": 0.95},
                ]
            },
        )

        policy = {
            "required_markets": ["o15", "u35", "dc_12", "c105"],
            "tradable_markets": ["o15", "u35", "dc_12", "c105"],
            "market_gating_path": str(gating_path),
            "odds_coverage_report_path": str(coverage_path),
            "odds_coverage_min_pct": 0.60,
        }

        resolved, summary = resolve_tradable_markets(policy)
        assert resolved == {"o15"}
        assert summary["coverage_applied"] is True
        assert summary["resolved_tradable_count"] == 1


def test_resolve_tradable_markets_without_coverage_uses_required_and_gating() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gating_path = tmp_path / "market_gating.json"
        _write_json(
            gating_path,
            {
                "markets": [
                    {"market_code": "o15", "eligible": True},
                    {"market_code": "u35", "eligible": True},
                    {"market_code": "dc_12", "eligible": False},
                ]
            },
        )

        policy = {
            "required_markets": ["o15", "u35", "dc_12"],
            "market_gating_path": str(gating_path),
            "odds_coverage_report_path": str(tmp_path / "missing.json"),
            "odds_coverage_min_pct": 0.60,
        }

        resolved, summary = resolve_tradable_markets(policy)
        assert resolved == {"o15", "u35"}
        assert summary["coverage_applied"] is False
