from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from src.modeling.v2.io.baseline_registry import (
    load_baseline_metrics,
    load_scope_markets,
    validate_baseline_coverage,
)


def test_load_scope_markets_from_yaml_fallback() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_scope_") as td:
        scope_path = Path(td) / "market_scope.yaml"
        scope_path.write_text(
            "\n".join(
                [
                    "version: 1",
                    "markets:",
                    "  - o15",
                    "  - u35",
                    "  - dc_12",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        markets = load_scope_markets(scope_path)
        assert markets == ["o15", "u35", "dc_12"]


def test_load_baseline_metrics_supports_metrics_wrapper() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_baseline_") as td:
        baseline_path = Path(td) / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "metrics": [
                        {"market_code": "o15", "auc": 0.61, "brier": 0.18},
                        {"market_code": "u35", "auc": 0.57, "brier": 0.21},
                    ],
                }
            ),
            encoding="utf-8",
        )

        out = load_baseline_metrics(baseline_path)
        assert set(out.keys()) == {"o15", "u35"}
        assert out["o15"].auc == 0.61
        assert out["u35"].brier == 0.21


def test_validate_baseline_coverage_strict_raises() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_guard_strict_") as td:
        baseline_path = Path(td) / "baseline.json"
        baseline_path.write_text(
            json.dumps([{"market_code": "o15", "auc": 0.61, "brier": 0.18}]),
            encoding="utf-8",
        )
        metrics = load_baseline_metrics(baseline_path)

        with pytest.raises(RuntimeError):
            validate_baseline_coverage(
                scope_markets=["o15", "u35", "dc_12"],
                baseline_metrics=metrics,
                strict=True,
                baseline_path=baseline_path,
            )


def test_validate_baseline_coverage_non_strict_returns_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_guard_non_strict_") as td:
        baseline_path = Path(td) / "baseline.json"
        baseline_path.write_text(
            json.dumps([{"market_code": "o15", "auc": 0.61, "brier": 0.18}]),
            encoding="utf-8",
        )
        metrics = load_baseline_metrics(baseline_path)
        missing = validate_baseline_coverage(
            scope_markets=["o15", "u35", "dc_12"],
            baseline_metrics=metrics,
            strict=False,
            baseline_path=baseline_path,
        )
        assert missing == ["dc_12", "u35"]
