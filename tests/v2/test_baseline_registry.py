from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from src.modeling.v2.io.baseline_registry import (
    load_baseline_metrics,
    load_scope_markets,
    validate_market_presence,
    validate_baseline_coverage,
    validate_markets_in_scope,
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


def test_load_baseline_metrics_expands_legacy_handicap_rows_to_canonical_v2_markets() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_baseline_expand_") as td:
        baseline_path = Path(td) / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "metrics": [
                        {"market_code": "ah_h05", "auc": 0.61, "brier": 0.18},
                        {"market_code": "ah_a05", "auc": 0.57, "brier": 0.21},
                        {"market_code": "ah_h15", "auc": 0.66, "brier": 0.15},
                        {"market_code": "ah_a15", "auc": 0.64, "brier": 0.16},
                        {"market_code": "dc_x2", "auc": 0.63, "brier": 0.20},
                        {"market_code": "dc_1x", "auc": 0.62, "brier": 0.19},
                        {"market_code": "eh_h1", "auc": 0.67, "brier": 0.14},
                        {"market_code": "eh_a1", "auc": 0.65, "brier": 0.17},
                    ],
                }
            ),
            encoding="utf-8",
        )

        out = load_baseline_metrics(baseline_path)

        assert out["ah2_home_m05"].auc == 0.61
        assert out["ah2_away_p05"].brier == 0.18
        assert out["ah2_away_m05"].auc == 0.57
        assert out["ah2_home_p05"].brier == 0.21
        assert out["ah2_home_m15"].auc == 0.66
        assert out["ah2_away_p15"].brier == 0.15
        assert out["ah2_away_m15"].auc == 0.64
        assert out["ah2_home_p15"].brier == 0.16
        assert out["eh3_0_1_home"].auc == 0.67
        assert out["eh3_0_1_away"].auc == 0.63
        assert out["eh3_1_0_home"].auc == 0.62
        assert out["eh3_1_0_away"].auc == 0.65


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


def test_validate_market_presence_allows_explicit_null_metric_rows() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_guard_null_metric_presence_") as td:
        baseline_path = Path(td) / "baseline.json"
        baseline_path.write_text(
            json.dumps(
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
                }
            ),
            encoding="utf-8",
        )
        metrics = load_baseline_metrics(baseline_path)

        missing = validate_market_presence(
            market_list=["eh3_0_1_draw"],
            baseline_metrics=metrics,
            strict=True,
            baseline_path=baseline_path,
            market_set_name="required",
        )

        assert missing == []


def test_validate_markets_in_scope_strict_raises_for_required_market_outside_scope() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_scope_required_guard_") as td:
        scope_path = Path(td) / "market_scope.yaml"
        scope_path.write_text("markets:\n  - o15\n  - u35\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Required markets are not present in scope"):
            validate_markets_in_scope(
                selected_markets=["eh3_0_1_draw"],
                scope_markets=load_scope_markets(scope_path),
                strict=True,
                scope_path=scope_path,
                market_set_name="required",
            )


def test_active_scope_uses_canonical_handicap_markets() -> None:
    markets = load_scope_markets(Path("model_v2/market_scope.yaml"))

    assert "ah2_home_m05" in markets
    assert "ah2_away_p15" in markets
    assert "eh3_0_1_draw" in markets
    assert "eh3_1_0_away" in markets
    assert "ah_h05" not in markets
    assert "eh_h1" not in markets
