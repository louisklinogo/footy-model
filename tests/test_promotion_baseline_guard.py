from pathlib import Path

import pytest

from src.modeling.layer2_markets.market_outcome_calibrator import (
    validate_baseline_coverage,
)


def test_validate_baseline_coverage_raises_when_missing_and_strict() -> None:
    with pytest.raises(RuntimeError):
        validate_baseline_coverage(
            selected_market_codes=["o15", "u35", "c85"],
            baseline_metrics={"o15": {"auc": 0.6, "brier": 0.18}},
            baseline_path=Path("model_artifacts/market_models/metrics_walkforward.json"),
            allow_missing=False,
        )


def test_validate_baseline_coverage_returns_missing_when_allowed() -> None:
    missing = validate_baseline_coverage(
        selected_market_codes=["o15", "u35", "c85"],
        baseline_metrics={"o15": {"auc": 0.6, "brier": 0.18}},
        baseline_path=Path("model_artifacts/market_models/metrics_walkforward.json"),
        allow_missing=True,
    )
    assert missing == {"u35", "c85"}
