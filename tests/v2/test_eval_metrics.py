from __future__ import annotations

import numpy as np
import pytest

from src.modeling.v2.eval.metrics import (
    aggregate_market_summary,
    binary_classification_row,
    build_prediction_frame,
    group_binary_classification_rows,
    summarize_binary_metric_rows,
)


def test_binary_classification_row_emits_extended_metrics() -> None:
    row = binary_classification_row(
        market="o15",
        y_true=np.array([0, 1, 1, 0, 1]),
        p_true=np.array([0.1, 0.9, 0.8, 0.2, 0.6]),
        extra_fields={"fold": 1},
    )
    assert row["market"] == "o15"
    assert row["fold"] == 1
    assert row["n"] == 5
    assert row["auc"] == pytest.approx(1.0)
    assert row["pr_auc"] == pytest.approx(1.0)
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["brier"] < 0.1
    assert row["log_loss"] < 0.5
    assert row["ece"] >= 0.0


def test_group_and_summary_rows_support_nested_grouping() -> None:
    rows = group_binary_classification_rows(
        market="o15",
        group_values=["EPL", "EPL", "EPL", "SA", "SA", "SA"],
        y_true=np.array([0, 1, 1, 0, 0, 1]),
        p_true=np.array([0.2, 0.7, 0.8, 0.1, 0.4, 0.9]),
        extra_fields={"fold": 2},
        min_count=3,
    )
    summary = summarize_binary_metric_rows(rows, group_keys=("market", "league_code"))
    assert set(summary["o15"].keys()) == {"EPL", "SA"}
    assert summary["o15"]["EPL"]["folds_used"] == 1
    assert summary["o15"]["EPL"]["n_total"] == 3
    assert summary["o15"]["SA"]["n_total"] == 3


def test_summarize_binary_metric_rows_uses_plain_market_keys_for_default_grouping() -> None:
    summary = summarize_binary_metric_rows(
        [
            {"market": "o15", "fold": 1, "auc": 0.6, "brier": 0.2, "n": 50},
            {"market": "o15", "fold": 2, "auc": 0.7, "brier": 0.1, "n": 60},
        ]
    )
    assert "o15" in summary
    assert "('o15',)" not in summary
    assert summary["o15"]["folds_used"] == 2
    assert summary["o15"]["n_total"] == 110


def test_aggregate_market_summary_combines_auc_and_brier() -> None:
    agg = aggregate_market_summary(
        {
            "o15": {"auc_mean": 0.61, "brier_mean": 0.19, "n_total": 100},
            "u35": {"auc_mean": 0.57, "brier_mean": 0.23, "n_total": 120},
        }
    )
    assert agg["markets_used"] == 2
    assert agg["auc_mean"] == pytest.approx(0.59)
    assert agg["brier_mean"] == pytest.approx(0.21)
    assert agg["n_total"] == 220


def test_build_prediction_frame_preserves_row_columns() -> None:
    frame = build_prediction_frame(
        label_key="market",
        label_value="o15",
        fixture_ids=[101, 102],
        y_true=np.array([0, 1]),
        p_true=np.array([0.0, 1.0]),
        extra_columns={"league_code": ["EPL", "SA"]},
    )
    assert list(frame.columns) == ["market", "fixture_id", "y_true", "p_model", "league_code"]
    assert frame.loc[0, "p_model"] == pytest.approx(0.001)
    assert frame.loc[1, "p_model"] == pytest.approx(0.999)
    assert frame.loc[1, "league_code"] == "SA"
