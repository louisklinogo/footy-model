from __future__ import annotations

import numpy as np

from src.modeling.evaluation.diagnose_prediction_slices import (
    _bucket_edge,
    _bucket_kickoff_hours,
    _bucket_odds,
    _compute_ece,
    _safe_auc,
    _weak_segments,
)


def test_bucket_helpers_cover_boundaries() -> None:
    assert _bucket_odds(None) == "missing"
    assert _bucket_odds(1.5) == "<=1.50"
    assert _bucket_odds(1.51) == "1.51-1.80"
    assert _bucket_odds(3.6) == ">3.50"

    assert _bucket_kickoff_hours(None) == "missing"
    assert _bucket_kickoff_hours(6.0) == "<=6h"
    assert _bucket_kickoff_hours(24.0) == "6-24h"
    assert _bucket_kickoff_hours(200.0) == ">168h"

    assert _bucket_edge(None) == "missing"
    assert _bucket_edge(-0.001) == "<0%"
    assert _bucket_edge(0.0) == "0-2%"
    assert _bucket_edge(0.02) == "2-4%"
    assert _bucket_edge(0.055) == "4-6%"
    assert _bucket_edge(0.2) == "6%+"


def test_safe_auc_and_ece_work_for_binary_data() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    p = np.array([0.2, 0.8, 0.3, 0.7], dtype=float)

    auc = _safe_auc(y, p)
    assert isinstance(auc, float)
    assert auc > 0.9

    ece = _compute_ece(y_true=y, p_pred=p, n_bins=4)
    assert isinstance(ece, float)
    assert ece >= 0.0


def test_weak_segments_filters_by_samples_and_class_balance() -> None:
    rows = [
        {
            "segment": "good",
            "n": 200,
            "positives": 120,
            "negatives": 80,
            "auc": 0.64,
            "brier_mean": 0.21,
        },
        {
            "segment": "weak",
            "n": 220,
            "positives": 110,
            "negatives": 110,
            "auc": 0.51,
            "brier_mean": 0.24,
        },
        {
            "segment": "too_small",
            "n": 80,
            "positives": 40,
            "negatives": 40,
            "auc": 0.49,
            "brier_mean": 0.25,
        },
    ]
    out = _weak_segments(rows, min_samples=120, top_k=5)
    assert len(out) == 2
    assert out[0]["segment"] == "weak"
