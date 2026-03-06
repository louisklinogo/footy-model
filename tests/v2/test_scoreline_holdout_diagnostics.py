from __future__ import annotations

import numpy as np

from src.modeling.v2.families.scoreline.holdout_diagnostics import (
    build_scoreline_slice_prediction_rows,
    summarize_scoreline_slice_prediction_rows,
)


def test_scoreline_slice_rows_and_summary_cover_low_score_failures() -> None:
    rows = build_scoreline_slice_prediction_rows(
        fixture_ids=[1, 2],
        league_codes=["EPL", "EPL"],
        home_goals=[0, 1],
        away_goals=[0, 1],
        score_matrices=[
            np.array([[0.30, 0.10], [0.20, 0.40]], dtype=float),
            np.array([[0.10, 0.20], [0.15, 0.55]], dtype=float),
        ],
    )
    assert len(rows) == 10
    slice_summary, by_league = summarize_scoreline_slice_prediction_rows(rows)
    assert slice_summary["exact_0_0"]["n_total"] == 2
    assert slice_summary["draw"]["base_rate_mean"] == 1.0
    assert by_league["exact_1_1"]["EPL"]["n_total"] == 2