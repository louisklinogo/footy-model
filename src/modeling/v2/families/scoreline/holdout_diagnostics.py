from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.modeling.v2.eval.metrics import (
    binary_classification_row,
    group_binary_classification_rows,
    summarize_binary_metric_rows,
)

LOW_SCORELINE_SLICES: dict[str, tuple[int, int] | None] = {
    "exact_0_0": (0, 0),
    "exact_1_0": (1, 0),
    "exact_0_1": (0, 1),
    "exact_1_1": (1, 1),
    "draw": None,
}


def build_scoreline_slice_prediction_rows(
    *,
    fixture_ids: Sequence[Any] | np.ndarray,
    league_codes: Iterable[Any] | None,
    home_goals: Sequence[int] | np.ndarray,
    away_goals: Sequence[int] | np.ndarray,
    score_matrices: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    league_values = list(league_codes) if league_codes is not None else [None] * len(score_matrices)
    fixture_array = np.asarray(fixture_ids)
    home_array = np.asarray(home_goals, dtype=int)
    away_array = np.asarray(away_goals, dtype=int)
    if not (
        len(fixture_array) == len(home_array) == len(away_array) == len(score_matrices) == len(league_values)
    ):
        raise ValueError("Scoreline slice inputs must have matching lengths")

    rows: list[dict[str, Any]] = []
    for fixture_id, league_code, actual_home, actual_away, matrix in zip(
        fixture_array,
        league_values,
        home_array,
        away_array,
        score_matrices,
        strict=False,
    ):
        league_label = "__missing__" if pd.isna(league_code) else str(league_code)
        draw_probability = float(np.trace(matrix))
        for slice_name, target in LOW_SCORELINE_SLICES.items():
            if target is None:
                p_model = draw_probability
                y_true = int(actual_home == actual_away)
            else:
                home_idx, away_idx = target
                in_bounds = home_idx < matrix.shape[0] and away_idx < matrix.shape[1]
                p_model = float(matrix[home_idx, away_idx]) if in_bounds else 0.0
                y_true = int(actual_home == home_idx and actual_away == away_idx)
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "league_code": league_label,
                    "slice": slice_name,
                    "y_true": y_true,
                    "p_model": float(np.clip(p_model, 0.001, 0.999)),
                }
            )
    return rows


def summarize_scoreline_slice_prediction_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        return {}, {}

    frame = pd.DataFrame(rows)
    metric_rows: list[dict[str, Any]] = []
    metric_rows_by_league: list[dict[str, Any]] = []
    for slice_name, group in frame.groupby("slice", sort=True, dropna=False):
        metric_rows.append(
            binary_classification_row(
                market=str(slice_name),
                y_true=group["y_true"].to_numpy(dtype=int),
                p_true=group["p_model"].to_numpy(dtype=float),
                extra_fields={"slice": str(slice_name)},
            )
        )
        metric_rows_by_league.extend(
            group_binary_classification_rows(
                market=str(slice_name),
                group_values=group["league_code"],
                group_key="league_code",
                y_true=group["y_true"].to_numpy(dtype=int),
                p_true=group["p_model"].to_numpy(dtype=float),
                extra_fields={"slice": str(slice_name)},
            )
        )

    return (
        summarize_binary_metric_rows(metric_rows, group_keys=("slice",)),
        summarize_binary_metric_rows(metric_rows_by_league, group_keys=("slice", "league_code")),
    )