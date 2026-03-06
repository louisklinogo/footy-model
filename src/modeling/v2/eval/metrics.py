from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

METRIC_FIELDS: tuple[str, ...] = (
    "auc",
    "pr_auc",
    "accuracy",
    "brier",
    "log_loss",
    "ece",
)


def _clip_probabilities(values: Sequence[float] | np.ndarray, clip_eps: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), clip_eps, 1.0 - clip_eps)


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    p_true: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
    clip_eps: float = 0.001,
) -> float | None:
    y = np.asarray(y_true, dtype=float)
    if y.size <= 0:
        return None
    p = _clip_probabilities(p_true, clip_eps)
    n_bins = max(1, int(bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y))
    ece = 0.0
    for idx in range(n_bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        if not mask.any():
            continue
        ece += (float(mask.sum()) / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def binary_metric_values(
    y_true: Sequence[int] | np.ndarray,
    p_true: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
    clip_eps: float = 0.001,
) -> dict[str, float | int | None]:
    y = np.asarray(y_true, dtype=int)
    if y.size <= 0:
        return {
            "n": 0,
            "base_rate": None,
            "auc": None,
            "pr_auc": None,
            "accuracy": None,
            "brier": None,
            "log_loss": None,
            "ece": None,
        }

    p = _clip_probabilities(p_true, clip_eps)
    pred = (p >= 0.5).astype(int)
    base_rate = float(np.mean(y))
    has_both_classes = np.unique(y).size > 1

    auc: float | None
    pr_auc: float | None
    if has_both_classes:
        auc = float(roc_auc_score(y, p))
        pr_auc = float(average_precision_score(y, p))
    else:
        auc = None
        pr_auc = None

    return {
        "n": int(len(y)),
        "base_rate": base_rate,
        "auc": auc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p, bins=bins, clip_eps=clip_eps),
    }


def binary_classification_row(
    *,
    market: str,
    y_true: Sequence[int] | np.ndarray,
    p_true: Sequence[float] | np.ndarray,
    bins: int = 10,
    clip_eps: float = 0.001,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {"market": str(market)}
    if extra_fields:
        row.update(extra_fields)
    row.update(binary_metric_values(y_true, p_true, bins=bins, clip_eps=clip_eps))
    return row


def build_prediction_frame(
    *,
    label_key: str,
    label_value: str,
    fixture_ids: Sequence[Any] | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    p_true: Sequence[float] | np.ndarray,
    extra_columns: dict[str, Sequence[Any] | np.ndarray] | None = None,
    clip_eps: float = 0.001,
) -> pd.DataFrame:
    fixture_array = np.asarray(fixture_ids)
    y_array = np.asarray(y_true, dtype=int)
    p_array = _clip_probabilities(p_true, clip_eps)
    if not (len(fixture_array) == len(y_array) == len(p_array)):
        raise ValueError("fixture_ids, y_true, and p_true must have the same length")

    payload: dict[str, Any] = {
        label_key: np.full(len(y_array), str(label_value), dtype=object),
        "fixture_id": fixture_array,
        "y_true": y_array,
        "p_model": p_array,
    }
    for column, values in (extra_columns or {}).items():
        value_array = np.asarray(values)
        if len(value_array) != len(y_array):
            raise ValueError(f"Column '{column}' length does not match prediction rows")
        payload[column] = value_array
    return pd.DataFrame(payload)


def group_binary_classification_rows(
    *,
    market: str,
    group_values: Iterable[Any],
    y_true: Sequence[int] | np.ndarray,
    p_true: Sequence[float] | np.ndarray,
    group_key: str = "league_code",
    min_count: int = 1,
    bins: int = 10,
    clip_eps: float = 0.001,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {
            group_key: pd.Series(list(group_values), dtype="object"),
            "y_true": np.asarray(y_true, dtype=int),
            "p_true": _clip_probabilities(p_true, clip_eps),
        }
    )
    if frame.empty:
        return []

    rows: list[dict[str, Any]] = []
    for group_name, group in frame.groupby(group_key, sort=True, dropna=False):
        if len(group) < int(min_count):
            continue
        label = "__missing__" if pd.isna(group_name) else str(group_name)
        fields = dict(extra_fields or {})
        fields[group_key] = label
        rows.append(
            binary_classification_row(
                market=market,
                y_true=group["y_true"].to_numpy(dtype=int),
                p_true=group["p_true"].to_numpy(dtype=float),
                bins=bins,
                clip_eps=clip_eps,
                extra_fields=fields,
            )
        )
    return rows


def summarize_binary_metric_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...] = ("market",),
    metric_fields: Sequence[str] = METRIC_FIELDS,
) -> dict[str, Any]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    if frame.empty or any(key not in frame.columns for key in group_keys):
        return {}

    out: dict[str, Any] = {}
    grouped = frame.groupby(list(group_keys), sort=True, dropna=False)
    for raw_keys, group in grouped:
        keys = tuple(raw_keys) if isinstance(raw_keys, tuple) else (raw_keys,)
        n_series = (
            pd.to_numeric(group["n"], errors="coerce")
            if "n" in group.columns
            else pd.Series(dtype=float)
        )
        payload: dict[str, float | int | None] = {
            "rows": int(len(group)),
            "folds_used": int(group["fold"].nunique()) if "fold" in group.columns else int(len(group)),
            "n_total": int(n_series.fillna(0).sum()) if not n_series.empty else 0,
        }
        for field in ("base_rate", *metric_fields):
            if field not in group.columns:
                payload[f"{field}_mean"] = None
                payload[f"{field}_std"] = None
                continue
            series = pd.to_numeric(group[field], errors="coerce")
            payload[f"{field}_mean"] = float(series.mean()) if series.notna().any() else None
            payload[f"{field}_std"] = (
                float(series.std(ddof=0)) if int(series.notna().sum()) > 1 else 0.0
            ) if series.notna().any() else None

        cursor = out
        for key in keys[:-1]:
            cursor = cursor.setdefault(str(key), {})
        cursor[str(keys[-1])] = payload
    return out


def aggregate_market_summary(
    summary: dict[str, dict[str, float | int | None]],
    *,
    metric_fields: Sequence[str] = METRIC_FIELDS,
) -> dict[str, float | int | None]:
    if not summary:
        return {"markets_used": 0, "n_total": 0}

    out: dict[str, float | int | None] = {
        "markets_used": int(len(summary)),
        "n_total": int(
            sum(int(row.get("n_total") or 0) for row in summary.values() if isinstance(row, dict))
        ),
    }
    for field in metric_fields:
        values: list[float] = []
        for row in summary.values():
            if not isinstance(row, dict):
                continue
            value = row.get(f"{field}_mean")
            if isinstance(value, (int, float)):
                values.append(float(value))
        out[f"{field}_mean"] = float(np.mean(values)) if values else None
    return out