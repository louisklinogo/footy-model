from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.modeling.v2.eval.metrics import binary_metric_values

CLIP_EPS = 0.001
ONE_SIDED_ANCHOR_MARKETS: frozenset[str] = frozenset(
    {
        "1x2_h",
        "dc_x2",
        "ah2_home_m05",
        "ah2_away_p05",
        "ah2_home_m15",
        "ah2_away_p15",
        "eh3_0_1_home",
        "eh3_0_1_away",
    }
)


def _clip_probabilities(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), CLIP_EPS, 1.0 - CLIP_EPS)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = _clip_probabilities(values)
    return np.log(clipped / (1.0 - clipped))


def _apply_one_sided_anchor_transform(
    values: pd.Series | np.ndarray | list[float],
    *,
    anchor: float,
    alpha: float,
) -> np.ndarray:
    clipped = _clip_probabilities(values)
    anchor_value = float(np.clip(float(anchor), CLIP_EPS, 1.0 - CLIP_EPS))
    alpha_value = float(np.clip(float(alpha), 0.0, 1.0))
    if anchor_value <= 0.5:
        adjusted = clipped - alpha_value * np.maximum(0.0, clipped - anchor_value)
    else:
        adjusted = clipped + alpha_value * np.maximum(0.0, anchor_value - clipped)
    return _clip_probabilities(adjusted)


def order_calibration_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy().reset_index(drop=True)
    ordered["__row_order"] = np.arange(len(ordered), dtype=int)
    sort_cols = ["__row_order"]
    if "match_datetime_utc" in ordered.columns:
        ordered["__match_datetime"] = pd.to_datetime(
            ordered["match_datetime_utc"], errors="coerce", utc=True
        )
        sort_cols = ["__match_datetime", "__row_order"]
        if "fixture_id" in ordered.columns:
            sort_cols = ["__match_datetime", "fixture_id", "__row_order"]
    ordered = ordered.sort_values(sort_cols, kind="stable", na_position="last")
    return ordered.drop(columns=["__row_order", "__match_datetime"], errors="ignore").reset_index(drop=True)


def split_calibration_frame(
    frame: pd.DataFrame,
    *,
    fit_fraction: float,
    min_fit_rows: int,
    min_eval_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = order_calibration_frame(frame)
    total = int(len(ordered))
    if total < int(min_fit_rows) + int(min_eval_rows):
        return ordered.iloc[0:0].copy(), ordered.iloc[0:0].copy()
    fit_n = max(int(min_fit_rows), int(total * float(fit_fraction)))
    fit_n = min(fit_n, total - int(min_eval_rows))
    if fit_n < int(min_fit_rows) or (total - fit_n) < int(min_eval_rows):
        return ordered.iloc[0:0].copy(), ordered.iloc[0:0].copy()
    return ordered.iloc[:fit_n].copy(), ordered.iloc[fit_n:].copy()


def fit_binary_calibrator(
    method: str,
    *,
    p_model: pd.Series | np.ndarray | list[float],
    y_true: pd.Series | np.ndarray | list[int],
    alpha: float | None = None,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    if np.unique(y).size < 2:
        raise ValueError("Calibration fit requires both classes")
    if method == "identity":
        return {"method": "identity"}
    if method == "one_sided_anchor":
        anchor = float(np.mean(y))
        return {
            "method": method,
            "anchor": anchor,
            "alpha": float(0.0 if alpha is None else alpha),
        }
    if method == "sigmoid":
        model = LogisticRegression(solver="lbfgs")
        model.fit(_logit(_clip_probabilities(p_model)).reshape(-1, 1), y)
        return {"method": method, "model": model}
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=CLIP_EPS, y_max=1.0 - CLIP_EPS)
        model.fit(_clip_probabilities(p_model), y)
        return {"method": method, "model": model}
    raise ValueError(f"Unsupported calibration method: {method}")


def apply_binary_calibrator(
    calibrator: dict[str, Any],
    p_model: pd.Series | np.ndarray | list[float],
) -> np.ndarray:
    method = str(calibrator.get("method") or "identity")
    clipped = _clip_probabilities(p_model)
    if method == "identity":
        return clipped
    if method == "one_sided_anchor":
        return _apply_one_sided_anchor_transform(
            clipped,
            anchor=float(calibrator.get("anchor") or 0.5),
            alpha=float(calibrator.get("alpha") or 0.0),
        )
    model = calibrator.get("model")
    if method == "sigmoid":
        return _clip_probabilities(model.predict_proba(_logit(clipped).reshape(-1, 1))[:, 1])
    if method == "isotonic":
        return _clip_probabilities(model.predict(clipped))
    raise ValueError(f"Unsupported calibration method: {method}")


def _to_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float, np.integer, np.floating)) else None


def _candidate_beats_raw(raw_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> bool:
    raw_brier = _to_float(raw_metrics.get("brier"))
    cand_brier = _to_float(candidate_metrics.get("brier"))
    if raw_brier is not None and cand_brier is not None and cand_brier > raw_brier + 1e-9:
        return False
    for field in ("brier", "log_loss", "ece"):
        raw_value = _to_float(raw_metrics.get(field))
        cand_value = _to_float(candidate_metrics.get(field))
        if raw_value is not None and cand_value is not None and cand_value < raw_value - 1e-6:
            return True
    return False


def _metric_rank(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    brier = _to_float(metrics.get("brier"))
    log_loss = _to_float(metrics.get("log_loss"))
    ece = _to_float(metrics.get("ece"))
    auc = _to_float(metrics.get("auc"))
    return (
        brier if brier is not None else float("inf"),
        log_loss if log_loss is not None else float("inf"),
        ece if ece is not None else float("inf"),
        -auc if auc is not None else float("inf"),
    )


def evaluate_market_calibration(
    frame: pd.DataFrame,
    *,
    fit_fraction: float,
    min_fit_rows: int,
    min_eval_rows: int,
    max_auc_drop: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    filtered = frame.loc[frame["y_true"].notna() & frame["p_model"].notna()].copy()
    report: dict[str, Any] = {
        "rows_total": int(len(filtered)),
        "fit_rows": 0,
        "eval_rows": 0,
        "status": "not_attempted",
        "selected_method": "identity",
        "candidate_metrics": {},
        "raw_eval_metrics": None,
        "calibrated_eval_metrics": None,
    }
    fit_df, eval_df = split_calibration_frame(
        filtered,
        fit_fraction=fit_fraction,
        min_fit_rows=min_fit_rows,
        min_eval_rows=min_eval_rows,
    )
    if fit_df.empty or eval_df.empty:
        report["status"] = "insufficient_rows"
        return report, None

    report["fit_rows"] = int(len(fit_df))
    report["eval_rows"] = int(len(eval_df))
    raw_eval = binary_metric_values(eval_df["y_true"], eval_df["p_model"])
    report["raw_eval_metrics"] = raw_eval
    report["candidate_metrics"]["identity"] = raw_eval

    fit_classes = pd.Series(fit_df["y_true"].to_numpy(dtype=int)).value_counts()
    eval_classes = pd.Series(eval_df["y_true"].to_numpy(dtype=int)).value_counts()
    if len(fit_classes) < 2 or len(eval_classes) < 2 or int(fit_classes.min()) < 3:
        report["status"] = "insufficient_class_diversity"
        return report, None

    best_method = "identity"
    best_metrics = raw_eval
    best_calibrator: dict[str, Any] | None = None
    raw_auc = _to_float(raw_eval.get("auc"))
    candidate_methods = ["sigmoid", "isotonic"]
    market_name: str | None = None
    if "market" in filtered.columns:
        market_values = filtered["market"].dropna().astype(str).unique().tolist()
        if len(market_values) == 1:
            market_name = market_values[0]
    if market_name in ONE_SIDED_ANCHOR_MARKETS:
        candidate_methods.append("one_sided_anchor")
    for method in candidate_methods:
        try:
            if method == "one_sided_anchor":
                anchor = float(np.mean(fit_df["y_true"].to_numpy(dtype=int)))
                best_method_metrics: dict[str, Any] | None = None
                best_method_calibrator: dict[str, Any] | None = None
                best_method_rank: tuple[float, float, float, float] | None = None
                for alpha in np.linspace(0.05, 1.0, 20):
                    calibrator = fit_binary_calibrator(
                        method,
                        p_model=fit_df["p_model"],
                        y_true=fit_df["y_true"],
                        alpha=float(alpha),
                    )
                    calibrated_eval = apply_binary_calibrator(calibrator, eval_df["p_model"])
                    metrics = binary_metric_values(eval_df["y_true"], calibrated_eval)
                    candidate_auc = _to_float(metrics.get("auc"))
                    auc_drop = (
                        float(raw_auc - candidate_auc)
                        if raw_auc is not None and candidate_auc is not None
                        else 0.0
                    )
                    if auc_drop > float(max_auc_drop):
                        continue
                    if not _candidate_beats_raw(raw_eval, metrics):
                        continue
                    rank = _metric_rank(metrics)
                    if best_method_rank is None or rank < best_method_rank:
                        best_method_rank = rank
                        best_method_calibrator = calibrator
                        best_method_metrics = {
                            **metrics,
                            "anchor": anchor,
                            "alpha": float(alpha),
                        }
                report["candidate_metrics"][method] = best_method_metrics or {
                    "anchor": anchor,
                    "status": "no_improving_alpha",
                }
                if best_method_calibrator is None or best_method_metrics is None:
                    continue
                calibrator = best_method_calibrator
                metrics = best_method_metrics
            else:
                calibrator = fit_binary_calibrator(method, p_model=fit_df["p_model"], y_true=fit_df["y_true"])
                calibrated_eval = apply_binary_calibrator(calibrator, eval_df["p_model"])
                metrics = binary_metric_values(eval_df["y_true"], calibrated_eval)
                report["candidate_metrics"][method] = metrics
            report["candidate_metrics"][method] = metrics
            candidate_auc = _to_float(metrics.get("auc"))
            auc_drop = (
                float(raw_auc - candidate_auc)
                if raw_auc is not None and candidate_auc is not None
                else 0.0
            )
            if auc_drop > float(max_auc_drop):
                continue
            if not _candidate_beats_raw(raw_eval, metrics):
                continue
            if _metric_rank(metrics) < _metric_rank(best_metrics):
                best_method = method
                best_metrics = metrics
                best_calibrator = calibrator
        except Exception as exc:
            report["candidate_metrics"][method] = {"error": str(exc)}

    if best_method == "identity" or best_calibrator is None:
        report["status"] = "identity_selected"
        return report, None

    report["status"] = "calibrated"
    report["selected_method"] = best_method
    report["calibrated_eval_metrics"] = best_metrics
    report["improvement"] = {
        "brier_delta": _to_float(best_metrics.get("brier")) - (_to_float(raw_eval.get("brier")) or 0.0),
        "log_loss_delta": _to_float(best_metrics.get("log_loss")) - (_to_float(raw_eval.get("log_loss")) or 0.0),
        "ece_delta": _to_float(best_metrics.get("ece")) - (_to_float(raw_eval.get("ece")) or 0.0),
        "auc_delta": (_to_float(best_metrics.get("auc")) or 0.0) - (_to_float(raw_eval.get("auc")) or 0.0),
    }
    return report, best_calibrator
