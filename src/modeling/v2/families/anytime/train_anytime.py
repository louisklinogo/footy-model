from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error


ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.layer2_markets import market_outcome_calibrator as legacy_calibrator
from src.modeling.v2.eval.metrics import (
    aggregate_market_summary,
    binary_classification_row,
    build_prediction_frame,
    group_binary_classification_rows,
    summarize_binary_metric_rows,
)
from src.modeling.v2.families.anytime.derive_markets import (
    derive_and_validate_anytime,
    derive_and_validate_anytime_phase_split,
)
from src.modeling.v2.families.anytime.features import build_anytime_features
from src.modeling.v2.io.artifact_identity import build_artifact_metadata, write_artifact_metadata
from src.modeling.v2.io.baseline_registry import load_scope_markets
from src.modeling.v2.io.contracts import load_feature_contract


DEFAULT_CONTRACT = ROOT_DIR / "model_v2" / "feature_contracts" / "anytime.yaml"
DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "anytime"
MODEL_NAME = "anytime_v2"
MODEL_VERSION = "markov_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v2 anytime family head.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional PIT dataset artifact (.parquet or .csv). Falls back to legacy fetch_dataset() when omitted.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="Path to anytime feature contract.",
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=DEFAULT_SCOPE,
        help="Path to v2 market scope.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for anytime artifacts.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=MODEL_VERSION,
        help="Persisted model_version to attach to this artifact set.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("histgb_poisson", "poisson_glm", "auto"),
        default="auto",
        help="Regressor family for home/away goal intensities. Use auto to select by walk-forward quality.",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=8,
        help="Markov state truncation cap.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on training rows (for smoke runs).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=6,
        help="Number of expanding walk-forward folds.",
    )
    parser.add_argument(
        "--min-fold-test-n",
        type=int,
        default=60,
        help="Minimum valid test rows per fold/market to score walk-forward metrics.",
    )
    parser.add_argument(
        "--path-version",
        type=str,
        choices=("constant", "phase_split"),
        default="constant",
        help="Anytime path derivation version.",
    )
    parser.add_argument(
        "--home-poisson-alpha",
        type=float,
        default=1.0,
        help="Poisson GLM regularization for home-goal heads.",
    )
    parser.add_argument(
        "--away-poisson-alpha",
        type=float,
        default=1.0,
        help="Poisson GLM regularization for away-goal heads.",
    )
    return parser.parse_args()


def _build_regressor(model_type: str, *, poisson_alpha: float = 1.0) -> Any:
    if model_type == "poisson_glm":
        return PoissonRegressor(alpha=max(0.0, float(poisson_alpha)), max_iter=1000)
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=40,
        random_state=42,
    )


def _load_training_frame(dataset_path: Path | None) -> tuple[pd.DataFrame, str]:
    if dataset_path is None:
        return legacy_calibrator.fetch_dataset(), "legacy_fetch_dataset"

    if not dataset_path.exists():
        raise RuntimeError(f"Dataset path does not exist: {dataset_path}")
    validation_path = dataset_path.parent / "pit_validation_report.json"
    if validation_path.exists():
        try:
            validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read PIT validation report: {validation_path}") from exc
        if validation_report.get("status") == "failed":
            raise RuntimeError(f"PIT dataset failed validation: {validation_path}")
    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(dataset_path)
    elif suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    else:
        raise RuntimeError(f"Unsupported dataset path format: {dataset_path}")
    return frame, str(dataset_path)


def _select_features(df: pd.DataFrame, contract_path: Path) -> list[str]:
    contract = load_feature_contract(contract_path)
    disabled = set(contract.disabled_features)
    required = [feat for feat in contract.required_features if feat not in disabled]
    optional = [feat for feat in contract.optional_features if feat not in disabled]
    missingness = [feat for feat in contract.missingness_indicators if feat not in disabled]
    missing_required = [feat for feat in required if feat not in df.columns]
    if missing_required:
        raise RuntimeError(
            "Missing required anytime contract features: " + ", ".join(sorted(missing_required))
        )
    selected = list(required)
    selected.extend(
        feat for feat in optional if feat in df.columns and feat not in selected
    )
    selected.extend(
        feat
        for feat in missingness
        if feat in df.columns and feat not in selected
    )
    return selected


def _fit_imputation(train_df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for feat in features:
        med = train_df[feat].median()
        medians[feat] = float(0.0 if pd.isna(med) else med)
    return medians


def _apply_imputation(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for feat, med in medians.items():
        out[feat] = out[feat].fillna(med)
    return out


def _prepare_phase_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["home_goals_p1"] = pd.to_numeric(out.get("home_goals_p1"), errors="coerce")
    out["away_goals_p1"] = pd.to_numeric(out.get("away_goals_p1"), errors="coerce")
    home_goals = pd.to_numeric(out["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(out["away_goals"], errors="coerce")
    out["home_goals_p2"] = (home_goals - out["home_goals_p1"]).where(
        out["home_goals_p1"].notna(), np.nan
    )
    out["away_goals_p2"] = (away_goals - out["away_goals_p1"]).where(
        out["away_goals_p1"].notna(), np.nan
    )
    out["home_goals_p2"] = pd.to_numeric(out["home_goals_p2"], errors="coerce").clip(lower=0.0)
    out["away_goals_p2"] = pd.to_numeric(out["away_goals_p2"], errors="coerce").clip(lower=0.0)
    return out


def _fit_goal_models(
    *,
    train_df: pd.DataFrame,
    features: list[str],
    model_type: str,
    path_version: str,
    home_poisson_alpha: float,
    away_poisson_alpha: float,
) -> dict[str, Any]:
    x_train = train_df[features]
    if path_version == "phase_split":
        models = {
            "home_p1": _build_regressor(model_type, poisson_alpha=home_poisson_alpha),
            "away_p1": _build_regressor(model_type, poisson_alpha=away_poisson_alpha),
            "home_p2": _build_regressor(model_type, poisson_alpha=home_poisson_alpha),
            "away_p2": _build_regressor(model_type, poisson_alpha=away_poisson_alpha),
        }
        models["home_p1"].fit(x_train, train_df["home_goals_p1"].astype(float))
        models["away_p1"].fit(x_train, train_df["away_goals_p1"].astype(float))
        models["home_p2"].fit(x_train, train_df["home_goals_p2"].astype(float))
        models["away_p2"].fit(x_train, train_df["away_goals_p2"].astype(float))
        return models

    models = {
        "home": _build_regressor(model_type, poisson_alpha=home_poisson_alpha),
        "away": _build_regressor(model_type, poisson_alpha=away_poisson_alpha),
    }
    models["home"].fit(x_train, train_df["home_goals"].astype(float))
    models["away"].fit(x_train, train_df["away_goals"].astype(float))
    return models


def _predict_goal_rates(
    *,
    models: dict[str, Any],
    x: pd.DataFrame,
    path_version: str,
) -> dict[str, np.ndarray]:
    if path_version == "phase_split":
        return {
            "home_p1": np.clip(models["home_p1"].predict(x), 0.01, 6.0),
            "away_p1": np.clip(models["away_p1"].predict(x), 0.01, 6.0),
            "home_p2": np.clip(models["home_p2"].predict(x), 0.01, 6.0),
            "away_p2": np.clip(models["away_p2"].predict(x), 0.01, 6.0),
        }

    return {
        "home": np.clip(models["home"].predict(x), 0.05, 8.0),
        "away": np.clip(models["away"].predict(x), 0.05, 8.0),
    }


def _derive_anytime_frame(
    *,
    preds: dict[str, np.ndarray],
    path_version: str,
    max_goals: int,
) -> pd.DataFrame:
    derived_rows: list[dict[str, float]] = []
    if path_version == "phase_split":
        for home_p1, away_p1, home_p2, away_p2 in zip(
            preds["home_p1"],
            preds["away_p1"],
            preds["home_p2"],
            preds["away_p2"],
            strict=True,
        ):
            derived_rows.append(
                derive_and_validate_anytime_phase_split(
                    lambda_home_p1=float(home_p1),
                    lambda_away_p1=float(away_p1),
                    lambda_home_p2=float(home_p2),
                    lambda_away_p2=float(away_p2),
                    max_goals=max(4, int(max_goals)),
                )
            )
        return pd.DataFrame(derived_rows)

    for home_mu, away_mu in zip(preds["home"], preds["away"], strict=True):
        derived_rows.append(
            derive_and_validate_anytime(
                lambda_home=float(home_mu),
                lambda_away=float(away_mu),
                max_goals=max(4, int(max_goals)),
            )
        )
    return pd.DataFrame(derived_rows)


def _market_metrics(market: str, y_true: np.ndarray, p_true: np.ndarray) -> dict[str, Any]:
    return binary_classification_row(market=market, y_true=y_true, p_true=p_true)


def _walkforward_ranges(total_rows: int, folds: int) -> list[tuple[int, int]]:
    if total_rows <= 0:
        return []
    steps = max(1, int(folds))
    points = np.linspace(0, total_rows, steps + 2, dtype=int)
    ranges: list[tuple[int, int]] = []
    for idx in range(1, len(points) - 1):
        train_end = int(points[idx])
        test_end = int(points[idx + 1])
        if train_end <= 0 or test_end <= train_end:
            continue
        ranges.append((train_end, test_end))
    return ranges


def _evaluate_walkforward(
    *,
    frame: pd.DataFrame,
    features: list[str],
    anytime_markets: set[str],
    model_type: str,
    max_goals: int,
    path_version: str,
    home_poisson_alpha: float,
    away_poisson_alpha: float,
    folds: int,
    min_fold_test_n: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, float | int | None]],
    list[dict[str, Any]],
]:
    ordered = frame.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(drop=True)
    fold_rows: list[dict[str, Any]] = []
    fold_league_rows: list[dict[str, Any]] = []
    for fold_idx, (train_end, test_end) in enumerate(
        _walkforward_ranges(len(ordered), folds), start=1
    ):
        train_df = ordered.iloc[:train_end].copy()
        test_df = ordered.iloc[train_end:test_end].copy()
        if train_df.empty or test_df.empty:
            continue

        imputation = _fit_imputation(train_df, features)
        train_df = _apply_imputation(train_df, imputation)
        test_df = _apply_imputation(test_df, imputation)

        x_test = test_df[features]
        models = _fit_goal_models(
            train_df=train_df,
            features=features,
            model_type=model_type,
            path_version=path_version,
            home_poisson_alpha=home_poisson_alpha,
            away_poisson_alpha=away_poisson_alpha,
        )
        preds = _predict_goal_rates(
            models=models,
            x=x_test,
            path_version=path_version,
        )
        pred_frame = _derive_anytime_frame(
            preds=preds,
            path_version=path_version,
            max_goals=max_goals,
        )

        for market in sorted(anytime_markets):
            if market not in pred_frame.columns:
                continue
            target_col = f"target_{market}"
            if target_col not in test_df.columns:
                continue
            valid = test_df[target_col].notna().to_numpy(dtype=bool)
            valid_n = int(valid.sum())
            if valid_n < int(min_fold_test_n):
                continue
            y_true = test_df[target_col].to_numpy(dtype=float)[valid].astype(int)
            p_true = np.clip(pred_frame[market].to_numpy(dtype=float)[valid], 0.001, 0.999)
            base_fields = {
                "fold": int(fold_idx),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
            }
            fold_rows.append(
                binary_classification_row(
                    market=market,
                    y_true=y_true,
                    p_true=p_true,
                    extra_fields=base_fields,
                )
            )
            if "league_code" in test_df.columns:
                fold_league_rows.extend(
                    group_binary_classification_rows(
                        market=market,
                        group_values=test_df.loc[valid, "league_code"],
                        y_true=y_true,
                        p_true=p_true,
                        extra_fields=base_fields,
                    )
                )

    if not fold_rows:
        return fold_rows, {}, fold_league_rows
    summary = summarize_binary_metric_rows(fold_rows)
    return fold_rows, summary, fold_league_rows


def _aggregate_walkforward_quality(
    summary: dict[str, dict[str, float | int | None]],
) -> dict[str, float | int | None]:
    return aggregate_market_summary(summary)


def _select_model_type(
    *,
    frame: pd.DataFrame,
    features: list[str],
    anytime_markets: set[str],
    max_goals: int,
    path_version: str,
    home_poisson_alpha: float,
    away_poisson_alpha: float,
    folds: int,
    min_fold_test_n: int,
) -> tuple[
    str,
    list[dict[str, Any]],
    dict[str, dict[str, float | int | None]],
    dict[str, dict[str, float | int | None]],
    list[dict[str, Any]],
]:
    candidates = ("histgb_poisson", "poisson_glm")
    collected: dict[
        str,
        tuple[
            list[dict[str, Any]],
            dict[str, dict[str, float | int | None]],
            dict[str, float | int | None],
            list[dict[str, Any]],
        ],
    ] = {}
    for model_type in candidates:
        folds_rows, summary, league_rows = _evaluate_walkforward(
            frame=frame,
            features=features,
            anytime_markets=anytime_markets,
            model_type=model_type,
            max_goals=max_goals,
            path_version=path_version,
            home_poisson_alpha=home_poisson_alpha,
            away_poisson_alpha=away_poisson_alpha,
            folds=folds,
            min_fold_test_n=min_fold_test_n,
        )
        collected[model_type] = (
            folds_rows,
            summary,
            _aggregate_walkforward_quality(summary),
            league_rows,
        )

    def _rank(model_type: str) -> tuple[float, float, int]:
        agg = collected[model_type][2]
        auc = agg.get("auc_mean")
        brier = agg.get("brier_mean")
        markets_used = int(agg.get("markets_used") or 0)
        auc_rank = float(auc) if isinstance(auc, (int, float)) else -1e9
        brier_rank = -float(brier) if isinstance(brier, (int, float)) else -1e9
        return auc_rank, brier_rank, markets_used

    selected = max(candidates, key=_rank)
    selected_rows, selected_summary, _selected_agg, selected_league_rows = collected[selected]
    model_selection_summary = {
        model_type: collected[model_type][2] for model_type in candidates
    }
    return selected, selected_rows, selected_summary, model_selection_summary, selected_league_rows


def main() -> None:
    args = parse_args()
    df, data_source = _load_training_frame(args.dataset_path)
    if df.empty:
        raise RuntimeError("No rows available for anytime training.")
    if args.max_rows is not None and args.max_rows > 0 and len(df) > args.max_rows:
        df = (
            df.sort_values(["match_datetime_utc", "fixture_id"])
            .tail(int(args.max_rows))
            .reset_index(drop=True)
        )

    if "prediction_time_utc" not in df.columns:
        df = legacy_calibrator.add_targets_and_derived(df)
    df["match_datetime_utc"] = pd.to_datetime(
        df["match_datetime_utc"], utc=True, errors="coerce"
    )
    if "home_goal_share" not in df.columns:
        df = build_anytime_features(df)
    path_version = str(args.path_version)
    if path_version == "phase_split":
        df = _prepare_phase_targets(df)
        df = df[df["home_goals_p1"].notna() & df["away_goals_p1"].notna()].copy()
        if df.empty:
            raise RuntimeError("No rows with phase-split goal labels available for anytime training.")

    features = _select_features(df, args.contract)
    scope_markets = set(load_scope_markets(args.scope))
    anytime_markets = {"h_1up", "a_1up", "h_2up", "a_2up"} & scope_markets
    model_type_requested = str(args.model_type)
    model_selection_summary: dict[str, dict[str, float | int | None]]
    walkforward_league_rows: list[dict[str, Any]]
    if model_type_requested == "auto":
        model_type_selected, walkforward_rows, walkforward_summary, model_selection_summary, walkforward_league_rows = (
            _select_model_type(
                frame=df,
                features=features,
                anytime_markets=anytime_markets,
                max_goals=int(args.max_goals),
                path_version=path_version,
                home_poisson_alpha=float(args.home_poisson_alpha),
                away_poisson_alpha=float(args.away_poisson_alpha),
                folds=int(args.folds),
                min_fold_test_n=int(args.min_fold_test_n),
            )
        )
    else:
        model_type_selected = model_type_requested
        walkforward_rows, walkforward_summary, walkforward_league_rows = _evaluate_walkforward(
            frame=df,
            features=features,
            anytime_markets=anytime_markets,
            model_type=model_type_selected,
            max_goals=int(args.max_goals),
            path_version=path_version,
            home_poisson_alpha=float(args.home_poisson_alpha),
            away_poisson_alpha=float(args.away_poisson_alpha),
            folds=int(args.folds),
            min_fold_test_n=int(args.min_fold_test_n),
        )
        model_selection_summary = {
            model_type_selected: _aggregate_walkforward_quality(walkforward_summary)
        }

    train_df, test_df = legacy_calibrator.split_time_respecting(df)
    imputation = _fit_imputation(train_df, features)
    train_df = _apply_imputation(train_df, imputation)
    test_df = _apply_imputation(test_df, imputation)

    x_test = test_df[features]
    models = _fit_goal_models(
        train_df=train_df,
        features=features,
        model_type=model_type_selected,
        path_version=path_version,
        home_poisson_alpha=float(args.home_poisson_alpha),
        away_poisson_alpha=float(args.away_poisson_alpha),
    )
    preds = _predict_goal_rates(
        models=models,
        x=x_test,
        path_version=path_version,
    )
    pred_frame = _derive_anytime_frame(
        preds=preds,
        path_version=path_version,
        max_goals=int(args.max_goals),
    )

    market_metrics: list[dict[str, Any]] = []
    holdout_league_rows: list[dict[str, Any]] = []
    holdout_prediction_frames: list[pd.DataFrame] = []
    for market in sorted(anytime_markets):
        if market not in pred_frame.columns:
            continue
        target_col = f"target_{market}"
        if target_col not in test_df.columns:
            continue
        valid = test_df[target_col].notna().to_numpy(dtype=bool)
        if int(valid.sum()) < 50:
            continue
        y_true = test_df[target_col].to_numpy(dtype=float)[valid].astype(int)
        p_true = np.clip(pred_frame[market].to_numpy(dtype=float)[valid], 0.001, 0.999)
        extra_columns: dict[str, Any] = {}
        if "league_code" in test_df.columns:
            extra_columns["league_code"] = (
                test_df.loc[valid, "league_code"].fillna("__missing__").astype(str).to_numpy()
            )
        if "match_datetime_utc" in test_df.columns:
            extra_columns["match_datetime_utc"] = (
                test_df.loc[valid, "match_datetime_utc"].astype(str).to_numpy()
            )
        holdout_prediction_frames.append(
            build_prediction_frame(
                label_key="market",
                label_value=market,
                fixture_ids=(
                    test_df.loc[valid, "fixture_id"].to_numpy()
                    if "fixture_id" in test_df.columns
                    else test_df.index.to_numpy()[valid]
                ),
                y_true=y_true,
                p_true=p_true,
                extra_columns=extra_columns,
            )
        )
        market_metrics.append(_market_metrics(market, y_true, p_true))
        if "league_code" in test_df.columns:
            holdout_league_rows.extend(
                group_binary_classification_rows(
                    market=market,
                    group_values=test_df.loc[valid, "league_code"],
                    y_true=y_true,
                    p_true=p_true,
                )
            )

    holdout_prediction_frame = (
        pd.concat(holdout_prediction_frames, ignore_index=True)
        if holdout_prediction_frames
        else pd.DataFrame(columns=["market", "fixture_id", "y_true", "p_model"])
    )

    diagnostics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": features,
        "model_name": MODEL_NAME,
        "model_version": str(args.model_version),
        "data_source": data_source,
        "model_type_requested": model_type_requested,
        "model_type_selected": model_type_selected,
        "path_version": path_version,
        "home_poisson_alpha": float(args.home_poisson_alpha),
        "away_poisson_alpha": float(args.away_poisson_alpha),
        "model_selection_summary": model_selection_summary,
        "max_goals": int(args.max_goals),
        "walkforward_folds": int(args.folds),
        "walkforward_min_fold_test_n": int(args.min_fold_test_n),
        "walkforward_markets_scored": int(len(walkforward_summary)),
        "walkforward_league_rows": int(len(walkforward_league_rows)),
        "holdout_league_rows": int(len(holdout_league_rows)),
        "holdout_prediction_rows": int(len(holdout_prediction_frame)),
        "trained_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    if path_version == "phase_split":
        diagnostics["home_goals_p1_mae"] = float(
            mean_absolute_error(test_df["home_goals_p1"].astype(float), preds["home_p1"])
        )
        diagnostics["away_goals_p1_mae"] = float(
            mean_absolute_error(test_df["away_goals_p1"].astype(float), preds["away_p1"])
        )
        diagnostics["home_goals_p2_mae"] = float(
            mean_absolute_error(test_df["home_goals_p2"].astype(float), preds["home_p2"])
        )
        diagnostics["away_goals_p2_mae"] = float(
            mean_absolute_error(test_df["away_goals_p2"].astype(float), preds["away_p2"])
        )
    else:
        diagnostics["home_goals_mae"] = float(
            mean_absolute_error(test_df["home_goals"].astype(float), preds["home"])
        )
        diagnostics["away_goals_mae"] = float(
            mean_absolute_error(test_df["away_goals"].astype(float), preds["away"])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_metadata(
        args.output_dir,
        build_artifact_metadata(
            family="anytime",
            model_name=MODEL_NAME,
            model_version=str(args.model_version),
            artifact_dir=args.output_dir,
            trained_at_utc=str(diagnostics["trained_at_utc"]),
            extra={
                "model_type_selected": model_type_selected,
                "path_version": path_version,
                "data_source": data_source,
            },
        ),
    )
    if path_version == "phase_split":
        joblib.dump(models["home_p1"], args.output_dir / "home_goals_p1_model.pkl")
        joblib.dump(models["away_p1"], args.output_dir / "away_goals_p1_model.pkl")
        joblib.dump(models["home_p2"], args.output_dir / "home_goals_p2_model.pkl")
        joblib.dump(models["away_p2"], args.output_dir / "away_goals_p2_model.pkl")
    else:
        joblib.dump(models["home"], args.output_dir / "home_goals_model.pkl")
        joblib.dump(models["away"], args.output_dir / "away_goals_model.pkl")
    (args.output_dir / "features.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )
    (args.output_dir / "imputation.json").write_text(
        json.dumps({"global_medians": imputation}, indent=2), encoding="utf-8"
    )
    (args.output_dir / "model_config.json").write_text(
        json.dumps(
            {"max_goals": int(args.max_goals), "path_version": path_version},
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "metrics_holdout.json").write_text(
        json.dumps(market_metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics_holdout_by_league.json").write_text(
        json.dumps(holdout_league_rows, indent=2), encoding="utf-8"
    )
    holdout_prediction_frame.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    (args.output_dir / "metrics_walkforward_folds.json").write_text(
        json.dumps(walkforward_rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics_walkforward_folds_by_league.json").write_text(
        json.dumps(walkforward_league_rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics_walkforward.json").write_text(
        json.dumps(walkforward_summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "training_report.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    print(f"Saved anytime artifacts to {args.output_dir}")
    if path_version == "phase_split":
        print(
            "Phase Goal MAE "
            f"home_p1={diagnostics['home_goals_p1_mae']:.3f} "
            f"away_p1={diagnostics['away_goals_p1_mae']:.3f} "
            f"home_p2={diagnostics['home_goals_p2_mae']:.3f} "
            f"away_p2={diagnostics['away_goals_p2_mae']:.3f}"
        )
    else:
        print(
            "Goal MAE "
            f"home={diagnostics['home_goals_mae']:.3f} away={diagnostics['away_goals_mae']:.3f}"
        )
    if model_type_requested == "auto":
        print(f"Auto-selected model_type={model_type_selected}")
    for row in market_metrics:
        print(
            f"{row['market']}: AUC={row['auc']}, Brier={row['brier']:.4f}, "
            f"Acc={row['accuracy']:.3f}, n={row['n']}"
        )


if __name__ == "__main__":
    main()
