from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from math import factorial
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
    group_binary_classification_rows,
    summarize_binary_metric_rows,
)
from src.modeling.v2.families.scoreline.derive_markets import derive_and_validate
from src.modeling.v2.io.artifact_identity import build_artifact_metadata, write_artifact_metadata
from src.modeling.v2.io.baseline_registry import load_scope_markets
from src.modeling.v2.io.contracts import load_feature_contract


DEFAULT_CONTRACT = ROOT_DIR / "model_v2" / "feature_contracts" / "scoreline.yaml"
DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
DEFAULT_OUT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "scoreline"
MODEL_NAME = "scoreline_v2"
MODEL_VERSION = "poisson_head_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v2 scoreline family head.")
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="Path to scoreline feature contract.",
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
        help="Output directory for scoreline artifacts.",
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
        help="Regressor family for home/away goals. Use auto to select by walk-forward quality.",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=10,
        help="Scoreline matrix truncation cap.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on training rows (for fast smoke runs).",
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
        default=80,
        help="Minimum valid test rows per fold/market to score walk-forward metrics.",
    )
    return parser.parse_args()


def _build_regressor(model_type: str) -> Any:
    if model_type == "poisson_glm":
        return PoissonRegressor(alpha=1.0, max_iter=500)
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=40,
        random_state=42,
    )


def _poisson_pmf(lmbda: float, k: int) -> float:
    lmbda = max(1e-6, float(lmbda))
    return float(np.exp(-lmbda) * (lmbda**k) / float(factorial(k)))


def _score_matrix_independent_poisson(
    lambda_home: float, lambda_away: float, max_goals: int
) -> np.ndarray:
    home = np.array([_poisson_pmf(lambda_home, g) for g in range(max_goals + 1)], dtype=float)
    away = np.array([_poisson_pmf(lambda_away, g) for g in range(max_goals + 1)], dtype=float)
    mat = np.outer(home, away)
    total = float(mat.sum())
    if total <= 0.0:
        return np.full((max_goals + 1, max_goals + 1), 1.0 / ((max_goals + 1) ** 2))
    return mat / total


def _select_features(df: pd.DataFrame, contract_path: Path) -> list[str]:
    contract = load_feature_contract(contract_path)
    disabled = set(contract.disabled_features)
    required = [feat for feat in contract.required_features if feat not in disabled]
    optional = [feat for feat in contract.optional_features if feat not in disabled]
    missingness = [feat for feat in contract.missingness_indicators if feat not in disabled]
    missing_required = [feat for feat in required if feat not in df.columns]
    if missing_required:
        missing_csv = ", ".join(sorted(missing_required))
        raise RuntimeError(f"Missing required scoreline contract features: {missing_csv}")

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
        if pd.isna(med):
            med = 0.0
        medians[feat] = float(med)
    return medians


def _apply_imputation(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for feat, med in medians.items():
        out[feat] = out[feat].fillna(med)
    return out


def _scoreline_market_metrics(
    *,
    market_code: str,
    y_true: np.ndarray,
    p_true: np.ndarray,
) -> dict[str, Any]:
    return binary_classification_row(market=market_code, y_true=y_true, p_true=p_true)


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
    scope_markets: set[str],
    model_type: str,
    max_goals: int,
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

        x_train = train_df[features]
        x_test = test_df[features]
        y_home_train = train_df["home_goals"].astype(float)
        y_away_train = train_df["away_goals"].astype(float)

        home_model = _build_regressor(model_type)
        away_model = _build_regressor(model_type)
        home_model.fit(x_train, y_home_train)
        away_model.fit(x_train, y_away_train)

        pred_home = np.clip(home_model.predict(x_test), 0.05, 8.0)
        pred_away = np.clip(away_model.predict(x_test), 0.05, 8.0)
        derived_rows: list[dict[str, float]] = []
        for lh, la in zip(pred_home, pred_away):
            mat = _score_matrix_independent_poisson(
                lambda_home=float(lh),
                lambda_away=float(la),
                max_goals=max(1, int(max_goals)),
            )
            derived_rows.append(derive_and_validate(mat))
        pred_frame = pd.DataFrame(derived_rows)

        for market in sorted(pred_frame.columns):
            if market not in scope_markets:
                continue
            target_col = f"target_{market}"
            if target_col not in test_df.columns:
                continue
            valid_mask = test_df[target_col].notna().to_numpy(dtype=bool)
            valid_n = int(valid_mask.sum())
            if valid_n < int(min_fold_test_n):
                continue
            y_true = test_df[target_col].to_numpy(dtype=float)[valid_mask].astype(int)
            p_true = np.clip(
                pred_frame[market].to_numpy(dtype=float)[valid_mask], 0.001, 0.999
            )
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
                        group_values=test_df.loc[valid_mask, "league_code"],
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
    scope_markets: set[str],
    max_goals: int,
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
            scope_markets=scope_markets,
            model_type=model_type,
            max_goals=max_goals,
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
    df = legacy_calibrator.fetch_dataset()
    if df.empty:
        raise RuntimeError("No rows available for scoreline training.")
    if args.max_rows is not None and args.max_rows > 0 and len(df) > args.max_rows:
        df = (
            df.sort_values(["match_datetime_utc", "fixture_id"])
            .tail(int(args.max_rows))
            .reset_index(drop=True)
        )

    df = legacy_calibrator.add_targets_and_derived(df)
    df["match_datetime_utc"] = pd.to_datetime(
        df["match_datetime_utc"], utc=True, errors="coerce"
    )

    features = _select_features(df, args.contract)
    scope_markets = set(load_scope_markets(args.scope))
    model_type_requested = str(args.model_type)
    model_selection_summary: dict[str, dict[str, float | int | None]]
    walkforward_league_rows: list[dict[str, Any]]
    if model_type_requested == "auto":
        model_type_selected, walkforward_rows, walkforward_summary, model_selection_summary, walkforward_league_rows = (
            _select_model_type(
                frame=df,
                features=features,
                scope_markets=scope_markets,
                max_goals=int(args.max_goals),
                folds=int(args.folds),
                min_fold_test_n=int(args.min_fold_test_n),
            )
        )
    else:
        model_type_selected = model_type_requested
        walkforward_rows, walkforward_summary, walkforward_league_rows = _evaluate_walkforward(
            frame=df,
            features=features,
            scope_markets=scope_markets,
            model_type=model_type_selected,
            max_goals=int(args.max_goals),
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

    x_train = train_df[features]
    x_test = test_df[features]
    y_home_train = train_df["home_goals"].astype(float)
    y_away_train = train_df["away_goals"].astype(float)
    y_home_test = test_df["home_goals"].astype(float)
    y_away_test = test_df["away_goals"].astype(float)

    home_model = _build_regressor(model_type_selected)
    away_model = _build_regressor(model_type_selected)
    home_model.fit(x_train, y_home_train)
    away_model.fit(x_train, y_away_train)

    pred_home = np.clip(home_model.predict(x_test), 0.05, 8.0)
    pred_away = np.clip(away_model.predict(x_test), 0.05, 8.0)

    derived_rows: list[dict[str, float]] = []
    for lh, la in zip(pred_home, pred_away):
        mat = _score_matrix_independent_poisson(
            lambda_home=float(lh), lambda_away=float(la), max_goals=max(1, int(args.max_goals))
        )
        markets = derive_and_validate(mat)
        derived_rows.append(markets)
    pred_frame = pd.DataFrame(derived_rows)

    market_metrics: list[dict[str, Any]] = []
    holdout_league_rows: list[dict[str, Any]] = []
    for market in sorted(pred_frame.columns):
        if market not in scope_markets:
            continue
        target_col = f"target_{market}"
        if target_col not in test_df.columns:
            continue
        y_true = test_df[target_col].astype(int).to_numpy()
        p_true = np.clip(pred_frame[market].to_numpy(dtype=float), 0.001, 0.999)
        market_metrics.append(
            _scoreline_market_metrics(
                market_code=market,
                y_true=y_true,
                p_true=p_true,
            )
        )
        if "league_code" in test_df.columns:
            holdout_league_rows.extend(
                group_binary_classification_rows(
                    market=market,
                    group_values=test_df["league_code"],
                    y_true=y_true,
                    p_true=p_true,
                )
            )

    diagnostics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": features,
        "model_name": MODEL_NAME,
        "model_version": str(args.model_version),
        "model_type_requested": model_type_requested,
        "model_type_selected": model_type_selected,
        "model_selection_summary": model_selection_summary,
        "max_goals": int(args.max_goals),
        "home_goals_mae": float(mean_absolute_error(y_home_test, pred_home)),
        "away_goals_mae": float(mean_absolute_error(y_away_test, pred_away)),
        "walkforward_folds": int(args.folds),
        "walkforward_min_fold_test_n": int(args.min_fold_test_n),
        "walkforward_markets_scored": int(len(walkforward_summary)),
        "walkforward_league_rows": int(len(walkforward_league_rows)),
        "holdout_league_rows": int(len(holdout_league_rows)),
        "trained_at_utc": datetime.now(tz=UTC).isoformat(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_metadata(
        args.output_dir,
        build_artifact_metadata(
            family="scoreline",
            model_name=MODEL_NAME,
            model_version=str(args.model_version),
            artifact_dir=args.output_dir,
            trained_at_utc=str(diagnostics["trained_at_utc"]),
            extra={"model_type_selected": model_type_selected},
        ),
    )
    joblib.dump(home_model, args.output_dir / "home_goals_model.pkl")
    joblib.dump(away_model, args.output_dir / "away_goals_model.pkl")
    (args.output_dir / "features.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )
    (args.output_dir / "imputation.json").write_text(
        json.dumps({"global_medians": imputation}, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics_holdout.json").write_text(
        json.dumps(market_metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics_holdout_by_league.json").write_text(
        json.dumps(holdout_league_rows, indent=2), encoding="utf-8"
    )
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
    print(f"Saved scoreline artifacts to {args.output_dir}")
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
