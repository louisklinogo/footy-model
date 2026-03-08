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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.layer2_markets import market_outcome_calibrator as legacy_calibrator
from src.modeling.v2.eval.metrics import binary_classification_row, build_prediction_frame, group_binary_classification_rows
from src.modeling.v2.families.scoreline.train_scoreline import (
    _apply_imputation,
    _build_regressor,
    _fit_imputation,
    _json_default,
    _load_training_frame,
    _score_matrix_independent_poisson,
    _walkforward_ranges,
)
from src.modeling.v2.io.artifact_identity import build_artifact_metadata, write_artifact_metadata


DEFAULT_OUT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "scoreline_o15_stacked_candidate_v1"
MODEL_NAME = "scoreline_o15_stacked_v1"
MODEL_VERSION = "stacked_o15_head_v1"
MARKET_CODE = "o15"
TARGET_COL = "target_o15"
STACKED_SCORELINE_FEATURES = [
    "scoreline_p_o15",
    "scoreline_lambda_home",
    "scoreline_lambda_away",
    "scoreline_lambda_total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a chrono-safe stacked o15 challenger.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--model-version", type=str, default=MODEL_VERSION)
    parser.add_argument("--model-kind", choices=("gbm", "hgbm", "auto"), default="auto")
    parser.add_argument("--scoreline-model-type", choices=("histgb_poisson", "poisson_glm"), default="histgb_poisson")
    parser.add_argument("--max-goals", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--min-fold-test-n", type=int, default=80)
    return parser.parse_args()


def _build_estimator(model_kind: str) -> object:
    if model_kind == "hgbm":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=350,
            min_samples_leaf=40,
            random_state=42,
        )
    return GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=40,
        random_state=42,
    )


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required {label} columns: {', '.join(sorted(missing))}")


def _scoreline_feature_row(lambda_home: float, lambda_away: float, max_goals: int) -> dict[str, float]:
    mat = _score_matrix_independent_poisson(
        lambda_home=float(lambda_home),
        lambda_away=float(lambda_away),
        max_goals=max(8, int(max_goals)),
    )
    under_or_push_o15 = float(mat[0, 0]) + float(mat[1, 0]) + float(mat[0, 1])
    p_o15 = float(np.clip(1.0 - under_or_push_o15, 0.001, 0.999))
    return {
        "scoreline_p_o15": p_o15,
        "scoreline_lambda_home": float(lambda_home),
        "scoreline_lambda_away": float(lambda_away),
        "scoreline_lambda_total": float(lambda_home + lambda_away),
    }


def _fit_scoreline_feature_models(
    *,
    train_df: pd.DataFrame,
    base_features: list[str],
    scoreline_model_type: str,
) -> tuple[object, object, dict[str, float]]:
    _require_columns(train_df, ["home_goals", "away_goals"], "scoreline target")
    imputation = _fit_imputation(train_df, base_features)
    train_imp = _apply_imputation(train_df, imputation)
    x_train = train_imp[base_features]
    home_model = _build_regressor(scoreline_model_type)
    away_model = _build_regressor(scoreline_model_type)
    home_model.fit(x_train, train_imp["home_goals"].astype(float))
    away_model.fit(x_train, train_imp["away_goals"].astype(float))
    return home_model, away_model, imputation


def _predict_scoreline_feature_frame(
    *,
    test_df: pd.DataFrame,
    base_features: list[str],
    home_model: object,
    away_model: object,
    imputation: dict[str, float],
    max_goals: int,
) -> pd.DataFrame:
    test_imp = _apply_imputation(test_df, imputation)
    pred_home = np.clip(home_model.predict(test_imp[base_features]), 0.05, 8.0)
    pred_away = np.clip(away_model.predict(test_imp[base_features]), 0.05, 8.0)
    return pd.DataFrame(
        [_scoreline_feature_row(lh, la, max_goals) for lh, la in zip(pred_home, pred_away)],
        index=test_df.index,
    )


def _build_oof_scoreline_feature_frame(
    *,
    frame: pd.DataFrame,
    base_features: list[str],
    scoreline_model_type: str,
    max_goals: int,
    folds: int,
) -> pd.DataFrame:
    ordered = frame.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for fold_idx, (train_end, test_end) in enumerate(_walkforward_ranges(len(ordered), folds), start=1):
        train_df = ordered.iloc[:train_end].copy()
        test_df = ordered.iloc[train_end:test_end].copy()
        if train_df.empty or test_df.empty:
            continue
        home_model, away_model, imputation = _fit_scoreline_feature_models(
            train_df=train_df,
            base_features=base_features,
            scoreline_model_type=scoreline_model_type,
        )
        stacked = _predict_scoreline_feature_frame(
            test_df=test_df,
            base_features=base_features,
            home_model=home_model,
            away_model=away_model,
            imputation=imputation,
            max_goals=max_goals,
        ).reset_index(drop=True)
        pieces.append(
            pd.concat(
                [
                    test_df[["fixture_id", "match_datetime_utc"]].reset_index(drop=True),
                    pd.Series(np.full(len(test_df), fold_idx, dtype=int), name="scoreline_oof_fold"),
                    stacked,
                ],
                axis=1,
            )
        )
    if not pieces:
        return pd.DataFrame(columns=["fixture_id", "match_datetime_utc", "scoreline_oof_fold", *STACKED_SCORELINE_FEATURES])
    return pd.concat(pieces, ignore_index=True)


def _fit_holdout_model(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model_kind: str,
) -> tuple[object, pd.DataFrame, np.ndarray, dict[str, Any]]:
    train = train_df[train_df[TARGET_COL].notna()].dropna(subset=features).copy()
    test = test_df[test_df[TARGET_COL].notna()].dropna(subset=features).copy()
    if train.empty or test.empty:
        raise ValueError(f"Insufficient rows for {MARKET_CODE}: train={len(train)}, test={len(test)}")

    y_train = train[TARGET_COL].astype(int)
    y_test = test[TARGET_COL].astype(int)
    calibrated = False

    if y_train.nunique() < 2:
        constant_class = int(y_train.iloc[0])
        model = DummyClassifier(strategy="constant", constant=constant_class)
        model.fit(train[features], y_train)
        probs = np.full(len(test), float(constant_class), dtype=float)
        auc: float | None = None
    else:
        base = _build_estimator(model_kind)
        class_counts = y_train.value_counts()
        can_calibrate = len(class_counts) > 1 and int(class_counts.min()) >= 3
        model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3) if can_calibrate else base
        model.fit(train[features], y_train)
        probs = np.clip(model.predict_proba(test[features])[:, 1], 0.001, 0.999)
        try:
            auc = float(roc_auc_score(y_test, probs))
        except ValueError:
            auc = None
        calibrated = bool(can_calibrate)

    metrics = binary_classification_row(
        market=MARKET_CODE,
        y_true=y_test.to_numpy(dtype=int),
        p_true=probs,
        extra_fields={
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "base_rate_train": float(y_train.mean()),
            "base_rate_test": float(y_test.mean()),
            "candidate_model": model_kind,
            "calibrated": calibrated,
            "train_start_utc": str(train["match_datetime_utc"].min()),
            "train_end_utc": str(train["match_datetime_utc"].max()),
            "test_start_utc": str(test["match_datetime_utc"].min()),
            "test_end_utc": str(test["match_datetime_utc"].max()),
        },
    )
    metrics["auc"] = auc
    return model, test, probs, metrics


def main() -> None:
    args = parse_args()
    df, data_source = _load_training_frame(args.dataset_path)
    if df.empty:
        raise RuntimeError("No rows available for scoreline o15 stacked training.")
    if args.max_rows is not None and args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sort_values(["match_datetime_utc", "fixture_id"]).tail(int(args.max_rows)).reset_index(drop=True)

    if TARGET_COL not in df.columns:
        df = legacy_calibrator.add_targets_and_derived(df)
    df["match_datetime_utc"] = pd.to_datetime(df["match_datetime_utc"], utc=True, errors="coerce")

    base_features = legacy_calibrator.feature_columns()
    missing_features = [feat for feat in base_features if feat not in df.columns]
    if missing_features:
        raise RuntimeError(f"Feature contract mismatch in o15 stacked dataset: {', '.join(sorted(missing_features))}")
    _require_columns(df, ["home_goals", "away_goals"], "scoreline target")

    train_df, test_df = legacy_calibrator.split_time_respecting(df)
    train_imp, test_imp, base_imputation = legacy_calibrator.impute_for_split(train_df, test_df, base_features)

    train_oof = _build_oof_scoreline_feature_frame(
        frame=train_df,
        base_features=base_features,
        scoreline_model_type=str(args.scoreline_model_type),
        max_goals=int(args.max_goals),
        folds=max(1, int(args.folds)),
    )
    if train_oof.empty:
        raise RuntimeError("Unable to generate chrono-safe scoreline OOF features for stacked training.")

    scoreline_home_model, scoreline_away_model, scoreline_imputation = _fit_scoreline_feature_models(
        train_df=train_df,
        base_features=base_features,
        scoreline_model_type=str(args.scoreline_model_type),
    )
    holdout_scoreline = _predict_scoreline_feature_frame(
        test_df=test_df,
        base_features=base_features,
        home_model=scoreline_home_model,
        away_model=scoreline_away_model,
        imputation=scoreline_imputation,
        max_goals=int(args.max_goals),
    ).reset_index(drop=True)

    train_meta = train_imp.merge(train_oof.drop(columns=["match_datetime_utc"]), on="fixture_id", how="left")
    test_meta = pd.concat([test_imp.reset_index(drop=True), holdout_scoreline], axis=1)
    meta_features = [*base_features, *STACKED_SCORELINE_FEATURES]

    stacked_train = train_meta[train_meta[TARGET_COL].notna()].dropna(subset=STACKED_SCORELINE_FEATURES).copy()
    walkforward_rows_all, candidate_summary, auto_selected_model = legacy_calibrator.evaluate_market_candidates_walkforward(
        frame=stacked_train,
        target_col=TARGET_COL,
        market_code=MARKET_CODE,
        features=meta_features,
        folds=max(1, int(args.folds)),
        min_fold_test_n=max(1, int(args.min_fold_test_n)),
    )
    model_kind_selected = auto_selected_model if str(args.model_kind) == "auto" else str(args.model_kind)
    walkforward_rows = [row for row in walkforward_rows_all if str(row.get("candidate_model")) == model_kind_selected]
    walkforward_summary = candidate_summary.get(model_kind_selected)
    walkforward_payload = {} if walkforward_summary is None else {MARKET_CODE: {**walkforward_summary, "candidate_model": model_kind_selected}}

    model, scored_holdout, probs, holdout_metrics = _fit_holdout_model(
        train_df=stacked_train,
        test_df=test_meta,
        features=meta_features,
        model_kind=model_kind_selected,
    )

    y_true = scored_holdout[TARGET_COL].astype(int).to_numpy()
    extra_columns: dict[str, Any] = {
        "league_code": scored_holdout["league_code"].fillna("__missing__").astype(str).to_numpy(),
        "match_datetime_utc": scored_holdout["match_datetime_utc"].astype(str).to_numpy(),
        "scoreline_p_o15": scored_holdout["scoreline_p_o15"].to_numpy(dtype=float),
        "scoreline_lambda_total": scored_holdout["scoreline_lambda_total"].to_numpy(dtype=float),
    }
    holdout_prediction_frame = build_prediction_frame(
        label_key="market",
        label_value=MARKET_CODE,
        fixture_ids=scored_holdout["fixture_id"].to_numpy(),
        y_true=y_true,
        p_true=probs,
        extra_columns=extra_columns,
    )
    holdout_league_rows = group_binary_classification_rows(
        market=MARKET_CODE,
        group_values=scored_holdout["league_code"],
        y_true=y_true,
        p_true=probs,
    )

    diagnostics = {
        "train_rows_total": int(len(train_imp)),
        "train_rows_stacked": int(len(stacked_train)),
        "train_rows_dropped_for_oof_warmup": int(len(train_imp) - len(stacked_train)),
        "test_rows": int(len(test_imp)),
        "base_features": base_features,
        "stacked_scoreline_features": STACKED_SCORELINE_FEATURES,
        "meta_features": meta_features,
        "model_name": MODEL_NAME,
        "model_version": str(args.model_version),
        "data_source": data_source,
        "model_kind_requested": str(args.model_kind),
        "model_kind_selected": model_kind_selected,
        "scoreline_model_type": str(args.scoreline_model_type),
        "max_goals": int(args.max_goals),
        "candidate_model_summary": candidate_summary,
        "walkforward_folds": int(args.folds),
        "walkforward_min_fold_test_n": int(args.min_fold_test_n),
        "walkforward_rows": int(len(walkforward_rows)),
        "scoreline_oof_rows": int(len(train_oof)),
        "holdout_prediction_rows": int(len(holdout_prediction_frame)),
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
            extra={
                "market": MARKET_CODE,
                "model_kind_selected": model_kind_selected,
                "data_source": data_source,
                "scoreline_model_type": str(args.scoreline_model_type),
            },
        ),
    )
    joblib.dump(model, args.output_dir / "o15_model.pkl")
    joblib.dump(scoreline_home_model, args.output_dir / "stack_scoreline_home_goals_model.pkl")
    joblib.dump(scoreline_away_model, args.output_dir / "stack_scoreline_away_goals_model.pkl")
    (args.output_dir / "features.json").write_text(json.dumps(meta_features, indent=2), encoding="utf-8")
    (args.output_dir / "imputation.json").write_text(json.dumps(base_imputation, indent=2), encoding="utf-8")
    (args.output_dir / "stack_scoreline_imputation.json").write_text(json.dumps(scoreline_imputation, indent=2), encoding="utf-8")
    (args.output_dir / "metrics_holdout.json").write_text(json.dumps([holdout_metrics], indent=2), encoding="utf-8")
    (args.output_dir / "metrics_holdout_by_league.json").write_text(json.dumps(holdout_league_rows, indent=2), encoding="utf-8")
    holdout_prediction_frame.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    (args.output_dir / "metrics_walkforward_folds.json").write_text(json.dumps(walkforward_rows, indent=2, default=_json_default), encoding="utf-8")
    (args.output_dir / "metrics_walkforward.json").write_text(json.dumps(walkforward_payload, indent=2, default=_json_default), encoding="utf-8")
    (args.output_dir / "training_report.json").write_text(json.dumps(diagnostics, indent=2, default=_json_default), encoding="utf-8")

    print(f"Saved scoreline o15 stacked artifacts to {args.output_dir}")
    print(
        f"{MARKET_CODE}: model={model_kind_selected} scoreline={args.scoreline_model_type} "
        f"AUC={holdout_metrics['auc']}, Brier={holdout_metrics['brier']:.4f}, "
        f"Acc={holdout_metrics['accuracy']:.3f}, n={holdout_metrics['n']}"
    )


if __name__ == "__main__":
    main()