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
from src.modeling.v2.eval.metrics import (
    binary_classification_row,
    build_prediction_frame,
    group_binary_classification_rows,
)
from src.modeling.v2.families.scoreline.train_scoreline import _json_default, _load_training_frame
from src.modeling.v2.io.artifact_identity import build_artifact_metadata, write_artifact_metadata


DEFAULT_OUT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "scoreline_o15_aux_candidate_v1"
MODEL_NAME = "scoreline_o15_aux_v1"
MODEL_VERSION = "direct_o15_head_v1"
MARKET_CODE = "o15"
TARGET_COL = "target_o15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an evaluation-only direct o15 auxiliary challenger.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--model-version", type=str, default=MODEL_VERSION)
    parser.add_argument("--model-kind", choices=("gbm", "hgbm", "auto"), default="auto")
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


def _fit_holdout_model(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model_kind: str,
) -> tuple[object, np.ndarray, dict[str, Any]]:
    train = train_df[train_df[TARGET_COL].notna()].copy()
    test = test_df[test_df[TARGET_COL].notna()].copy()
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
    return model, probs, metrics


def main() -> None:
    args = parse_args()
    df, data_source = _load_training_frame(args.dataset_path)
    if df.empty:
        raise RuntimeError("No rows available for scoreline o15 auxiliary training.")
    if args.max_rows is not None and args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sort_values(["match_datetime_utc", "fixture_id"]).tail(int(args.max_rows)).reset_index(drop=True)

    if TARGET_COL not in df.columns:
        df = legacy_calibrator.add_targets_and_derived(df)
    df["match_datetime_utc"] = pd.to_datetime(df["match_datetime_utc"], utc=True, errors="coerce")

    features = legacy_calibrator.feature_columns()
    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        missing_csv = ", ".join(sorted(missing_features))
        raise RuntimeError(f"Feature contract mismatch in o15 auxiliary dataset: {missing_csv}")

    train_df, test_df = legacy_calibrator.split_time_respecting(df)
    train_imp, test_imp, imputation = legacy_calibrator.impute_for_split(train_df, test_df, features)
    market_df = pd.concat([train_imp, test_imp], ignore_index=True)

    walkforward_rows_all, candidate_summary, auto_selected_model = (
        legacy_calibrator.evaluate_market_candidates_walkforward(
            frame=market_df,
            target_col=TARGET_COL,
            market_code=MARKET_CODE,
            features=features,
            folds=max(1, int(args.folds)),
            min_fold_test_n=max(1, int(args.min_fold_test_n)),
        )
    )
    model_kind_selected = auto_selected_model if str(args.model_kind) == "auto" else str(args.model_kind)
    walkforward_rows = [
        row for row in walkforward_rows_all if str(row.get("candidate_model")) == model_kind_selected
    ]
    walkforward_summary = candidate_summary.get(model_kind_selected)
    walkforward_payload = {} if walkforward_summary is None else {
        MARKET_CODE: {**walkforward_summary, "candidate_model": model_kind_selected}
    }

    model, probs, holdout_metrics = _fit_holdout_model(
        train_df=train_imp,
        test_df=test_imp,
        features=features,
        model_kind=model_kind_selected,
    )

    y_true = test_imp[TARGET_COL].astype(int).to_numpy()
    extra_columns: dict[str, Any] = {}
    if "league_code" in test_imp.columns:
        extra_columns["league_code"] = test_imp["league_code"].fillna("__missing__").astype(str).to_numpy()
    if "match_datetime_utc" in test_imp.columns:
        extra_columns["match_datetime_utc"] = test_imp["match_datetime_utc"].astype(str).to_numpy()
    holdout_prediction_frame = build_prediction_frame(
        label_key="market",
        label_value=MARKET_CODE,
        fixture_ids=(test_imp["fixture_id"].to_numpy() if "fixture_id" in test_imp.columns else test_imp.index.to_numpy()),
        y_true=y_true,
        p_true=probs,
        extra_columns=extra_columns,
    )
    holdout_league_rows = (
        group_binary_classification_rows(
            market=MARKET_CODE,
            group_values=test_imp["league_code"],
            y_true=y_true,
            p_true=probs,
        )
        if "league_code" in test_imp.columns
        else []
    )

    diagnostics = {
        "train_rows": int(len(train_imp)),
        "test_rows": int(len(test_imp)),
        "features": features,
        "model_name": MODEL_NAME,
        "model_version": str(args.model_version),
        "data_source": data_source,
        "model_kind_requested": str(args.model_kind),
        "model_kind_selected": model_kind_selected,
        "candidate_model_summary": candidate_summary,
        "walkforward_folds": int(args.folds),
        "walkforward_min_fold_test_n": int(args.min_fold_test_n),
        "walkforward_rows": int(len(walkforward_rows)),
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
            extra={"market": MARKET_CODE, "model_kind_selected": model_kind_selected, "data_source": data_source},
        ),
    )
    joblib.dump(model, args.output_dir / "o15_model.pkl")
    (args.output_dir / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (args.output_dir / "imputation.json").write_text(json.dumps(imputation, indent=2), encoding="utf-8")
    (args.output_dir / "metrics_holdout.json").write_text(json.dumps([holdout_metrics], indent=2), encoding="utf-8")
    (args.output_dir / "metrics_holdout_by_league.json").write_text(json.dumps(holdout_league_rows, indent=2), encoding="utf-8")
    holdout_prediction_frame.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    (args.output_dir / "metrics_walkforward_folds.json").write_text(
        json.dumps(walkforward_rows, indent=2, default=_json_default), encoding="utf-8"
    )
    (args.output_dir / "metrics_walkforward.json").write_text(
        json.dumps(walkforward_payload, indent=2, default=_json_default), encoding="utf-8"
    )
    (args.output_dir / "training_report.json").write_text(
        json.dumps(diagnostics, indent=2, default=_json_default), encoding="utf-8"
    )

    print(f"Saved scoreline o15 auxiliary artifacts to {args.output_dir}")
    print(
        f"{MARKET_CODE}: model={model_kind_selected} AUC={holdout_metrics['auc']}, "
        f"Brier={holdout_metrics['brier']:.4f}, Acc={holdout_metrics['accuracy']:.3f}, n={holdout_metrics['n']}"
    )


if __name__ == "__main__":
    main()