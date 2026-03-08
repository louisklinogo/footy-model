from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling.layer2_markets import market_outcome_calibrator as legacy_calibrator
from src.modeling.v2.eval.metrics import (
    binary_classification_row,
    build_prediction_frame,
    group_binary_classification_rows,
    summarize_binary_metric_rows,
)
from src.modeling.v2.families.scoreline.residual_utils import (
    BINARY_RESIDUAL_MARKETS,
    DOUBLE_CHANCE_MARKETS,
    ONE_X_TWO_MARKETS,
    apply_residual_bundle,
    build_residual_source_frame,
    prepare_residual_feature_frame,
)
from src.modeling.v2.families.scoreline.train_scoreline import (
    _apply_imputation,
    _build_regressor,
    _fit_imputation,
    _json_default,
    _load_training_frame,
    _score_matrix_independent_poisson,
    _walkforward_ranges,
)
from src.modeling.v2.io.baseline_registry import load_scope_markets


DEFAULT_ARTIFACT_DIR = ROOT_DIR / "model_artifacts" / "v2" / "scoreline"
DEFAULT_SCOPE = ROOT_DIR / "model_v2" / "market_scope.yaml"
MARKET_TARGETS = {
    "1x2_h": "target_1x2_h",
    "1x2_d": "target_1x2_d",
    "1x2_a": "target_1x2_a",
    "o15": "target_o15",
    "u35": "target_u35",
}


def _resolve_changed_markets(
    scope_markets: set[str],
    available_markets: list[str] | tuple[str, ...] | pd.Index,
) -> list[str]:
    available = {str(market) for market in available_markets}
    return [
        market
        for market in list(ONE_X_TWO_MARKETS) + list(DOUBLE_CHANCE_MARKETS) + list(BINARY_RESIDUAL_MARKETS)
        if market in scope_markets and market in available
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scoreline residual overlays.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--model-kind", choices=("hgbm",), default="hgbm")
    parser.add_argument("--max-goals", type=int, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_base_artifacts(artifact_dir: Path) -> tuple[object, object, list[str], dict[str, float], dict[str, Any]]:
    home_model = joblib.load(artifact_dir / "home_goals_model.pkl")
    away_model = joblib.load(artifact_dir / "away_goals_model.pkl")
    features = json.loads((artifact_dir / "features.json").read_text(encoding="utf-8"))
    imputation_payload = json.loads((artifact_dir / "imputation.json").read_text(encoding="utf-8"))
    medians = {
        str(k): float(v)
        for k, v in (imputation_payload.get("global_medians") or {}).items()
        if isinstance(v, (int, float))
    }
    report = _read_json(artifact_dir / "training_report.json")
    return home_model, away_model, list(features), medians, report


def _ensure_base_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for feature in features:
        if feature not in out.columns:
            out[feature] = pd.NA
    return out


def _fit_fold_base_models(train_df: pd.DataFrame, base_features: list[str], model_type: str) -> tuple[object, object, dict[str, float]]:
    train_df = _ensure_base_features(train_df, base_features)
    imputation = _fit_imputation(train_df, base_features)
    train_imp = _apply_imputation(train_df, imputation)
    x_train = train_imp[base_features]
    home_model = _build_regressor(model_type)
    away_model = _build_regressor(model_type)
    home_model.fit(x_train, train_imp["home_goals"].astype(float))
    away_model.fit(x_train, train_imp["away_goals"].astype(float))
    return home_model, away_model, imputation


def _predict_base_market_frame(
    *,
    frame: pd.DataFrame,
    home_model: object,
    away_model: object,
    base_features: list[str],
    medians: dict[str, float],
    max_goals: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    featured = _ensure_base_features(frame, base_features)
    featured = _apply_imputation(featured, medians)
    x = featured[base_features]
    lambda_home = pd.Series(home_model.predict(x), index=frame.index, dtype=float).clip(0.05, 8.0)
    lambda_away = pd.Series(away_model.predict(x), index=frame.index, dtype=float).clip(0.05, 8.0)
    rows: list[dict[str, float]] = []
    for lh, la in zip(lambda_home.to_numpy(dtype=float), lambda_away.to_numpy(dtype=float), strict=True):
        matrix = _score_matrix_independent_poisson(
            lambda_home=float(lh),
            lambda_away=float(la),
            max_goals=max(1, int(max_goals)),
        )
        from src.modeling.v2.families.scoreline.derive_markets import derive_and_validate

        rows.append(derive_and_validate(matrix))
    return pd.DataFrame(rows, index=frame.index), lambda_home, lambda_away


def _build_oof_training_frame(
    *,
    frame: pd.DataFrame,
    base_features: list[str],
    model_type: str,
    max_goals: int,
    folds: int,
) -> pd.DataFrame:
    ordered = frame.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for fold_idx, (train_end, test_end) in enumerate(_walkforward_ranges(len(ordered), folds), start=1):
        fold_train = ordered.iloc[:train_end].copy()
        fold_valid = ordered.iloc[train_end:test_end].copy()
        if fold_train.empty or fold_valid.empty:
            continue
        home_model, away_model, medians = _fit_fold_base_models(fold_train, base_features, model_type)
        base_markets, lambda_home, lambda_away = _predict_base_market_frame(
            frame=fold_valid,
            home_model=home_model,
            away_model=away_model,
            base_features=base_features,
            medians=medians,
            max_goals=max_goals,
        )
        piece = build_residual_source_frame(
            raw_frame=fold_valid,
            base_market_frame=base_markets,
            lambda_home=lambda_home.to_numpy(dtype=float),
            lambda_away=lambda_away.to_numpy(dtype=float),
        )
        piece["residual_fold"] = int(fold_idx)
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _fit_binary_head(x: pd.DataFrame, y: pd.Series) -> object:
    target = y.astype(int)
    if target.nunique() < 2:
        constant_class = int(target.iloc[0])
        model = DummyClassifier(strategy="constant", constant=constant_class)
        model.fit(x, target)
        return model
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=5,
        max_iter=250,
        min_samples_leaf=35,
        random_state=42,
    )
    model.fit(x, target)
    return model


def _fit_residual_models(
    *,
    oof_train: pd.DataFrame,
    scope_markets: set[str],
) -> tuple[dict[str, object], list[str], dict[str, float], dict[str, Any]]:
    train_features, residual_features, residual_imputation = prepare_residual_feature_frame(oof_train)
    models: dict[str, object] = {}
    model_diagnostics: dict[str, Any] = {}
    for market, target_col in MARKET_TARGETS.items():
        if market not in scope_markets or target_col not in oof_train.columns:
            continue
        market_mask = oof_train[target_col].notna()
        if int(market_mask.sum()) < 50:
            continue
        models[market] = _fit_binary_head(
            train_features.loc[market_mask].reset_index(drop=True),
            oof_train.loc[market_mask, target_col].reset_index(drop=True),
        )
        model_diagnostics[market] = {
            "train_rows": int(market_mask.sum()),
            "positive_rate": float(oof_train.loc[market_mask, target_col].astype(float).mean()),
        }
    return models, residual_features, residual_imputation, model_diagnostics


def _build_holdout_rows(
    *,
    test_df: pd.DataFrame,
    market_frame: pd.DataFrame,
    markets: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    league_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for market in markets:
        target_col = MARKET_TARGETS.get(market, f"target_{market}")
        if target_col not in test_df.columns or market not in market_frame.columns:
            continue
        y_true = test_df[target_col].astype(int).to_numpy()
        p_true = market_frame[market].to_numpy(dtype=float)
        metrics_rows.append(binary_classification_row(market=market, y_true=y_true, p_true=p_true))
        if "league_code" in test_df.columns:
            league_rows.extend(
                group_binary_classification_rows(
                    market=market,
                    group_values=test_df["league_code"],
                    y_true=y_true,
                    p_true=p_true,
                )
            )
        extra_columns: dict[str, Any] = {}
        if "league_code" in test_df.columns:
            extra_columns["league_code"] = test_df["league_code"].fillna("__missing__").astype(str).to_numpy()
        if "match_datetime_utc" in test_df.columns:
            extra_columns["match_datetime_utc"] = test_df["match_datetime_utc"].astype(str).to_numpy()
        extra_columns["residual_overlay_applied"] = [True] * len(test_df)
        extra_columns["p_model_base"] = market_frame.get(f"base::{market}", market_frame[market]).to_numpy(dtype=float)
        prediction_frames.append(
            build_prediction_frame(
                label_key="market",
                label_value=market,
                fixture_ids=test_df["fixture_id"].to_numpy(),
                y_true=y_true,
                p_true=p_true,
                extra_columns=extra_columns,
            )
        )
    prediction_frame = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(columns=["market", "fixture_id", "y_true", "p_model"])
    )
    return metrics_rows, league_rows, prediction_frame


def _evaluate_residual_walkforward(
    *,
    frame: pd.DataFrame,
    base_features: list[str],
    scope_markets: set[str],
    model_type: str,
    max_goals: int,
    folds: int,
    min_fold_test_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    ordered = frame.sort_values(["match_datetime_utc", "fixture_id"]).reset_index(drop=True)
    fold_rows: list[dict[str, Any]] = []
    fold_league_rows: list[dict[str, Any]] = []
    for fold_idx, (train_end, test_end) in enumerate(_walkforward_ranges(len(ordered), folds), start=1):
        fold_train = ordered.iloc[:train_end].copy()
        fold_test = ordered.iloc[train_end:test_end].copy()
        if fold_train.empty or fold_test.empty:
            continue
        if int(len(fold_test)) < int(min_fold_test_n):
            continue

        home_model, away_model, base_imputation = _fit_fold_base_models(
            fold_train,
            base_features,
            model_type,
        )
        inner_oof = _build_oof_training_frame(
            frame=fold_train,
            base_features=base_features,
            model_type=model_type,
            max_goals=max_goals,
            folds=folds,
        )
        models: dict[str, object] = {}
        residual_feature_columns: list[str] = []
        residual_imputation: dict[str, float] = {}
        if not inner_oof.empty:
            models, residual_feature_columns, residual_imputation, _ = _fit_residual_models(
                oof_train=inner_oof,
                scope_markets=scope_markets,
            )

        base_markets, lambda_home, lambda_away = _predict_base_market_frame(
            frame=fold_test,
            home_model=home_model,
            away_model=away_model,
            base_features=base_features,
            medians=base_imputation,
            max_goals=max_goals,
        )
        adjusted_markets = base_markets.copy()
        overlay_applied = bool(models)
        if overlay_applied:
            residual_source = build_residual_source_frame(
                raw_frame=fold_test,
                base_market_frame=base_markets,
                lambda_home=lambda_home.to_numpy(dtype=float),
                lambda_away=lambda_away.to_numpy(dtype=float),
            )
            residual_features, _, _ = prepare_residual_feature_frame(
                residual_source,
                feature_columns=residual_feature_columns,
                imputation=residual_imputation,
            )
            adjusted_markets = apply_residual_bundle(
                base_market_frame=base_markets,
                residual_features=residual_features,
                residual_bundle={"models": models},
                direct_market_overrides=True,
            )

        changed_markets = _resolve_changed_markets(scope_markets, adjusted_markets.columns)
        base_fields = {
            "fold": int(fold_idx),
            "train_rows": int(len(fold_train)),
            "test_rows": int(len(fold_test)),
            "residual_overlay_applied": bool(overlay_applied),
            "residual_train_rows": int(len(inner_oof)),
        }
        for market in changed_markets:
            target_col = MARKET_TARGETS.get(market, f"target_{market}")
            if target_col not in fold_test.columns or market not in adjusted_markets.columns:
                continue
            y_true = fold_test[target_col].astype(int).to_numpy()
            p_true = adjusted_markets[market].to_numpy(dtype=float)
            fold_rows.append(
                binary_classification_row(
                    market=market,
                    y_true=y_true,
                    p_true=p_true,
                    extra_fields=base_fields,
                )
            )
            if "league_code" in fold_test.columns:
                fold_league_rows.extend(
                    group_binary_classification_rows(
                        market=market,
                        group_values=fold_test["league_code"],
                        y_true=y_true,
                        p_true=p_true,
                        extra_fields=base_fields,
                    )
                )

    return fold_rows, summarize_binary_metric_rows(fold_rows), fold_league_rows


def _replace_market_rows(existing_path: Path, replacement: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    if existing.empty:
        return replacement.copy()
    if "market" not in existing.columns:
        return replacement.copy()
    keep = existing.loc[~existing["market"].isin(replacement["market"].unique())].copy()
    merged = pd.concat([keep, replacement], ignore_index=True)
    sort_cols = [col for col in ["market", "fixture_id"] if col in merged.columns]
    return merged.sort_values(sort_cols).reset_index(drop=True) if sort_cols else merged.reset_index(drop=True)


def _replace_metric_rows(existing_path: Path, replacement_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not replacement_rows:
        return json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else []
    existing = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else []
    replacement_markets = {str(row.get("market")) for row in replacement_rows}
    kept = [row for row in existing if str(row.get("market")) not in replacement_markets]
    return kept + replacement_rows


def main() -> None:
    args = parse_args()
    df, data_source = _load_training_frame(args.dataset_path)
    if df.empty:
        raise RuntimeError("No rows available for scoreline residual training.")
    if args.max_rows is not None and args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sort_values(["match_datetime_utc", "fixture_id"]).tail(int(args.max_rows)).reset_index(drop=True)
    if "target_1x2_h" not in df.columns:
        df = legacy_calibrator.add_targets_and_derived(df)
    df["match_datetime_utc"] = pd.to_datetime(df["match_datetime_utc"], utc=True, errors="coerce")

    scope_markets = set(load_scope_markets(args.scope))
    home_model, away_model, base_features, base_medians, base_report = _load_base_artifacts(args.artifact_dir)
    model_type = str(base_report.get("model_type_selected") or "histgb_poisson")
    max_goals = int(args.max_goals or base_report.get("max_goals") or 10)

    train_df, test_df = legacy_calibrator.split_time_respecting(df)
    oof_train = _build_oof_training_frame(
        frame=train_df,
        base_features=base_features,
        model_type=model_type,
        max_goals=max_goals,
        folds=int(args.folds),
    )
    if oof_train.empty:
        raise RuntimeError("No OOF rows available for scoreline residual training.")

    models, residual_features, residual_imputation, model_diagnostics = _fit_residual_models(
        oof_train=oof_train,
        scope_markets=scope_markets,
    )
    if not models:
        raise RuntimeError("No scoreline residual heads were trained.")

    base_holdout, lambda_home, lambda_away = _predict_base_market_frame(
        frame=test_df,
        home_model=home_model,
        away_model=away_model,
        base_features=base_features,
        medians=base_medians,
        max_goals=max_goals,
    )
    holdout_source = build_residual_source_frame(
        raw_frame=test_df,
        base_market_frame=base_holdout,
        lambda_home=lambda_home.to_numpy(dtype=float),
        lambda_away=lambda_away.to_numpy(dtype=float),
    )
    holdout_features, _, _ = prepare_residual_feature_frame(
        holdout_source,
        feature_columns=residual_features,
        imputation=residual_imputation,
    )
    adjusted_holdout = apply_residual_bundle(
        base_market_frame=base_holdout,
        residual_features=holdout_features,
        residual_bundle={"models": models},
        direct_market_overrides=True,
    )
    for market in base_holdout.columns:
        adjusted_holdout[f"base::{market}"] = base_holdout[market].to_numpy(dtype=float)

    changed_markets = _resolve_changed_markets(scope_markets, adjusted_holdout.columns)
    holdout_metrics, holdout_league_rows, holdout_prediction_frame = _build_holdout_rows(
        test_df=test_df,
        market_frame=adjusted_holdout,
        markets=changed_markets,
    )

    walkforward_folds = int(base_report.get("walkforward_folds") or args.folds)
    walkforward_min_fold_test_n = int(base_report.get("walkforward_min_fold_test_n") or 1)
    residual_walkforward_rows, residual_walkforward_summary, residual_walkforward_league_rows = (
        _evaluate_residual_walkforward(
            frame=df,
            base_features=base_features,
            scope_markets=scope_markets,
            model_type=model_type,
            max_goals=max_goals,
            folds=walkforward_folds,
            min_fold_test_n=walkforward_min_fold_test_n,
        )
    )

    holdout_prediction_frame.to_csv(args.artifact_dir / "residual_holdout_predictions.csv", index=False)

    (args.artifact_dir / "residual_metrics_walkforward_folds.json").write_text(
        json.dumps(residual_walkforward_rows, indent=2, default=_json_default), encoding="utf-8"
    )
    (args.artifact_dir / "residual_metrics_walkforward_folds_by_league.json").write_text(
        json.dumps(residual_walkforward_league_rows, indent=2, default=_json_default), encoding="utf-8"
    )
    (args.artifact_dir / "residual_metrics_walkforward.json").write_text(
        json.dumps(residual_walkforward_summary, indent=2, default=_json_default), encoding="utf-8"
    )

    residual_bundle = {
        "models": models,
        "feature_columns": residual_features,
        "imputation": residual_imputation,
        "publish_to_canonical_surface": False,
        "serving_markets": [],
        "trained_markets": sorted(models.keys()),
        "generated_markets": changed_markets,
    }
    joblib.dump(residual_bundle, args.artifact_dir / "residual_models.joblib")

    report = {
        "trained_at_utc": datetime.now(tz=UTC).isoformat(),
        "data_source": data_source,
        "artifact_dir": str(args.artifact_dir),
        "base_model_type": model_type,
        "max_goals": max_goals,
        "oof_rows": int(len(oof_train)),
        "holdout_rows": int(len(test_df)),
        "residual_features": residual_features,
        "trained_markets": sorted(models.keys()),
        "generated_markets": changed_markets,
        "walkforward_folds": walkforward_folds,
        "walkforward_min_fold_test_n": walkforward_min_fold_test_n,
        "walkforward_rows": int(len(residual_walkforward_rows)),
        "walkforward_league_rows": int(len(residual_walkforward_league_rows)),
        "walkforward_markets": sorted(residual_walkforward_summary.keys()),
        "model_diagnostics": model_diagnostics,
    }
    (args.artifact_dir / "residual_features.json").write_text(
        json.dumps(residual_features, indent=2), encoding="utf-8"
    )
    (args.artifact_dir / "residual_imputation.json").write_text(
        json.dumps(residual_imputation, indent=2), encoding="utf-8"
    )
    (args.artifact_dir / "residual_training_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )

    training_report_path = args.artifact_dir / "training_report.json"
    training_report = _read_json(training_report_path)
    training_report["residual_overlay"] = {
        "published_to_canonical_surface": False,
        "serving_markets": [],
        "trained_markets": sorted(models.keys()),
        "generated_markets": changed_markets,
        "feature_count": int(len(residual_features)),
        "holdout_prediction_rows": int(len(holdout_prediction_frame)),
        "walkforward_rows": int(len(residual_walkforward_rows)),
        "walkforward_markets": sorted(residual_walkforward_summary.keys()),
        "walkforward_league_rows": int(len(residual_walkforward_league_rows)),
    }
    training_report_path.write_text(
        json.dumps(training_report, indent=2, default=_json_default), encoding="utf-8"
    )
    print(f"Saved scoreline residual overlay artifacts to {args.artifact_dir}")


if __name__ == "__main__":
    main()