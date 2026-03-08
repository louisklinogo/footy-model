from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.modeling.v2.families.scoreline import train_scoreline_residuals


class _ConstantProbabilityModel:
    def __init__(self, positive_prob: float) -> None:
        self.positive_prob = float(positive_prob)

    def predict_proba(self, x: pd.DataFrame):
        probs = [self.positive_prob] * len(x)
        return [[1.0 - prob, prob] for prob in probs]


def _base_eval_frame() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        home_win = 1 if idx % 3 == 0 else 0
        draw = 1 if idx % 3 == 1 else 0
        away_win = 1 if idx % 3 == 2 else 0
        rows.append(
            {
                "fixture_id": idx + 1,
                "match_datetime_utc": f"2026-01-{idx + 1:02d}T00:00:00+00:00",
                "league_code": "L1",
                "target_1x2_h": home_win,
                "target_1x2_d": draw,
                "target_1x2_a": away_win,
                "target_dc_1x": int(home_win or draw),
                "target_dc_x2": int(draw or away_win),
                "target_dc_12": int(home_win or away_win),
                "target_o15": int(idx % 2 == 0),
                "target_u35": int(idx % 2 == 1),
            }
        )
    return pd.DataFrame(rows)


def _oof_frame(rows: int = 80) -> pd.DataFrame:
    payload = []
    for idx in range(rows):
        payload.append(
            {
                "base_lambda_home": 1.2,
                "base_lambda_away": 0.9,
                "base_lambda_total": 2.1,
                "base_1x2_h": 0.42,
                "base_1x2_d": 0.28,
                "base_1x2_a": 0.30,
                "base_dc_1x": 0.70,
                "base_dc_x2": 0.58,
                "base_dc_12": 0.72,
                "base_o15": 0.63,
                "base_u35": 0.67,
                "odds_over_15": 1.55,
                "target_1x2_h": int(idx % 3 == 0),
                "target_1x2_d": int(idx % 3 == 1),
                "target_1x2_a": int(idx % 3 == 2),
                "target_o15": int(idx % 2 == 0),
                "target_u35": int(idx % 2 == 1),
            }
        )
    return pd.DataFrame(payload)


def test_evaluate_residual_walkforward_emits_overlay_rows(monkeypatch) -> None:
    frame = _base_eval_frame()

    monkeypatch.setattr(
        train_scoreline_residuals,
        "_fit_fold_base_models",
        lambda train_df, base_features, model_type: (object(), object(), {}),
    )
    monkeypatch.setattr(
        train_scoreline_residuals,
        "_build_oof_training_frame",
        lambda **kwargs: _oof_frame(),
    )
    monkeypatch.setattr(
        train_scoreline_residuals,
        "_fit_residual_models",
        lambda **kwargs: (
            {
                "1x2_h": _ConstantProbabilityModel(0.55),
                "1x2_d": _ConstantProbabilityModel(0.25),
                "1x2_a": _ConstantProbabilityModel(0.35),
                "o15": _ConstantProbabilityModel(0.72),
                "u35": _ConstantProbabilityModel(0.41),
            },
            ["base_1x2_h", "odds_over_15"],
            {"base_1x2_h": 0.42, "odds_over_15": 1.55},
            {"o15": {"train_rows": 80}},
        ),
    )

    def _predict_base_market_frame(**kwargs):
        test_frame = kwargs["frame"]
        n = len(test_frame)
        return (
            pd.DataFrame(
                {
                    "1x2_h": [0.40] * n,
                    "1x2_d": [0.30] * n,
                    "1x2_a": [0.30] * n,
                    "dc_1x": [0.70] * n,
                    "dc_x2": [0.60] * n,
                    "dc_12": [0.70] * n,
                    "o15": [0.60] * n,
                    "u35": [0.65] * n,
                }
            ),
            pd.Series([1.25] * n),
            pd.Series([0.95] * n),
        )

    monkeypatch.setattr(
        train_scoreline_residuals,
        "_predict_base_market_frame",
        _predict_base_market_frame,
    )

    fold_rows, summary, league_rows = train_scoreline_residuals._evaluate_residual_walkforward(
        frame=frame,
        base_features=["feature_a"],
        scope_markets={"1x2_h", "1x2_d", "1x2_a", "dc_1x", "dc_x2", "dc_12", "o15", "u35"},
        model_type="histgb_poisson",
        max_goals=6,
        folds=3,
        min_fold_test_n=1,
    )

    assert fold_rows
    assert any(bool(row["residual_overlay_applied"]) for row in fold_rows)
    assert {row["market"] for row in fold_rows}.issuperset({"1x2_h", "dc_1x", "o15", "u35"})
    assert "dc_1x" in summary
    assert league_rows


def test_main_keeps_canonical_metrics_and_writes_residual_research_artifacts(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = tmp_path / "scoreline"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scope_path = tmp_path / "scope.yaml"
    scope_path.write_text("version: 1\nmarkets:\n  - o15\n", encoding="utf-8")
    (artifact_dir / "metrics_holdout.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "metrics_holdout_by_league.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "metrics_walkforward_folds.json").write_text(
        json.dumps(
            [
                {
                    "market": "o15",
                    "fold": 1,
                    "auc": 0.5,
                    "brier": 0.2,
                    "log_loss": 0.7,
                    "ece": 0.1,
                    "accuracy": 0.5,
                    "base_rate": 0.5,
                    "n": 4,
                }
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "metrics_walkforward_folds_by_league.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "training_report.json").write_text("{}", encoding="utf-8")

    holdout_df = pd.DataFrame(
        {
            "fixture_id": [101, 102],
            "match_datetime_utc": ["2026-02-01T00:00:00+00:00", "2026-02-02T00:00:00+00:00"],
            "league_code": ["L1", "L1"],
            "home_goals": [2, 0],
            "away_goals": [0, 2],
            "target_1x2_h": [1, 0],
            "target_o15": [1, 1],
            "target_u35": [1, 1],
        }
    )
    train_df = pd.DataFrame(
        {
            "fixture_id": [1, 2],
            "match_datetime_utc": ["2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"],
            "league_code": ["L1", "L1"],
            "home_goals": [1, 0],
            "away_goals": [0, 1],
            "target_1x2_h": [1, 0],
            "target_o15": [0, 1],
            "target_u35": [1, 1],
        }
    )
    full_df = pd.concat([train_df, holdout_df], ignore_index=True)

    monkeypatch.setattr(
        train_scoreline_residuals,
        "parse_args",
        lambda: argparse.Namespace(
            artifact_dir=artifact_dir,
            scope=scope_path,
            dataset_path=None,
            max_rows=None,
            folds=3,
            model_kind="hgbm",
            max_goals=6,
        ),
    )
    monkeypatch.setattr(
        train_scoreline_residuals,
        "_load_training_frame",
        lambda dataset_path: (full_df.copy(), "synthetic"),
    )
    monkeypatch.setattr(
        train_scoreline_residuals.legacy_calibrator,
        "split_time_respecting",
        lambda df: (train_df.copy(), holdout_df.copy()),
    )
    monkeypatch.setattr(
        train_scoreline_residuals,
        "_load_base_artifacts",
        lambda artifact_dir: (object(), object(), [], {}, {"model_type_selected": "histgb_poisson", "walkforward_folds": 2, "walkforward_min_fold_test_n": 1}),
    )
    monkeypatch.setattr(
        train_scoreline_residuals,
        "_build_oof_training_frame",
        lambda **kwargs: _oof_frame(),
    )
    monkeypatch.setattr(
        train_scoreline_residuals,
        "_fit_residual_models",
        lambda **kwargs: (
            {"o15": _ConstantProbabilityModel(0.82)},
            ["base_o15"],
            {"base_o15": 0.63},
            {"o15": {"train_rows": 80, "positive_rate": 0.5}},
        ),
    )

    def _predict_base_market_frame(**kwargs):
        test_frame = kwargs["frame"]
        n = len(test_frame)
        return pd.DataFrame({"o15": [0.61] * n, "u35": [0.74] * n}), pd.Series([1.2] * n), pd.Series([0.9] * n)

    monkeypatch.setattr(
        train_scoreline_residuals,
        "_predict_base_market_frame",
        _predict_base_market_frame,
    )
    monkeypatch.setattr(
        train_scoreline_residuals,
        "_evaluate_residual_walkforward",
        lambda **kwargs: (
            [
                {
                    "market": "o15",
                    "fold": 1,
                    "auc": 0.91,
                    "brier": 0.11,
                    "log_loss": 0.31,
                    "ece": 0.02,
                    "accuracy": 0.81,
                    "base_rate": 0.5,
                    "n": 4,
                    "train_rows": 2,
                    "test_rows": 2,
                    "residual_overlay_applied": True,
                    "residual_train_rows": 80,
                }
            ],
            {
                "o15": {
                    "rows": 1,
                    "folds_used": 1,
                    "n_total": 4,
                    "base_rate_mean": 0.5,
                    "base_rate_std": 0.0,
                    "auc_mean": 0.91,
                    "auc_std": 0.0,
                    "brier_mean": 0.11,
                    "brier_std": 0.0,
                    "log_loss_mean": 0.31,
                    "log_loss_std": 0.0,
                    "ece_mean": 0.02,
                    "ece_std": 0.0,
                    "accuracy_mean": 0.81,
                    "accuracy_std": 0.0,
                }
            },
            [],
        ),
    )
    monkeypatch.setattr(train_scoreline_residuals.joblib, "dump", lambda payload, path: None)

    train_scoreline_residuals.main()

    walkforward_rows = json.loads((artifact_dir / "metrics_walkforward_folds.json").read_text(encoding="utf-8"))
    residual_walkforward_rows = json.loads(
        (artifact_dir / "residual_metrics_walkforward_folds.json").read_text(encoding="utf-8")
    )
    residual_report = json.loads((artifact_dir / "residual_training_report.json").read_text(encoding="utf-8"))
    training_report = json.loads((artifact_dir / "training_report.json").read_text(encoding="utf-8"))

    assert walkforward_rows[0]["market"] == "o15"
    assert walkforward_rows[0]["auc"] == 0.5
    assert residual_walkforward_rows[0]["auc"] == 0.91
    assert (artifact_dir / "residual_metrics_walkforward_folds.json").exists()
    assert (artifact_dir / "residual_holdout_predictions.csv").exists()
    assert residual_report["walkforward_rows"] == 1
    assert training_report["residual_overlay"]["walkforward_rows"] == 1
    assert training_report["residual_overlay"]["published_to_canonical_surface"] is False
    assert training_report["residual_overlay"]["serving_markets"] == []