from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest
from sklearn.linear_model import PoissonRegressor

from src.modeling.v2.families.anytime import train_anytime
from src.modeling.v2.families.anytime.train_anytime import (
    _aggregate_walkforward_quality,
    _build_regressor,
    _derive_anytime_frame,
    _fit_blend_weight,
    _fit_goal_models,
    _predict_goal_rates,
    _prepare_phase_targets,
    _select_features,
)
from src.modeling.v2.families.anytime.labels import prepare_state_ladder_labels


def test_anytime_select_features_raises_when_required_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_anytime_contract_") as td:
        contract = Path(td) / "anytime.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: anytime",
                    "required_features:",
                    "  - lambda_home_l1",
                    "  - lambda_away_l1",
                    "optional_features:",
                    "  - rule_fired_home",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame([{"lambda_home_l1": 1.2}])
        with pytest.raises(RuntimeError):
            _select_features(frame, contract)


def test_anytime_select_features_includes_optional_when_present() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_anytime_contract_") as td:
        contract = Path(td) / "anytime.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: anytime",
                    "required_features:",
                    "  - lambda_home_l1",
                    "optional_features:",
                    "  - rule_fired_home",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame([{"lambda_home_l1": 1.2, "rule_fired_home": 1}])
        selected = _select_features(frame, contract)
        assert selected == ["lambda_home_l1", "rule_fired_home"]


def test_anytime_aggregate_walkforward_quality_summarizes_markets() -> None:
    summary = {
        "h_1up": {"auc_mean": 0.60, "brier_mean": 0.23, "n_total": 140},
        "a_1up": {"auc_mean": 0.58, "brier_mean": 0.24, "n_total": 130},
    }
    agg = _aggregate_walkforward_quality(summary)
    assert agg["markets_used"] == 2
    assert abs(float(agg["auc_mean"]) - 0.59) < 1e-9
    assert abs(float(agg["brier_mean"]) - 0.235) < 1e-9
    assert agg["n_total"] == 270


def test_anytime_select_features_excludes_disabled_entries() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_anytime_contract_") as td:
        contract = Path(td) / "anytime.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: anytime",
                    "required_features:",
                    "  - lambda_home_l1",
                    "  - lambda_away_l1",
                    "optional_features:",
                    "  - home_recent_xg_mean_5",
                    "disabled_features:",
                    "  - home_recent_xg_mean_5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame(
            [
                {
                    "lambda_home_l1": 1.2,
                    "lambda_away_l1": 1.1,
                    "home_recent_xg_mean_5": 1.4,
                }
            ]
        )
        selected = _select_features(frame, contract)
        assert selected == ["lambda_home_l1", "lambda_away_l1"]


def test_anytime_select_features_supports_dbreuse_player_context_contract() -> None:
    contract = Path("model_v2/feature_contracts/experiments/anytime_dbreuse_player_context_v1.yaml")
    frame = pd.DataFrame(
        [
            {
                "home_sample_size": 10,
                "away_sample_size": 11,
                "lambda_home_l1": 1.4,
                "lambda_away_l1": 1.0,
                "adj_lambda_home_resolved": 1.5,
                "adj_lambda_away_resolved": 1.1,
                "home_rolling_xg": 1.7,
                "away_rolling_xg": 1.2,
                "home_rolling_sot": 4.2,
                "away_rolling_sot": 3.9,
                "goal_diff_proxy": 0.5,
                "xg_net_diff": 0.4,
                "home_missing_market_value_total": 20_000_000,
                "home_top2_attack_xga_share": 0.72,
                "lineup_completeness_home": 1.36,
                "home_missing_market_value_total_is_missing": 0,
                "home_availability_refresh_hours_dbreuse_is_missing": 0,
            }
        ]
    )

    selected = _select_features(frame, contract)

    assert "home_missing_market_value_total" in selected
    assert "home_top2_attack_xga_share" in selected
    assert "lineup_completeness_home" in selected
    assert "home_missing_market_value_total_is_missing" in selected
    assert "home_availability_refresh_hours_dbreuse_is_missing" in selected


def test_anytime_prepare_phase_targets_derives_second_half_goals() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_goals": 3,
                "away_goals": 1,
                "home_goals_p1": 1,
                "away_goals_p1": 1,
            },
            {
                "home_goals": 2,
                "away_goals": 0,
                "home_goals_p1": None,
                "away_goals_p1": None,
            },
        ]
    )
    out = _prepare_phase_targets(frame)
    assert out.loc[0, "home_goals_p2"] == 2.0
    assert out.loc[0, "away_goals_p2"] == 0.0
    assert pd.isna(out.loc[1, "home_goals_p2"])
    assert pd.isna(out.loc[1, "away_goals_p2"])


def test_anytime_build_regressor_uses_custom_poisson_alpha() -> None:
    model = _build_regressor("poisson_glm", poisson_alpha=0.3)
    assert isinstance(model, PoissonRegressor)
    assert model.alpha == pytest.approx(0.3)


def test_load_training_frame_reads_pit_dataset_and_enforces_validation() -> None:
    with tempfile.TemporaryDirectory(prefix="anytime_pit_dataset_") as td:
        root = Path(td)
        dataset_path = root / "dataset.csv"
        pd.DataFrame(
            [{"fixture_id": 2, "prediction_time_utc": "2026-03-01T00:00:00+00:00"}]
        ).to_csv(dataset_path, index=False)
        (root / "pit_validation_report.json").write_text(
            json.dumps({"status": "passed"}), encoding="utf-8"
        )

        frame, source = train_anytime._load_training_frame(dataset_path)

        assert source == str(dataset_path)
        assert list(frame["fixture_id"]) == [2]


def test_load_training_frame_raises_when_validation_failed() -> None:
    with tempfile.TemporaryDirectory(prefix="anytime_pit_invalid_") as td:
        root = Path(td)
        dataset_path = root / "dataset.csv"
        pd.DataFrame([{"fixture_id": 2}]).to_csv(dataset_path, index=False)
        (root / "pit_validation_report.json").write_text(
            json.dumps({"status": "failed"}), encoding="utf-8"
        )

        with pytest.raises(RuntimeError, match="PIT dataset failed validation"):
            train_anytime._load_training_frame(dataset_path)


def test_prepare_state_ladder_labels_adds_conditional_targets() -> None:
    frame = pd.DataFrame(
        [
            {
                "target_h_1up": 1.0,
                "target_h_2up": 1.0,
                "target_a_1up": 0.0,
                "target_a_2up": 0.0,
                "first_home_lead_minute": 18,
                "first_away_lead_minute": None,
            },
            {
                "target_h_1up": 0.0,
                "target_h_2up": 0.0,
                "target_a_1up": 1.0,
                "target_a_2up": 0.0,
                "first_home_lead_minute": None,
                "first_away_lead_minute": 54,
            },
        ]
    )
    out = prepare_state_ladder_labels(frame)
    assert out.loc[0, "target_h_2up_given_h_1up"] == 1.0
    assert pd.isna(out.loc[0, "target_a_2up_given_a_1up"])
    assert out.loc[0, "target_home_first_lead"] == 1.0
    assert out.loc[0, "target_home_first_lead_early"] == 1.0
    assert out.loc[1, "target_a_2up_given_a_1up"] == 0.0
    assert out.loc[1, "target_away_first_lead"] == 1.0


def test_state_ladder_goal_models_emit_valid_anytime_predictions() -> None:
    rows: list[dict[str, object]] = []
    for idx in range(48):
        strong_home = idx % 4 in {0, 1}
        home_l1 = 1.9 if strong_home else 1.0
        away_l1 = 0.9 if strong_home else 1.6
        home_p1 = 1 if strong_home else 0
        away_p1 = 0 if strong_home else 1
        home_goals = 2 if strong_home else 1
        away_goals = 0 if strong_home else 2
        rows.append(
            {
                "fixture_id": idx + 1,
                "lambda_home_l1": float(home_l1),
                "lambda_away_l1": float(away_l1),
                "home_goals": float(home_goals),
                "away_goals": float(away_goals),
                "home_goals_p1": float(home_p1),
                "away_goals_p1": float(away_p1),
                "target_h_1up": float(strong_home),
                "target_a_1up": float(not strong_home),
                "target_h_2up": float(strong_home),
                "target_a_2up": 0.0,
                "first_home_lead_minute": 20.0 if strong_home else None,
                "first_away_lead_minute": None if strong_home else 24.0,
            }
        )
    frame = prepare_state_ladder_labels(_prepare_phase_targets(pd.DataFrame(rows)))
    features = ["lambda_home_l1", "lambda_away_l1"]
    models = _fit_goal_models(
        train_df=frame,
        features=features,
        model_type="poisson_glm",
        path_version="state_ladder",
        home_poisson_alpha=1.0,
        away_poisson_alpha=1.0,
    )
    preds = _predict_goal_rates(models=models, x=frame[features], path_version="state_ladder")
    derived = _derive_anytime_frame(preds=preds, path_version="state_ladder", max_goals=8)
    assert set(derived.columns) == {"h_1up", "a_1up", "h_2up", "a_2up"}
    assert (derived["h_2up"] <= derived["h_1up"] + 1e-9).all()
    assert (derived["a_2up"] <= derived["a_1up"] + 1e-9).all()


def test_state_ladder_constant_model_prior_emits_valid_anytime_predictions() -> None:
    rows: list[dict[str, object]] = []
    for idx in range(40):
        strong_home = idx % 3 == 0
        rows.append(
            {
                "fixture_id": idx + 1,
                "lambda_home_l1": 1.8 if strong_home else 1.2,
                "lambda_away_l1": 0.8 if strong_home else 1.4,
                "home_goals": 2.0 if strong_home else 1.0,
                "away_goals": 0.0 if strong_home else 1.0,
                "home_goals_p1": 1.0 if strong_home else 0.0,
                "away_goals_p1": 0.0 if strong_home else 0.0,
                "target_h_1up": float(strong_home),
                "target_a_1up": float(not strong_home),
                "target_h_2up": float(strong_home),
                "target_a_2up": 0.0,
            }
        )
    frame = prepare_state_ladder_labels(_prepare_phase_targets(pd.DataFrame(rows)))
    features = ["lambda_home_l1", "lambda_away_l1"]
    models = _fit_goal_models(
        train_df=frame,
        features=features,
        model_type="poisson_glm",
        path_version="state_ladder",
        home_poisson_alpha=1.0,
        away_poisson_alpha=1.0,
        state_ladder_prior_version="constant_model",
    )
    preds = _predict_goal_rates(
        models=models,
        x=frame[features],
        path_version="state_ladder",
        state_ladder_prior_version="constant_model",
    )
    derived = _derive_anytime_frame(preds=preds, path_version="state_ladder", max_goals=8)
    assert set(derived.columns) == {"h_1up", "a_1up", "h_2up", "a_2up"}
    assert (derived["h_2up"] <= derived["h_1up"] + 1e-9).all()


def test_direct_monotone_goal_models_emit_valid_anytime_predictions() -> None:
    rows: list[dict[str, object]] = []
    for idx in range(42):
        strong_home = idx % 2 == 0
        rows.append(
            {
                "fixture_id": idx + 1,
                "match_datetime_utc": f"2024-01-{(idx % 28) + 1:02d}T12:00:00Z",
                "lambda_home_l1": 1.9 if strong_home else 1.1,
                "lambda_away_l1": 0.8 if strong_home else 1.5,
                "home_goals": 2.0 if strong_home else 1.0,
                "away_goals": 0.0 if strong_home else 2.0,
                "target_h_1up": float(strong_home),
                "target_a_1up": float(not strong_home),
                "target_h_2up": float(strong_home),
                "target_a_2up": 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    features = ["lambda_home_l1", "lambda_away_l1"]
    models = _fit_goal_models(
        train_df=frame,
        features=features,
        model_type="poisson_glm",
        path_version="direct_monotone",
        home_poisson_alpha=1.0,
        away_poisson_alpha=1.0,
    )
    preds = _predict_goal_rates(models=models, x=frame[features], path_version="direct_monotone")
    derived = _derive_anytime_frame(preds=preds, path_version="direct_monotone", max_goals=8)
    assert set(derived.columns) == {"h_1up", "a_1up", "h_2up", "a_2up"}
    assert (derived["h_2up"] <= derived["h_1up"] + 1e-9).all()
    assert (derived["a_2up"] <= derived["a_1up"] + 1e-9).all()


def test_fit_blend_weight_prefers_shrinkage_when_head_is_noisier() -> None:
    prior = [0.20, 0.25, 0.75, 0.80]
    head = [0.05, 0.95, 0.10, 0.90]
    y_true = [0, 0, 1, 1]
    alpha = _fit_blend_weight(prior_probs=prior, head_probs=head, y_true=y_true)
    assert 0.0 <= alpha <= 1.0
    assert alpha < 1.0
