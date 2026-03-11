from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.modeling.v2.families.corners.features import (
    add_corners_context_features,
    apply_league_regime,
    fit_league_regime,
)
from src.modeling.v2.families.corners import train_corners
from src.modeling.v2.families.corners.train_corners import (
    _derive_corner_frame,
    _fit_corner_models,
    _predict_corner_rates,
    _aggregate_walkforward_quality,
    _select_features,
)


def test_corners_select_features_raises_when_required_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_corners_contract_") as td:
        contract = Path(td) / "corners.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: corners",
                    "required_features:",
                    "  - home_rolling_corners",
                    "  - away_rolling_corners",
                    "optional_features:",
                    "  - odds_c95_over",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame([{"home_rolling_corners": 5.1, "odds_c95_over": 1.9}])
        with pytest.raises(RuntimeError):
            _select_features(frame, contract)


def test_corners_select_features_includes_optional_when_available() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_corners_contract_") as td:
        contract = Path(td) / "corners.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: corners",
                    "required_features:",
                    "  - home_rolling_corners",
                    "optional_features:",
                    "  - odds_c95_over",
                    "missingness_indicators:",
                    "  - odds_c95_over_is_missing",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame(
            [
                {
                    "home_rolling_corners": 5.1,
                    "odds_c95_over": 1.9,
                    "odds_c95_over_is_missing": 0,
                }
            ]
        )
        selected = _select_features(frame, contract)
        assert selected == [
            "home_rolling_corners",
            "odds_c95_over",
            "odds_c95_over_is_missing",
        ]


def test_corners_aggregate_walkforward_quality_summarizes_markets() -> None:
    summary = {
        "c85": {"auc_mean": 0.57, "brier_mean": 0.24, "n_total": 200},
        "c95": {"auc_mean": 0.55, "brier_mean": 0.25, "n_total": 180},
    }
    agg = _aggregate_walkforward_quality(summary)
    assert agg["markets_used"] == 2
    assert abs(float(agg["auc_mean"]) - 0.56) < 1e-9
    assert abs(float(agg["brier_mean"]) - 0.245) < 1e-9
    assert agg["n_total"] == 380


def test_corners_select_features_excludes_disabled_entries() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_corners_contract_") as td:
        contract = Path(td) / "corners.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: corners",
                    "required_features:",
                    "  - home_rolling_corners",
                    "  - away_rolling_corners",
                    "optional_features:",
                    "  - home_rolling_crosses",
                    "disabled_features:",
                    "  - home_rolling_crosses",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame(
            [
                {
                    "home_rolling_corners": 5.1,
                    "away_rolling_corners": 4.8,
                    "home_rolling_crosses": 14.2,
                }
            ]
        )
        selected = _select_features(frame, contract)
        assert selected == ["home_rolling_corners", "away_rolling_corners"]


def test_load_training_frame_reads_pit_dataset_and_enforces_validation() -> None:
    with tempfile.TemporaryDirectory(prefix="corners_pit_dataset_") as td:
        root = Path(td)
        dataset_path = root / "dataset.csv"
        pd.DataFrame(
            [{"fixture_id": 1, "prediction_time_utc": "2026-03-01T00:00:00+00:00"}]
        ).to_csv(dataset_path, index=False)
        (root / "pit_validation_report.json").write_text(
            json.dumps({"status": "passed"}), encoding="utf-8"
        )

        frame, source = train_corners._load_training_frame(dataset_path)

        assert source == str(dataset_path)
        assert list(frame["fixture_id"]) == [1]


def test_load_training_frame_raises_when_validation_failed() -> None:
    with tempfile.TemporaryDirectory(prefix="corners_pit_invalid_") as td:
        root = Path(td)
        dataset_path = root / "dataset.csv"
        pd.DataFrame([{"fixture_id": 1}]).to_csv(dataset_path, index=False)
        (root / "pit_validation_report.json").write_text(
            json.dumps({"status": "failed"}), encoding="utf-8"
        )

        with pytest.raises(RuntimeError, match="PIT dataset failed validation"):
            train_corners._load_training_frame(dataset_path)


def test_totals_first_corner_models_emit_valid_markets() -> None:
    rows: list[dict[str, float | str | int]] = []
    for idx in range(36):
        high_total = idx % 3 == 0
        home_corners = 6.0 if high_total else 4.0
        away_corners = 5.0 if high_total else 3.0
        rows.append(
            {
                "fixture_id": idx + 1,
                "match_datetime_utc": f"2024-02-{(idx % 28) + 1:02d}T12:00:00Z",
                "home_rolling_corners": 6.4 if high_total else 4.8,
                "away_rolling_corners": 5.9 if high_total else 3.9,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": home_corners + away_corners,
            }
        )
    frame = pd.DataFrame(rows)
    features = ["home_rolling_corners", "away_rolling_corners"]
    models = _fit_corner_models(
        train_df=frame,
        features=features,
        model_type="poisson_glm",
        path_version="totals_first",
    )
    preds = _predict_corner_rates(models=models, x=frame[features], path_version="totals_first")
    derived = _derive_corner_frame(
        preds=preds,
        total_r=None,
        home_r=None,
        away_r=None,
        path_version="totals_first",
    )
    assert {"c75", "c85", "c95", "c105", "hc25", "hc35", "hc45", "hc55", "ac25", "ac35", "ac45", "ac55"} == set(derived.columns)
    assert (derived["c75"] >= derived["c85"]).all()
    assert (derived["c85"] >= derived["c95"]).all()
    assert (derived["c95"] >= derived["c105"]).all()
    assert (derived["hc25"] >= derived["hc35"]).all()
    assert (derived["ac25"] >= derived["ac35"]).all()


def test_totals_first_residual_corner_models_preserve_total_and_emit_valid_markets() -> None:
    rows: list[dict[str, float | str | int]] = []
    for idx in range(48):
        high_total = idx % 4 in {0, 1}
        home_bias = idx % 2 == 0
        home_corners = 7.0 if (high_total and home_bias) else 5.0 if high_total else 4.0 if home_bias else 3.0
        away_corners = 4.0 if (high_total and home_bias) else 6.0 if high_total else 2.0 if home_bias else 4.0
        rows.append(
            {
                "fixture_id": idx + 1,
                "match_datetime_utc": f"2024-03-{(idx % 28) + 1:02d}T12:00:00Z",
                "home_rolling_corners": 6.8 if home_bias else 4.6,
                "away_rolling_corners": 4.9 if home_bias else 5.7,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": home_corners + away_corners,
            }
        )
    frame = pd.DataFrame(rows)
    features = ["home_rolling_corners", "away_rolling_corners"]
    models = _fit_corner_models(
        train_df=frame,
        features=features,
        model_type="poisson_glm",
        path_version="totals_first_residual",
    )
    preds = _predict_corner_rates(
        models=models,
        x=frame[features],
        path_version="totals_first_residual",
    )
    derived = _derive_corner_frame(
        preds=preds,
        total_r=None,
        home_r=None,
        away_r=None,
        path_version="totals_first_residual",
    )
    assert "home_delta" in preds
    assert np.allclose(preds["home"] + preds["away"], preds["total"])
    assert (derived["c75"] >= derived["c85"]).all()
    assert (derived["c85"] >= derived["c95"]).all()
    assert (derived["hc25"] >= derived["hc35"]).all()


def test_totals_first_market_heads_emit_direct_team_market_probs() -> None:
    rows: list[dict[str, float | str | int]] = []
    for idx in range(60):
        high_total = idx % 3 != 2
        home_bias = idx % 4 in {0, 1}
        home_corners = 7.0 if (high_total and home_bias) else 5.0 if high_total else 4.0 if home_bias else 2.0
        away_corners = 4.0 if (high_total and home_bias) else 6.0 if high_total else 2.0 if home_bias else 5.0
        rows.append(
            {
                "fixture_id": idx + 1,
                "match_datetime_utc": f"2024-04-{(idx % 28) + 1:02d}T12:00:00Z",
                "home_rolling_corners": 6.9 if home_bias else 4.2,
                "away_rolling_corners": 4.4 if home_bias else 5.8,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": home_corners + away_corners,
                "target_hc25": int(home_corners > 2.5),
                "target_hc35": int(home_corners > 3.5),
                "target_hc45": int(home_corners > 4.5),
                "target_hc55": int(home_corners > 5.5),
                "target_ac25": int(away_corners > 2.5),
                "target_ac35": int(away_corners > 3.5),
                "target_ac45": int(away_corners > 4.5),
                "target_ac55": int(away_corners > 5.5),
            }
        )
    frame = pd.DataFrame(rows)
    features = ["home_rolling_corners", "away_rolling_corners"]
    models = _fit_corner_models(
        train_df=frame,
        features=features,
        model_type="poisson_glm",
        path_version="totals_first_market_heads",
    )
    preds = _predict_corner_rates(
        models=models,
        x=frame[features],
        path_version="totals_first_market_heads",
    )
    derived = _derive_corner_frame(
        preds=preds,
        total_r=None,
        home_r=None,
        away_r=None,
        path_version="totals_first_market_heads",
    )
    assert "team_market_probs" in preds
    assert set(preds["team_market_probs"].keys()) == {
        "hc25", "hc35", "hc45", "hc55", "ac25", "ac35", "ac45", "ac55"
    }
    assert np.allclose(preds["home"] + preds["away"], preds["total"])
    assert (derived["hc25"] >= derived["hc35"]).all()
    assert (derived["hc35"] >= derived["hc45"]).all()
    assert (derived["ac25"] >= derived["ac35"]).all()
    assert (derived["ac35"] >= derived["ac45"]).all()


def test_totals_first_team_market_calibrated_emits_monotone_team_market_probs() -> None:
    rows: list[dict[str, float | str | int]] = []
    for idx in range(84):
        high_home = idx % 4 in (0, 1)
        high_away = idx % 6 in (0, 3)
        home_corners = 7.0 if high_home else 4.0 if idx % 3 else 2.0
        away_corners = 6.0 if high_away else 3.0 if idx % 2 else 1.0
        rows.append(
            {
                "fixture_id": idx + 1,
                "league_code": "A" if idx % 2 == 0 else "B",
                "match_datetime_utc": f"2024-06-{(idx % 28) + 1:02d}T12:00:00Z",
                "home_sample_size": 12.0,
                "away_sample_size": 12.0,
                "home_rolling_corners": 6.5 if high_home else 4.2,
                "home_rolling_corners_against": 3.2,
                "away_rolling_corners": 5.8 if high_away else 3.4,
                "away_rolling_corners_against": 3.1,
                "home_rolling_box_touches": 28.0 if high_home else 20.0,
                "away_rolling_box_touches": 25.0 if high_away else 17.0,
                "home_rolling_xg": 1.7 if high_home else 1.1,
                "home_rolling_xg_against": 1.0,
                "away_rolling_xg": 1.5 if high_away else 0.9,
                "away_rolling_xg_against": 1.0,
                "home_rolling_sot": 5.7 if high_home else 3.8,
                "home_rolling_sot_against": 3.1,
                "away_rolling_sot": 5.0 if high_away else 3.0,
                "away_rolling_sot_against": 3.0,
                "xg_net_diff": (1.7 if high_home else 1.1) - (1.5 if high_away else 0.9),
                "home_season_baseline_xg": 1.4,
                "away_season_baseline_xg": 1.3,
                "home_recent_xg_mean_5": 1.6 if high_home else 1.2,
                "away_recent_xg_mean_5": 1.5 if high_away else 1.0,
                "home_regressed_recent_xg": 1.5 if high_home else 1.15,
                "away_regressed_recent_xg": 1.45 if high_away else 1.0,
                "regressed_xg_diff": (1.5 if high_home else 1.15) - (1.45 if high_away else 1.0),
                "home_recent_vs_baseline_zscore": 0.6 if high_home else -0.2,
                "away_recent_vs_baseline_zscore": 0.5 if high_away else -0.3,
                "recent_vs_baseline_gap": (0.6 if high_home else -0.2) - (0.5 if high_away else -0.3),
                "style_delta": 1.0 if high_home and not high_away else -0.5 if high_away and not high_home else 0.0,
                "home_rolling_possession": 56.0 if high_home else 49.0,
                "home_rolling_possession_against": 44.0,
                "away_rolling_possession": 54.0 if high_away else 47.0,
                "away_rolling_possession_against": 46.0,
                "league_total_corners_mean": 10.5 if idx % 2 == 0 else 8.4,
                "league_home_corners_mean": 5.6 if idx % 2 == 0 else 4.4,
                "league_away_corners_mean": 4.9 if idx % 2 == 0 else 4.0,
                "league_c75_rate": 0.72 if idx % 2 == 0 else 0.58,
                "league_c85_rate": 0.61 if idx % 2 == 0 else 0.46,
                "league_c95_rate": 0.49 if idx % 2 == 0 else 0.34,
                "league_c105_rate": 0.37 if idx % 2 == 0 else 0.24,
                "league_fixture_count": 40.0 if idx % 2 == 0 else 35.0,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": home_corners + away_corners,
            }
        )
    frame = pd.DataFrame(rows)
    features = [
        "league_total_corners_mean",
        "league_home_corners_mean",
        "league_away_corners_mean",
        "league_c75_rate",
        "league_c85_rate",
        "league_c95_rate",
        "league_c105_rate",
        "league_fixture_count",
        "home_sample_size",
        "away_sample_size",
        "home_rolling_corners",
        "home_rolling_corners_against",
        "away_rolling_corners",
        "away_rolling_corners_against",
        "home_rolling_box_touches",
        "away_rolling_box_touches",
        "home_rolling_xg",
        "home_rolling_xg_against",
        "away_rolling_xg",
        "away_rolling_xg_against",
        "home_rolling_sot",
        "home_rolling_sot_against",
        "away_rolling_sot",
        "away_rolling_sot_against",
        "xg_net_diff",
        "home_season_baseline_xg",
        "away_season_baseline_xg",
        "home_recent_xg_mean_5",
        "away_recent_xg_mean_5",
        "home_regressed_recent_xg",
        "away_regressed_recent_xg",
        "regressed_xg_diff",
        "home_recent_vs_baseline_zscore",
        "away_recent_vs_baseline_zscore",
        "recent_vs_baseline_gap",
        "style_delta",
        "home_rolling_possession",
        "home_rolling_possession_against",
        "away_rolling_possession",
        "away_rolling_possession_against",
    ]
    models = _fit_corner_models(
        train_df=frame,
        features=features,
        model_type="poisson_glm",
        path_version="totals_first_team_market_calibrated",
    )
    preds = _predict_corner_rates(
        models=models,
        x=frame[features],
        path_version="totals_first_team_market_calibrated",
        total_r=None,
        home_r=None,
        away_r=None,
    )
    derived = _derive_corner_frame(
        preds=preds,
        total_r=None,
        home_r=None,
        away_r=None,
        path_version="totals_first_team_market_calibrated",
    )
    assert "team_market_probs" in preds
    assert set(preds["team_market_probs"].keys()) == {"hc25", "hc35", "hc45", "hc55", "ac25", "ac35", "ac45", "ac55"}
    assert (derived["hc25"] >= derived["hc35"]).all()
    assert (derived["hc35"] >= derived["hc45"]).all()
    assert (derived["hc45"] >= derived["hc55"]).all()
    assert (derived["ac25"] >= derived["ac35"]).all()
    assert (derived["ac35"] >= derived["ac45"]).all()
    assert (derived["ac45"] >= derived["ac55"]).all()


def test_totals_first_league_share_residual_emits_prior_aware_team_means() -> None:
    rows: list[dict[str, float | str | int]] = []
    for idx in range(84):
        league_code = "A" if idx % 2 == 0 else "B"
        home_bias = league_code == "A"
        total_corners = 11.0 if idx % 3 == 0 else 9.0 if idx % 3 == 1 else 7.0
        home_share = 0.62 if home_bias else 0.42
        home_corners = float(round(total_corners * home_share))
        away_corners = float(total_corners - home_corners)
        rows.append(
            {
                "fixture_id": idx + 1,
                "league_code": league_code,
                "match_datetime_utc": f"2024-07-{(idx % 28) + 1:02d}T12:00:00Z",
                "home_sample_size": 12.0,
                "away_sample_size": 12.0,
                "home_rolling_corners": 6.4 if home_bias else 4.1,
                "home_rolling_corners_against": 3.2,
                "away_rolling_corners": 4.0 if home_bias else 5.8,
                "away_rolling_corners_against": 3.1,
                "home_rolling_box_touches": 27.0 if home_bias else 21.0,
                "away_rolling_box_touches": 20.0 if home_bias else 25.0,
                "home_rolling_xg": 1.6 if home_bias else 1.1,
                "home_rolling_xg_against": 1.0,
                "away_rolling_xg": 1.0 if home_bias else 1.5,
                "away_rolling_xg_against": 1.0,
                "home_rolling_sot": 5.2 if home_bias else 3.8,
                "home_rolling_sot_against": 3.0,
                "away_rolling_sot": 3.6 if home_bias else 4.9,
                "away_rolling_sot_against": 3.0,
                "xg_net_diff": 0.4 if home_bias else -0.4,
                "home_season_baseline_xg": 1.4,
                "away_season_baseline_xg": 1.3,
                "home_recent_xg_mean_5": 1.55 if home_bias else 1.15,
                "away_recent_xg_mean_5": 1.05 if home_bias else 1.45,
                "home_regressed_recent_xg": 1.5 if home_bias else 1.1,
                "away_regressed_recent_xg": 1.0 if home_bias else 1.4,
                "regressed_xg_diff": 0.5 if home_bias else -0.3,
                "home_recent_vs_baseline_zscore": 0.5 if home_bias else -0.2,
                "away_recent_vs_baseline_zscore": -0.1 if home_bias else 0.4,
                "recent_vs_baseline_gap": 0.6 if home_bias else -0.6,
                "style_delta": 0.8 if home_bias else -0.6,
                "home_rolling_possession": 55.0 if home_bias else 48.0,
                "home_rolling_possession_against": 45.0,
                "away_rolling_possession": 47.0 if home_bias else 54.0,
                "away_rolling_possession_against": 46.0,
                "league_total_corners_mean": 9.8 if home_bias else 8.6,
                "league_home_corners_mean": 5.9 if home_bias else 3.8,
                "league_away_corners_mean": 3.9 if home_bias else 4.8,
                "league_home_share_mean": 0.60 if home_bias else 0.44,
                "league_c75_rate": 0.68 if home_bias else 0.57,
                "league_c85_rate": 0.58 if home_bias else 0.45,
                "league_c95_rate": 0.47 if home_bias else 0.32,
                "league_c105_rate": 0.34 if home_bias else 0.22,
                "league_fixture_count": 40.0 if home_bias else 36.0,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": total_corners,
            }
        )
    frame = pd.DataFrame(rows)
    features = [
        "league_total_corners_mean", "league_home_corners_mean", "league_away_corners_mean", "league_home_share_mean",
        "league_c75_rate", "league_c85_rate", "league_c95_rate", "league_c105_rate", "league_fixture_count",
        "home_sample_size", "away_sample_size", "home_rolling_corners", "home_rolling_corners_against", "away_rolling_corners", "away_rolling_corners_against",
        "home_rolling_box_touches", "away_rolling_box_touches", "home_rolling_xg", "home_rolling_xg_against", "away_rolling_xg", "away_rolling_xg_against",
        "home_rolling_sot", "home_rolling_sot_against", "away_rolling_sot", "away_rolling_sot_against", "xg_net_diff",
        "home_season_baseline_xg", "away_season_baseline_xg", "home_recent_xg_mean_5", "away_recent_xg_mean_5",
        "home_regressed_recent_xg", "away_regressed_recent_xg", "regressed_xg_diff", "home_recent_vs_baseline_zscore", "away_recent_vs_baseline_zscore",
        "recent_vs_baseline_gap", "style_delta", "home_rolling_possession", "home_rolling_possession_against", "away_rolling_possession", "away_rolling_possession_against",
    ]
    models = _fit_corner_models(
        train_df=frame,
        features=features,
        model_type="poisson_glm",
        path_version="totals_first_league_share_residual",
    )
    preds = _predict_corner_rates(
        models=models,
        x=frame[features],
        path_version="totals_first_league_share_residual",
        total_r=None,
        home_r=None,
        away_r=None,
    )
    derived = _derive_corner_frame(
        preds=preds,
        total_r=None,
        home_r=None,
        away_r=None,
        path_version="totals_first_league_share_residual",
    )
    assert "home_share_prior" in preds
    assert 0.0 <= float(models["home_share_prior_blend"]) <= 1.0
    mask_a = frame["league_code"] == "A"
    mask_b = frame["league_code"] == "B"
    assert float(np.mean(preds["home_share"][mask_a])) > float(np.mean(preds["home_share"][mask_b]))
    assert (derived["hc25"] >= derived["hc35"]).all()
    assert (derived["hc35"] >= derived["hc45"]).all()
    assert (derived["ac25"] >= derived["ac35"]).all()
    assert (derived["ac35"] >= derived["ac45"]).all()


def test_pmf_surface_blended_emits_coherent_total_and_team_market_probs() -> None:
    rows: list[dict[str, float | str | int]] = []
    for idx in range(96):
        high_total = idx % 4 in {0, 1}
        home_bias = idx % 3 == 0
        total_corners = 12.0 if high_total else 8.0 if idx % 4 == 2 else 6.0
        home_corners = 7.0 if (high_total and home_bias) else 5.0 if high_total else 4.0 if home_bias else 2.0
        away_corners = float(total_corners - home_corners)
        rows.append(
            {
                "fixture_id": idx + 1,
                "league_code": "A" if idx % 2 == 0 else "B",
                "match_datetime_utc": f"2024-08-{(idx % 28) + 1:02d}T12:00:00Z",
                "home_sample_size": 12.0,
                "away_sample_size": 12.0,
                "home_rolling_corners": 6.8 if home_bias else 4.3,
                "home_rolling_corners_against": 3.0,
                "away_rolling_corners": 4.1 if home_bias else 5.7,
                "away_rolling_corners_against": 3.1,
                "home_rolling_box_touches": 27.0 if home_bias else 20.0,
                "away_rolling_box_touches": 20.0 if home_bias else 25.0,
                "home_rolling_xg": 1.6 if home_bias else 1.0,
                "home_rolling_xg_against": 1.0,
                "away_rolling_xg": 1.0 if home_bias else 1.4,
                "away_rolling_xg_against": 1.0,
                "home_rolling_sot": 5.4 if home_bias else 3.8,
                "home_rolling_sot_against": 3.0,
                "away_rolling_sot": 3.5 if home_bias else 4.9,
                "away_rolling_sot_against": 3.0,
                "xg_net_diff": 0.4 if home_bias else -0.3,
                "home_season_baseline_xg": 1.4,
                "away_season_baseline_xg": 1.3,
                "home_recent_xg_mean_5": 1.5 if home_bias else 1.1,
                "away_recent_xg_mean_5": 1.0 if home_bias else 1.4,
                "home_regressed_recent_xg": 1.45 if home_bias else 1.1,
                "away_regressed_recent_xg": 1.0 if home_bias else 1.35,
                "regressed_xg_diff": 0.45 if home_bias else -0.25,
                "home_recent_vs_baseline_zscore": 0.5 if home_bias else -0.1,
                "away_recent_vs_baseline_zscore": -0.1 if home_bias else 0.3,
                "recent_vs_baseline_gap": 0.6 if home_bias else -0.4,
                "style_delta": 0.7 if home_bias else -0.5,
                "home_rolling_possession": 55.0 if home_bias else 48.0,
                "home_rolling_possession_against": 45.0,
                "away_rolling_possession": 47.0 if home_bias else 54.0,
                "away_rolling_possession_against": 46.0,
                "league_total_corners_mean": 10.1 if idx % 2 == 0 else 8.4,
                "league_home_corners_mean": 5.7 if idx % 2 == 0 else 4.0,
                "league_away_corners_mean": 4.4 if idx % 2 == 0 else 4.4,
                "league_home_share_mean": 0.57 if idx % 2 == 0 else 0.47,
                "league_c75_rate": 0.69 if idx % 2 == 0 else 0.55,
                "league_c85_rate": 0.58 if idx % 2 == 0 else 0.43,
                "league_c95_rate": 0.45 if idx % 2 == 0 else 0.31,
                "league_c105_rate": 0.33 if idx % 2 == 0 else 0.20,
                "league_fixture_count": 44.0 if idx % 2 == 0 else 38.0,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": total_corners,
            }
        )
    frame = pd.DataFrame(rows)
    features = [
        "league_total_corners_mean", "league_home_corners_mean", "league_away_corners_mean", "league_home_share_mean",
        "league_c75_rate", "league_c85_rate", "league_c95_rate", "league_c105_rate", "league_fixture_count",
        "home_sample_size", "away_sample_size", "home_rolling_corners", "home_rolling_corners_against", "away_rolling_corners", "away_rolling_corners_against",
        "home_rolling_box_touches", "away_rolling_box_touches", "home_rolling_xg", "home_rolling_xg_against", "away_rolling_xg", "away_rolling_xg_against",
        "home_rolling_sot", "home_rolling_sot_against", "away_rolling_sot", "away_rolling_sot_against", "xg_net_diff",
        "home_season_baseline_xg", "away_season_baseline_xg", "home_recent_xg_mean_5", "away_recent_xg_mean_5",
        "home_regressed_recent_xg", "away_regressed_recent_xg", "regressed_xg_diff", "home_recent_vs_baseline_zscore", "away_recent_vs_baseline_zscore",
        "recent_vs_baseline_gap", "style_delta", "home_rolling_possession", "home_rolling_possession_against", "away_rolling_possession", "away_rolling_possession_against",
    ]
    models = _fit_corner_models(
        train_df=frame,
        features=features,
        model_type="poisson_glm",
        path_version="pmf_surface_blended",
    )
    preds = _predict_corner_rates(
        models=models,
        x=frame[features],
        path_version="pmf_surface_blended",
        total_r=None,
        home_r=None,
        away_r=None,
    )
    derived = _derive_corner_frame(
        preds=preds,
        total_r=None,
        home_r=None,
        away_r=None,
        path_version="pmf_surface_blended",
    )
    assert "total_market_probs" in preds
    assert "team_market_probs" in preds
    assert set(preds["total_market_probs"].keys()) == {"c75", "c85", "c95", "c105"}
    assert set(preds["team_market_probs"].keys()) == {"hc25", "hc35", "hc45", "hc55", "ac25", "ac35", "ac45", "ac55"}
    assert (derived["c75"] >= derived["c85"]).all()
    assert (derived["c85"] >= derived["c95"]).all()
    assert (derived["c95"] >= derived["c105"]).all()
    assert (derived["hc25"] >= derived["hc35"]).all()
    assert (derived["hc35"] >= derived["hc45"]).all()
    assert (derived["hc45"] >= derived["hc55"]).all()
    assert (derived["ac25"] >= derived["ac35"]).all()
    assert (derived["ac35"] >= derived["ac45"]).all()
    assert (derived["ac45"] >= derived["ac55"]).all()


def test_totals_surface_calibrated_emits_direct_total_market_probs() -> None:
    rows: list[dict[str, float | str | int]] = []
    for idx in range(72):
        high_total = idx % 3 == 0
        medium_total = idx % 3 == 1
        total_corners = 12.0 if high_total else 9.0 if medium_total else 6.0
        home_corners = 7.0 if high_total else 5.0 if medium_total else 3.0
        away_corners = total_corners - home_corners
        rows.append(
            {
                "fixture_id": idx + 1,
                "match_datetime_utc": f"2024-05-{(idx % 28) + 1:02d}T12:00:00Z",
                "home_rolling_corners": 6.8 if high_total else 5.4 if medium_total else 4.1,
                "away_rolling_corners": 5.1 if high_total else 4.4 if medium_total else 3.2,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": total_corners,
                "target_c75": int(total_corners > 7.5),
                "target_c85": int(total_corners > 8.5),
                "target_c95": int(total_corners > 9.5),
                "target_c105": int(total_corners > 10.5),
            }
        )
    frame = pd.DataFrame(rows)
    features = ["home_rolling_corners", "away_rolling_corners"]
    models = _fit_corner_models(
        train_df=frame,
        features=features,
        model_type="poisson_glm",
        path_version="totals_surface_calibrated",
    )
    preds = _predict_corner_rates(
        models=models,
        x=frame[features],
        path_version="totals_surface_calibrated",
        total_r=None,
    )
    derived = _derive_corner_frame(
        preds=preds,
        total_r=None,
        home_r=None,
        away_r=None,
        path_version="totals_surface_calibrated",
    )
    assert "total_market_probs" in preds
    assert set(preds["total_market_probs"].keys()) == {"c75", "c85", "c95", "c105"}
    assert (derived["c75"] >= derived["c85"]).all()
    assert (derived["c85"] >= derived["c95"]).all()
    assert (derived["c95"] >= derived["c105"]).all()


def test_add_corners_context_features_derives_style_delta_from_formation() -> None:
    frame = pd.DataFrame(
        {
            "fixture_id": [1, 2],
            "home_formation": ["4-3-3", "3-5-2"],
            "away_formation": ["4-4-2", "4-2-3"],
        }
    )
    enriched = add_corners_context_features(frame)
    assert "style_delta" in enriched.columns
    assert enriched["style_delta"].notna().all()
    assert enriched.loc[0, "style_delta"] > 0.0


def test_apply_league_regime_adds_smoothed_league_features_with_fallback() -> None:
    train = pd.DataFrame(
        {
            "fixture_id": [1, 2, 3, 4],
            "league_code": ["A", "A", "B", "B"],
            "total_corners": [11.0, 10.0, 7.0, 6.0],
            "home_corners": [6.0, 5.0, 4.0, 3.0],
            "away_corners": [5.0, 5.0, 3.0, 3.0],
        }
    )
    regime = fit_league_regime(train, prior_strength=2.0)
    scored = apply_league_regime(
        pd.DataFrame({"fixture_id": [10, 11], "league_code": ["A", "Z"]}),
        regime,
    )
    assert scored.loc[0, "league_total_corners_mean"] > scored.loc[1, "league_total_corners_mean"]
    assert scored.loc[0, "league_c95_rate"] > scored.loc[1, "league_c95_rate"]
    assert scored.loc[1, "league_fixture_count"] == pytest.approx(regime["global"]["league_fixture_count"])
