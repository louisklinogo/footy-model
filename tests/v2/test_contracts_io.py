from __future__ import annotations

from pathlib import Path
import tempfile

from src.modeling.v2.io.contracts import load_feature_contract


def test_load_feature_contract_reads_lists() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_contract_") as td:
        path = Path(td) / "scoreline.yaml"
        path.write_text(
            "\n".join(
                [
                    "family: scoreline",
                    "required_features:",
                    "  - home_rolling_xg",
                    "optional_features:",
                    "  - odds_over_15",
                    "missingness_indicators:",
                    "  - odds_over_15_is_missing",
                    "forbidden_cross_family_features:",
                    "  - corners_gap_95",
                    "disabled_features:",
                    "  - home_rolling_crosses",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        contract = load_feature_contract(path)
        assert contract.family == "scoreline"
        assert contract.required_features == ["home_rolling_xg"]
        assert contract.optional_features == ["odds_over_15"]
        assert contract.missingness_indicators == ["odds_over_15_is_missing"]
        assert contract.forbidden_cross_family_features == ["corners_gap_95"]
        assert contract.disabled_features == ["home_rolling_crosses"]


def test_load_scoreline_live_parity_plus_contract() -> None:
    path = Path("model_v2/feature_contracts/experiments/scoreline_live_parity_plus.yaml")
    contract = load_feature_contract(path)

    assert contract.family == "scoreline"
    assert "adj_lambda_home_final" in contract.required_features
    assert "adj_lambda_away_final" in contract.required_features
    assert "rule_fired_home" in contract.optional_features
    assert "odds_over_15" in contract.optional_features
    assert "implied_over15" in contract.optional_features
    assert "home_availability_known" in contract.optional_features
    assert "home_lineup_known" in contract.optional_features
    assert "adj_lambda_home_final_is_missing" in contract.missingness_indicators


def test_load_scoreline_live_parity_strict_contract() -> None:
    path = Path("model_v2/feature_contracts/experiments/scoreline_live_parity_strict.yaml")
    contract = load_feature_contract(path)

    assert contract.family == "scoreline"
    assert "adj_lambda_home_final" in contract.required_features
    assert "adj_lambda_away_final" in contract.required_features
    assert "rule_fired_home" in contract.optional_features
    assert "odds_over_15" in contract.optional_features
    assert "implied_over15" in contract.optional_features
    assert "home_availability_known" not in contract.optional_features
    assert "home_lineup_known" not in contract.optional_features
    assert "home_recent_xg_mean_5" not in contract.optional_features
    assert "adj_lambda_home_final_is_missing" in contract.missingness_indicators
