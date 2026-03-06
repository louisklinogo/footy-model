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
