from __future__ import annotations

from pathlib import Path
import tempfile

from src.modeling.v2.data.feature_lineage import build_lineage


def test_build_lineage_reads_contracts_and_flags_required_optional() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_lineage_") as td:
        contracts_dir = Path(td)
        (contracts_dir / "sample.yaml").write_text(
            "\n".join(
                [
                    "family: sample_family",
                    "required_features:",
                    "  - home_rolling_xg",
                    "optional_features:",
                    "  - odds_over_15",
                    "missingness_indicators:",
                    "  - odds_over_15_is_missing",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        payload = build_lineage(contracts_dir)
        assert "sample_family" in payload["families"]
        rows = payload["families"]["sample_family"]
        by_feature = {row["feature"]: row for row in rows}

        assert by_feature["home_rolling_xg"]["required"] is True
        assert by_feature["home_rolling_xg"]["source"] == "team_premium_snapshots"
        assert by_feature["odds_over_15"]["optional"] is True
        assert by_feature["odds_over_15"]["source"] == "fixture_odds_markets"
        assert by_feature["odds_over_15_is_missing"]["missingness_indicator"] is True

