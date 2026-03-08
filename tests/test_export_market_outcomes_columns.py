from src.modeling.export.export_market_outcomes_fixtures_first import OUT_COLUMNS


def test_export_columns_include_legacy_and_canonical_handicap_predictions() -> None:
    expected = {
        "p_ah_h05",
        "p_ah_a05",
        "p_ah_h15",
        "p_ah_a15",
        "p_eh_h1",
        "p_eh_a1",
        "p_ah2_home_m05",
        "p_ah2_away_p05",
        "p_ah2_away_m05",
        "p_ah2_home_p05",
        "p_ah2_home_m15",
        "p_ah2_away_p15",
        "p_ah2_away_m15",
        "p_ah2_home_p15",
        "p_eh3_0_1_home",
        "p_eh3_0_1_draw",
        "p_eh3_0_1_away",
        "p_eh3_1_0_home",
        "p_eh3_1_0_draw",
        "p_eh3_1_0_away",
    }
    assert expected.issubset(set(OUT_COLUMNS))