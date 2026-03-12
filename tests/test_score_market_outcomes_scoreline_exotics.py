from src.betting.settlement import settle_market, supported_market_codes
from src.modeling.evaluation.score_market_outcomes_fixtures_first import MARKETS, compute_actual


def _row(market_code: str, home_goals: int, away_goals: int):
    return {
        "market_code": market_code,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def test_scoreline_multigoals_actuals_settle_correctly() -> None:
    assert compute_actual(_row("mg_1_3", home_goals=2, away_goals=1)) == 1.0
    assert compute_actual(_row("mg_4_6", home_goals=2, away_goals=1)) == 0.0
    assert compute_actual(_row("hmg_1_3", home_goals=2, away_goals=1)) == 1.0
    assert compute_actual(_row("amg_1_3", home_goals=2, away_goals=1)) == 1.0
    assert compute_actual(_row("amg_4p", home_goals=2, away_goals=1)) == 0.0


def test_scoreline_multiscore_actuals_settle_correctly() -> None:
    assert compute_actual(_row("ms_h_1_0_2_0_3_0", home_goals=2, away_goals=0)) == 1.0
    assert compute_actual(_row("ms_a_0_1_0_2_0_3", home_goals=0, away_goals=2)) == 1.0
    assert compute_actual(_row("ms_draw", home_goals=1, away_goals=1)) == 1.0
    assert compute_actual(_row("ms_other_homewin", home_goals=4, away_goals=3)) == 1.0
    assert compute_actual(_row("ms_other_awaywin", home_goals=3, away_goals=4)) == 1.0
    assert compute_actual(_row("ms_other_homewin", home_goals=2, away_goals=1)) == 0.0


def test_settlement_support_and_score_inventory_include_scoreline_exotics() -> None:
    expected = {
        "mg_1_3",
        "hmg_1_3",
        "amg_1_3",
        "ms_h_1_0_2_0_3_0",
        "ms_draw",
        "ms_other_homewin",
    }
    assert expected.issubset(set(MARKETS))
    assert expected.issubset(set(supported_market_codes()))


def test_settle_market_handles_scoreline_exotics() -> None:
    assert settle_market("mg_1_3", odds=2.0, home_goals=2, away_goals=1).actual == 1.0
    assert settle_market("ms_other_awaywin", odds=2.0, home_goals=3, away_goals=4).actual == 1.0
