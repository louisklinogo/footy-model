from src.modeling.evaluation.score_market_outcomes_fixtures_first import (
    CANONICAL_HANDICAP_MARKETS,
    MARKETS,
    compute_actual,
    is_push_outcome,
)


def _row(market_code: str, home_goals: int, away_goals: int):
    return {
        "market_code": market_code,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def test_legacy_eh_proxy_still_treats_exact_margin_as_push() -> None:
    row = _row("eh_h1", home_goals=2, away_goals=1)
    assert compute_actual(row) is None
    assert is_push_outcome(row) is True


def test_canonical_eh_draw_is_real_class_not_push() -> None:
    row = _row("eh3_0_1_draw", home_goals=2, away_goals=1)
    assert compute_actual(row) == 1.0
    assert is_push_outcome(row) is False


def test_canonical_ah_whole_line_push_is_visible_to_scoring() -> None:
    row = _row("ah2_home_p10", home_goals=1, away_goals=2)
    assert compute_actual(row) is None
    assert is_push_outcome(row) is True


def test_score_market_inventory_includes_canonical_handicap_codes() -> None:
    assert set(CANONICAL_HANDICAP_MARKETS).issubset(set(MARKETS))