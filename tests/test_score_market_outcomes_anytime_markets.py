from src.modeling.evaluation.score_market_outcomes_fixtures_first import compute_actual


def _row(market_code: str, **kwargs):
    base = {
        "market_code": market_code,
        "home_goals": 2,
        "away_goals": 1,
        "home_led_by_1_any": True,
        "away_led_by_1_any": False,
        "home_led_by_2_any": True,
        "away_led_by_2_any": False,
    }
    base.update(kwargs)
    return base


def test_anytime_markets_use_incident_flags():
    assert compute_actual(_row("h_1up")) == 1.0
    assert compute_actual(_row("a_1up")) == 0.0
    assert compute_actual(_row("h_2up")) == 1.0
    assert compute_actual(_row("a_2up")) == 0.0


def test_anytime_markets_return_none_when_missing_flags():
    assert compute_actual(_row("h_1up", home_led_by_1_any=None)) is None
    assert compute_actual(_row("a_1up", away_led_by_1_any=None)) is None
    assert compute_actual(_row("h_2up", home_led_by_2_any=None)) is None
    assert compute_actual(_row("a_2up", away_led_by_2_any=None)) is None
