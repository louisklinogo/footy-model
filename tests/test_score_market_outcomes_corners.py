from src.modeling.evaluation.score_market_outcomes_fixtures_first import compute_actual


def _row(market_code: str, **kwargs):
    base = {
        "market_code": market_code,
        "home_goals": 2,
        "away_goals": 1,
        "h_corners": 6,
        "a_corners": 4,
    }
    base.update(kwargs)
    return base


def test_corner_total_lines_settle_correctly():
    assert compute_actual(_row("c75")) == 1.0
    assert compute_actual(_row("c85")) == 1.0
    assert compute_actual(_row("c95")) == 1.0
    assert compute_actual(_row("c105")) == 0.0


def test_corner_team_lines_settle_correctly():
    assert compute_actual(_row("hc25")) == 1.0
    assert compute_actual(_row("hc35")) == 1.0
    assert compute_actual(_row("hc45")) == 1.0
    assert compute_actual(_row("hc55")) == 1.0
    assert compute_actual(_row("ac25")) == 1.0
    assert compute_actual(_row("ac35")) == 1.0
    assert compute_actual(_row("ac45")) == 0.0
    assert compute_actual(_row("ac55")) == 0.0


def test_corner_lines_require_corner_inputs():
    assert compute_actual(_row("c95", h_corners=None)) is None
    assert compute_actual(_row("hc35", h_corners=None)) is None
    assert compute_actual(_row("ac35", a_corners=None)) is None
