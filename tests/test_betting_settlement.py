# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.market_contract import get_linchpin_market_codes
from src.betting.settlement import settle_asian_handicap
from src.betting.settlement import settle_market
from src.betting.settlement import supported_market_codes


def test_all_linchpin_markets_are_supported() -> None:
    contract_codes = set(get_linchpin_market_codes())
    supported_codes = set(supported_market_codes())
    assert contract_codes == supported_codes


def test_ah_h15_win_and_loss() -> None:
    win_result = settle_market("ah_h15", odds=2.2, home_goals=3, away_goals=1)
    loss_result = settle_market("ah_h15", odds=2.2, home_goals=2, away_goals=1)

    assert win_result.actual == 1.0
    assert win_result.return_factor == 2.2
    assert win_result.outcome == "full_win"

    assert loss_result.actual == 0.0
    assert loss_result.return_factor == 0.0
    assert loss_result.outcome == "full_loss"


def test_ah_h05_binary() -> None:
    result = settle_market("ah_h05", odds=1.95, home_goals=1, away_goals=0)
    assert result.actual == 1.0
    assert result.return_factor == 1.95
    assert result.outcome == "full_win"


def test_eh_h1_push_case() -> None:
    result = settle_market("eh_h1", odds=2.4, home_goals=2, away_goals=1)
    assert result.actual is None
    assert result.return_factor == 1.0
    assert result.outcome == "push"


def test_canonical_eh3_draw_is_true_win_not_push() -> None:
    result = settle_market("eh3_0_1_draw", odds=3.8, home_goals=2, away_goals=1)
    assert result.actual == 1.0
    assert result.return_factor == 3.8
    assert result.outcome == "full_win"


def test_canonical_ah2_whole_line_push_is_supported() -> None:
    result = settle_market("ah2_home_p10", odds=1.9, home_goals=1, away_goals=2)
    assert result.actual is None
    assert result.return_factor == 1.0
    assert result.outcome == "push"


def test_canonical_ah2_away_minus_half_win_is_supported() -> None:
    result = settle_market("ah2_away_m05", odds=3.6, home_goals=1, away_goals=2)
    assert result.actual == 1.0
    assert result.return_factor == 3.6
    assert result.outcome == "full_win"


def test_non_handicap_binary_markets() -> None:
    dc_result = settle_market("dc_x2", odds=1.6, home_goals=1, away_goals=2)
    o15_result = settle_market("o15", odds=1.7, home_goals=1, away_goals=1)
    u35_result = settle_market("u35", odds=1.8, home_goals=2, away_goals=1)
    incident_result = settle_market(
        "h_1up",
        odds=1.9,
        incident_lead_states={"home_led_by_1_any": True},
    )

    assert dc_result.actual == 1.0
    assert dc_result.return_factor == 1.6

    assert o15_result.actual == 1.0
    assert o15_result.return_factor == 1.7

    assert u35_result.actual == 1.0
    assert u35_result.return_factor == 1.8

    assert incident_result.actual == 1.0
    assert incident_result.return_factor == 1.9


def test_asian_handicap_quarter_line_half_loss() -> None:
    result = settle_asian_handicap(
        market_code="ah_custom_h025",
        selected_goals=1,
        opponent_goals=1,
        handicap=-0.25,
        odds=2.0,
    )
    assert result.actual is None
    assert result.return_factor == 0.5
    assert result.outcome == "half_loss"


def test_asian_handicap_quarter_line_half_win() -> None:
    result = settle_asian_handicap(
        market_code="ah_custom_h025",
        selected_goals=2,
        opponent_goals=1,
        handicap=-0.75,
        odds=2.4,
    )
    assert result.actual is None
    assert result.return_factor == 1.7
    assert result.outcome == "half_win"


def test_missing_required_inputs_void() -> None:
    corners_void = settle_market("c75", odds=1.8)
    incident_void = settle_market("h_1up", odds=1.9, incident_lead_states={})

    assert corners_void.actual is None
    assert corners_void.return_factor == 1.0
    assert corners_void.outcome == "void"

    assert incident_void.actual is None
    assert incident_void.return_factor == 1.0
    assert incident_void.outcome == "void"
