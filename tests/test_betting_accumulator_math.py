# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

import sys
from pathlib import Path
from math import isclose

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.settlement import settle_market, settle_asian_handicap


def calculate_accumulator_return(return_factors: list[float]) -> float:
    """
    Pure helper to multiply return factors.
    In a real system, this might live in src/betting/settlement.py or a backtester.
    For this task, we are testing the math logic.
    """
    total_return = 1.0
    for factor in return_factors:
        total_return *= factor
    return total_return


def test_accumulator_all_wins() -> None:
    # Leg 1: Home win (odds 2.0)
    res1 = settle_market("dc_1x", odds=2.0, home_goals=1, away_goals=0)
    # Leg 2: Over 1.5 (odds 1.5)
    res2 = settle_market("o15", odds=1.5, home_goals=1, away_goals=1)

    factors = [res1.return_factor, res2.return_factor]
    total_return = calculate_accumulator_return(factors)

    assert res1.return_factor == 2.0
    assert res2.return_factor == 1.5
    assert isclose(total_return, 3.0)


def test_accumulator_with_void_leg() -> None:
    # Leg 1: Win (odds 2.0)
    res1 = settle_market("dc_1x", odds=2.0, home_goals=1, away_goals=0)
    # Leg 2: Void (missing data)
    res2 = settle_market("o15", odds=1.5, home_goals=None, away_goals=None)

    factors = [res1.return_factor, res2.return_factor]
    total_return = calculate_accumulator_return(factors)

    assert res1.return_factor == 2.0
    assert res2.return_factor == 1.0  # Void factor is 1.0
    assert isclose(total_return, 2.0)


def test_accumulator_with_push_leg() -> None:
    # Leg 1: Win (odds 2.0)
    res1 = settle_market("dc_1x", odds=2.0, home_goals=1, away_goals=0)
    # Leg 2: Push (EH -1 with 1 goal margin)
    res2 = settle_market("eh_h1", odds=3.0, home_goals=2, away_goals=1)

    factors = [res1.return_factor, res2.return_factor]
    total_return = calculate_accumulator_return(factors)

    assert res1.return_factor == 2.0
    assert res2.return_factor == 1.0  # Push factor is 1.0
    assert isclose(total_return, 2.0)


def test_accumulator_with_half_loss_leg() -> None:
    # Leg 1: Win (odds 2.0)
    res1 = settle_market("dc_1x", odds=2.0, home_goals=1, away_goals=0)
    # Leg 2: Half Loss (AH -0.25 with draw)
    res2 = settle_asian_handicap(
        market_code="ah_custom_h025",
        selected_goals=1,
        opponent_goals=1,
        handicap=-0.25,
        odds=2.0,
    )

    factors = [res1.return_factor, res2.return_factor]
    total_return = calculate_accumulator_return(factors)

    assert res1.return_factor == 2.0
    assert res2.return_factor == 0.5  # Half loss factor is 0.5
    assert isclose(total_return, 1.0)


def test_accumulator_with_half_win_leg() -> None:
    # Leg 1: Win (odds 2.0)
    res1 = settle_market("dc_1x", odds=2.0, home_goals=1, away_goals=0)
    # Leg 2: Half Win (AH -0.75 with 1 goal margin)
    # (odds + 1) / 2 = (2.4 + 1) / 2 = 1.7
    res2 = settle_asian_handicap(
        market_code="ah_custom_h075",
        selected_goals=2,
        opponent_goals=1,
        handicap=-0.75,
        odds=2.4,
    )

    factors = [res1.return_factor, res2.return_factor]
    total_return = calculate_accumulator_return(factors)

    assert res1.return_factor == 2.0
    assert isclose(res2.return_factor, 1.7)
    assert isclose(total_return, 3.4)


def test_accumulator_full_loss() -> None:
    # Leg 1: Win (odds 2.0)
    res1 = settle_market("dc_1x", odds=2.0, home_goals=1, away_goals=0)
    # Leg 2: Loss (odds 1.5)
    res2 = settle_market("o15", odds=1.5, home_goals=0, away_goals=0)

    factors = [res1.return_factor, res2.return_factor]
    total_return = calculate_accumulator_return(factors)

    assert res1.return_factor == 2.0
    assert res2.return_factor == 0.0
    assert total_return == 0.0
