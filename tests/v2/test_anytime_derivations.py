from __future__ import annotations

import pytest

from src.modeling.v2.families.anytime.derive_markets import (
    derive_and_validate_anytime,
    derive_and_validate_anytime_direct_monotone,
    derive_and_validate_anytime_phase_split,
    derive_and_validate_anytime_state_ladder,
)


def test_anytime_probs_are_valid_and_monotonic() -> None:
    probs = derive_and_validate_anytime(lambda_home=1.4, lambda_away=1.1, max_goals=8)
    assert set(probs.keys()) == {"h_1up", "a_1up", "h_2up", "a_2up"}
    for value in probs.values():
        assert 0.0 <= value <= 1.0
    assert probs["h_2up"] <= probs["h_1up"] + 1e-9
    assert probs["a_2up"] <= probs["a_1up"] + 1e-9


def test_anytime_symmetry_on_equal_lambdas() -> None:
    probs = derive_and_validate_anytime(lambda_home=1.25, lambda_away=1.25, max_goals=8)
    assert probs["h_1up"] == pytest.approx(probs["a_1up"], abs=0.02)
    assert probs["h_2up"] == pytest.approx(probs["a_2up"], abs=0.02)


def test_anytime_favourite_has_higher_1up() -> None:
    probs = derive_and_validate_anytime(lambda_home=2.0, lambda_away=0.8, max_goals=8)
    assert probs["h_1up"] > probs["a_1up"]


def test_anytime_phase_split_probs_are_valid_and_monotonic() -> None:
    probs = derive_and_validate_anytime_phase_split(
        lambda_home_p1=0.7,
        lambda_away_p1=0.5,
        lambda_home_p2=0.9,
        lambda_away_p2=0.6,
        max_goals=8,
    )
    assert set(probs.keys()) == {"h_1up", "a_1up", "h_2up", "a_2up"}
    for value in probs.values():
        assert 0.0 <= value <= 1.0
    assert probs["h_2up"] <= probs["h_1up"] + 1e-9
    assert probs["a_2up"] <= probs["a_1up"] + 1e-9


def test_anytime_phase_split_matches_constant_when_rates_match() -> None:
    phase_split = derive_and_validate_anytime_phase_split(
        lambda_home_p1=0.65,
        lambda_away_p1=0.55,
        lambda_home_p2=0.65,
        lambda_away_p2=0.55,
        max_goals=8,
    )
    constant = derive_and_validate_anytime(lambda_home=1.3, lambda_away=1.1, max_goals=8)
    for market in constant:
        assert phase_split[market] == pytest.approx(constant[market], abs=1e-6)


def test_anytime_state_ladder_probs_are_valid_and_monotonic() -> None:
    probs = derive_and_validate_anytime_state_ladder(
        p_home_1up=0.63,
        p_away_1up=0.41,
        p_home_2up_given_1up=0.36,
        p_away_2up_given_1up=0.22,
    )
    assert set(probs.keys()) == {"h_1up", "a_1up", "h_2up", "a_2up"}
    for value in probs.values():
        assert 0.0 <= value <= 1.0
    assert probs["h_2up"] <= probs["h_1up"] + 1e-9
    assert probs["a_2up"] <= probs["a_1up"] + 1e-9


def test_anytime_direct_monotone_reconciles_2up_to_1up() -> None:
    probs = derive_and_validate_anytime_direct_monotone(
        p_home_1up=0.55,
        p_away_1up=0.46,
        p_home_2up=0.71,
        p_away_2up=0.18,
    )
    assert probs["h_2up"] == pytest.approx(0.55)
    assert probs["a_2up"] == pytest.approx(0.18)
