from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.pricing.markov import MarkovPricer


ANYTIME_MARKETS = ("h_1up", "a_1up", "h_2up", "a_2up")


@lru_cache(maxsize=8)
def _get_pricer(max_goals: int) -> MarkovPricer:
    return MarkovPricer(max_goals=max(4, int(max_goals)))


@lru_cache(maxsize=8192)
def _derive_cached(lambda_home: float, lambda_away: float, max_goals: int) -> tuple[float, ...]:
    pricer = _get_pricer(max_goals)
    probs = pricer.calculate_lead_probs(lambda_home, lambda_away)
    return tuple(float(np.clip(probs.get(market, 0.0), 0.0, 1.0)) for market in ANYTIME_MARKETS)


@lru_cache(maxsize=8192)
def _derive_phase_split_cached(
    lambda_home_p1: float,
    lambda_away_p1: float,
    lambda_home_p2: float,
    lambda_away_p2: float,
    max_goals: int,
) -> tuple[float, ...]:
    pricer = _get_pricer(max_goals)
    probs = pricer.calculate_lead_probs_phase_split(
        lambda_h_p1=lambda_home_p1,
        lambda_a_p1=lambda_away_p1,
        lambda_h_p2=lambda_home_p2,
        lambda_a_p2=lambda_away_p2,
    )
    return tuple(float(np.clip(probs.get(market, 0.0), 0.0, 1.0)) for market in ANYTIME_MARKETS)


def _validate_anytime_probs(probs: dict[str, float]) -> None:
    missing = [market for market in ANYTIME_MARKETS if market not in probs]
    if missing:
        raise ValueError(f"Missing anytime markets: {', '.join(missing)}")

    for market in ANYTIME_MARKETS:
        value = float(probs[market])
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{market} out of range [0,1]: {value}")

    # Path-dependent monotonicity constraints.
    if float(probs["h_2up"]) > float(probs["h_1up"]) + 1e-9:
        raise ValueError("Invalid anytime monotonicity: h_2up > h_1up")
    if float(probs["a_2up"]) > float(probs["a_1up"]) + 1e-9:
        raise ValueError("Invalid anytime monotonicity: a_2up > a_1up")


def derive_and_validate_anytime(
    *,
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 8,
) -> dict[str, float]:
    lh = max(0.01, float(lambda_home))
    la = max(0.01, float(lambda_away))
    max_goals_int = max(4, int(max_goals))
    # Round for stable caching; this is still much finer than model uncertainty.
    probs_tuple = _derive_cached(round(lh, 4), round(la, 4), max_goals_int)
    out = {market: float(value) for market, value in zip(ANYTIME_MARKETS, probs_tuple, strict=True)}
    _validate_anytime_probs(out)
    return out


def derive_and_validate_anytime_phase_split(
    *,
    lambda_home_p1: float,
    lambda_away_p1: float,
    lambda_home_p2: float,
    lambda_away_p2: float,
    max_goals: int = 8,
) -> dict[str, float]:
    lh_p1 = max(0.01, float(lambda_home_p1))
    la_p1 = max(0.01, float(lambda_away_p1))
    lh_p2 = max(0.01, float(lambda_home_p2))
    la_p2 = max(0.01, float(lambda_away_p2))
    max_goals_int = max(4, int(max_goals))
    probs_tuple = _derive_phase_split_cached(
        round(lh_p1, 4),
        round(la_p1, 4),
        round(lh_p2, 4),
        round(la_p2, 4),
        max_goals_int,
    )
    out = {market: float(value) for market, value in zip(ANYTIME_MARKETS, probs_tuple, strict=True)}
    _validate_anytime_probs(out)
    return out
