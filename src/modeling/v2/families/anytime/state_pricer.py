from __future__ import annotations

import numpy as np


def _clip_prob(value: float) -> float:
    return float(np.clip(float(value), 0.001, 0.999))


def derive_state_ladder_probs(
    *,
    p_home_1up: float,
    p_away_1up: float,
    p_home_2up_given_1up: float,
    p_away_2up_given_1up: float,
) -> dict[str, float]:
    h1 = _clip_prob(p_home_1up)
    a1 = _clip_prob(p_away_1up)
    h2_cond = _clip_prob(p_home_2up_given_1up)
    a2_cond = _clip_prob(p_away_2up_given_1up)
    h2 = _clip_prob(min(h1, h1 * h2_cond))
    a2 = _clip_prob(min(a1, a1 * a2_cond))
    return {
        "h_1up": h1,
        "a_1up": a1,
        "h_2up": h2,
        "a_2up": a2,
    }