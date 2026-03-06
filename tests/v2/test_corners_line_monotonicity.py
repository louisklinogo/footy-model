from __future__ import annotations

import numpy as np
import pytest

from src.modeling.v2.families.corners.derive_lines import (
    assert_corners_monotonicity,
    derive_and_validate_corners,
    estimate_nb_dispersion,
    nbinom_over_probability,
    poisson_over_probability,
)


def test_poisson_and_nbinom_probabilities_in_range() -> None:
    p1 = poisson_over_probability(9.1, 8.5)
    p2 = nbinom_over_probability(9.1, 8.5, 5.0)
    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p2 <= 1.0


def test_derive_corners_monotonicity_holds() -> None:
    probs = derive_and_validate_corners(
        home_mu=5.2,
        away_mu=4.4,
        total_r=6.0,
        home_r=4.0,
        away_r=4.5,
    )
    assert probs["c75"] >= probs["c85"] >= probs["c95"] >= probs["c105"]
    assert probs["hc25"] >= probs["hc35"] >= probs["hc45"] >= probs["hc55"]
    assert probs["ac25"] >= probs["ac35"] >= probs["ac45"] >= probs["ac55"]


def test_assert_corners_monotonicity_raises_on_invalid_payload() -> None:
    bad = {
        "c75": 0.40,
        "c85": 0.60,
        "c95": 0.30,
        "c105": 0.20,
        "hc25": 0.7,
        "hc35": 0.6,
        "hc45": 0.5,
        "hc55": 0.4,
        "ac25": 0.7,
        "ac35": 0.6,
        "ac45": 0.5,
        "ac55": 0.4,
    }
    with pytest.raises(ValueError):
        assert_corners_monotonicity(bad)


def test_estimate_nb_dispersion_detects_overdispersion() -> None:
    rng = np.random.default_rng(42)
    values = rng.negative_binomial(n=4, p=0.35, size=5000).astype(float)
    r = estimate_nb_dispersion(values)
    assert r is not None
    assert r > 0.0

