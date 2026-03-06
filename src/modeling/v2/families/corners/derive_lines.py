from __future__ import annotations

from math import exp, floor, lgamma, log, sqrt
from typing import Any

import numpy as np


TOTAL_MARKETS: dict[str, float] = {
    "c75": 7.5,
    "c85": 8.5,
    "c95": 9.5,
    "c105": 10.5,
}

HOME_MARKETS: dict[str, float] = {
    "hc25": 2.5,
    "hc35": 3.5,
    "hc45": 4.5,
    "hc55": 5.5,
}

AWAY_MARKETS: dict[str, float] = {
    "ac25": 2.5,
    "ac35": 3.5,
    "ac45": 4.5,
    "ac55": 5.5,
}


def estimate_nb_dispersion(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 100:
        return None
    mu = float(np.mean(arr))
    var = float(np.var(arr))
    if mu <= 1e-9:
        return None
    if var <= mu:
        return None
    r = (mu * mu) / max(var - mu, 1e-9)
    if not np.isfinite(r) or r <= 1e-6:
        return None
    return float(r)


def poisson_over_probability(mu: float, line: float) -> float:
    mu = max(float(mu), 1e-6)
    threshold = int(floor(line)) + 1
    if threshold <= 0:
        return 1.0
    pmf = exp(-mu)  # P(X=0)
    cdf = pmf
    for k in range(1, threshold):
        pmf *= mu / float(k)
        cdf += pmf
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def _nbinom_logpmf(k: int, mu: float, r: float) -> float:
    # NB parameterized by mean mu and dispersion r.
    # Var = mu + mu^2 / r
    if k < 0:
        return float("-inf")
    mu = max(float(mu), 1e-9)
    r = max(float(r), 1e-9)
    p = r / (r + mu)
    q = mu / (r + mu)
    return (
        lgamma(k + r)
        - lgamma(r)
        - lgamma(k + 1.0)
        + r * log(p)
        + k * log(q)
    )


def nbinom_over_probability(mu: float, line: float, r: float) -> float:
    threshold = int(floor(line)) + 1
    if threshold <= 0:
        return 1.0
    mu = max(float(mu), 1e-6)
    r = max(float(r), 1e-6)
    # dynamic truncation for safety in tail calculations
    var = mu + (mu * mu) / r
    max_k = max(threshold + 30, int(mu + 10.0 * sqrt(max(var, 1e-9))) + 20)
    cdf = 0.0
    for k in range(0, threshold):
        cdf += exp(_nbinom_logpmf(k, mu, r))
    # residual tail not accounted above is naturally in 1-cdf
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def over_probability(mu: float, line: float, r: float | None) -> float:
    if r is None or not np.isfinite(r) or r <= 0.0:
        return poisson_over_probability(mu, line)
    return nbinom_over_probability(mu, line, r)


def derive_corners_markets(
    *,
    home_mu: float,
    away_mu: float,
    total_r: float | None,
    home_r: float | None,
    away_r: float | None,
) -> dict[str, float]:
    home_mu = max(float(home_mu), 0.05)
    away_mu = max(float(away_mu), 0.05)
    total_mu = home_mu + away_mu

    out: dict[str, float] = {}
    for market, line in TOTAL_MARKETS.items():
        out[market] = over_probability(total_mu, line, total_r)
    for market, line in HOME_MARKETS.items():
        out[market] = over_probability(home_mu, line, home_r)
    for market, line in AWAY_MARKETS.items():
        out[market] = over_probability(away_mu, line, away_r)

    return {k: float(np.clip(v, 0.0, 1.0)) for k, v in out.items()}


def assert_corners_monotonicity(probs: dict[str, float], tol: float = 1e-9) -> None:
    total = [probs["c75"], probs["c85"], probs["c95"], probs["c105"]]
    home = [probs["hc25"], probs["hc35"], probs["hc45"], probs["hc55"]]
    away = [probs["ac25"], probs["ac35"], probs["ac45"], probs["ac55"]]

    for seq_name, seq in (("total", total), ("home", home), ("away", away)):
        for i in range(len(seq) - 1):
            if seq[i + 1] > seq[i] + tol:
                raise ValueError(f"{seq_name} line monotonicity violated at idx={i}")


def derive_and_validate_corners(
    *,
    home_mu: float,
    away_mu: float,
    total_r: float | None,
    home_r: float | None,
    away_r: float | None,
) -> dict[str, Any]:
    probs = derive_corners_markets(
        home_mu=home_mu,
        away_mu=away_mu,
        total_r=total_r,
        home_r=home_r,
        away_r=away_r,
    )
    assert_corners_monotonicity(probs)
    return probs

