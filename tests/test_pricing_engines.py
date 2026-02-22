"""
Unit tests for the src/ pricing engines, value finder, and EWMA.
No database required — these are pure mathematical correctness tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pricing.poisson import PoissonPricer
from src.pricing.markov import MarkovPricer
from src.pricing.simulators import MatchSimulator
from src.betting.value_finder import ValueFinder
from src.features.build_team_premium_snapshots_v1 import _ewma


# ---------------------------------------------------------------------------
# PoissonPricer
# ---------------------------------------------------------------------------

class TestPoissonPricer:
    def setup_method(self):
        self.pricer = PoissonPricer(max_goals=10)

    def test_matrix_sums_to_one(self):
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        assert matrix.sum() == pytest.approx(1.0, abs=1e-6)

    def test_matrix_sums_to_one_with_rho(self):
        matrix = self.pricer.generate_matrix(1.5, 1.2, rho=-0.1)
        assert matrix.sum() == pytest.approx(1.0, abs=1e-6)

    def test_1x2_sums_to_one(self):
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        probs = self.pricer.get_1x2(matrix)
        total = probs["home"] + probs["draw"] + probs["away"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_home_favourite_has_higher_win_prob(self):
        matrix = self.pricer.generate_matrix(2.5, 0.8)
        probs = self.pricer.get_1x2(matrix)
        assert probs["home"] > probs["away"]

    def test_symmetric_lambdas_give_symmetric_1x2(self):
        matrix = self.pricer.generate_matrix(1.3, 1.3)
        probs = self.pricer.get_1x2(matrix)
        assert probs["home"] == pytest.approx(probs["away"], abs=1e-6)

    def test_over_under_monotonic(self):
        """Higher total lambda should give higher Over 2.5 probability."""
        matrix_low = self.pricer.generate_matrix(0.8, 0.7)
        matrix_high = self.pricer.generate_matrix(2.0, 1.8)
        assert self.pricer.get_over_under(matrix_high, 2.5) > self.pricer.get_over_under(matrix_low, 2.5)

    def test_over_under_boundaries(self):
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        # Over -0.5 should be ~1.0 (almost all outcomes have >= 0 goals)
        assert self.pricer.get_over_under(matrix, -0.5) == pytest.approx(1.0, abs=1e-4)
        # Over 100 should be ~0.0
        assert self.pricer.get_over_under(matrix, 100) == pytest.approx(0.0, abs=1e-6)

    def test_btts_between_zero_and_one(self):
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        btts = self.pricer.get_btts(matrix)
        assert 0.0 < btts < 1.0

    def test_btts_high_lambda_gives_high_btts(self):
        low = self.pricer.get_btts(self.pricer.generate_matrix(0.5, 0.4))
        high = self.pricer.get_btts(self.pricer.generate_matrix(2.5, 2.0))
        assert high > low

    def test_dixon_coles_negative_rho_increases_draw(self):
        """Negative rho should increase draw probability (the DC correction)."""
        matrix_no_rho = self.pricer.generate_matrix(1.3, 1.1, rho=0.0)
        matrix_rho = self.pricer.generate_matrix(1.3, 1.1, rho=-0.15)
        draw_no = np.sum(np.diag(matrix_no_rho))
        draw_rho = np.sum(np.diag(matrix_rho))
        assert draw_rho > draw_no

    # --- Asian Handicap ---

    def test_ah_standard_line_probs_sum_to_one(self):
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        ah = self.pricer.get_asian_handicap(matrix, -0.5)
        total = ah["full_win"] + ah["push"] + ah["full_loss"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_ah_half_line_no_push(self):
        """Half-goal lines (e.g. -0.5) cannot push."""
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        ah = self.pricer.get_asian_handicap(matrix, -0.5)
        assert ah["push"] == pytest.approx(0.0, abs=1e-10)

    def test_ah_whole_line_can_push(self):
        """Whole-number lines (e.g. 0.0) can push on draws."""
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        ah = self.pricer.get_asian_handicap(matrix, 0.0)
        assert ah["push"] > 0.0

    def test_ah_quarter_line_returns_five_keys(self):
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        ah = self.pricer.get_asian_handicap(matrix, -0.25)
        expected_keys = {"full_win", "half_win", "push", "half_loss", "full_loss"}
        assert set(ah.keys()) == expected_keys

    def test_ah_quarter_line_probs_sum_to_one(self):
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        ah = self.pricer.get_asian_handicap(matrix, -0.25)
        total = sum(ah.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_ah_quarter_line_consistent_keys_with_standard(self):
        """Quarter lines should return the same keys as standard lines."""
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        ah_std = self.pricer.get_asian_handicap(matrix, -0.5)
        ah_qtr = self.pricer.get_asian_handicap(matrix, -0.25)
        assert set(ah_std.keys()) == set(ah_qtr.keys())

    def test_team_over_under_boundaries(self):
        matrix = self.pricer.generate_matrix(1.5, 1.2)
        # Home scoring over -0.5 = P(home >= 0) ≈ 1.0
        assert self.pricer.get_team_over_under(matrix, -0.5, is_home=True) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# MarkovPricer
# ---------------------------------------------------------------------------

class TestMarkovPricer:
    def setup_method(self):
        self.markov = MarkovPricer(max_goals=5)

    def test_symmetric_lambdas_give_symmetric_lead_probs(self):
        probs = self.markov.calculate_lead_probs(1.3, 1.3)
        assert probs["h_1up"] == pytest.approx(probs["a_1up"], abs=0.01)
        assert probs["h_2up"] == pytest.approx(probs["a_2up"], abs=0.01)

    def test_lead_probs_in_valid_range(self):
        probs = self.markov.calculate_lead_probs(1.5, 1.0)
        for key, val in probs.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of range"

    def test_1up_greater_than_2up(self):
        """1UP is always at least as likely as 2UP."""
        probs = self.markov.calculate_lead_probs(1.5, 1.0)
        assert probs["h_1up"] >= probs["h_2up"]
        assert probs["a_1up"] >= probs["a_2up"]

    def test_favourite_has_higher_1up(self):
        probs = self.markov.calculate_lead_probs(2.0, 0.8)
        assert probs["h_1up"] > probs["a_1up"]

    def test_zero_lambda_away_gives_home_1up_very_high(self):
        """If away can barely score, home 1UP should be very likely."""
        probs = self.markov.calculate_lead_probs(2.0, 0.01)
        assert probs["h_1up"] > 0.85


# ---------------------------------------------------------------------------
# MatchSimulator
# ---------------------------------------------------------------------------

class TestMatchSimulator:
    def test_reproducibility_with_seed(self):
        sim1 = MatchSimulator(iterations=10000, seed=42)
        sim2 = MatchSimulator(iterations=10000, seed=42)
        r1 = sim1.simulate_match(1.5, 1.0)
        r2 = sim2.simulate_match(1.5, 1.0)
        assert r1["h_1up_pct"] == pytest.approx(r2["h_1up_pct"], abs=1e-10)

    def test_values_in_valid_range(self):
        sim = MatchSimulator(iterations=10000, seed=0)
        r = sim.simulate_match(1.5, 1.0)
        for key, val in r.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of range"

    def test_higher_lambda_gives_higher_1up(self):
        sim = MatchSimulator(iterations=20000, seed=123)
        r = sim.simulate_match(2.0, 0.8)
        assert r["h_1up_pct"] > r["a_1up_pct"]

    def test_1up_at_least_as_likely_as_2up(self):
        sim = MatchSimulator(iterations=10000, seed=7)
        r = sim.simulate_match(1.5, 1.0)
        assert r["h_1up_pct"] >= r["h_2up_pct"]
        assert r["a_1up_pct"] >= r["a_2up_pct"]

    def test_mc_converges_to_poisson_ft_win(self):
        """MC home FT win should approximate Poisson 1X2 home win."""
        pricer = PoissonPricer()
        matrix = pricer.generate_matrix(1.5, 1.0)
        poisson_home = pricer.get_1x2(matrix)["home"]

        sim = MatchSimulator(iterations=50000, seed=99)
        mc_result = sim.simulate_match(1.5, 1.0)

        # 50k sims should give ~0.5% precision
        assert mc_result["h_win_ft"] == pytest.approx(poisson_home, abs=0.015)


# ---------------------------------------------------------------------------
# ValueFinder
# ---------------------------------------------------------------------------

class TestValueFinder:
    def test_no_edge_at_fair_odds(self):
        """If odds exactly match fair probability, EV < min_ev so no edges returned."""
        finder = ValueFinder(min_ev=0.02)
        pricer = PoissonPricer()
        matrix = pricer.generate_matrix(1.5, 1.0)
        home_prob = pricer.get_1x2(matrix)["home"]
        fair_odds = 1.0 / home_prob

        # Offering fair odds should yield EV = 0, so no edges
        edges = finder.find_edges(1.5, 1.0, {"home_win": fair_odds})
        assert len(edges) == 0

    def test_edge_found_when_odds_exceed_fair(self):
        """Odds significantly above fair value should produce an edge."""
        finder = ValueFinder(min_ev=0.02)
        pricer = PoissonPricer()
        matrix = pricer.generate_matrix(1.5, 1.0)
        home_prob = pricer.get_1x2(matrix)["home"]
        generous_odds = (1.0 / home_prob) * 1.20  # 20% above fair

        edges = finder.find_edges(1.5, 1.0, {"home_win": generous_odds})
        assert len(edges) >= 1
        assert edges[0]["ev"] > 0

    def test_kelly_returns_zero_for_negative_ev(self):
        finder = ValueFinder()
        assert finder.apply_kelly_staking(ev=-0.05, odds=2.0) == 0.0

    def test_kelly_returns_zero_for_bad_odds(self):
        finder = ValueFinder()
        assert finder.apply_kelly_staking(ev=0.1, odds=1.0) == 0.0
        assert finder.apply_kelly_staking(ev=0.1, odds=0.5) == 0.0

    def test_kelly_positive_for_positive_ev(self):
        finder = ValueFinder()
        stake = finder.apply_kelly_staking(ev=0.10, odds=2.50, fraction=0.25)
        assert stake > 0.0

    def test_edge_ev_calculation_is_correct(self):
        """Verify EV = (prob * odds) - 1."""
        finder = ValueFinder(min_ev=0.0)
        edges = finder.find_edges(2.0, 0.8, {"home_win": 3.0})
        if edges:
            edge = edges[0]
            expected_ev = (edge["model_prob"] * edge["odds"]) - 1.0
            assert edge["ev"] == pytest.approx(expected_ev, abs=1e-6)


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------

class TestEWMA:
    def test_empty_list_returns_none(self):
        assert _ewma([], 0.2) is None

    def test_all_none_returns_none(self):
        assert _ewma([None, None, None], 0.2) is None

    def test_single_value(self):
        assert _ewma([5.0], 0.2) == pytest.approx(5.0)

    def test_recent_values_weighted_more(self):
        """EWMA with a recent spike should be pulled towards the spike."""
        stable = _ewma([1.0, 1.0, 1.0, 1.0, 1.0], 0.3)
        spiked = _ewma([1.0, 1.0, 1.0, 1.0, 3.0], 0.3)
        assert spiked > stable

    def test_nones_are_filtered(self):
        """None values should be ignored, not treated as zeros."""
        result_with_nones = _ewma([1.5, None, 2.0], 0.3)
        result_clean = _ewma([1.5, 2.0], 0.3)
        assert result_with_nones == pytest.approx(result_clean)

    def test_matches_manual_calculation(self):
        """Verify against hand-calculated EWMA."""
        alpha = 0.2
        values = [10.0, 12.0, 11.0]
        # S0 = 10.0
        # S1 = 0.2 * 12.0 + 0.8 * 10.0 = 2.4 + 8.0 = 10.4
        # S2 = 0.2 * 11.0 + 0.8 * 10.4 = 2.2 + 8.32 = 10.52
        expected = 10.52
        assert _ewma(values, alpha) == pytest.approx(expected, abs=1e-6)
