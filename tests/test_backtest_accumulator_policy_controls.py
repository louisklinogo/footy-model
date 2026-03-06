from __future__ import annotations

from datetime import UTC, datetime

from src.betting.backtest_accumulators import CandidateLeg, build_slips_walk_forward


def _candidate(
    *,
    prediction_id: int,
    fixture_id: int,
    market_code: str,
    league_code: str,
    kickoff_day: int,
    home_team_id: int,
    away_team_id: int,
) -> CandidateLeg:
    return CandidateLeg(
        prediction_id=prediction_id,
        fixture_id=fixture_id,
        market_code=market_code,
        league_code=league_code,
        kickoff_utc=datetime(2026, 3, kickoff_day, 15, 0, tzinfo=UTC),
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        odds_used=1.3,
        p_model=0.68,
        p_conservative=0.68,
        edge_adjusted=0.03,
        risk_score=30.0,
        line_num=1.5,
        risk_flags=[],
        home_goals=2,
        away_goals=0,
        home_corners=5,
        away_corners=4,
        home_led_by_1_any=True,
        away_led_by_1_any=False,
        home_led_by_2_any=False,
        away_led_by_2_any=False,
    )


def test_limited_market_uniqueness_blocks_duplicate_market_codes_by_default() -> None:
    slips, _, metrics = build_slips_walk_forward(
        candidates=[
            _candidate(
                prediction_id=1,
                fixture_id=11,
                market_code="o15",
                league_code="E0",
                kickoff_day=1,
                home_team_id=100,
                away_team_id=101,
            ),
            _candidate(
                prediction_id=2,
                fixture_id=12,
                market_code="o15",
                league_code="D1",
                kickoff_day=1,
                home_team_id=102,
                away_team_id=103,
            ),
        ],
        slip_size=2,
        policy={
            "max_tickets_per_day": 1,
            "stake_fraction_per_ticket": 0.02,
            "daily_stop_loss_fraction": 0.05,
            "ticket_total_odds_min": 1.0,
            "ticket_total_odds_max": 5.0,
            "correlation_policy": {
                "forbid_same_fixture": True,
                "forbid_same_team_across_legs": True,
                "max_legs_per_league": 1,
                "same_league_probability_haircut": 1.0,
            },
        },
        limited_to_one_leg_markets={"o15"},
        initial_bankroll=100.0,
    )

    assert slips == []
    assert metrics["slip_count"] == 0


def test_policy_can_allow_duplicate_market_codes_across_fixtures() -> None:
    slips, _, metrics = build_slips_walk_forward(
        candidates=[
            _candidate(
                prediction_id=1,
                fixture_id=11,
                market_code="o15",
                league_code="E0",
                kickoff_day=1,
                home_team_id=100,
                away_team_id=101,
            ),
            _candidate(
                prediction_id=2,
                fixture_id=12,
                market_code="o15",
                league_code="D1",
                kickoff_day=1,
                home_team_id=102,
                away_team_id=103,
            ),
        ],
        slip_size=2,
        policy={
            "max_tickets_per_day": 1,
            "stake_fraction_per_ticket": 0.02,
            "daily_stop_loss_fraction": 0.05,
            "ticket_total_odds_min": 1.0,
            "ticket_total_odds_max": 5.0,
            "enforce_limited_market_uniqueness": False,
            "correlation_policy": {
                "forbid_same_fixture": True,
                "forbid_same_team_across_legs": True,
                "max_legs_per_league": 1,
                "same_league_probability_haircut": 1.0,
            },
        },
        limited_to_one_leg_markets={"o15"},
        initial_bankroll=100.0,
    )

    assert len(slips) == 1
    assert metrics["slip_count"] == 1
