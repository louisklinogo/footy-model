from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.run_small_bankroll_acca_campaign import (
    resolve_campaign_markets,
    simulate_target_paths,
)


def test_resolve_campaign_markets_intersects_with_eligible_and_blocked() -> None:
    resolved, summary = resolve_campaign_markets(
        eligible_markets={"dc_1x", "h_1up", "c85"},
        whitelist_payload={
            "tradable_markets": ["dc_1x", "h_1up", "dc_12"],
            "watchlist_markets": ["c85"],
            "blocked_markets": ["h_1up"],
        },
        include_watchlist=False,
    )

    assert resolved == {"dc_1x"}
    assert summary["resolved_market_count"] == 1
    assert summary["requested_but_ineligible"] == ["dc_12"]


def test_resolve_campaign_markets_can_include_watchlist() -> None:
    resolved, summary = resolve_campaign_markets(
        eligible_markets={"dc_1x", "c85"},
        whitelist_payload={
            "tradable_markets": ["dc_1x"],
            "watchlist_markets": ["c85"],
            "blocked_markets": [],
        },
        include_watchlist=True,
    )

    assert resolved == {"dc_1x", "c85"}
    assert summary["include_watchlist"] is True


def test_simulate_target_paths_all_wins_hit_targets() -> None:
    summary = simulate_target_paths(
        return_factors=[3.0],
        stake_fraction=0.1,
        initial_bankroll=100.0,
        bankroll_floor=20.0,
        milestones=[120.0, 150.0],
        simulation_count=50,
        campaign_slips=3,
        seed=7,
    )

    assert summary.ruin_probability == 0.0
    assert summary.milestone_hit_probability["120"] == 1.0
    assert summary.milestone_hit_probability["150"] == 1.0
    assert summary.median_ending_bankroll > 150.0


def test_simulate_target_paths_all_losses_ruin() -> None:
    summary = simulate_target_paths(
        return_factors=[0.0],
        stake_fraction=0.5,
        initial_bankroll=100.0,
        bankroll_floor=20.0,
        milestones=[120.0],
        simulation_count=40,
        campaign_slips=10,
        seed=11,
    )

    assert summary.ruin_probability == 1.0
    assert summary.milestone_hit_probability["120"] == 0.0
    assert summary.mean_ending_bankroll <= 20.0
