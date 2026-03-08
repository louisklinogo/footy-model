from __future__ import annotations

from src.modeling.v2.market_family_registry import DEFAULT_FAMILY_MARKETS, SCORELINE_DIRECTIONAL_MARKETS


def test_scoreline_directional_registry_uses_canonical_handicap_contract() -> None:
    assert "ah2_home_m05" in SCORELINE_DIRECTIONAL_MARKETS
    assert "ah2_away_p15" in SCORELINE_DIRECTIONAL_MARKETS
    assert "eh3_0_1_draw" in SCORELINE_DIRECTIONAL_MARKETS
    assert "eh3_1_0_away" in SCORELINE_DIRECTIONAL_MARKETS
    assert "ah_h05" not in SCORELINE_DIRECTIONAL_MARKETS
    assert "eh_h1" not in SCORELINE_DIRECTIONAL_MARKETS


def test_default_scoreline_directional_family_matches_registry_subset() -> None:
    assert DEFAULT_FAMILY_MARKETS["scoreline_directional"] == SCORELINE_DIRECTIONAL_MARKETS