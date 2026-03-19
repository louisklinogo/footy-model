"""
opportunity_finder.py - Ranks markets by exploitability based on match story.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.match_profile import MatchProfile, OddsContext
from src.betting.story_generator import MatchStory

# Story signals and their market implications
SETTLEABLE_MARKETS = {"ah2_home_m05", "ah2_home_p05", "ah2_home_m15", "ah2_home_p15", "ah2_away_m05", "ah2_away_p05", "ah2_away_m15", "ah2_away_p15", "ah_h05", "ah_a05", "ah_h15", "ah_a15", "hc25", "ac25", "h_1up", "a_1up", "h_2up", "a_2up"}

STORY_MARKET_ALIGNMENT = {
    "possession_vs_direct": {"aligned": ["corners_ou_9.5_over", "ou_2.5_over"], "contradictory": ["corners_ou_9.5_under"]},
    "possession_vs_block": {"aligned": ["ou_2.5_under", "corners_ou_9.5_over", "1x2_h"], "contradictory": ["ou_2.5_over"]},
    "hot_home_vs_cold_away": {"aligned": ["1x2_h", "ah_home_-0.5", "ou_2.5_over", "eh_home_-1", "ah2_home_m05", "ah2_home_p05"], "contradictory": ["1x2_a"]},
    "hot_away_vs_cold_home": {"aligned": ["1x2_a", "ah_away_+0.5", "ah2_away_m05", "ah2_away_p05"], "contradictory": ["1x2_h"]},
    "hot_vs_hot": {"aligned": ["btts_yes", "ou_2.5_over"], "contradictory": ["btts_no"]},
    "cold_vs_cold": {"aligned": ["ou_2.5_under", "btts_no"], "contradictory": ["ou_2.5_over"]},
    "title_race_home": {"aligned": ["1x2_h", "eh_home_-1", "ah2_home_m05", "ah2_home_p05"], "contradictory": ["1x2_a"]},
    "relegation_battle": {"aligned": ["cards_ou_3.5_over", "btts_yes"], "contradictory": ["cards_ou_3.5_under"]},
    "expected_2_0": {"aligned": ["1x2_h", "eh_home_-1", "ou_3.5_under", "ah2_home_m15"], "contradictory": ["btts_yes", "1x2_a"]},
    "expected_1_1": {"aligned": ["1x2_d", "btts_yes", "ou_2.5_under", "ah2_home_p05", "ah2_away_p05"], "contradictory": ["eh_home_-1"]},
    "expected_0_1": {"aligned": ["1x2_a", "ou_2.5_under", "ah2_away_m05", "ah2_away_p05"], "contradictory": ["1x2_h", "btts_yes"]},
    "expected_1_0": {"aligned": ["1x2_h", "ou_2.5_under", "btts_no", "ah2_home_m05", "ah2_home_p05"], "contradictory": ["ou_3.5_over"]},
    "expected_2_1": {"aligned": ["1x2_h", "btts_yes", "ou_2.5_over", "ah2_home_m05", "ah2_home_p05"], "contradictory": ["eh_home_-1"]},
    "derby": {"aligned": ["cards_ou_3.5_over", "btts_yes", "1x2_d"], "contradictory": ["cards_ou_3.5_under"]},
    "home_dominant": {"aligned": ["1x2_h", "corners_ou_9.5_over", "eh_home_-1", "ah2_home_m05", "ah2_home_p05"], "contradictory": ["1x2_a"]},
    "away_dominant": {"aligned": ["1x2_a", "ah_away_+0.5", "ah2_away_m05", "ah2_away_p05"], "contradictory": ["1x2_h"]},
    "high_corners_expected": {"aligned": ["corners_ou_9.5_over", "corners_ou_10.5_over"], "contradictory": ["corners_ou_9.5_under"]},
    "low_corners_expected": {"aligned": ["corners_ou_9.5_under"], "contradictory": ["corners_ou_9.5_over"]},
    "top_vs_bottom": {"aligned": ["ah_home_-1.5", "eh_home_-2", "ou_3.5_over"], "contradictory": ["1x2_a"]},
}

# Only use signals that have demonstrated 70%+ win rate in backtests
HIGH_PERFORMANCE_SIGNALS = {
    "title_race_home",
    "expected_2_0", 
    "expected_1_0",
    "relegation_battle",
    "expected_2_1",
    "hot_home_vs_cold_away",
    "home_dominant",
}

# Signals to avoid (below 70% win rate)
AVOID_SIGNALS = {
    "expected_0_1",
    "expected_1_1",
    "cold_vs_cold",
}

@dataclass(frozen=True)
class MarketOpportunity:
    market_code: str
    line: float | None
    selection: str
    model_prob: float
    odds: float | None
    edge: float | None
    exploitability_score: float
    exploitability: str
    reason: str
    story_alignment: str
    story_signals: tuple[str, ...]

@dataclass(frozen=True)
class MatchOpportunities:
    fixture_id: int
    teams: tuple[str, str]
    opportunities: tuple[MarketOpportunity, ...]
    top_picks: tuple[MarketOpportunity, ...]
    avoid: tuple[MarketOpportunity, ...]
    story_signals: tuple[str, ...]

def extract_story_signals(story: MatchStory) -> list[str]:
    signals = []
    style = story.style_matchup.matchup_type
    if style in STORY_MARKET_ALIGNMENT:
        signals.append(style)
    if story.form_clash.advantage == "home" and story.form_clash.home_form == "Hot":
        if story.form_clash.away_form == "Cold":
            signals.append("hot_home_vs_cold_away")
    if story.form_clash.home_form == "Hot" and story.form_clash.away_form == "Hot":
        signals.append("hot_vs_hot")
    if story.form_clash.home_form == "Cold" and story.form_clash.away_form == "Cold":
        signals.append("cold_vs_cold")
    if "Chasing Europe" in story.position_context.home_situation:
        signals.append("title_race_home")
    if "Relegation" in story.position_context.home_situation or "Relegation" in story.position_context.away_situation:
        signals.append("relegation_battle")
    score = story.expected_scoreline.replace("-", "_")
    score_sig = f"expected_{score}"
    if score_sig in STORY_MARKET_ALIGNMENT:
        signals.append(score_sig)
    if "home dominant" in story.game_flow.lower():
        signals.append("home_dominant")
    if "away dominant" in story.game_flow.lower():
        signals.append("away_dominant")
    if "High" in story.style_matchup.expected_corners:
        signals.append("high_corners_expected")
    if "Low" in story.style_matchup.expected_corners:
        signals.append("low_corners_expected")
    if story.derby_context:
        signals.append("derby")
    return signals

def check_story_alignment(market_code: str, story_signals: list[str]) -> tuple[str, list[str]]:
    aligned, contradictory = [], []
    for sig in story_signals:
        # Immediately reject any signal in AVOID_SIGNALS
        if sig in AVOID_SIGNALS:
            return "contradictory", [sig]
        if sig in STORY_MARKET_ALIGNMENT:
            rules = STORY_MARKET_ALIGNMENT[sig]
            if market_code in rules.get("aligned", []):
                aligned.append(sig)
            if market_code in rules.get("contradictory", []):
                contradictory.append(sig)
    if contradictory:
        return "contradictory", contradictory
    # Only consider "aligned" if signal is high-performance
    if aligned and any(sig in HIGH_PERFORMANCE_SIGNALS for sig in aligned):
        return "aligned", aligned
    return "neutral", []

def find_opportunities(profile: MatchProfile, story: MatchStory, top_n: int = 10) -> MatchOpportunities:
    signals = extract_story_signals(story)
    opps = []
    for mkt, prob in profile.predictions.items():
        # Skip markets we cannot settle
        if mkt not in SETTLEABLE_MARKETS:
            continue
        align, align_sigs = check_story_alignment(mkt, signals)
        edge = None
        odds = None
        for o in profile.odds:
            if o.market_code in mkt or mkt.startswith(o.market_code):
                if o.odds_home:
                    odds = o.odds_home
                elif o.odds_over:
                    odds = o.odds_over
                break
        if odds and odds > 1.0:
            implied = 1.0 / odds
            edge = prob - implied
        score = 0.3 * (1.0 if align == "aligned" else 0.0 if align == "contradictory" else 0.5)
        if edge:
            score += 0.4 * min(abs(edge), 0.15) / 0.15
        score += 0.2 * (1.0 if odds else 0.5)
        score += 0.1 * abs(prob - 0.5) * 2
        exp = "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"
        opps.append(MarketOpportunity(market_code=mkt, line=None, selection="", model_prob=prob, odds=odds, edge=edge, exploitability_score=score, exploitability=exp, reason=align, story_alignment=align, story_signals=tuple(align_sigs)))
    opps.sort(key=lambda x: -x.exploitability_score)
    top = [o for o in opps if o.exploitability != "low"][:top_n]
    avoid = [o for o in opps if o.story_alignment == "contradictory"]
    return MatchOpportunities(profile.fixture_id, (profile.home_team.name, profile.away_team.name), tuple(opps), tuple(top), tuple(avoid), tuple(signals))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=int, required=True)
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    from src.betting.match_profile import build_match_profile
    from src.betting.story_generator import generate_story
    profile = build_match_profile(args.fixture)
    if not profile:
        print(f"No profile for {args.fixture}")
        return
    story = generate_story(profile)
    opps = find_opportunities(profile, story, args.top)
    if args.output == "json":
        print(json.dumps({"signals": list(opps.story_signals), "top": [{"mkt": o.market_code, "odds": o.odds, "edge": o.edge} for o in opps.top_picks]}, indent=2))
    else:
        print(f"Story signals: {', '.join(opps.story_signals)}")
        print("\nTop picks:")
        for o in opps.top_picks:
            print(f"  {o.market_code}: prob={o.model_prob:.1%}, odds={o.odds}, edge={o.edge:+.1%}, {o.reason}")

if __name__ == "__main__":
    main()
