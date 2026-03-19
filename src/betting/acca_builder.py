"""
acca_builder.py - Builds accumulator combinations with coherent narratives.

Creates multi-leg accumulators where legs tell a coherent story, not just
high probability picks. Validates that legs don't contradict each other.

Usage:
    python src/betting/acca_builder.py --today
    python src/betting/acca_builder.py --today --strategy conservative
    python src/betting/acca_builder.py --today --strategy balanced --max-legs 4 --min-odds 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.opportunity_finder import (
    MatchOpportunities,
    MarketOpportunity,
    STORY_MARKET_ALIGNMENT,
)


# =============================================================================
# Enums
# =============================================================================


class AccaStrategy(str, Enum):
    """Strategy for building accumulators."""
    ULTRA = "ultra"                  # 80%+ prob, story aligned, max 2 legs
    CONSERVATIVE = "conservative"   # 70%+ prob, story aligned
    BALANCED = "balanced"           # 65-80% prob, story aligned
    AGGRESSIVE = "aggressive"       # 60-70% prob, story aligned


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class AccaLeg:
    """A single leg in an accumulator."""
    fixture_id: int
    teams: tuple[str, str]
    market_code: str
    selection: str
    odds: float
    model_prob: float
    story_summary: str
    story_signals: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "fixture_id": self.fixture_id,
            "teams": list(self.teams),
            "market_code": self.market_code,
            "selection": self.selection,
            "odds": self.odds,
            "model_prob": self.model_prob,
            "story_summary": self.story_summary,
            "story_signals": list(self.story_signals),
        }


@dataclass
class Acca:
    """An accumulator bet with multiple legs."""
    legs: list[AccaLeg]
    combined_odds: float
    combined_prob: float
    expected_value: float
    narrative: str
    strategy: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "legs": [leg.to_dict() for leg in self.legs],
            "combined_odds": self.combined_odds,
            "combined_prob": self.combined_prob,
            "expected_value": self.expected_value,
            "narrative": self.narrative,
            "strategy": self.strategy,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Market Contradiction Rules
# =============================================================================

# Markets that contradict each other when combined in same fixture
# Format: (market_a, selection_a, market_b, selection_b) -> contradiction
MARKET_CONTRADICTIONS: list[tuple[str, str, str, str]] = [
    # Result contradictions
    ("1x2", "home", "1x2", "away"),
    ("1x2", "home", "1x2", "draw"),
    ("1x2", "away", "1x2", "draw"),
    
    # Goals contradictions
    ("ou_2.5", "over", "ou_2.5", "under"),
    ("ou_3.5", "over", "ou_3.5", "under"),
    ("ou_1.5", "over", "ou_1.5", "under"),
    
    # BTTS contradictions
    ("btts", "yes", "btts", "no"),
    
    # Cross-market contradictions (same fixture)
    ("1x2", "home", "btts", "no"),  # Home win + no BTTS possible
    ("1x2", "away", "btts", "no"),  # Away win + no BTTS possible
    ("ou_0.5", "under", "btts", "yes"),  # Under 0.5 goals + BTTS yes
    ("ou_1.5", "under", "btts", "yes"),  # Under 1.5 goals + BTTS yes
    ("ou_2.5", "under", "btts", "yes"),  # Under 2.5 + BTTS yes (possible but risky)
    
    # Home dominant contradictions
    ("1x2", "home", "ou_3.5", "under"),  # Home win but under 3.5 (possible but story mismatch)
    ("eh_home_-1", "home", "ou_2.5", "under"),  # Home -1 EH + under 2.5 (need 2+ goal win)
    
    # Asian handicap contradictions
    ("ah_home_-0.5", "home", "1x2", "draw"),
    ("ah_home_-0.5", "home", "1x2", "away"),
    ("ah_away_+0.5", "away", "1x2", "home"),
]

# Market code patterns for extracting selection
MARKET_SELECTION_PATTERNS = {
    "1x2": ["home", "draw", "away"],
    "btts": ["yes", "no"],
    "ou_2.5": ["over", "under"],
    "ou_3.5": ["over", "under"],
    "ou_1.5": ["over", "under"],
    "ou_0.5": ["over", "under"],
    "eh_home_-1": ["home", "draw", "away"],
    "eh_home_-2": ["home", "draw", "away"],
    "ah_home_-0.5": ["home"],
    "ah_away_+0.5": ["away"],
    "corners_ou_9.5": ["over", "under"],
    "corners_ou_10.5": ["over", "under"],
    "cards_ou_3.5": ["over", "under"],
}


# =============================================================================
# Helper Functions
# =============================================================================


def extract_selection(market_code: str) -> str:
    """Extract the selection from a market code.
    
    Examples:
        '1x2_h' -> 'home'
        'ou_2.5_over' -> 'over'
        'btts_yes' -> 'yes'
    """
    market_code_lower = market_code.lower()
    
    # Check for suffix patterns
    if market_code_lower.endswith("_h") or market_code_lower.endswith("_home"):
        return "home"
    if market_code_lower.endswith("_a") or market_code_lower.endswith("_away"):
        return "away"
    if market_code_lower.endswith("_d") or market_code_lower.endswith("_draw"):
        return "draw"
    if market_code_lower.endswith("_over"):
        return "over"
    if market_code_lower.endswith("_under"):
        return "under"
    if market_code_lower.endswith("_yes"):
        return "yes"
    if market_code_lower.endswith("_no"):
        return "no"
    
    # Default: return the last part after underscore
    parts = market_code_lower.split("_")
    if len(parts) > 1:
        return parts[-1]
    
    return "unknown"


def get_base_market(market_code: str) -> str:
    """Extract the base market type from a market code.
    
    Examples:
        '1x2_h' -> '1x2'
        'ou_2.5_over' -> 'ou_2.5'
        'corners_ou_9.5_over' -> 'corners_ou_9.5'
    """
    market_code_lower = market_code.lower()
    
    # Known suffixes to strip
    suffixes = ["_h", "_a", "_d", "_home", "_away", "_draw", 
                "_over", "_under", "_yes", "_no"]
    
    for suffix in suffixes:
        if market_code_lower.endswith(suffix):
            return market_code_lower[:-len(suffix)]
    
    return market_code_lower


def check_market_contradiction(
    market_a: str, selection_a: str,
    market_b: str, selection_b: str
) -> bool:
    """Check if two market selections contradict each other."""
    base_a = get_base_market(market_a)
    base_b = get_base_market(market_b)
    
    # Check direct contradictions
    for check in MARKET_CONTRADICTIONS:
        if (
            (base_a == check[0] and selection_a == check[1] and
             base_b == check[2] and selection_b == check[3])
            or
            (base_b == check[0] and selection_b == check[1] and
             base_a == check[2] and selection_a == check[3])
        ):
            return True
    
    return False


def check_story_alignment_conflict(
    story_signals_a: tuple[str, ...],
    story_signals_b: tuple[str, ...],
    market_a: str,
    market_b: str
) -> bool:
    """Check if story signals from two legs conflict.
    
    A conflict exists if one leg's story signals contradict the other leg's market.
    """
    # Check if leg A's story signals contradict leg B's market
    for signal in story_signals_a:
        if signal in STORY_MARKET_ALIGNMENT:
            rules = STORY_MARKET_ALIGNMENT[signal]
            if market_b in rules.get("contradictory", []):
                return True
    
    # Check if leg B's story signals contradict leg A's market
    for signal in story_signals_b:
        if signal in STORY_MARKET_ALIGNMENT:
            rules = STORY_MARKET_ALIGNMENT[signal]
            if market_a in rules.get("contradictory", []):
                return True
    
    return False


# =============================================================================
# Validation
# =============================================================================


def validate_acca(legs: list[AccaLeg]) -> tuple[bool, list[str]]:
    """Validate that accumulator legs don't contradict each other.
    
    Checks:
    1. No duplicate fixtures
    2. No market contradictions within same fixture
    3. No story alignment conflicts
    
    Returns:
        Tuple of (is_valid, list of validation messages)
    """
    issues: list[str] = []
    
    # Check for duplicate fixtures
    fixture_ids = [leg.fixture_id for leg in legs]
    seen = set()
    for fid in fixture_ids:
        if fid in seen:
            issues.append(f"Duplicate fixture: {fid}")
        seen.add(fid)
    
    # Check for contradictions between legs
    for i, leg_a in enumerate(legs):
        selection_a = extract_selection(leg_a.market_code)
        
        for leg_b in legs[i+1:]:
            selection_b = extract_selection(leg_b.market_code)
            
            # Check market contradiction
            if check_market_contradiction(
                leg_a.market_code, selection_a,
                leg_b.market_code, selection_b
            ):
                issues.append(
                    f"Market contradiction: {leg_a.teams[0]} vs {leg_a.teams[1]} "
                    f"{leg_a.market_code} vs {leg_b.market_code}"
                )
            
            # Check story alignment conflict
            if check_story_alignment_conflict(
                leg_a.story_signals,
                leg_b.story_signals,
                leg_a.market_code,
                leg_b.market_code
            ):
                issues.append(
                    f"Story conflict: {leg_a.teams[0]} vs {leg_a.teams[1]} "
                    f"{leg_a.market_code} clashes with {leg_b.teams[0]} vs {leg_b.teams[1]} "
                    f"{leg_b.market_code}"
                )
    
    return len(issues) == 0, issues


# =============================================================================
# Leg Selection by Strategy
# =============================================================================


def filter_legs_by_strategy(
    match_opps: list[MatchOpportunities],
    strategy: AccaStrategy,
    min_odds: float = 1.0
) -> list[AccaLeg]:
    """Filter market opportunities into potential legs based on strategy.
    
    Args:
        match_opps: List of match opportunities
        strategy: Accumulator strategy (conservative/balanced/aggressive)
        min_odds: Minimum odds for a leg
    
    Returns:
        List of potential legs that meet strategy criteria
    """
    legs: list[AccaLeg] = []
    
    for match in match_opps:
        for opp in match.top_picks:
            # Skip if no odds
            if opp.odds is None or opp.odds < min_odds:
                continue
            
            # Probability thresholds by strategy
            if strategy == AccaStrategy.ULTRA:
                # 80%+ probability, story aligned, max 2 legs
                if opp.model_prob < 0.80:
                    continue
                if opp.story_alignment != "aligned":
                    continue
            elif strategy == AccaStrategy.CONSERVATIVE:
                # 70%+ probability, story aligned
                if opp.model_prob < 0.70:
                    continue
                if opp.story_alignment != "aligned":
                    continue
            elif strategy == AccaStrategy.BALANCED:
                # 65-80% probability, story aligned
                if opp.model_prob < 0.65 or opp.model_prob > 0.80:
                    continue
                if opp.story_alignment != "aligned":
                    continue
            elif strategy == AccaStrategy.AGGRESSIVE:
                # 60-70% probability, story aligned
                if opp.model_prob < 0.60 or opp.model_prob > 0.70:
                    continue
                if opp.story_alignment != "aligned":
                    continue
            
            # Create leg
            selection = extract_selection(opp.market_code)
            story_summary = _build_story_summary(opp, match)
            
            leg = AccaLeg(
                fixture_id=match.fixture_id,
                teams=match.teams,
                market_code=opp.market_code,
                selection=selection,
                odds=opp.odds,
                model_prob=opp.model_prob,
                story_summary=story_summary,
                story_signals=match.story_signals,
            )
            legs.append(leg)
    
    return legs


def _build_story_summary(opp: MarketOpportunity, match: MatchOpportunities) -> str:
    """Build a one-line story summary for a leg."""
    parts = []
    
    # Team matchup
    teams_str = f"{match.teams[0]} vs {match.teams[1]}"
    parts.append(teams_str)
    
    # Market and selection
    parts.append(f"→ {opp.market_code}")
    
    # Key story signals (max 2)
    if opp.story_signals:
        key_signals = list(opp.story_signals[:2])
        parts.append(f"[{', '.join(key_signals)}]")
    
    # Edge info
    if opp.edge:
        parts.append(f"(edge: {opp.edge:+.1%})")
    
    return " ".join(parts)


# =============================================================================
# Acca Building
# =============================================================================


def build_accas(
    match_opps: list[MatchOpportunities],
    strategy: AccaStrategy = AccaStrategy.BALANCED,
    max_legs: int = 5,
    min_legs: int = 2,
    min_odds: float = 1.5,
    max_accas: int = 10,
) -> list[Acca]:
    """Build accumulator combinations from match opportunities.
    
    Creates accas where legs tell a coherent story, not just high probability picks.
    Validates that legs don't contradict each other.
    
    Args:
        match_opps: List of match opportunities
        strategy: Accumulator strategy
        max_legs: Maximum legs per acca
        min_legs: Minimum legs per acca
        min_odds: Minimum odds per leg
        max_accas: Maximum number of accas to return
    
    Returns:
        List of valid Acca objects sorted by expected value
    """
    # Filter potential legs by strategy
    potential_legs = filter_legs_by_strategy(match_opps, strategy, min_odds)
    
    if len(potential_legs) < min_legs:
        return []
    
    # Group legs by fixture for combination building
    legs_by_fixture: dict[int, list[AccaLeg]] = {}
    for leg in potential_legs:
        if leg.fixture_id not in legs_by_fixture:
            legs_by_fixture[leg.fixture_id] = []
        legs_by_fixture[leg.fixture_id].append(leg)
    
    # Build combinations
    all_accas: list[Acca] = []
    
    # Try different leg counts
    for num_legs in range(min_legs, min(max_legs + 1, len(legs_by_fixture) + 1)):
        # Get combinations of fixtures
        fixture_combos = combinations(legs_by_fixture.keys(), num_legs)
        
        for fixture_combo in fixture_combos:
            # For each fixture combination, get all leg combinations
            fixture_leg_lists = [legs_by_fixture[fid] for fid in fixture_combo]
            
            # Build all possible leg combinations (one leg per fixture)
            for leg_combo in _build_leg_combinations(fixture_leg_lists):
                # Validate the combination
                is_valid, issues = validate_acca(list(leg_combo))
                
                if not is_valid:
                    continue
                
                # Calculate combined metrics
                combined_odds = 1.0
                combined_prob = 1.0
                
                for leg in leg_combo:
                    combined_odds *= leg.odds
                    combined_prob *= leg.model_prob
                
                expected_value = (combined_prob * combined_odds) - 1.0
                
                # Build narrative
                narrative = _build_narrative(list(leg_combo), strategy)
                
                acca = Acca(
                    legs=list(leg_combo),
                    combined_odds=round(combined_odds, 2),
                    combined_prob=round(combined_prob, 4),
                    expected_value=round(expected_value, 4),
                    narrative=narrative,
                    strategy=strategy.value,
                )
                
                all_accas.append(acca)
    
    # Sort by expected value descending
    all_accas.sort(key=lambda a: a.expected_value, reverse=True)
    
    # Return top N
    return all_accas[:max_accas]


def _build_leg_combinations(
    leg_lists: list[list[AccaLeg]]
) -> list[tuple[AccaLeg, ...]]:
    """Build all combinations taking one leg from each list."""
    if not leg_lists:
        return []
    
    if len(leg_lists) == 1:
        return [(leg,) for leg in leg_lists[0]]
    
    result = []
    first_list = leg_lists[0]
    rest_combos = _build_leg_combinations(leg_lists[1:])
    
    for leg in first_list:
        for rest_combo in rest_combos:
            result.append((leg,) + rest_combo)
    
    return result


def _build_narrative(legs: list[AccaLeg], strategy: AccaStrategy) -> str:
    """Build a coherent narrative for the accumulator."""
    if not legs:
        return "Empty accumulator"
    
    parts = []
    
    # Strategy header
    strategy_desc = {
        AccaStrategy.ULTRA: "Ultra-safe build",
        AccaStrategy.CONSERVATIVE: "High-probability build",
        AccaStrategy.BALANCED: "Value-focused build",
        AccaStrategy.AGGRESSIVE: "Story-aligned build",
    }
    parts.append(f"[{strategy_desc.get(strategy, strategy.value)}]")
    
    # Identify common story themes across legs
    all_signals: list[str] = []
    for leg in legs:
        all_signals.extend(leg.story_signals)
    
    # Find shared signals
    signal_counts: dict[str, int] = {}
    for signal in all_signals:
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    shared_signals = [s for s, c in signal_counts.items() if c >= 2]
    
    if shared_signals:
        parts.append(f"Theme: {', '.join(shared_signals[:3])}")
    
    # Leg summaries
    parts.append("")
    for i, leg in enumerate(legs, 1):
        parts.append(f"  {i}. {leg.story_summary}")
    
    # Summary stats
    parts.append("")
    parts.append(
        f"Combined: {legs[0].odds:.2f}" + 
        "".join(f" x {leg.odds:.2f}" for leg in legs[1:]) +
        f" = {legs[0].odds * legs[1].odds:.2f}" if len(legs) > 1 else f" = {legs[0].odds:.2f}"
    )
    
    # Actually calculate combined odds properly
    combined_odds = 1.0
    for leg in legs:
        combined_odds *= leg.odds
    parts[-1] = f"Combined odds: {combined_odds:.2f}"
    
    return "\n".join(parts)


# =============================================================================
# CLI Functions
# =============================================================================


def get_todays_match_opportunities(league: str | None = None) -> list[MatchOpportunities]:
    """Get match opportunities for today's fixtures."""
    from src.betting.match_profile import build_match_profile, get_todays_fixtures
    from src.betting.story_generator import generate_story
    from src.betting.opportunity_finder import find_opportunities
    
    fixture_ids = get_todays_fixtures(league=league)
    
    match_opps = []
    for fixture_id in fixture_ids:
        try:
            profile = build_match_profile(fixture_id)
            story = generate_story(profile)
            opps = find_opportunities(profile, story)
            match_opps.append(opps)
        except Exception as e:
            print(f"Warning: Failed to process fixture {fixture_id}: {e}")
    
    return match_opps


def write_accas_json(accas: list[Acca], output_file: str, strategy: str) -> None:
    """Write accas to JSON file compatible with sportybet_booker.py.
    
    Args:
        accas: List of accumulators to write
        output_file: Path to output JSON file
        strategy: Strategy name used to generate accas
    """
    data = {
        "accas": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
    }
    for acca in accas:
        acca_data = {
            "name": f"Acca {len(data['accas']) + 1}",
            "legs": [
                {
                    "home_team": leg.teams[0],
                    "away_team": leg.teams[1],
                    "market_code": leg.market_code,
                    "odds": leg.odds,
                }
                for leg in acca.legs
            ]
        }
        data["accas"].append(acca_data)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written {len(accas)} accas to {output_file}")


def format_acca_summary(acca: Acca) -> str:
    """Format an accumulator for display."""
    lines = [
        f"{'=' * 60}",
        f"ACCUMULATOR ({acca.strategy.upper()})",
        f"{'=' * 60}",
        "",
    ]
    
    # Legs
    for i, leg in enumerate(acca.legs, 1):
        lines.append(f"Leg {i}: {leg.teams[0]} vs {leg.teams[1]}")
        lines.append(f"       Market: {leg.market_code}")
        lines.append(f"       Odds: {leg.odds:.2f} | Prob: {leg.model_prob:.1%}")
        lines.append(f"       Story: {leg.story_summary}")
        lines.append("")
    
    # Summary
    lines.extend([
        "--- SUMMARY ---",
        f"  Combined Odds: {acca.combined_odds:.2f}",
        f"  Combined Prob: {acca.combined_prob:.1%}",
        f"  Expected Value: {acca.expected_value:+.1%}",
        "",
        "--- NARRATIVE ---",
    ])
    
    for line in acca.narrative.split("\n"):
        lines.append(f"  {line}")
    
    lines.append("")
    lines.append(f"Created: {acca.created_at.isoformat()}")
    
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build accumulator combinations with coherent narratives"
    )
    
    parser.add_argument(
        "--today",
        action="store_true",
        help="Build accas for today's fixtures",
    )
    
    parser.add_argument(
        "--fixtures",
        type=int,
        nargs="+",
        help="Specific fixture IDs to analyze",
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in AccaStrategy],
        default=AccaStrategy.BALANCED.value,
        help="Accumulator strategy (default: balanced)",
    )
    
    parser.add_argument(
        "--max-legs",
        type=int,
        default=5,
        help="Maximum legs per acca (default: 5)",
    )
    
    parser.add_argument(
        "--min-legs",
        type=int,
        default=2,
        help="Minimum legs per acca (default: 2)",
    )
    
    parser.add_argument(
        "--min-odds",
        type=float,
        default=1.5,
        help="Minimum odds per leg (default: 1.5)",
    )
    
    parser.add_argument(
        "--max-accas",
        type=int,
        default=10,
        help="Maximum number of accas to return (default: 10)",
    )
    
    parser.add_argument(
        "--league",
        type=str,
        help="Filter by league code (for --today)",
    )
    
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Write accas to JSON file for sportybet_booker.py",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    if not args.today and not args.fixtures:
        print("Error: Must specify --today or --fixtures")
        return 1
    
    # Get match opportunities
    if args.today:
        print("Fetching today's match opportunities...")
        match_opps = get_todays_match_opportunities(league=args.league)
    else:
        print(f"Building profiles for {len(args.fixtures)} fixtures...")
        from src.betting.match_profile import build_match_profile
        from src.betting.story_generator import generate_story
        from src.betting.opportunity_finder import find_opportunities
        
        match_opps = []
        for fixture_id in args.fixtures:
            try:
                profile = build_match_profile(fixture_id)
                story = generate_story(profile)
                opps = find_opportunities(profile, story)
                match_opps.append(opps)
            except Exception as e:
                print(f"Warning: Failed to process fixture {fixture_id}: {e}")
    
    if not match_opps:
        print("No match opportunities found")
        return 0
    
    print(f"Found {len(match_opps)} match opportunities")
    
    # Build accas
    strategy = AccaStrategy(args.strategy)
    
    print(f"\nBuilding accas with {strategy.value} strategy...")
    print(f"  Max legs: {args.max_legs}")
    print(f"  Min legs: {args.min_legs}")
    print(f"  Min odds: {args.min_odds}")
    
    accas = build_accas(
        match_opps=match_opps,
        strategy=strategy,
        max_legs=args.max_legs,
        min_legs=args.min_legs,
        min_odds=args.min_odds,
        max_accas=args.max_accas,
    )
    
    if not accas:
        print("\nNo valid accumulators found with current criteria")
        return 0
    
    print(f"\nGenerated {len(accas)} valid accumulators\n")
    
    # Output
    if args.output_file:
        write_accas_json(accas, args.output_file, strategy.value)
    elif args.output == "json":
        output = {
            "strategy": strategy.value,
            "total_accas": len(accas),
            "accas": [acca.to_dict() for acca in accas],
        }
        print(json.dumps(output, indent=2))
    else:
        for i, acca in enumerate(accas, 1):
            print(f"\n--- ACCA #{i} ---")
            print(format_acca_summary(acca))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
