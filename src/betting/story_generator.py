"""
story_generator.py - Converts match profiles to narratives.

Takes a MatchProfile and generates a human-readable story with style matchups,
form analysis, position context, and simulated scorelines.

Usage:
    python src/betting/story_generator.py --fixture 12345
    python src/betting/story_generator.py --fixture 12345 --output markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from math import factorial
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.match_profile import (
    MatchProfile,
    TeamProfile,
    MatchContext,
    build_match_profile,
)


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class StyleMatchup:
    """Analysis of how two team styles match up."""
    home_style: str
    away_style: str
    matchup_type: str  # "possession_vs_direct", "press_vs_block", etc.
    description: str  # "Open game expected"
    expected_corners: str  # "High", "Medium", "Low"
    expected_goals: str  # "High-scoring", "Low-scoring", "Average"


@dataclass(frozen=True)
class FormClash:
    """Analysis of form advantage between two teams."""
    home_form: str  # "Hot", "Cold", "Mixed"
    away_form: str
    advantage: str  # "home", "away", "neutral"
    description: str


@dataclass(frozen=True)
class PositionContext:
    """Analysis of motivation based on table positions."""
    home_situation: str  # "Title race", "Relegation battle", "Mid-table", "Chasing Europe"
    away_situation: str
    motivation: str  # "Home needs win more"
    description: str


@dataclass(frozen=True)
class MatchStory:
    """Complete narrative story for a match."""
    fixture_id: int
    teams: tuple[str, str]

    # Core narrative elements
    style_matchup: StyleMatchup
    form_clash: FormClash
    position_context: PositionContext

    # Additional context
    h2h_narrative: str
    availability_impact: str
    derby_context: str | None

    # Simulation outputs
    expected_scoreline: str
    likely_scorelines: list[str]
    expected_corners: str
    expected_cards: str
    game_flow: str

    # Full narrative
    narrative: str  # Human-readable paragraph


# =============================================================================
# Style Matchup Analysis
# =============================================================================

# Style matchup rules: (home_pattern, away_pattern) -> (matchup_type, description, corners, goals)
STYLE_MATCHUP_RULES: list[tuple[str, str, str, str, str, str]] = [
    # Possession vs Direct
    ("Possession", "Direct", "possession_vs_direct",
     "Possession side will dominate the ball against direct counter-attackers. Expect an open game with spaces to exploit.",
     "High", "High-scoring"),
    ("Direct", "Possession", "direct_vs_possession",
     "Direct side will look to hit on the break against possession-heavy opponents. Counter-attack opportunities.",
     "High", "High-scoring"),

    # Possession vs Press
    ("Possession", "Balanced_Press", "possession_vs_press",
     "Possession side faces organized pressing. Ball retention will be tested, potential for mistakes.",
     "Medium", "Medium"),
    ("Possession", "Press", "possession_vs_press",
     "Possession side faces intense pressing. Ball retention will be tested, potential for mistakes.",
     "Medium", "Medium"),
    ("Balanced_Press", "Possession", "press_vs_possession",
     "Pressing side will look to disrupt possession play. High-intensity contest expected.",
     "Medium", "Medium"),

    # Direct vs Balanced
    ("Direct", "Balanced", "direct_vs_balanced",
     "Direct approach against organized opposition. May struggle to break down defensive structure.",
     "Low", "Low-scoring"),
    ("Balanced", "Direct", "balanced_vs_direct",
     "Balanced side faces direct opposition. Need to manage counter-attack threats.",
     "Low", "Low-scoring"),

    # Press vs Block
    ("Press", "Block", "press_vs_block",
     "High press meets low block. Pressing side will need patience to break down organized defense.",
     "Medium", "Medium"),
    ("Block", "Press", "block_vs_press",
     "Low block faces high press. Defensive organization will be tested under sustained pressure.",
     "Medium", "Medium"),
    ("Balanced_Press", "Block", "press_vs_block",
     "Moderate press meets low block. Patient buildup required.",
     "Medium", "Medium"),
    ("Block", "Balanced_Press", "block_vs_press",
     "Low block faces moderate press. Counter-attack opportunities may be limited.",
     "Medium", "Medium"),

    # Possession vs Block
    ("Possession", "Block", "possession_vs_block",
     "Possession side will dominate territory against deep defense. Patience and width will be key.",
     "High", "Low-scoring"),
    ("Block", "Possession", "block_vs_possession",
     "Deep block against possession side. Counter-attack specialists may find space.",
     "High", "Low-scoring"),

    # Direct vs Press
    ("Direct", "Press", "direct_vs_press",
     "Direct side faces high press. Long balls may bypass pressure effectively.",
     "Medium", "Medium"),
    ("Press", "Direct", "press_vs_direct",
     "High press against direct side. Look to force mistakes in opposition half.",
     "Medium", "Medium"),

    # Balanced vs Balanced
    ("Balanced", "Balanced", "balanced_vs_balanced",
     "Two balanced sides meet. Tactical flexibility from both teams expected.",
     "Medium", "Average"),
]


def analyze_style_matchup(home_style: str, away_style: str) -> StyleMatchup:
    """
    Analyze what happens when two styles meet.

    Uses pattern matching on style cluster names to determine the matchup type,
    expected corners, and expected goals.

    Args:
        home_style: Style cluster name for home team
        away_style: Style cluster name for away team

    Returns:
        StyleMatchup with analysis of the tactical battle
    """
    # Handle unknown styles
    if home_style == "Unknown" or away_style == "Unknown":
        return StyleMatchup(
            home_style=home_style,
            away_style=away_style,
            matchup_type="unknown",
            description="Unable to analyze matchup due to missing style data.",
            expected_corners="Medium",
            expected_goals="Average",
        )

    # Try to match against rules
    for home_pattern, away_pattern, matchup_type, description, corners, goals in STYLE_MATCHUP_RULES:
        if home_pattern in home_style and away_pattern in away_style:
            return StyleMatchup(
                home_style=home_style,
                away_style=away_style,
                matchup_type=matchup_type,
                description=description,
                expected_corners=corners,
                expected_goals=goals,
            )

    # Default fallback for unmatched combinations
    return StyleMatchup(
        home_style=home_style,
        away_style=away_style,
        matchup_type="standard",
        description="Standard tactical matchup. Both teams will look to impose their style.",
        expected_corners="Medium",
        expected_goals="Average",
    )


# =============================================================================
# Form Analysis
# =============================================================================


def analyze_form(home_form_pts: int, away_form_pts: int) -> FormClash:
    """
    Analyze form advantage between two teams.

    Form categories:
    - Hot: 11+ pts from last 5 (avg 2.2+ pts per game)
    - Cold: 0-4 pts from last 5 (avg <1 pt per game)
    - Mixed: 5-10 pts

    Args:
        home_form_pts: Points from last 5 games for home team
        away_form_pts: Points from last 5 games for away team

    Returns:
        FormClash with form categories and advantage assessment
    """
    def categorize_form(pts: int) -> str:
        if pts >= 11:
            return "Hot"
        elif pts <= 4:
            return "Cold"
        else:
            return "Mixed"

    home_form = categorize_form(home_form_pts)
    away_form = categorize_form(away_form_pts)

    # Determine advantage
    form_diff = home_form_pts - away_form_pts

    if form_diff >= 5:
        advantage = "home"
        description = f"Home side in significantly better form ({home_form_pts} vs {away_form_pts} pts). Strong momentum advantage."
    elif form_diff <= -5:
        advantage = "away"
        description = f"Away side in significantly better form ({away_form_pts} vs {home_form_pts} pts). Momentum on their side."
    elif form_diff >= 2:
        advantage = "home"
        description = f"Home side has slight form edge ({home_form_pts} vs {away_form_pts} pts)."
    elif form_diff <= -2:
        advantage = "away"
        description = f"Away side has slight form edge ({away_form_pts} vs {home_form_pts} pts)."
    else:
        advantage = "neutral"
        description = f"Both sides in similar form ({home_form_pts} vs {away_form_pts} pts). Even contest expected."

    return FormClash(
        home_form=home_form,
        away_form=away_form,
        advantage=advantage,
        description=description,
    )


# =============================================================================
# Position Context Analysis
# =============================================================================


def analyze_position_context(home_pos: int, away_pos: int, league_size: int = 20) -> PositionContext:
    """
    Analyze motivation based on table position.

    Position categories:
    - Top 4: "Title race" / "Chasing Europe"
    - Bottom 4: "Relegation battle"
    - Else: "Mid-table"

    Args:
        home_pos: League position of home team
        away_pos: League position of away team
        league_size: Total teams in league (default 20)

    Returns:
        PositionContext with situation analysis and motivation
    """
    def get_situation(pos: int) -> str:
        if pos == 0:  # Unknown position
            return "Unknown position"
        if pos <= 4:
            return "Chasing Europe"
        if pos <= 6:
            return "European places in reach"
        if pos >= league_size - 3:
            return "Relegation battle"
        if pos >= league_size - 6:
            return "Looking over shoulder"
        return "Mid-table"

    home_situation = get_situation(home_pos)
    away_situation = get_situation(away_pos)

    # Determine motivation
    home_need = _calculate_need(home_pos, league_size)
    away_need = _calculate_need(away_pos, league_size)

    if home_need > away_need + 1:
        motivation = "Home needs win more"
        description = f"{home_situation} side faces {away_situation.lower()} opponents. Home side has more to play for."
    elif away_need > home_need + 1:
        motivation = "Away needs win more"
        description = f"{away_situation} side travels to {home_situation.lower()} hosts. Away side desperate for points."
    else:
        motivation = "Even motivation"
        description = f"Both sides in similar situations: {home_situation} vs {away_situation}."

    return PositionContext(
        home_situation=home_situation,
        away_situation=away_situation,
        motivation=motivation,
        description=description,
    )


def _calculate_need(pos: int, league_size: int) -> int:
    """Calculate a 'need' score based on position (higher = more desperate)."""
    if pos == 0:
        return 0
    if pos <= 4:
        return 3  # European places - high motivation
    if pos >= league_size - 3:
        return 5  # Relegation zone - desperate
    if pos >= league_size - 6:
        return 3  # Near drop zone - concerned
    return 1  # Mid-table comfort


# =============================================================================
# Scoreline Simulation
# =============================================================================


def _poisson_pmf(k: int, lambda_val: float) -> float:
    """Calculate Poisson probability mass function."""
    return (lambda_val ** k) * (2.718281828 ** (-lambda_val)) / factorial(k)


def simulate_scoreline(home_lambda: float, away_lambda: float) -> tuple[str, list[str]]:
    """
    Simulate expected scoreline from lambda values using Poisson distribution.

    Calculates probabilities for scorelines 0-0 through 5-5 and returns:
    - The most likely scoreline (mode)
    - Top 4 most likely scorelines

    Args:
        home_lambda: Expected goals for home team
        away_lambda: Expected goals for away team

    Returns:
        Tuple of (most_likely_scoreline, top_4_scorelines)
    """
    # Calculate probabilities for all scorelines 0-0 to 5-5
    scorelines: list[tuple[float, str]] = []

    for home_goals in range(6):
        for away_goals in range(6):
            prob = _poisson_pmf(home_goals, home_lambda) * _poisson_pmf(away_goals, away_lambda)
            scoreline = f"{home_goals}-{away_goals}"
            scorelines.append((prob, scoreline))

    # Sort by probability descending
    scorelines.sort(key=lambda x: x[0], reverse=True)

    # Get most likely and top 4
    most_likely = scorelines[0][1]
    top_4 = [s[1] for s in scorelines[:4]]

    return most_likely, top_4


def _estimate_corners(home_corners: float, away_corners: float) -> str:
    """Estimate total corners for a match."""
    total = home_corners + away_corners
    if total >= 12:
        return "High (11+)"
    elif total >= 9:
        return "Medium (8-11)"
    else:
        return "Low (<8)"


def _estimate_cards(style_matchup: StyleMatchup, is_derby: bool) -> str:
    """Estimate card expectation for a match."""
    base = "Average"

    if is_derby:
        base = "High (derby intensity)"

    if style_matchup.matchup_type in ("possession_vs_press", "press_vs_possession", "press_vs_block", "block_vs_press"):
        if base == "High (derby intensity)":
            base = "Very High (derby + intensity)"
        else:
            base = "Above Average (tactical fouling)"

    return base


def _generate_game_flow(style_matchup: StyleMatchup, possession_home: float, possession_away: float) -> str:
    """Generate a description of expected game flow."""
    possession_diff = possession_home - possession_away

    if possession_diff > 10:
        return f"Home-dominated possession ({possession_home:.0f}% expected). Looks to control tempo."
    elif possession_diff < -10:
        return f"Away-dominated possession ({possession_away:.0f}% expected). Visitors to control the ball."
    elif style_matchup.matchup_type == "possession_vs_direct":
        return "Contrasting styles: possession vs counter-attack. Potential for end-to-end action."
    elif style_matchup.matchup_type in ("press_vs_block", "block_vs_press"):
        return "Tactical chess match: press vs block. Patience required."
    else:
        return "Balanced contest expected. Both sides capable of controlling phases."


# =============================================================================
# H2H Narrative
# =============================================================================


def _generate_h2h_narrative(match_context: MatchContext, home_name: str, away_name: str) -> str:
    """Generate H2H narrative from match context."""
    h2h = match_context.h2h_matches_count

    if h2h == 0:
        return f"No recent head-to-head data between {home_name} and {away_name}."

    home_wins = match_context.h2h_home_wins
    draws = match_context.h2h_draws
    away_wins = match_context.h2h_away_wins

    total = home_wins + draws + away_wins

    if total == 0:
        return f"No recent head-to-head data between {home_name} and {away_name}."

    home_pct = home_wins / total * 100
    away_pct = away_wins / total * 100

    if home_pct > 60:
        return f"{home_name} has dominated this fixture recently ({home_wins}W-{draws}D-{away_wins}L in {h2h} matches)."
    elif away_pct > 60:
        return f"{away_name} has had the upper hand recently ({away_wins}W-{draws}D-{home_wins}L in {h2h} matches)."
    else:
        return f"Even head-to-head record: {home_wins} wins each, {draws} draws from {h2h} recent meetings."


# =============================================================================
# Availability Impact
# =============================================================================


def _generate_availability_impact(home_team: TeamProfile, away_team: TeamProfile) -> str:
    """Generate narrative about player availability impact."""
    home_missing = len(home_team.missing_players)
    away_missing = len(away_team.missing_players)
    home_value = home_team.missing_market_value
    away_value = away_team.missing_market_value

    if home_missing == 0 and away_missing == 0:
        return "Both teams at full strength. No significant absences."

    parts = []

    if home_missing > 0:
        eur_value = f"EUR {home_value:,.0f}"
        if home_value > 50_000_000:
            parts.append(f"Home side severely depleted ({home_missing} out, {eur_value} value)")
        elif home_value > 20_000_000:
            parts.append(f"Home side with notable absences ({home_missing} out, {eur_value} value)")
        else:
            parts.append(f"Home side with minor absences ({home_missing} out)")

    if away_missing > 0:
        eur_value = f"EUR {away_value:,.0f}"
        if away_value > 50_000_000:
            parts.append(f"away side severely depleted ({away_missing} out, {eur_value} value)")
        elif away_value > 20_000_000:
            parts.append(f"away side with notable absences ({away_missing} out, {eur_value} value)")
        else:
            parts.append(f"away side with minor absences ({away_missing} out)")

    if len(parts) == 2:
        return f"{parts[0]}, {parts[1]}."
    else:
        return parts[0] + "."


# =============================================================================
# Main Story Generation
# =============================================================================


def generate_story(profile: MatchProfile) -> MatchStory:
    """
    Convert raw profile to narrative story.

    Takes a MatchProfile and generates a comprehensive MatchStory with:
    - Style matchup analysis
    - Form clash analysis
    - Position context
    - H2H narrative
    - Availability impact
    - Simulated scorelines
    - Full narrative paragraph

    Args:
        profile: MatchProfile with all fixture data

    Returns:
        MatchStory with complete narrative
    """
    # Extract data
    home_team = profile.home_team
    away_team = profile.away_team
    match_context = profile.match_context

    # Analyze style matchup
    style_matchup = analyze_style_matchup(
        home_team.style_cluster,
        away_team.style_cluster
    )

    # Analyze form
    form_clash = analyze_form(
        home_team.form_points_last5,
        away_team.form_points_last5
    )

    # Analyze position context
    position_context = analyze_position_context(
        home_team.league_position,
        away_team.league_position
    )

    # Simulate scoreline
    expected_scoreline, likely_scorelines = simulate_scoreline(
        home_team.lambda_value,
        away_team.lambda_value
    )

    # Generate H2H narrative
    h2h_narrative = _generate_h2h_narrative(
        match_context,
        home_team.name,
        away_team.name
    )

    # Generate availability impact
    availability_impact = _generate_availability_impact(home_team, away_team)

    # Derby context
    derby_context = None
    if match_context.is_derby and match_context.rivalry_name:
        derby_context = match_context.rivalry_name

    # Estimate corners and cards
    expected_corners = _estimate_corners(
        home_team.rolling_corners,
        away_team.rolling_corners
    )
    expected_cards = _estimate_cards(style_matchup, match_context.is_derby)

    # Game flow
    game_flow = _generate_game_flow(
        style_matchup,
        home_team.rolling_possession,
        away_team.rolling_possession
    )

    # Build full narrative
    narrative = _build_narrative(
        home_team=home_team,
        away_team=away_team,
        style_matchup=style_matchup,
        form_clash=form_clash,
        position_context=position_context,
        h2h_narrative=h2h_narrative,
        availability_impact=availability_impact,
        derby_context=derby_context,
        expected_scoreline=expected_scoreline,
        game_flow=game_flow,
    )

    return MatchStory(
        fixture_id=profile.fixture_id,
        teams=(home_team.name, away_team.name),
        style_matchup=style_matchup,
        form_clash=form_clash,
        position_context=position_context,
        h2h_narrative=h2h_narrative,
        availability_impact=availability_impact,
        derby_context=derby_context,
        expected_scoreline=expected_scoreline,
        likely_scorelines=likely_scorelines,
        expected_corners=expected_corners,
        expected_cards=expected_cards,
        game_flow=game_flow,
        narrative=narrative,
    )


def _build_narrative(
    home_team: TeamProfile,
    away_team: TeamProfile,
    style_matchup: StyleMatchup,
    form_clash: FormClash,
    position_context: PositionContext,
    h2h_narrative: str,
    availability_impact: str,
    derby_context: str | None,
    expected_scoreline: str,
    game_flow: str,
) -> str:
    """Build the full narrative paragraph."""
    parts = []

    # Opening - teams and context
    derby_str = f" ({derby_context})" if derby_context else ""
    parts.append(f"{home_team.name} hosts {away_team.name}{derby_str}.")

    # Style matchup
    parts.append(style_matchup.description)

    # Form
    parts.append(form_clash.description)

    # Position context
    parts.append(position_context.description)

    # H2H
    parts.append(h2h_narrative)

    # Availability
    parts.append(availability_impact)

    # Prediction
    parts.append(f"Model projects {expected_scoreline}, with game flow: {game_flow.lower()}")

    return " ".join(parts)


# =============================================================================
# Output Formatting
# =============================================================================


def _story_to_dict(story: MatchStory) -> dict[str, Any]:
    """Convert MatchStory to a JSON-serializable dict."""
    data = asdict(story)
    # Convert nested dataclasses
    data["style_matchup"] = asdict(story.style_matchup)
    data["form_clash"] = asdict(story.form_clash)
    data["position_context"] = asdict(story.position_context)
    return data


def _format_summary(story: MatchStory) -> str:
    """Format a human-readable summary of the story."""
    lines = [
        f"{'=' * 60}",
        f"MATCH STORY: {story.teams[0]} vs {story.teams[1]}",
        f"Fixture ID: {story.fixture_id}",
        f"{'=' * 60}",
        "",
        "--- STYLE MATCHUP ---",
        f"  Home Style: {story.style_matchup.home_style}",
        f"  Away Style: {story.style_matchup.away_style}",
        f"  Type: {story.style_matchup.matchup_type}",
        f"  {story.style_matchup.description}",
        f"  Expected Corners: {story.style_matchup.expected_corners}",
        f"  Expected Goals: {story.style_matchup.expected_goals}",
        "",
        "--- FORM CLASH ---",
        f"  Home Form: {story.form_clash.home_form}",
        f"  Away Form: {story.form_clash.away_form}",
        f"  Advantage: {story.form_clash.advantage}",
        f"  {story.form_clash.description}",
        "",
        "--- POSITION CONTEXT ---",
        f"  Home: {story.position_context.home_situation}",
        f"  Away: {story.position_context.away_situation}",
        f"  Motivation: {story.position_context.motivation}",
        f"  {story.position_context.description}",
        "",
        "--- ADDITIONAL CONTEXT ---",
        f"  H2H: {story.h2h_narrative}",
        f"  Availability: {story.availability_impact}",
        f"  Derby: {story.derby_context or 'No'}",
        "",
        "--- SIMULATION ---",
        f"  Expected Scoreline: {story.expected_scoreline}",
        f"  Likely Scorelines: {', '.join(story.likely_scorelines)}",
        f"  Expected Corners: {story.expected_corners}",
        f"  Expected Cards: {story.expected_cards}",
        f"  Game Flow: {story.game_flow}",
        "",
        "--- NARRATIVE ---",
        f"  {story.narrative}",
    ]
    return "\n".join(lines)


def _format_markdown(story: MatchStory) -> str:
    """Format story as markdown."""
    lines = [
        f"# Match Story: {story.teams[0]} vs {story.teams[1]}",
        "",
        f"**Fixture ID:** {story.fixture_id}",
        "",
        "## Style Matchup",
        "",
        f"| Aspect | Home | Away |",
        f"|--------|------|------|",
        f"| Style | {story.style_matchup.home_style} | {story.style_matchup.away_style} |",
        f"| Form | {story.form_clash.home_form} | {story.form_clash.away_form} |",
        f"| Position | {story.position_context.home_situation} | {story.position_context.away_situation} |",
        "",
        f"**Matchup Type:** {story.style_matchup.matchup_type}",
        "",
        f"_{story.style_matchup.description}_",
        "",
        f"- **Expected Corners:** {story.style_matchup.expected_corners}",
        f"- **Expected Goals:** {story.style_matchup.expected_goals}",
        "",
        "## Form Analysis",
        "",
        f"_{story.form_clash.description}_",
        "",
        f"**Advantage:** {story.form_clash.advantage.capitalize()}",
        "",
        "## Position Context",
        "",
        f"_{story.position_context.description}_",
        "",
        f"**Motivation:** {story.position_context.motivation}",
        "",
        "## Match Context",
        "",
        f"- **H2H:** {story.h2h_narrative}",
        f"- **Availability:** {story.availability_impact}",
        f"- **Derby:** {story.derby_context or 'No'}",
        "",
        "## Simulation",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Expected Scoreline | {story.expected_scoreline} |",
        f"| Likely Scorelines | {', '.join(story.likely_scorelines)} |",
        f"| Expected Corners | {story.expected_corners} |",
        f"| Expected Cards | {story.expected_cards} |",
        f"| Game Flow | {story.game_flow} |",
        "",
        "## Narrative",
        "",
        f"> {story.narrative}",
    ]
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate match stories from profiles"
    )
    parser.add_argument(
        "--fixture",
        type=int,
        required=True,
        help="Fixture ID to analyze",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        # Build profile
        profile = build_match_profile(args.fixture)

        # Generate story
        story = generate_story(profile)

        # Output
        if args.output == "json":
            print(json.dumps(_story_to_dict(story), indent=2))
        elif args.output == "markdown":
            print(_format_markdown(story))
        else:
            print(_format_summary(story))

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
