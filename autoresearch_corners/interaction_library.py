from __future__ import annotations

from dataclasses import dataclass

from src.modeling.v2.families.corners.features import (
    AXIS_INTERACTION_FEATURES,
    STYLE_MATCHUP_INTERACTION_FEATURES,
)


@dataclass(frozen=True)
class InteractionBlock:
    name: str
    short_name: str
    description: str
    feature_names: tuple[str, ...]


def _matchup_block(*source_columns: str) -> tuple[str, ...]:
    suffixes = tuple(f"__x__{column}" for column in source_columns)
    return tuple(
        feature
        for feature in STYLE_MATCHUP_INTERACTION_FEATURES
        if feature.endswith(suffixes)
    )


BLOCKS: tuple[InteractionBlock, ...] = (
    InteractionBlock(
        name="style_matchup_possession",
        short_name="sm_poss",
        description="Style-matchup interactions gated by rolling possession.",
        feature_names=_matchup_block("home_rolling_possession", "away_rolling_possession"),
    ),
    InteractionBlock(
        name="style_matchup_possession_against",
        short_name="sm_possa",
        description="Style-matchup interactions gated by rolling possession-against.",
        feature_names=_matchup_block(
            "home_rolling_possession_against",
            "away_rolling_possession_against",
        ),
    ),
    InteractionBlock(
        name="style_matchup_corners",
        short_name="sm_corn",
        description="Style-matchup interactions gated by rolling corners.",
        feature_names=_matchup_block("home_rolling_corners", "away_rolling_corners"),
    ),
    InteractionBlock(
        name="style_matchup_corners_against",
        short_name="sm_corna",
        description="Style-matchup interactions gated by rolling corners conceded.",
        feature_names=_matchup_block(
            "home_rolling_corners_against",
            "away_rolling_corners_against",
        ),
    ),
    InteractionBlock(
        name="style_matchup_xg",
        short_name="sm_xg",
        description="Style-matchup interactions gated by rolling xG.",
        feature_names=_matchup_block("home_rolling_xg", "away_rolling_xg"),
    ),
    InteractionBlock(
        name="style_matchup_box_touches",
        short_name="sm_bt",
        description="Style-matchup interactions gated by rolling box touches.",
        feature_names=_matchup_block("home_rolling_box_touches", "away_rolling_box_touches"),
    ),
    InteractionBlock(
        name="style_matchup_box_touches_against",
        short_name="sm_bta",
        description="Style-matchup interactions gated by rolling box touches conceded.",
        feature_names=_matchup_block(
            "home_rolling_box_touches_against",
            "away_rolling_box_touches_against",
        ),
    ),
    InteractionBlock(
        name="style_matchup_crosses",
        short_name="sm_cross",
        description="Style-matchup interactions gated by rolling crosses.",
        feature_names=_matchup_block("home_rolling_crosses", "away_rolling_crosses"),
    ),
    InteractionBlock(
        name="style_matchup_crosses_against",
        short_name="sm_crossa",
        description="Style-matchup interactions gated by rolling crosses conceded.",
        feature_names=_matchup_block(
            "home_rolling_crosses_against",
            "away_rolling_crosses_against",
        ),
    ),
    InteractionBlock(
        name="style_matchup_sot",
        short_name="sm_sot",
        description="Style-matchup interactions gated by rolling shots on target.",
        feature_names=_matchup_block("home_rolling_sot", "away_rolling_sot"),
    ),
    InteractionBlock(
        name="style_matchup_sot_against",
        short_name="sm_sota",
        description="Style-matchup interactions gated by rolling shots on target conceded.",
        feature_names=_matchup_block(
            "home_rolling_sot_against",
            "away_rolling_sot_against",
        ),
    ),
    InteractionBlock(
        name="style_matchup_xg_against",
        short_name="sm_xga",
        description="Style-matchup interactions gated by rolling xG against.",
        feature_names=_matchup_block("home_rolling_xg_against", "away_rolling_xg_against"),
    ),
    InteractionBlock(
        name="axis_attack_vs_defense",
        short_name="ax_advd",
        description="Axis-level attack-vs-defense interactions.",
        feature_names=tuple(
            feature
            for feature in AXIS_INTERACTION_FEATURES
            if "style_attack_axis" in feature
        ),
    ),
    InteractionBlock(
        name="axis_press_vs_territory",
        short_name="ax_prterr",
        description="Axis-level press-vs-territory interactions.",
        feature_names=tuple(
            feature
            for feature in AXIS_INTERACTION_FEATURES
            if "style_press_axis" in feature
        ),
    ),
)

BLOCKS_BY_NAME: dict[str, InteractionBlock] = {block.name: block for block in BLOCKS}


def list_blocks() -> tuple[InteractionBlock, ...]:
    return BLOCKS


def get_block(name: str) -> InteractionBlock:
    try:
        return BLOCKS_BY_NAME[str(name)]
    except KeyError as exc:
        raise KeyError(f"Unknown interaction block: {name}") from exc


def selected_feature_names(block_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    selected: list[str] = []
    for block_name in block_names:
        for feature_name in get_block(block_name).feature_names:
            if feature_name not in selected:
                selected.append(feature_name)
    return tuple(selected)


def managed_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for block in BLOCKS:
        for feature_name in block.feature_names:
            if feature_name not in names:
                names.append(feature_name)
    return tuple(names)
