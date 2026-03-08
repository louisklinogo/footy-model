from __future__ import annotations

from typing import Final


SCORELINE_DIRECTIONAL_MARKETS: Final[frozenset[str]] = frozenset(
    {
        "1x2_h",
        "1x2_d",
        "1x2_a",
        "dc_1x",
        "dc_x2",
        "dc_12",
        "ah2_home_m05",
        "ah2_away_p05",
        "ah2_away_m05",
        "ah2_home_p05",
        "ah2_home_m15",
        "ah2_away_p15",
        "ah2_away_m15",
        "ah2_home_p15",
        "eh3_0_1_home",
        "eh3_0_1_draw",
        "eh3_0_1_away",
        "eh3_1_0_home",
        "eh3_1_0_draw",
        "eh3_1_0_away",
    }
)

SCORELINE_TOTALS_CORE_MARKETS: Final[frozenset[str]] = frozenset({"o15", "u35"})

SCORELINE_MULTIGOALS_MARKETS: Final[frozenset[str]] = frozenset(
    {
        "mg_0",
        "mg_1_2",
        "mg_1_3",
        "mg_1_4",
        "mg_1_5",
        "mg_1_6",
        "mg_2_3",
        "mg_2_4",
        "mg_2_5",
        "mg_2_6",
        "mg_3_4",
        "mg_3_5",
        "mg_3_6",
        "mg_4_5",
        "mg_4_6",
        "mg_5_6",
        "mg_7p",
        "hmg_0",
        "hmg_1_2",
        "hmg_1_3",
        "hmg_2_3",
        "hmg_4p",
        "amg_0",
        "amg_1_2",
        "amg_1_3",
        "amg_2_3",
        "amg_4p",
    }
)

SCORELINE_MULTISCORE_MARKETS: Final[frozenset[str]] = frozenset(
    {
        "ms_h_1_0_2_0_3_0",
        "ms_a_0_1_0_2_0_3",
        "ms_h_4_0_5_0_6_0",
        "ms_a_0_4_0_5_0_6",
        "ms_h_2_1_3_1_4_1",
        "ms_h_1_2_1_3_1_4",
        "ms_h_3_2_4_2_5_1",
        "ms_a_2_3_2_4_1_5",
        "ms_other_homewin",
        "ms_other_awaywin",
        "ms_draw",
    }
)

SCORELINE_SUBFAMILY_MARKETS: Final[dict[str, frozenset[str]]] = {
    "scoreline_directional": SCORELINE_DIRECTIONAL_MARKETS,
    "scoreline_totals_core": SCORELINE_TOTALS_CORE_MARKETS,
    "scoreline_multigoals": SCORELINE_MULTIGOALS_MARKETS,
    "scoreline_multiscore": SCORELINE_MULTISCORE_MARKETS,
}

CORNERS_MARKETS: Final[frozenset[str]] = frozenset(
    {"c75", "c85", "c95", "c105", "hc25", "hc35", "hc45", "hc55", "ac25", "ac35", "ac45", "ac55"}
)

ANYTIME_MARKETS: Final[frozenset[str]] = frozenset({"h_1up", "a_1up", "h_2up", "a_2up"})

DEFAULT_FAMILY_MARKETS: Final[dict[str, frozenset[str]]] = {
    **SCORELINE_SUBFAMILY_MARKETS,
    "corners": CORNERS_MARKETS,
    "anytime": ANYTIME_MARKETS,
}

DEFAULT_FAMILY_ARTIFACT_SOURCES: Final[dict[str, str]] = {
    "scoreline_directional": "scoreline",
    "scoreline_totals_core": "scoreline",
    "scoreline_multigoals": "scoreline",
    "scoreline_multiscore": "scoreline",
    "corners": "corners",
    "anytime": "anytime",
}