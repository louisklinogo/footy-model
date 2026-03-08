from __future__ import annotations

from typing import Any

import numpy as np


MULTIGOALS_TOTAL_RANGES: dict[str, tuple[int, int | None]] = {
    "mg_0": (0, 0),
    "mg_1_2": (1, 2),
    "mg_1_3": (1, 3),
    "mg_1_4": (1, 4),
    "mg_1_5": (1, 5),
    "mg_1_6": (1, 6),
    "mg_2_3": (2, 3),
    "mg_2_4": (2, 4),
    "mg_2_5": (2, 5),
    "mg_2_6": (2, 6),
    "mg_3_4": (3, 4),
    "mg_3_5": (3, 5),
    "mg_3_6": (3, 6),
    "mg_4_5": (4, 5),
    "mg_4_6": (4, 6),
    "mg_5_6": (5, 6),
    "mg_7p": (7, None),
}

MULTIGOALS_HOME_RANGES: dict[str, tuple[int, int | None]] = {
    "hmg_0": (0, 0),
    "hmg_1_2": (1, 2),
    "hmg_1_3": (1, 3),
    "hmg_2_3": (2, 3),
    "hmg_4p": (4, None),
}

MULTIGOALS_AWAY_RANGES: dict[str, tuple[int, int | None]] = {
    "amg_0": (0, 0),
    "amg_1_2": (1, 2),
    "amg_1_3": (1, 3),
    "amg_2_3": (2, 3),
    "amg_4p": (4, None),
}

MULTISCORE_GROUPS: dict[str, set[tuple[int, int]]] = {
    "ms_h_1_0_2_0_3_0": {(1, 0), (2, 0), (3, 0)},
    "ms_a_0_1_0_2_0_3": {(0, 1), (0, 2), (0, 3)},
    "ms_h_4_0_5_0_6_0": {(4, 0), (5, 0), (6, 0)},
    "ms_a_0_4_0_5_0_6": {(0, 4), (0, 5), (0, 6)},
    "ms_h_2_1_3_1_4_1": {(2, 1), (3, 1), (4, 1)},
    "ms_h_1_2_1_3_1_4": {(1, 2), (1, 3), (1, 4)},
    "ms_h_3_2_4_2_5_1": {(3, 2), (4, 2), (5, 1)},
    "ms_a_2_3_2_4_1_5": {(2, 3), (2, 4), (1, 5)},
}


def _prob_range(mask_source: np.ndarray, low: int, high: int | None, mat: np.ndarray) -> float:
    if high is None:
        mask = mask_source >= low
    else:
        mask = (mask_source >= low) & (mask_source <= high)
    return float(mat[mask].sum())


def _prob_exact_group(
    *,
    mat: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    group_scores: set[tuple[int, int]],
) -> tuple[float, np.ndarray]:
    mask = np.zeros_like(mat, dtype=bool)
    for home_goals, away_goals in group_scores:
        mask = mask | ((home_idx == home_goals) & (away_idx == away_goals))
    return float(mat[mask].sum()), mask


def normalize_score_matrix(score_matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(score_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("score_matrix must be 2D")
    if np.any(mat < 0.0):
        raise ValueError("score_matrix cannot have negative probabilities")
    total = float(mat.sum())
    if total <= 0.0:
        raise ValueError("score_matrix sum must be positive")
    return mat / total


def derive_markets_from_score_matrix(score_matrix: np.ndarray) -> dict[str, float]:
    mat = normalize_score_matrix(score_matrix)
    max_home, max_away = mat.shape
    home_idx, away_idx = np.indices((max_home, max_away))
    goal_total = home_idx + away_idx
    goal_diff = home_idx - away_idx

    p_home = float(mat[home_idx > away_idx].sum())
    p_draw = float(mat[home_idx == away_idx].sum())
    p_away = float(mat[home_idx < away_idx].sum())

    p_o15 = float(mat[goal_total >= 2].sum())
    p_u35 = float(mat[goal_total <= 3].sum())

    p_dc_1x = p_home + p_draw
    p_dc_x2 = p_draw + p_away
    p_dc_12 = p_home + p_away

    p_ah_h05 = p_home
    p_ah_a05 = p_away
    p_ah_h15 = float(mat[goal_diff >= 2].sum())
    p_ah_a15 = float(mat[goal_diff <= -2].sum())

    p_ah2_home_m05 = p_ah_h05
    p_ah2_away_p05 = float(mat[goal_diff <= 0].sum())
    p_ah2_away_m05 = p_ah_a05
    p_ah2_home_p05 = float(mat[goal_diff >= 0].sum())
    p_ah2_home_m15 = p_ah_h15
    p_ah2_away_p15 = float(mat[goal_diff <= 1].sum())
    p_ah2_away_m15 = p_ah_a15
    p_ah2_home_p15 = float(mat[goal_diff >= -1].sum())

    p_eh3_0_1_home = float(mat[goal_diff >= 2].sum())
    p_eh3_0_1_draw = float(mat[goal_diff == 1].sum())
    p_eh3_0_1_away = float(mat[goal_diff <= 0].sum())
    p_eh3_1_0_home = float(mat[goal_diff >= 0].sum())
    p_eh3_1_0_draw = float(mat[goal_diff == -1].sum())
    p_eh3_1_0_away = float(mat[goal_diff <= -2].sum())

    # Legacy repo EH aliases remain binary proxies for the home/away win branches.
    p_eh_h1 = p_eh3_0_1_home
    p_eh_a1 = p_eh3_1_0_away

    out = {
        "1x2_h": p_home,
        "1x2_d": p_draw,
        "1x2_a": p_away,
        "dc_1x": p_dc_1x,
        "dc_x2": p_dc_x2,
        "dc_12": p_dc_12,
        "o15": p_o15,
        "u35": p_u35,
        # Legacy runtime handicap aliases.
        "ah_h05": p_ah_h05,
        "ah_a05": p_ah_a05,
        "ah_h15": p_ah_h15,
        "ah_a15": p_ah_a15,
        "eh_h1": p_eh_h1,
        "eh_a1": p_eh_a1,
        # Canonical AH current-line selections.
        "ah2_home_m05": p_ah2_home_m05,
        "ah2_away_p05": p_ah2_away_p05,
        "ah2_away_m05": p_ah2_away_m05,
        "ah2_home_p05": p_ah2_home_p05,
        "ah2_home_m15": p_ah2_home_m15,
        "ah2_away_p15": p_ah2_away_p15,
        "ah2_away_m15": p_ah2_away_m15,
        "ah2_home_p15": p_ah2_home_p15,
        # Canonical EH 3-way selections for the current one-goal displayed lines.
        "eh3_0_1_home": p_eh3_0_1_home,
        "eh3_0_1_draw": p_eh3_0_1_draw,
        "eh3_0_1_away": p_eh3_0_1_away,
        "eh3_1_0_home": p_eh3_1_0_home,
        "eh3_1_0_draw": p_eh3_1_0_draw,
        "eh3_1_0_away": p_eh3_1_0_away,
    }

    # Multigoals totals (range bins)
    for code, (low, high) in MULTIGOALS_TOTAL_RANGES.items():
        out[code] = _prob_range(goal_total, low, high, mat)

    # Team multigoals (home / away)
    for code, (low, high) in MULTIGOALS_HOME_RANGES.items():
        out[code] = _prob_range(home_idx, low, high, mat)
    for code, (low, high) in MULTIGOALS_AWAY_RANGES.items():
        out[code] = _prob_range(away_idx, low, high, mat)

    # Multiscore grouped exact-score markets
    used_home = np.zeros_like(mat, dtype=bool)
    used_away = np.zeros_like(mat, dtype=bool)
    for code, scores in MULTISCORE_GROUPS.items():
        prob, mask = _prob_exact_group(
            mat=mat, home_idx=home_idx, away_idx=away_idx, group_scores=scores
        )
        out[code] = prob
        score_list = list(scores)
        is_home_group = all(home_goals > away_goals for home_goals, away_goals in score_list)
        is_away_group = all(home_goals < away_goals for home_goals, away_goals in score_list)
        if is_home_group:
            used_home = used_home | mask
        elif is_away_group:
            used_away = used_away | mask

    draw_mask = home_idx == away_idx
    home_win_mask = home_idx > away_idx
    away_win_mask = home_idx < away_idx
    out["ms_draw"] = float(mat[draw_mask].sum())
    out["ms_other_homewin"] = float(mat[home_win_mask & (~used_home)].sum())
    out["ms_other_awaywin"] = float(mat[away_win_mask & (~used_away)].sum())

    return {market: float(np.clip(prob, 0.0, 1.0)) for market, prob in out.items()}


def assert_probability_identities(probs: dict[str, float], tol: float = 1e-9) -> None:
    def _close(a: float, b: float) -> bool:
        return abs(a - b) <= tol

    if not _close(probs["1x2_h"] + probs["1x2_d"] + probs["1x2_a"], 1.0):
        raise ValueError("1x2 probabilities do not sum to 1.")
    if not _close(probs["dc_1x"], probs["1x2_h"] + probs["1x2_d"]):
        raise ValueError("dc_1x identity failed.")
    if not _close(probs["dc_x2"], probs["1x2_d"] + probs["1x2_a"]):
        raise ValueError("dc_x2 identity failed.")
    if not _close(probs["dc_12"], probs["1x2_h"] + probs["1x2_a"]):
        raise ValueError("dc_12 identity failed.")
    if not _close(probs["o15"] + (1.0 - probs["o15"]), 1.0):
        raise ValueError("o15 complement identity failed.")
    if not _close(probs["u35"] + (1.0 - probs["u35"]), 1.0):
        raise ValueError("u35 complement identity failed.")
    canonical_ah_pairs = (
        ("ah2_home_m05", "ah2_away_p05"),
        ("ah2_away_m05", "ah2_home_p05"),
        ("ah2_home_m15", "ah2_away_p15"),
        ("ah2_away_m15", "ah2_home_p15"),
    )
    for left, right in canonical_ah_pairs:
        if left in probs and right in probs and not _close(probs[left] + probs[right], 1.0):
            raise ValueError(f"{left}/{right} complement identity failed.")
    canonical_eh_lines = (
        ("eh3_0_1_home", "eh3_0_1_draw", "eh3_0_1_away"),
        ("eh3_1_0_home", "eh3_1_0_draw", "eh3_1_0_away"),
    )
    for home_key, draw_key, away_key in canonical_eh_lines:
        if {home_key, draw_key, away_key}.issubset(probs.keys()):
            if not _close(probs[home_key] + probs[draw_key] + probs[away_key], 1.0):
                raise ValueError(f"{home_key}/{draw_key}/{away_key} partition identity failed.")
    if "mg_0" in probs and "mg_7p" in probs:
        if not (
            probs["mg_0"] <= probs["mg_1_2"] + tol
            and probs["mg_1_2"] <= probs["mg_1_3"] + tol
            and probs["mg_1_3"] <= probs["mg_1_4"] + tol
        ):
            raise ValueError("multigoals total monotonic range identity failed.")
    multiscore_keys = {
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
    if multiscore_keys.issubset(probs.keys()):
        total = float(sum(probs[key] for key in multiscore_keys))
        if not _close(total, 1.0):
            raise ValueError("multiscore partition does not sum to 1.")


def derive_and_validate(score_matrix: np.ndarray) -> dict[str, Any]:
    probs = derive_markets_from_score_matrix(score_matrix)
    assert_probability_identities(probs)
    return probs
