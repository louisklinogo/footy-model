"""Utilities for pre-match ELO feature generation."""

from __future__ import annotations

from typing import Dict
import pandas as pd


ELO_BASE = 1500.0
ELO_K = 20.0
HOME_ADVANTAGE = 60.0


def _expected_home_score(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(home_elo + HOME_ADVANTAGE - away_elo) / 400.0))


def add_pre_match_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-free pre-match ELO columns to historical matches.

    Required columns:
      - home_team, away_team, fthg, ftag
    Assumes rows are in chronological order.
    """
    out = df.copy()
    ratings: Dict[str, float] = {}

    home_pre = []
    away_pre = []
    elo_diff = []
    home_win_prob = []

    for row in out.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        h_elo = ratings.get(home, ELO_BASE)
        a_elo = ratings.get(away, ELO_BASE)
        exp_h = _expected_home_score(h_elo, a_elo)

        home_pre.append(h_elo)
        away_pre.append(a_elo)
        elo_diff.append(h_elo - a_elo)
        home_win_prob.append(exp_h)

        if row.fthg > row.ftag:
            s_h = 1.0
        elif row.fthg == row.ftag:
            s_h = 0.5
        else:
            s_h = 0.0

        delta = ELO_K * (s_h - exp_h)
        ratings[home] = h_elo + delta
        ratings[away] = a_elo - delta

    out["home_elo"] = home_pre
    out["away_elo"] = away_pre
    out["elo_diff"] = elo_diff
    out["elo_home_win_prob"] = home_win_prob
    return out


def build_latest_elo_ratings(df_completed: pd.DataFrame) -> Dict[str, float]:
    """
    Build latest team ELO map from completed historical matches.

    Required columns:
      - home_team, away_team, fthg, ftag
    Assumes rows are in chronological order.
    """
    ratings: Dict[str, float] = {}
    for row in df_completed.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        h_elo = ratings.get(home, ELO_BASE)
        a_elo = ratings.get(away, ELO_BASE)
        exp_h = _expected_home_score(h_elo, a_elo)

        if row.fthg > row.ftag:
            s_h = 1.0
        elif row.fthg == row.ftag:
            s_h = 0.5
        else:
            s_h = 0.0

        delta = ELO_K * (s_h - exp_h)
        ratings[home] = h_elo + delta
        ratings[away] = a_elo - delta

    return ratings


def expected_from_ratings(home_elo: float, away_elo: float) -> float:
    return _expected_home_score(home_elo, away_elo)
