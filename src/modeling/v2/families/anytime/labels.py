from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def prepare_state_ladder_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    target_h1 = _numeric_series(out, "target_h_1up")
    target_a1 = _numeric_series(out, "target_a_1up")
    target_h2 = _numeric_series(out, "target_h_2up")
    target_a2 = _numeric_series(out, "target_a_2up")

    out["target_h_2up_given_h_1up"] = np.where(target_h1 >= 0.5, target_h2, np.nan)
    out["target_a_2up_given_a_1up"] = np.where(target_a1 >= 0.5, target_a2, np.nan)

    first_home = _numeric_series(out, "first_home_lead_minute")
    first_away = _numeric_series(out, "first_away_lead_minute")
    home_first = first_home.notna() & (first_away.isna() | (first_home < first_away))
    away_first = first_away.notna() & (first_home.isna() | (first_away < first_home))

    out["target_home_first_lead"] = np.where(home_first | away_first, home_first.astype(float), np.nan)
    out["target_away_first_lead"] = np.where(home_first | away_first, away_first.astype(float), np.nan)
    out["target_home_first_lead_early"] = np.where(home_first, (first_home <= 30).astype(float), np.nan)
    out["target_away_first_lead_early"] = np.where(away_first, (first_away <= 30).astype(float), np.nan)
    return out