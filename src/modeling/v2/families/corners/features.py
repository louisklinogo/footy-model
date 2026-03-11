from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LEAGUE_REGIME_FEATURES: tuple[str, ...] = (
    "league_total_corners_mean",
    "league_home_corners_mean",
    "league_away_corners_mean",
    "league_home_share_mean",
    "league_c75_rate",
    "league_c85_rate",
    "league_c95_rate",
    "league_c105_rate",
    "league_fixture_count",
)


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _parse_form(value: object) -> tuple[float, float, float]:
    try:
        pieces = [float(part) for part in str(value).split("-")[:3]]
        if len(pieces) >= 3:
            return pieces[0], pieces[1], pieces[2]
    except Exception:
        pass
    return np.nan, np.nan, np.nan


def add_corners_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "style_delta" not in out.columns:
        home_style = _safe_numeric(out, "home_style_score")
        away_style = _safe_numeric(out, "away_style_score")
        if home_style.notna().any() or away_style.notna().any():
            out["style_delta"] = home_style - away_style
        elif "home_formation" in out.columns and "away_formation" in out.columns:
            home_parts = out["home_formation"].apply(_parse_form)
            away_parts = out["away_formation"].apply(_parse_form)
            home_defs = home_parts.apply(lambda vals: vals[0])
            home_mids = home_parts.apply(lambda vals: vals[1])
            home_fwds = home_parts.apply(lambda vals: vals[2])
            away_defs = away_parts.apply(lambda vals: vals[0])
            away_mids = away_parts.apply(lambda vals: vals[1])
            away_fwds = away_parts.apply(lambda vals: vals[2])
            out["style_delta"] = (
                (home_defs + 2.0 * home_mids + 3.0 * home_fwds)
                - (away_defs + 2.0 * away_mids + 3.0 * away_fwds)
            )
        else:
            out["style_delta"] = np.nan
    for feat in LEAGUE_REGIME_FEATURES:
        if feat not in out.columns:
            out[feat] = np.nan
    return out


def fit_league_regime(df: pd.DataFrame, prior_strength: float = 30.0) -> dict[str, Any]:
    required = {"league_code", "fixture_id", "total_corners", "home_corners", "away_corners"}
    if not required.issubset(df.columns):
        return {"prior_strength": float(prior_strength), "global": {}, "by_league": {}}
    base = df[list(required)].copy()
    base["total_corners"] = pd.to_numeric(base["total_corners"], errors="coerce")
    base["home_corners"] = pd.to_numeric(base["home_corners"], errors="coerce")
    base["away_corners"] = pd.to_numeric(base["away_corners"], errors="coerce")
    base["league_code"] = base["league_code"].astype("string")
    base = base.dropna(subset=["league_code", "total_corners", "home_corners", "away_corners"]).copy()
    if base.empty:
        return {"prior_strength": float(prior_strength), "global": {}, "by_league": {}}

    global_n = float(len(base))
    global_stats = {
        "league_total_corners_mean": float(base["total_corners"].mean()),
        "league_home_corners_mean": float(base["home_corners"].mean()),
        "league_away_corners_mean": float(base["away_corners"].mean()),
        "league_home_share_mean": float(
            (base["home_corners"].sum() + 0.5 * global_n)
            / max(float(base["total_corners"].sum() + global_n), 1.0)
        ),
        "league_c75_rate": float((base["total_corners"] > 7.5).mean()),
        "league_c85_rate": float((base["total_corners"] > 8.5).mean()),
        "league_c95_rate": float((base["total_corners"] > 9.5).mean()),
        "league_c105_rate": float((base["total_corners"] > 10.5).mean()),
        "league_fixture_count": global_n,
    }
    grouped = base.groupby("league_code", observed=True).agg(
        fixture_count=("fixture_id", "size"),
        total_sum=("total_corners", "sum"),
        home_sum=("home_corners", "sum"),
        away_sum=("away_corners", "sum"),
        c75_hits=("total_corners", lambda s: int((s > 7.5).sum())),
        c85_hits=("total_corners", lambda s: int((s > 8.5).sum())),
        c95_hits=("total_corners", lambda s: int((s > 9.5).sum())),
        c105_hits=("total_corners", lambda s: int((s > 10.5).sum())),
    )
    by_league: dict[str, dict[str, float]] = {}
    for league_code, row in grouped.iterrows():
        n = float(row["fixture_count"])
        denom = n + float(prior_strength)
        by_league[str(league_code)] = {
            "league_total_corners_mean": float((row["total_sum"] + prior_strength * global_stats["league_total_corners_mean"]) / denom),
            "league_home_corners_mean": float((row["home_sum"] + prior_strength * global_stats["league_home_corners_mean"]) / denom),
            "league_away_corners_mean": float((row["away_sum"] + prior_strength * global_stats["league_away_corners_mean"]) / denom),
            "league_home_share_mean": float((row["home_sum"] + prior_strength * global_stats["league_home_share_mean"]) / max(float(row["total_sum"] + prior_strength), 1.0)),
            "league_c75_rate": float((row["c75_hits"] + prior_strength * global_stats["league_c75_rate"]) / denom),
            "league_c85_rate": float((row["c85_hits"] + prior_strength * global_stats["league_c85_rate"]) / denom),
            "league_c95_rate": float((row["c95_hits"] + prior_strength * global_stats["league_c95_rate"]) / denom),
            "league_c105_rate": float((row["c105_hits"] + prior_strength * global_stats["league_c105_rate"]) / denom),
            "league_fixture_count": n,
        }
    return {
        "prior_strength": float(prior_strength),
        "global": global_stats,
        "by_league": by_league,
    }


def apply_league_regime(df: pd.DataFrame, league_regime: dict[str, Any] | None) -> pd.DataFrame:
    out = add_corners_context_features(df)
    payload = league_regime or {}
    by_league = payload.get("by_league") or {}
    global_stats = payload.get("global") or {}
    league_codes = out.get("league_code", pd.Series(pd.NA, index=out.index)).astype("string")
    for feat in LEAGUE_REGIME_FEATURES:
        default = float(global_stats.get(feat, 0.0))
        mapping = {league: float(values.get(feat, default)) for league, values in by_league.items()}
        out[feat] = pd.to_numeric(league_codes.map(mapping), errors="coerce").fillna(default)
    return out