from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[5]
STYLE_CLUSTER_PATH = ROOT_DIR / "model_artifacts" / "style_clusters" / "cluster_labels.parquet"
STYLE_CLUSTER_JOIN_COLUMNS: tuple[str, ...] = (
    "home_style_cluster",
    "home_style_cluster_confidence",
    "away_style_cluster",
    "away_style_cluster_confidence",
)
STYLE_CLUSTER_AXIS_FEATURES: tuple[str, ...] = (
    "home_style_possession_axis",
    "away_style_possession_axis",
    "home_style_press_axis",
    "away_style_press_axis",
    "home_style_attack_axis",
    "away_style_attack_axis",
    "style_possession_delta",
    "style_press_delta",
    "style_attack_delta",
    "style_cluster_same",
    "home_style_cluster_confidence",
    "away_style_cluster_confidence",
)
STYLE_CLUSTER_LABELS: tuple[str, ...] = (
    "Balanced_MidBlock_Measured",
    "Balanced_Press_Measured",
    "Direct_MidBlock_Reactive",
    "Possession_LowBlock_FrontFoot",
)


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


@lru_cache(maxsize=1)
def _load_style_cluster_fixture_frame() -> pd.DataFrame | None:
    if not STYLE_CLUSTER_PATH.exists():
        return None
    frame = pd.read_parquet(STYLE_CLUSTER_PATH, columns=["fixture_id", *STYLE_CLUSTER_JOIN_COLUMNS])
    if "fixture_id" not in frame.columns or frame.empty:
        return None
    return frame.drop_duplicates(subset=["fixture_id"]).copy()


def _merge_style_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    if "fixture_id" not in df.columns:
        return df
    missing_cols = [col for col in STYLE_CLUSTER_JOIN_COLUMNS if col not in df.columns]
    if not missing_cols:
        return df
    fixture_style = _load_style_cluster_fixture_frame()
    if fixture_style is None or fixture_style.empty:
        return df
    available_cols = ["fixture_id", *[col for col in missing_cols if col in fixture_style.columns]]
    if len(available_cols) == 1:
        return df
    return df.merge(fixture_style[available_cols], on="fixture_id", how="left")


def _cluster_axis(label_series: pd.Series, *, positive: str, neutral: str, negative: str) -> pd.Series:
    labels = label_series.astype("string")
    out = pd.Series(np.nan, index=labels.index, dtype=float)
    out = out.mask(labels.str.contains(positive, na=False), 1.0)
    out = out.mask(labels.str.contains(neutral, na=False), 0.0)
    out = out.mask(labels.str.contains(negative, na=False), -1.0)
    return out


def _style_cluster_slug(label: str) -> str:
    return label.strip().lower().replace("_", "_")


def _style_matchup_feature_name(home_label: str, away_label: str) -> str:
    return f"style_matchup_{_style_cluster_slug(home_label)}__{_style_cluster_slug(away_label)}"


def _interaction_feature_name(left: str, right: str) -> str:
    return f"{left}__x__{right}"


STYLE_MATCHUP_FEATURES: tuple[str, ...] = tuple(
    _style_matchup_feature_name(home_label, away_label)
    for home_label in STYLE_CLUSTER_LABELS
    for away_label in STYLE_CLUSTER_LABELS
)
STYLE_MATCHUP_INTERACTION_SOURCE_COLUMNS: tuple[str, ...] = (
    "home_rolling_corners",
    "away_rolling_corners",
    "home_rolling_corners_against",
    "away_rolling_corners_against",
    "home_rolling_possession",
    "home_rolling_possession_against",
    "away_rolling_possession",
    "away_rolling_possession_against",
    "home_rolling_box_touches",
    "away_rolling_box_touches",
    "home_rolling_box_touches_against",
    "away_rolling_box_touches_against",
    "home_rolling_crosses",
    "away_rolling_crosses",
    "home_rolling_crosses_against",
    "away_rolling_crosses_against",
    "home_rolling_xg",
    "away_rolling_xg",
    "home_rolling_xg_against",
    "away_rolling_xg_against",
    "home_rolling_sot",
    "away_rolling_sot",
    "home_rolling_sot_against",
    "away_rolling_sot_against",
)
STYLE_MATCHUP_INTERACTION_FEATURES: tuple[str, ...] = tuple(
    _interaction_feature_name(matchup_feature, source_column)
    for matchup_feature in STYLE_MATCHUP_FEATURES
    for source_column in STYLE_MATCHUP_INTERACTION_SOURCE_COLUMNS
)
AXIS_INTERACTION_SPECIFICATIONS: tuple[tuple[str, str], ...] = (
    ("home_style_attack_axis", "away_rolling_xg_against"),
    ("away_style_attack_axis", "home_rolling_xg_against"),
    ("home_style_press_axis", "away_rolling_box_touches"),
    ("away_style_press_axis", "home_rolling_box_touches"),
)
AXIS_INTERACTION_FEATURES: tuple[str, ...] = tuple(
    _interaction_feature_name(left, right) for left, right in AXIS_INTERACTION_SPECIFICATIONS
)


def _add_style_cluster_axes(df: pd.DataFrame) -> pd.DataFrame:
    if "home_style_cluster" not in df.columns or "away_style_cluster" not in df.columns:
        return df
    out = df.copy()
    out["home_style_possession_axis"] = _cluster_axis(
        out["home_style_cluster"],
        positive="Possession",
        neutral="Balanced",
        negative="Direct",
    )
    out["away_style_possession_axis"] = _cluster_axis(
        out["away_style_cluster"],
        positive="Possession",
        neutral="Balanced",
        negative="Direct",
    )
    out["home_style_press_axis"] = _cluster_axis(
        out["home_style_cluster"],
        positive="Press",
        neutral="MidBlock",
        negative="LowBlock",
    )
    out["away_style_press_axis"] = _cluster_axis(
        out["away_style_cluster"],
        positive="Press",
        neutral="MidBlock",
        negative="LowBlock",
    )
    out["home_style_attack_axis"] = _cluster_axis(
        out["home_style_cluster"],
        positive="FrontFoot",
        neutral="Measured",
        negative="Reactive",
    )
    out["away_style_attack_axis"] = _cluster_axis(
        out["away_style_cluster"],
        positive="FrontFoot",
        neutral="Measured",
        negative="Reactive",
    )
    out["style_possession_delta"] = out["home_style_possession_axis"] - out["away_style_possession_axis"]
    out["style_press_delta"] = out["home_style_press_axis"] - out["away_style_press_axis"]
    out["style_attack_delta"] = out["home_style_attack_axis"] - out["away_style_attack_axis"]
    home_cluster = out["home_style_cluster"].astype("string")
    away_cluster = out["away_style_cluster"].astype("string")
    out["style_cluster_same"] = np.where(
        home_cluster.notna() & away_cluster.notna(),
        (home_cluster == away_cluster).astype(float),
        np.nan,
    )
    for confidence_col in ("home_style_cluster_confidence", "away_style_cluster_confidence"):
        if confidence_col in out.columns:
            out[confidence_col] = pd.to_numeric(out[confidence_col], errors="coerce")
    return out


def _add_style_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    if "home_style_cluster" not in df.columns or "away_style_cluster" not in df.columns:
        return df
    if all(feature_name in df.columns for feature_name in STYLE_MATCHUP_FEATURES):
        return df
    out = df.copy()
    home_cluster = out["home_style_cluster"].astype("string")
    away_cluster = out["away_style_cluster"].astype("string")
    matchup_known = home_cluster.isin(STYLE_CLUSTER_LABELS) & away_cluster.isin(STYLE_CLUSTER_LABELS)
    feature_frame: dict[str, pd.Series] = {}
    for home_label in STYLE_CLUSTER_LABELS:
        for away_label in STYLE_CLUSTER_LABELS:
            feature_name = _style_matchup_feature_name(home_label, away_label)
            if feature_name in out.columns:
                continue
            values = (
                (home_cluster == home_label) & (away_cluster == away_label) & matchup_known
            ).astype(float)
            feature_frame[feature_name] = values.where(matchup_known, np.nan)
    if not feature_frame:
        return out
    return pd.concat([out, pd.DataFrame(feature_frame, index=out.index)], axis=1)


def _add_style_matchup_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    present_sources = [
        column for column in STYLE_MATCHUP_INTERACTION_SOURCE_COLUMNS if column in df.columns
    ]
    if not present_sources:
        return df
    if all(feature_name in df.columns for feature_name in STYLE_MATCHUP_INTERACTION_FEATURES):
        return df
    if not any(feature in df.columns for feature in STYLE_MATCHUP_FEATURES):
        return df
    out = df.copy()
    feature_frame: dict[str, pd.Series] = {}
    for matchup_feature in STYLE_MATCHUP_FEATURES:
        if matchup_feature not in out.columns:
            continue
        matchup_source = out[matchup_feature]
        if isinstance(matchup_source, pd.DataFrame):
            matchup_source = matchup_source.iloc[:, 0]
        matchup_values = pd.to_numeric(matchup_source, errors="coerce")
        for source_column in present_sources:
            interaction_feature = f"{matchup_feature}__x__{source_column}"
            if interaction_feature in out.columns:
                continue
            base_values = pd.to_numeric(out[source_column], errors="coerce")
            feature_frame[interaction_feature] = matchup_values * base_values
    if not feature_frame:
        return out
    return pd.concat([out, pd.DataFrame(feature_frame, index=out.index)], axis=1)


def _add_axis_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    present_pairs = [
        (left, right)
        for left, right in AXIS_INTERACTION_SPECIFICATIONS
        if left in df.columns and right in df.columns
    ]
    if not present_pairs:
        return df
    if all(_interaction_feature_name(left, right) in df.columns for left, right in present_pairs):
        return df
    out = df.copy()
    feature_frame: dict[str, pd.Series] = {}
    for left, right in present_pairs:
        feature_name = _interaction_feature_name(left, right)
        if feature_name in out.columns:
            continue
        left_values = pd.to_numeric(out[left], errors="coerce")
        right_values = pd.to_numeric(out[right], errors="coerce")
        feature_frame[feature_name] = left_values * right_values
    if not feature_frame:
        return out
    return pd.concat([out, pd.DataFrame(feature_frame, index=out.index)], axis=1)


def add_corners_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _merge_style_cluster_features(df.copy())
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
    out = _add_style_cluster_axes(out)
    out = _add_style_matchup_features(out)
    out = _add_style_matchup_interaction_features(out)
    out = _add_axis_interaction_features(out)
    missing_league_features = {
        feat: pd.Series(np.nan, index=out.index, dtype=float)
        for feat in LEAGUE_REGIME_FEATURES
        if feat not in out.columns
    }
    if missing_league_features:
        out = pd.concat([out, pd.DataFrame(missing_league_features, index=out.index)], axis=1)
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
