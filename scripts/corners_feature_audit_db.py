from __future__ import annotations

from src.db.db_utils import connect_db
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

RAW_METRICS = [
    "xg", "xgot", "xa", "big_chances", "crosses", "sot", "box_touches", "possession",
    "goals_prevented", "tackles_pct", "errors_lead_to_shot", "final_third_entries",
    "blocked_shots", "shots_inside_box", "through_passes", "ball_recoveries", "duels_won",
    "ground_duels_won", "aerial_duels_won", "accurate_long_balls", "total_long_balls",
    "yellow_cards", "fouls", "dispossessed", "hit_woodwork", "dribbles_success", "dribbles_total",
]
SNAP_METRICS = [
    "sample_size", "rolling_xg", "rolling_xgot", "rolling_xa", "rolling_big_chances",
    "rolling_crosses", "rolling_sot", "rolling_box_touches", "rolling_possession",
    "rolling_goals_prevented", "rolling_tackles_pct", "rolling_errors_lead_to_shot",
    "rolling_rest_days", "rolling_xg_p1", "rolling_sot_p1", "rolling_xg_h2_delta",
    "rolling_sot_h2_delta", "rolling_lead_rate_1up", "rolling_lead_rate_2up", "fidelity_score",
]

RAW_SQL = """
SELECT fixture_id, h_corners, a_corners, (h_corners + a_corners) AS total_corners,
       h_xg, a_xg, h_xgot, a_xgot, h_xa, a_xa, h_big_chances, a_big_chances,
       h_crosses, a_crosses, h_sot, a_sot, h_box_touches, a_box_touches,
       h_possession, a_possession, h_goals_prevented, a_goals_prevented,
       h_tackles_pct, a_tackles_pct, h_errors_lead_to_shot, a_errors_lead_to_shot,
       h_final_third_entries, a_final_third_entries, h_blocked_shots, a_blocked_shots,
       h_shots_inside_box, a_shots_inside_box, h_through_passes, a_through_passes,
       h_ball_recoveries, a_ball_recoveries, h_duels_won, a_duels_won,
       h_ground_duels_won, a_ground_duels_won, h_aerial_duels_won, a_aerial_duels_won,
       h_accurate_long_balls, a_accurate_long_balls, h_total_long_balls, a_total_long_balls,
       h_yellow_cards, a_yellow_cards, h_fouls, a_fouls, h_dispossessed, a_dispossessed,
       h_hit_woodwork, a_hit_woodwork, h_dribbles_success, a_dribbles_success,
       h_dribbles_total, a_dribbles_total, fidelity_score
FROM fixture_stats_premium
WHERE h_corners IS NOT NULL AND a_corners IS NOT NULL
"""

SNAP_SQL = """
SELECT f.fixture_id, f.league_code,
       home.sample_size AS home_sample_size, away.sample_size AS away_sample_size,
       home.rolling_xg AS home_rolling_xg, away.rolling_xg AS away_rolling_xg,
       home.rolling_xgot AS home_rolling_xgot, away.rolling_xgot AS away_rolling_xgot,
       home.rolling_xa AS home_rolling_xa, away.rolling_xa AS away_rolling_xa,
       home.rolling_big_chances AS home_rolling_big_chances, away.rolling_big_chances AS away_rolling_big_chances,
       home.rolling_crosses AS home_rolling_crosses, away.rolling_crosses AS away_rolling_crosses,
       home.rolling_sot AS home_rolling_sot, away.rolling_sot AS away_rolling_sot,
       home.rolling_box_touches AS home_rolling_box_touches, away.rolling_box_touches AS away_rolling_box_touches,
       home.rolling_possession AS home_rolling_possession, away.rolling_possession AS away_rolling_possession,
       home.rolling_goals_prevented AS home_rolling_goals_prevented, away.rolling_goals_prevented AS away_rolling_goals_prevented,
       home.rolling_tackles_pct AS home_rolling_tackles_pct, away.rolling_tackles_pct AS away_rolling_tackles_pct,
       home.rolling_errors_lead_to_shot AS home_rolling_errors_lead_to_shot, away.rolling_errors_lead_to_shot AS away_rolling_errors_lead_to_shot,
       home.rolling_rest_days AS home_rolling_rest_days, away.rolling_rest_days AS away_rolling_rest_days,
       home.rolling_xg_p1 AS home_rolling_xg_p1, away.rolling_xg_p1 AS away_rolling_xg_p1,
       home.rolling_sot_p1 AS home_rolling_sot_p1, away.rolling_sot_p1 AS away_rolling_sot_p1,
       home.rolling_xg_h2_delta AS home_rolling_xg_h2_delta, away.rolling_xg_h2_delta AS away_rolling_xg_h2_delta,
       home.rolling_sot_h2_delta AS home_rolling_sot_h2_delta, away.rolling_sot_h2_delta AS away_rolling_sot_h2_delta,
       home.rolling_lead_rate_1up AS home_rolling_lead_rate_1up, away.rolling_lead_rate_1up AS away_rolling_lead_rate_1up,
       home.rolling_lead_rate_2up AS home_rolling_lead_rate_2up, away.rolling_lead_rate_2up AS away_rolling_lead_rate_2up,
       home.fidelity_score AS home_fidelity_score, away.fidelity_score AS away_fidelity_score,
       p.h_corners AS home_corners, p.a_corners AS away_corners, (p.h_corners + p.a_corners) AS total_corners
FROM fixtures f
JOIN team_premium_snapshots home ON home.fixture_id = f.fixture_id AND home.is_home = TRUE
JOIN team_premium_snapshots away ON away.fixture_id = f.fixture_id AND away.is_home = FALSE
JOIN fixture_stats_premium p ON p.fixture_id = f.fixture_id
WHERE p.h_corners IS NOT NULL AND p.a_corners IS NOT NULL
"""


def corr_rank(df: pd.DataFrame, metrics: list[str], prefix: str, home_y: str, away_y: str, total_y: str) -> pd.DataFrame:
    rows = []
    for m in metrics:
        hcol, acol = (f"h_{m}", f"a_{m}") if prefix == "raw" else (f"home_{m}", f"away_{m}")
        if hcol not in df.columns or acol not in df.columns:
            continue
        features = {
            f"{m}_sum": pd.to_numeric(df[hcol], errors="coerce") + pd.to_numeric(df[acol], errors="coerce"),
            f"{m}_diff": pd.to_numeric(df[hcol], errors="coerce") - pd.to_numeric(df[acol], errors="coerce"),
            f"home_{m}": pd.to_numeric(df[hcol], errors="coerce"),
            f"away_{m}": pd.to_numeric(df[acol], errors="coerce"),
        }
        for name, s in features.items():
            mask = s.notna() & df[total_y].notna()
            if int(mask.sum()) < 500:
                continue
            rows.append({
                "feature": name, "n": int(mask.sum()),
                "corr_total": float(s[mask].corr(df.loc[mask, total_y])),
                "corr_home": float(s[mask].corr(df.loc[mask, home_y])),
                "corr_away": float(s[mask].corr(df.loc[mask, away_y])),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_corr_total"] = out["corr_total"].abs()
    return out.sort_values(["abs_corr_total", "n"], ascending=[False, False])


def main() -> None:
    conn = connect_db()
    try:
        raw = pd.read_sql_query(RAW_SQL, conn)
        print("RAW_ROWS", len(raw))
        raw_rank = corr_rank(raw, RAW_METRICS, "raw", "h_corners", "a_corners", "total_corners")
        print("\nRAW_POSTMATCH_TOP_TOTAL")
        print(raw_rank[["feature", "n", "corr_total", "corr_home", "corr_away"]].head(20).to_string(index=False))

        snap = pd.read_sql_query(SNAP_SQL, conn)
        print("\nSNAP_ROWS", len(snap))
        snap_rank = corr_rank(snap, SNAP_METRICS, "home", "home_corners", "away_corners", "total_corners")
        print("\nPREMATCH_SNAPSHOT_TOP_TOTAL")
        print(snap_rank[["feature", "n", "corr_total", "corr_home", "corr_away"]].head(25).to_string(index=False))

        feat_cols = [c for c in snap.columns if c.startswith(("home_", "away_")) and c not in {"home_corners", "away_corners"}]
        X = snap[feat_cols].apply(pd.to_numeric, errors="coerce")
        valid = [c for c in X.columns if X[c].notna().sum() >= 500 and X[c].nunique(dropna=True) > 5]
        X = X[valid].fillna(X[valid].median(numeric_only=True))
        mi_total = mutual_info_regression(X, snap["total_corners"].astype(float), random_state=42)
        mi_c95 = mutual_info_classif(X, (snap["total_corners"].astype(float) > 9.5).astype(int), random_state=42)
        mi = pd.DataFrame({"feature": valid, "mi_total": mi_total, "mi_c95": mi_c95})
        print("\nPREMATCH_SNAPSHOT_TOP_MI_TOTAL")
        print(mi.sort_values("mi_total", ascending=False).head(20).to_string(index=False))
        print("\nPREMATCH_SNAPSHOT_TOP_MI_C95")
        print(mi.sort_values("mi_c95", ascending=False).head(20).to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()

