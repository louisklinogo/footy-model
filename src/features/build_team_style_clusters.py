"""
build_team_style_clusters.py
----------------------------
Cluster teams into style-of-play archetypes using rolling tactical stats
and formation data. Outputs a point-in-time cluster label per team per fixture.

Usage:
    python src/features/build_team_style_clusters.py [--k 5] [--method kmeans]
    python src/features/build_team_style_clusters.py --evaluate  # silhouette + ARI only

Outputs:
    model_artifacts/style_clusters/cluster_model.pkl
    model_artifacts/style_clusters/cluster_labels.parquet
    model_artifacts/style_clusters/cluster_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

CLUSTER_FEATURES = [
    "rolling_possession",
    "rolling_tackles_pct",
    "rolling_xg_p1",
    "rolling_xg_h2_delta",
    "rolling_corners",
]

ARCHETYPE_NAMES_K4 = {0: "Possession", 1: "CounterAttack", 2: "HighPress", 3: "LowBlock"}
ARCHETYPE_NAMES_K5 = {0: "Possession", 1: "CounterAttack", 2: "HighPress", 3: "LowBlock", 4: "Direct"}
ARCHETYPE_NAMES_K6 = {0: "Possession", 1: "CounterAttack", 2: "HighPress", 3: "LowBlock", 4: "Direct", 5: "Balanced"}

OUT_DIR = ROOT_DIR / "model_artifacts" / "style_clusters"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build team style-of-play clusters.")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters (default: 5).")
    parser.add_argument(
        "--method",
        choices=("kmeans", "gmm"),
        default="kmeans",
        help="Clustering algorithm (default: kmeans).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run silhouette + temporal ARI stability check only. Do not save model.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for cluster artifacts.",
    )
    parser.add_argument(
        "--min-silhouette",
        type=float,
        default=0.3,
        help="Minimum silhouette score required to save clusters (default: 0.3).",
    )
    parser.add_argument(
        "--min-ari",
        type=float,
        default=0.3,
        help="Minimum Adjusted Rand Index (temporal stability) required to save clusters (default: 0.3).",
    )
    return parser.parse_args()


def _parse_formation(form_str: object) -> tuple[int, int, int]:
    """Parse '4-3-3' style strings into (defenders, midfielders, forwards)."""
    if not isinstance(form_str, str) or "-" not in form_str:
        return (4, 4, 2)
    parts = [int(p) for p in form_str.split("-") if p.isdigit()]
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    if len(parts) == 4:
        return (parts[0], parts[1] + parts[2], parts[3])
    if len(parts) >= 5:
        return (parts[0], sum(parts[1:-1]), parts[-1])
    return (4, 4, 2)


def load_data() -> pd.DataFrame:
    """Load per-team per-fixture rolling style profile."""
    conn = connect_db()

    # Load snapshots (both home and away perspectives)
    print("Loading team_premium_snapshots...")
    snap = pd.read_sql(
        """
        SELECT
            tps.fixture_id,
            tps.team_id,
            tps.is_home,
            tps.rolling_possession,
            tps.rolling_tackles_pct,
            tps.rolling_xg_p1,
            tps.rolling_xg_h2_delta,
            tps.rolling_corners
        FROM team_premium_snapshots tps
        """,
        conn,
    )
    print(f"  {len(snap)} snapshot rows")

    # Load fixtures for datetime + league context
    print("Loading fixtures...")
    fixtures = pd.read_sql(
        """
        SELECT f.fixture_id, f.match_datetime_utc, f.league_code,
               f.home_team_id, f.away_team_id
        FROM fixtures f
        WHERE f.status = 'ft'
        """,
        conn,
    )
    fixtures["match_datetime_utc"] = pd.to_datetime(
        fixtures["match_datetime_utc"], utc=True, errors="coerce"
    )

    # Load formations
    print("Loading fixture_formations...")
    formations = pd.read_sql(
        "SELECT fixture_id, home_formation, away_formation FROM fixture_formations",
        conn,
    )
    conn.close()

    # Merge fixture context onto snapshots
    snap = snap.merge(
        fixtures[["fixture_id", "match_datetime_utc", "league_code",
                   "home_team_id", "away_team_id"]],
        on="fixture_id",
        how="left",
    )

    # Attach formation per team side
    snap = snap.merge(formations, on="fixture_id", how="left")
    snap["formation_str"] = np.where(
        snap["is_home"], snap["home_formation"], snap["away_formation"]
    )
    defenders, midfielders, forwards = zip(
        *snap["formation_str"].apply(_parse_formation)
    )
    snap["defenders"] = defenders
    snap["midfielders"] = midfielders
    snap["forwards"] = forwards

    # Add season key (year of second half of season)
    snap["season"] = snap["match_datetime_utc"].dt.year.where(
        snap["match_datetime_utc"].dt.month >= 7,
        snap["match_datetime_utc"].dt.year - 1,
    )

    return snap


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build standardised feature matrix for clustering.
    Returns (X_scaled, df_valid) where df_valid has no NaNs in cluster features.
    """
    cluster_cols = CLUSTER_FEATURES + ["defenders", "midfielders", "forwards"]
    available = [c for c in cluster_cols if c in df.columns]
    if not available:
        raise RuntimeError(f"None of {cluster_cols} found in dataframe columns: {list(df.columns)}")

    df_valid = df.dropna(subset=available).copy()
    print(f"  Rows with all cluster features: {len(df_valid)} / {len(df)} ({len(df_valid)/len(df):.1%})")

    # Z-score standardise within league-season to remove league-specific biases
    scaler = StandardScaler()
    X = df_valid[available].values.astype(float)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df_valid, available, scaler


def fit_clusters(X: np.ndarray, k: int, method: str) -> object:
    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
    else:
        model = GaussianMixture(n_components=k, random_state=42, n_init=5)
        model.fit(X)
    return model


def predict_labels(model: object, X: np.ndarray, method: str) -> np.ndarray:
    if method == "kmeans":
        return model.predict(X)
    else:
        return model.predict(X)


def assess_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels)) < 2:
        return 0.0
    sample_size = min(5000, len(X))
    idx = np.random.default_rng(42).choice(len(X), size=sample_size, replace=False)
    return float(silhouette_score(X[idx], labels[idx]))


def assess_temporal_ari(df: pd.DataFrame, k: int, method: str) -> float:
    """
    Fit clusters on seasons <= N-1, predict on season N, measure ARI.
    Returns mean ARI across available season splits.
    """
    seasons = sorted(df["season"].dropna().unique())
    if len(seasons) < 2:
        print("  Not enough seasons for temporal ARI. Skipping.")
        return 0.0

    ari_scores = []
    for i in range(1, len(seasons)):
        train_season = seasons[i - 1]
        test_season = seasons[i]
        train_df = df[df["season"] == train_season]
        test_df = df[df["season"] == test_season]

        if len(train_df) < 50 or len(test_df) < 50:
            continue

        available = CLUSTER_FEATURES + ["defenders", "midfielders", "forwards"]
        available = [c for c in available if c in df.columns]
        train_valid = train_df.dropna(subset=available)
        test_valid = test_df.dropna(subset=available)
        if len(train_valid) < 50 or len(test_valid) < 50:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_valid[available].values.astype(float))
        X_test = scaler.transform(test_valid[available].values.astype(float))

        model_train = fit_clusters(X_train, k, method)
        model_test = fit_clusters(X_test, k, method)

        labels_train_on_test = predict_labels(model_train, X_test, method)
        labels_test = predict_labels(model_test, X_test, method)
        ari = float(adjusted_rand_score(labels_test, labels_train_on_test))
        ari_scores.append(ari)
        print(f"    Seasons {train_season}→{test_season}: ARI={ari:.3f}")

    return float(np.mean(ari_scores)) if ari_scores else 0.0


def label_clusters(centroids: np.ndarray, feature_names: list[str]) -> dict[int, str]:
    """
    Auto-label clusters from centroids by ranking by possession and pressing features.
    Returns a dict of {cluster_id: label_string}.
    """
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    poss_idx = feat_idx.get("rolling_possession", None)
    press_idx = feat_idx.get("rolling_tackles_pct", None)
    h2_idx = feat_idx.get("rolling_xg_h2_delta", None)

    k = len(centroids)
    labels: dict[int, str] = {}
    assigned: set[str] = set()

    for i, centroid in enumerate(centroids):
        poss = centroid[poss_idx] if poss_idx is not None else 0
        press = centroid[press_idx] if press_idx is not None else 0
        h2 = centroid[h2_idx] if h2_idx is not None else 0

        if poss > 0.5 and "Possession" not in assigned:
            label = "Possession"
        elif press > 0.5 and h2 > 0 and "HighPress" not in assigned:
            label = "HighPress"
        elif press < -0.3 and "LowBlock" not in assigned:
            label = "LowBlock"
        elif h2 < -0.3 and "CounterAttack" not in assigned:
            label = "CounterAttack"
        elif "Direct" not in assigned:
            label = "Direct"
        else:
            label = f"Style_{i}"
        assigned.add(label)
        labels[i] = label

    return labels


def main() -> None:
    args = parse_args()

    print("Loading data...")
    df = load_data()

    print("Building feature matrix...")
    X_scaled, df_valid, feature_names, scaler = build_feature_matrix(df)

    print(f"\nFitting {args.method} clusters (k={args.k})...")
    model = fit_clusters(X_scaled, args.k, args.method)
    labels = predict_labels(model, X_scaled, args.method)
    df_valid = df_valid.copy()
    df_valid["style_cluster_raw"] = labels

    print("\nEvaluating silhouette score...")
    sil = assess_silhouette(X_scaled, labels)
    print(f"  Silhouette: {sil:.4f}  (gate: >= {args.min_silhouette})")

    print("\nEvaluating temporal ARI stability...")
    ari = assess_temporal_ari(df, args.k, args.method)
    print(f"  Mean ARI:   {ari:.4f}  (gate: >= {args.min_ari})")

    gate_passed = sil >= args.min_silhouette and ari >= args.min_ari
    print(f"\n{'✅' if gate_passed else '❌'} Gate: {'PASSED' if gate_passed else 'FAILED'}")

    if args.evaluate:
        print("\n--evaluate mode: not saving model.")
        return

    if not gate_passed:
        print(f"\nClusters did not pass quality gates. Adjust k or features. Exiting without saving.")
        return

    # Auto-label clusters
    if args.method == "kmeans":
        centroids = model.cluster_centers_
    else:
        centroids = model.means_

    cluster_labels = label_clusters(centroids, feature_names)
    df_valid["style_cluster"] = df_valid["style_cluster_raw"].map(cluster_labels)

    # Build the output: one row per (fixture_id, team_id, side)
    output = df_valid[
        ["fixture_id", "team_id", "is_home", "match_datetime_utc",
         "league_code", "season", "style_cluster_raw", "style_cluster"]
    ].copy()

    # Also produce a fixture-level frame with home/away labels for easy join
    home_style = (
        output[output["is_home"]]
        .rename(columns={"style_cluster": "home_style_cluster", "style_cluster_raw": "home_style_cluster_raw"})
        [["fixture_id", "home_style_cluster", "home_style_cluster_raw"]]
    )
    away_style = (
        output[~output["is_home"]]
        .rename(columns={"style_cluster": "away_style_cluster", "style_cluster_raw": "away_style_cluster_raw"})
        [["fixture_id", "away_style_cluster", "away_style_cluster_raw"]]
    )
    fixture_style = home_style.merge(away_style, on="fixture_id", how="outer")
    fixture_style["style_matchup"] = (
        fixture_style["home_style_cluster"].fillna("Unknown")
        + "_vs_"
        + fixture_style["away_style_cluster"].fillna("Unknown")
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save model artifacts
    model_blob = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "cluster_labels": cluster_labels,
        "k": args.k,
        "method": args.method,
        "silhouette": sil,
        "temporal_ari": ari,
    }
    model_path = args.out_dir / "cluster_model.pkl"
    joblib.dump(model_blob, model_path)
    print(f"Saved model: {model_path}")

    labels_path = args.out_dir / "cluster_labels.parquet"
    fixture_style.to_parquet(labels_path, index=False)
    print(f"Saved labels: {labels_path}")

    # Save report
    cluster_sizes = df_valid["style_cluster"].value_counts().to_dict()
    report_lines = [
        "# Style Cluster Report",
        "",
        f"- Method: `{args.method}`",
        f"- k: `{args.k}`",
        f"- Silhouette: `{sil:.4f}` (gate: >= {args.min_silhouette})",
        f"- Temporal ARI: `{ari:.4f}` (gate: >= {args.min_ari})",
        f"- Gate: `{'PASSED' if gate_passed else 'FAILED'}`",
        "",
        "## Cluster Sizes",
        "| Cluster | Label | Count |",
        "| --- | --- | --- |",
    ]
    for raw_id, label in sorted(cluster_labels.items()):
        n = int(cluster_sizes.get(label, 0))
        report_lines.append(f"| {raw_id} | {label} | {n} |")

    report_lines += [
        "",
        "## Centroid Profiles",
        f"| Cluster | {' | '.join(feature_names)} |",
        "| --- | " + " | ".join(["---"] * len(feature_names)) + " |",
    ]
    for raw_id, centroid in enumerate(centroids):
        label = cluster_labels.get(raw_id, str(raw_id))
        vals = " | ".join(f"{v:.3f}" for v in centroid)
        report_lines.append(f"| {label} | {vals} |")

    report_path = args.out_dir / "cluster_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved report: {report_path}")

    meta = {
        "k": args.k,
        "method": args.method,
        "features": feature_names,
        "silhouette": sil,
        "temporal_ari": ari,
        "gate_passed": gate_passed,
        "cluster_labels": cluster_labels,
    }
    meta_path = args.out_dir / "cluster_model.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
