import numpy as np
import pandas as pd

from src.features import build_team_style_clusters as clusters


def test_add_style_axes_builds_net_features() -> None:
    frame = pd.DataFrame(
        {
            "rolling_possession": [55.0, 48.0],
            "rolling_tackles_pct": [0.62, 0.47],
            "rolling_xg": [1.8, 1.1],
            "rolling_xg_against": [0.9, 1.4],
            "rolling_corners": [6.5, 4.0],
            "rolling_corners_against": [3.5, 5.0],
            "defenders": [4, 5],
            "midfielders": [3, 4],
            "forwards": [3, 1],
        }
    )

    out = clusters.add_style_axes(frame)

    np.testing.assert_allclose(out["xg_net"].to_numpy(), np.array([0.9, -0.3]))
    np.testing.assert_allclose(out["corners_net"].to_numpy(), np.array([3.0, -1.0]))


def test_build_feature_matrix_uses_hybrid_feature_recipe() -> None:
    frame = pd.DataFrame(
        {
            "rolling_possession": [55.0, 48.0, np.nan],
            "rolling_tackles_pct": [0.62, 0.47, 0.51],
            "rolling_xg": [1.8, 1.1, 1.2],
            "rolling_xg_against": [0.9, 1.4, 1.0],
            "rolling_corners": [6.5, 4.0, 5.0],
            "rolling_corners_against": [3.5, 5.0, 4.0],
            "defenders": [4, 5, 4],
            "midfielders": [3, 4, 4],
            "forwards": [3, 1, 2],
        }
    )

    X, valid, feature_names, _ = clusters.build_feature_matrix(frame)

    assert feature_names == [
        "rolling_possession",
        "rolling_tackles_pct",
        "xg_net",
        "corners_net",
        "defenders",
        "midfielders",
        "forwards",
    ]
    assert len(valid) == 2
    assert X.shape == (2, 7)


def test_label_clusters_uses_existing_feature_names_only() -> None:
    feature_names = [
        "rolling_possession",
        "rolling_tackles_pct",
        "xg_net",
        "corners_net",
        "defenders",
        "midfielders",
        "forwards",
    ]
    centroids = np.array(
        [
            [1.5, 1.2, 1.1, 0.8, 0.0, 0.0, 0.0],
            [-1.2, -1.1, -0.8, 0.6, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.0, -0.2, 0.0, 0.0, 0.0],
            [0.4, -0.6, 0.7, 0.3, 0.0, 0.0, 0.0],
        ]
    )

    labels = clusters.label_clusters(centroids, feature_names)

    assert set(labels.keys()) == {0, 1, 2, 3}
    assert all("rolling_xg_h2_delta" not in label for label in labels.values())
    assert any(label.startswith("Possession_") for label in labels.values())
    assert any(label.startswith("Direct_") for label in labels.values())


def test_build_fixture_style_frame_keeps_home_away_contract() -> None:
    output = pd.DataFrame(
        {
            "fixture_id": [1, 1, 2],
            "is_home": [True, False, True],
            "style_cluster": ["Possession_Press_FrontFoot", "Direct_LowBlock_Reactive", "Balanced_MidBlock_Measured"],
            "style_cluster_raw": [0, 1, 2],
            "style_cluster_confidence": [0.91, 0.82, 0.73],
        }
    )

    fixture = clusters.build_fixture_style_frame(output)

    assert set(fixture.columns) == {
        "fixture_id",
        "home_style_cluster",
        "home_style_cluster_raw",
        "home_style_cluster_confidence",
        "away_style_cluster",
        "away_style_cluster_raw",
        "away_style_cluster_confidence",
        "style_matchup",
    }
    row_one = fixture.loc[fixture["fixture_id"] == 1].iloc[0]
    assert row_one["style_matchup"] == "Possession_Press_FrontFoot_vs_Direct_LowBlock_Reactive"


def test_assess_min_cluster_share_uses_smallest_normalized_cluster() -> None:
    labels = np.array([0, 0, 0, 1, 1, 2])
    assert clusters.assess_min_cluster_share(labels) == 1 / 6