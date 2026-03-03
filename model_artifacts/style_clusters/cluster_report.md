# Style Cluster Report

- Method: `gmm`
- k: `4`
- Silhouette: `0.2397` (gate: >= 0.15)
- Temporal ARI: `0.5810` (gate: >= 0.3)
- Gate: `PASSED`

## Cluster Sizes
| Cluster | Label | Count |
| --- | --- | --- |
| 0 | LowBlock | 1011 |
| 1 | Direct | 675 |
| 2 | Style_2 | 3637 |
| 3 | Style_3 | 2046 |

## Centroid Profiles
| Cluster | rolling_possession | rolling_tackles_pct | rolling_xg_p1 | rolling_xg_h2_delta | rolling_corners | defenders | midfielders | forwards |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LowBlock | -0.220 | -0.407 | 0.000 | 0.069 | -0.088 | -2.285 | 1.246 | -0.031 |
| Direct | 0.071 | -0.900 | 0.065 | 0.029 | -0.011 | 0.649 | -1.843 | 1.748 |
| Style_2 | 0.019 | 0.574 | -0.002 | -0.067 | 0.021 | 0.284 | -0.472 | 0.374 |
| Style_3 | 0.052 | -0.522 | -0.019 | 0.075 | 0.010 | 0.410 | 0.832 | -1.227 |