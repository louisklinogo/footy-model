# Style Cluster Report

- Method: `gmm`
- k: `4`
- Feature version: `hybrid_net_v2`
- Fit statuses: `['ft']`
- Fit rows (valid): `14417`
- Scored rows (valid): `19269`
- Silhouette: `0.2070` (gate: >= 0.15)
- Temporal ARI: `0.5010` (gate: >= 0.3)
- Min cluster share: `0.0557` (gate: >= 0.05)
- Gate: `PASSED`

## Cluster Sizes
| Cluster | Label | Count |
| --- | --- | --- |
| 0 | Possession_LowBlock_FrontFoot | 5770 |
| 1 | Balanced_Press_Measured | 4621 |
| 2 | Direct_MidBlock_Reactive | 803 |
| 3 | Balanced_MidBlock_Measured | 3223 |

## Centroid Profiles
| Cluster | rolling_possession | rolling_tackles_pct | xg_net | corners_net | defenders | midfielders | forwards |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Possession_LowBlock_FrontFoot | 0.156 | -0.106 | 0.075 | 0.073 | 0.479 | 0.402 | -0.816 |
| Balanced_Press_Measured | -0.179 | 0.130 | -0.059 | -0.081 | -1.329 | 0.657 | 0.201 |
| Direct_MidBlock_Reactive | -0.642 | 0.106 | -0.432 | -0.392 | 2.286 | -1.411 | -0.019 |
| Balanced_MidBlock_Measured | 0.138 | -0.024 | 0.057 | 0.083 | 0.479 | -1.309 | 1.177 |