# CLV Proxy Report
- Generated at: 2026-03-03T19:31:37.648486+00:00
- Lookback: 90 days
- Samples evaluated: 43336
- CLV available: 22380
- CLV unavailable: 20956
- Missing opening prices: 0
- Missing latest prices: 0

## Mean CLV by Action
| Action | Count | Mean CLV |
| :--- | :--- | :--- |
| bet | 22 | -0.0066 |
| bet_small | 17 | +0.0030 |
| pass | 1563 | -0.0007 |
| unknown | 20772 | -0.0006 |
| watch | 6 | +0.0075 |

## Mean CLV by Market
| Market | Count | Mean CLV |
| :--- | :--- | :--- |
| 1x2_a | 3171 | -0.0023 |
| 1x2_d | 3171 | -0.0040 |
| 1x2_h | 3171 | -0.0043 |
| c85 | 370 | +0.0141 |
| dc_12 | 3125 | +0.0037 |
| dc_1x | 3125 | +0.0016 |
| dc_x2 | 3125 | +0.0015 |
| o15 | 3122 | -0.0022 |

## CLV Unavailable Breakdown
| Reason | Count |
| :--- | :--- |
| market_unmapped | 11944 |
| selection_unavailable | 9012 |
| opening_missing | 0 |
| latest_missing | 0 |
| both_missing | 0 |

**Overall Mean CLV: -0.0006**

Note: CLV proxy = implied_prob_latest - implied_prob_opening using the same selected side/line from one leakage-safe pre-match snapshot row.