# Market Deployment Readiness (2026-02-27)

- Decision: **GO_WITH_MONITORING**

| Gate | Status | Evidence |
|---|---|---|
| sample_size_per_market_ge_3000 | pass | {'a_1up': 7030, 'a_2up': 7030, 'h_1up': 7030, 'h_2up': 7030} |
| quality_vs_markov_non_negative_brier_delta | pass | see champion-challenger deltas |
| fallback_rate_1up2up_le_1pct | pass | fallback_rows=20, anytime_total=28968, rate=0.0690% |
| scoring_skip_volume_reasonable | pass | skipped_total=272 |

## Remaining Settlement Gaps

- `c85` missing corners inputs: 272
- anytime missing incident lead-state rows: 0
