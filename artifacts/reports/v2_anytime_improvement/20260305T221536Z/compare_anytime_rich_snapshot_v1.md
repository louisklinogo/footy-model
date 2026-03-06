# Anytime Rich Snapshot V1 Compare

Baseline: `artifacts/v2/experiments/anytime_baseline_refresh_20260305T214459Z`
Candidate: `artifacts/v2/experiments/anytime_rich_snapshot_v1_20260305T221536Z`

## Aggregate
- Mean AUC: 0.626786 -> 0.631488 (+0.004703)
- Mean Brier: 0.201665 -> 0.200259 (-0.001406)
- Markets with AUC up: 3/4
- Markets with Brier down: 4/4

## Market Deltas
- `a_1up`: AUC 0.614256 -> 0.609454 (-0.004802); Brier 0.241061 -> 0.240317 (-0.000744)
- `a_2up`: AUC 0.630486 -> 0.637611 (+0.007125); Brier 0.150882 -> 0.150390 (-0.000492)
- `h_1up`: AUC 0.639174 -> 0.644044 (+0.004869); Brier 0.228710 -> 0.226176 (-0.002534)
- `h_2up`: AUC 0.623226 -> 0.634845 (+0.011619); Brier 0.186007 -> 0.184155 (-0.001853)

## Decision
- Promote candidate for anytime family refresh.
- Reason: stronger aggregate discrimination and calibration with `3/4` markets improving on AUC and all `4/4` improving on Brier.
- Caveat: `a_1up` AUC remains weaker than baseline and should stay on the next diagnostics queue.
