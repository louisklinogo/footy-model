# Anytime Reality Check (2026-03-05)

Purpose: verify whether simple alternatives beat current anytime v2 before deeper re-architecture.

## Data/Quality Snapshot
1. Last 365d settled fixtures checked: `7364`
2. Incident lead-state label availability:
   1. `h_1up/a_1up/h_2up/a_2up`: `7080` each (`~96%` coverage in this window query)
3. Training dataset (`fetch_dataset + add_targets + anytime features`) had:
   1. rows: `8167`
   2. label non-null share for anytime targets: `~94.7%`

## Benchmarks Run

### 1) Direct classifier benchmark (time split) vs current anytime holdout baseline
- Result: underperformed baseline on all four anytime markets.
- Mean delta:
  1. AUC: negative
  2. Brier: worse (higher)

### 2) Simple recalibration (`isotonic`, `logit+league`) on scored predictions (`h_1up/a_1up`)
- Result: underperformed baseline.
- `h_1up`:
  1. baseline AUC `0.644342`, isotonic `0.643622`, league-logit `0.641763`
  2. baseline Brier `0.225371`, isotonic `0.240403`, league-logit `0.241855`
- `a_1up`:
  1. baseline AUC `0.605112`, isotonic `0.603555`, league-logit `0.599316`
  2. baseline Brier `0.242776`, isotonic `0.265707`, league-logit `0.265414`

## Conclusion
1. We are not blocked by labels alone.
2. We are blocked by model form for path events.
3. Best next step is structural: phase-aware anytime modeling (`Path V3`), not more generic feature toggles.
