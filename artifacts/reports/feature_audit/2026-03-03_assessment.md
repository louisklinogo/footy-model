# Feature System Assessment (Senior DS Report) - 2026-03-03

## Executive Verdict
Your model stack is not failing because of model class choice first; it is constrained by feature-market alignment and data coverage.

Current weak-market underperformance is primarily caused by:
1. sparse/incorrect line coverage for corner markets,
2. a global feature contract applied to all markets (instead of family-specific contracts),
3. missing high-signal 1X2/DC odds features despite high availability.

## Evidence from Audit
Source artifacts:
- `artifacts/reports/feature_audit/2026-03-03.json`
- `artifacts/reports/feature_audit/2026-03-03.md`
- `artifacts/reports/feature_audit/2026-03-03_ablation_fast.json`

Dataset used by trainer:
- FT fixtures: 8,082
- Train: 6,465
- Test: 1,617
- Current feature count: 108

### 1) Core weak-market behavior
From full weak-family retrain output:
- `1x2_d`: AUC 0.532, Brier 0.195 (not promoted)
- `dc_12`: AUC 0.532, Brier 0.195 (not promoted)
- `o15`: AUC 0.596 (not promoted under current baseline/gate)
- `u35`: AUC 0.567
- `c75`: AUC 0.568
- `c85`: AUC 0.550
- `c95`: AUC 0.555
- `c105`: AUC 0.545

### 2) Coverage mismatch is severe for corners lines
Historical FT coverage (SofaScore, pre-match/closing <= kickoff):
- `corners_ou 7.5`: 3 / 8,082 (0.0%)
- `corners_ou 8.5`: 881 / 8,082 (10.9%)
- `corners_ou 9.5`: 5,088 / 8,082 (63.0%)
- `corners_ou 10.5`: 1,615 / 8,082 (20.0%)

But your weak set currently includes `c75`, `c85`, `c95`, `c105`.
This means `c75`, `c85`, `c105` are trained with mostly missing line features, forcing poor proxies.

### 3) Missingness hotspots in current 108-feature contract
High missing rates in training frame:
- `odds_c75_over/under`, `implied_c75_over`, `corners_gap_75`: 100% missing
- `odds_c85_over/under`, `implied_c85_over`, `corners_gap_85`: 89.1% missing
- `odds_c105_over/under`, `implied_c105_over`, `corners_gap_105`: 80.0% missing
- `adj_lambda_home_final`, `adj_lambda_away_final`: 96.7% missing
- several phase-2 rolling split features: ~72.8% missing

This is currently diluted through imputation and fed into every market model.

### 4) Signal concentration: odds features dominate
Cross-market repeated top univariate features are almost all OU odds-derived features:
- `odds_under_25`, `implied_under25`, `odds_gap_25`, `odds_under_15`, `odds_gap_15`, `odds_under_35`, etc.

Interpretation: market prices carry most immediate predictive signal in your dataset.

### 5) You are currently not using high-coverage/high-signal 1X2 odds for 1X2/DC models
Coverage on FT fixtures:
- `1x2` odds available for 7,906 / 8,082 fixtures (97.8%)
- `dc` odds available for 7,620 / 8,082 fixtures (94.3%)

Univariate draw-odds signal for `1x2_d` / `dc_12`:
- `odds_draw` (or `implied_draw`) AUC magnitude ~0.572

These features are not currently in `feature_columns()` for market calibrator.

### 6) Injury/availability features exist but are weak alone
`player_availability` coverage is high (~97.4% fixture-level), but simple univariate AUC for injury-derived counts/impact is typically ~0.50-0.52.
These are likely interaction/context features, not standalone primary signals.

### 7) Ablation confirms dependency on odds signal
Fast ablation (`HistGradientBoosting`, same time split):
- Removing odds usually drops AUC by ~0.007 to ~0.022 across weak markets.
- For corners, odds-only often outperforms all-features model, indicating noisy non-odds features are hurting.
- For `o15/u35`, removing corner-line features sometimes improves AUC slightly, showing cross-family noise.

## What To Do (Priority Order)

## P0 (Immediate): Stop training markets with structurally missing data
1. Keep only corner lines with stable coverage in tradable set.
2. Based on current evidence, `c95` is the only corners line with acceptable historical coverage (~63%).
3. Move `c75`, `c85`, `c105` to `predict_only` until coverage improves (or provider mix changes).

## P1 (High impact): Split feature contracts by market family
1. Implement separate feature lists for:
- `1x2_dc`
- `goals_totals`
- `corners_totals`
2. Do not feed corner-line sparse features into goals/1x2 models.
3. Do not feed irrelevant formation/style features into every market by default.

Expected effect: less imputation noise, better generalization, more stable Brier.

## P2 (High impact): Add 1X2/DC odds-derived features (currently missing)
1. Add pre-match `1x2` features:
- `odds_home`, `odds_draw`, `odds_away`
- implied probs and overround-normalized probs
- `draw_vs_home_gap`, `draw_vs_away_gap`, entropy-like concentration
2. Add `dc` features similarly where available.

Expected effect: most realistic path to lift `1x2_d` and `dc_12` AUC.

## P3 (Medium-high impact): Tighten corners market design
1. Train only for lines with sufficient coverage threshold (e.g., >=60%).
2. If you need multiple corners products, dynamically map to nearest available line per fixture and store line distance as a feature.
3. Keep line-specific models only where line data exists at scale.

## P4 (Medium impact): Use missingness as signal, not just a problem
1. Add missingness indicators for high-impact odds features.
2. Separate "odds unavailable" vs "odds extreme" regimes explicitly.

## P5 (Medium impact): Integrate availability/injury context carefully
1. Add engineered context features (delta and interaction with team strength), not raw counts only.
2. Gate these features by data quality timestamps (pre-kickoff only).
3. Expect modest lift; do not treat as primary signal.

## P6 (Governance): Keep promotion logic strict and transparent
1. Keep hard gate: `AUC_delta >= 0.02` and `Brier_delta <= 0`.
2. Enforce complete baseline coverage (already added), no silent `None` deltas.

## Concrete Next Sprint (Recommended)
1. Implement family-specific feature contracts.
2. Add `1x2/dc` odds features to calibrator fetch + derived features.
3. Re-scope corners weak set to supported lines by coverage threshold.
4. Retrain weak family with complete baseline and evaluate.
5. Promote only markets passing gate; keep others `predict_only`.

## Reality Check
- Better features and better market selection can improve metrics materially.
- No method can guarantee "no losing stakes" in football betting.
- Your best practical edge is: only trade markets where both signal quality and odds availability are strong.
