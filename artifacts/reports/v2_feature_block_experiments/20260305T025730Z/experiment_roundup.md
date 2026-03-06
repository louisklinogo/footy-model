# Feature Block Experiment Roundup

Date: 2026-03-05

## Experiment 2: scoreline_draw_variance_block

- Compare file: `compare_scoreline_draw_variance_block.md`
- Family summary:
  - `auc_delta_mean=+0.000161`
  - `brier_delta_mean=-0.000157`
- Focus segment summary (`1x2_d`, `dc_12`, `ms_draw`):
  - `focus_auc_delta_mean=-0.003522`
  - `focus_brier_delta_mean=+0.000584`
- Decision: reject for promotion (core weak segment got worse).

## Experiment 3: anytime_momentum_path_block

- Compare file: `compare_anytime_momentum_path_block.md`
- Family summary:
  - `auc_delta_mean=+0.000082`
  - `brier_delta_mean=-0.000009`
- Focus segment summary (`h_1up`, `a_1up`, `h_2up`, `a_2up`):
  - `focus_auc_delta_mean=+0.000082`
  - `focus_brier_delta_mean=-0.000009`
- Decision: hold (statistically tiny/flat; not meaningful enough to promote).

## Recommendation

1. Stop broad feature expansion for draw recovery; move to a draw-specific branch/calibration strategy.
2. Focus next cycle on segment-aware modeling (weak leagues + draw markets), not global feature bloat.
