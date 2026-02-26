# Layer 2 Rule-Layer Key-Absence Sweep

Generated: 2026-02-26T20:58:14.723808+00:00

## Gate Definition
- `triggered_home_events >= 30`
- `delta_triggered_home_vs_l2 > 0`
- `delta_global_home_vs_l2 >= 0`
- `delta_global_away_vs_l2 >= 0` (tolerance `-1e-6`)

## Top Variants
| Variant | Family | Home Key % | Triggered Home Delta vs L2 | Global Home Delta vs L2 | Global Away Delta vs L2 | Home Hit Delta vs L2 | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| keyabs_only_home_minus0.12 | keyabs_only | -12.00% | +2.32% | +0.08% | +0.00% | +24.00% | pass |
| keyabs_only_home_minus0.10 | keyabs_only | -10.00% | +2.09% | +0.08% | +0.00% | +24.00% | pass |
| keyabs_only_home_minus0.08 | keyabs_only | -8.00% | +1.79% | +0.06% | +0.00% | +24.00% | pass |
| keyabs_only_home_minus0.07 | keyabs_only | -7.00% | +1.62% | +0.06% | +0.00% | +22.00% | pass |
| home_only_v1_reference | reference | - | +1.25% | +0.06% | +0.00% | +13.85% | pass |
| keyabs_only_home_minus0.05 | keyabs_only | -5.00% | +1.23% | +0.04% | +0.00% | +16.00% | pass |
| keyabs_only_home_minus0.03 | keyabs_only | -3.00% | +0.79% | +0.03% | +0.00% | +8.00% | pass |

## Decision
- Best key-absence variant: `keyabs_only_home_minus0.12` (gate: `pass`).
- Triggered home delta vs Layer2-only: +2.32%.
- Global home delta vs Layer2-only: +0.08%.
- Global away delta vs Layer2-only: +0.00%.
- Promotion: not applied.