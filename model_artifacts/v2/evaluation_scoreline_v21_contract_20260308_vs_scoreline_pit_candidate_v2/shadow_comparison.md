# Shadow comparison report

- Generated at: 2026-03-08T14:57:55.094064+00:00
- Decision: failed
- Recommended scope: none
- Required markets passed: 2/22

## Artifact versions

- anytime: baseline `markov_head_v1` vs challenger `markov_head_v1`
- corners: baseline `distribution_head_v1` vs challenger `distribution_head_v1`
- scoreline: baseline `scoreline_pit_candidate_v2` vs challenger `scoreline_v21_contract_20260308`

## Market deltas

| Market | Family | Status | ΔAUC | ΔBrier | ΔLogLoss | ΔECE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1x2_h | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| 1x2_d | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1x2_a | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| dc_1x | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| dc_x2 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| dc_12 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ah2_home_m05 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| ah2_away_p05 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| ah2_away_m05 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| ah2_home_p05 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| ah2_home_m15 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ah2_away_p15 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ah2_away_m15 | scoreline | failed | 0.0000 | 0.0000 | -0.0000 | -0.0000 |
| ah2_home_p15 | scoreline | failed | 0.0000 | 0.0000 | -0.0000 | -0.0000 |
| eh3_0_1_home | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| eh3_0_1_draw | scoreline | passed | - | - | - | - |
| eh3_0_1_away | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| eh3_1_0_home | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| eh3_1_0_draw | scoreline | passed | - | - | - | - |
| eh3_1_0_away | scoreline | failed | 0.0000 | 0.0000 | -0.0000 | -0.0000 |
| o15 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| u35 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| mg_0 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mg_1_2 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| mg_1_3 | scoreline | failed | -0.0000 | -0.0000 | -0.0000 | -0.0000 |
| mg_1_4 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mg_1_5 | scoreline | passed | 0.0000 | -0.0000 | -0.0000 | 0.0000 |
| mg_1_6 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mg_2_3 | scoreline | failed | -0.0000 | -0.0000 | -0.0000 | -0.0000 |
| mg_2_4 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| mg_2_5 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| mg_2_6 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| mg_3_4 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| mg_3_5 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| mg_3_6 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| mg_4_5 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| mg_4_6 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| mg_5_6 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mg_7p | scoreline | passed | 0.0001 | -0.0000 | -0.0000 | -0.0000 |
| hmg_0 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| hmg_1_2 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | -0.0000 |
| hmg_1_3 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hmg_2_3 | scoreline | passed | 0.0000 | -0.0000 | -0.0000 | -0.0000 |
| hmg_4p | scoreline | failed | 0.0000 | -0.0000 | 0.0000 | 0.0000 |
| amg_0 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| amg_1_2 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| amg_1_3 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| amg_2_3 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| amg_4p | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ms_h_1_0_2_0_3_0 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ms_a_0_1_0_2_0_3 | scoreline | failed | -0.0000 | -0.0000 | -0.0000 | -0.0000 |
| ms_h_4_0_5_0_6_0 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| ms_a_0_4_0_5_0_6 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ms_h_2_1_3_1_4_1 | scoreline | passed | 0.0000 | -0.0000 | -0.0000 | -0.0000 |
| ms_h_1_2_1_3_1_4 | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ms_h_3_2_4_2_5_1 | scoreline | failed | -0.0001 | 0.0000 | 0.0000 | -0.0000 |
| ms_a_2_3_2_4_1_5 | scoreline | failed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ms_other_homewin | scoreline | passed | 0.0000 | -0.0000 | -0.0000 | 0.0000 |
| ms_other_awaywin | scoreline | failed | -0.0001 | -0.0000 | -0.0000 | -0.0000 |
| ms_draw | scoreline | failed | -0.0000 | 0.0000 | 0.0000 | 0.0000 |
| c75 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| c85 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| c95 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| c105 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hc25 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hc35 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hc45 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hc55 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ac25 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ac35 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ac45 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ac55 | corners | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| h_1up | anytime | failed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| a_1up | anytime | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| h_2up | anytime | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| a_2up | anytime | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
