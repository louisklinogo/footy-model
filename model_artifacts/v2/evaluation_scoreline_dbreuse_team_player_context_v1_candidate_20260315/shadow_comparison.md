# Shadow comparison report

- Generated at: 2026-03-15T11:13:37.290342+00:00
- Decision: failed
- Recommended scope: none
- Required markets passed: 2/22

## Artifact versions

- anytime: baseline `anytime_direct_monotone_v1_candidate_20260308` vs challenger `markov_head_v1`
- corners: baseline `distribution_head_v1` vs challenger `live_verification_corners_20260306`
- scoreline: baseline `poisson_head_v1` vs challenger `scoreline_dbreuse_team_player_context_v1_candidate_20260315`

## Market deltas

| Market | Family | Status | ΔAUC | ΔBrier | ΔLogLoss | ΔECE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1x2_h | scoreline | failed | -0.0157 | 0.0016 | 0.0032 | -0.0124 |
| 1x2_d | scoreline | failed | 0.0254 | 0.0022 | 0.0043 | 0.0062 |
| 1x2_a | scoreline | failed | -0.0187 | 0.0028 | 0.0069 | -0.0041 |
| dc_1x | scoreline | failed | -0.0187 | 0.0028 | 0.0069 | -0.0041 |
| dc_x2 | scoreline | failed | -0.0157 | 0.0016 | 0.0032 | -0.0124 |
| dc_12 | scoreline | failed | 0.0254 | 0.0022 | 0.0043 | 0.0062 |
| ah2_home_m05 | scoreline | failed | -0.0157 | 0.0016 | 0.0032 | -0.0124 |
| ah2_away_p05 | scoreline | failed | -0.0157 | 0.0016 | 0.0032 | -0.0124 |
| ah2_away_m05 | scoreline | failed | -0.0187 | 0.0028 | 0.0069 | -0.0041 |
| ah2_home_p05 | scoreline | failed | -0.0187 | 0.0028 | 0.0069 | -0.0041 |
| ah2_home_m15 | scoreline | failed | -0.0171 | 0.0002 | 0.0014 | -0.0062 |
| ah2_away_p15 | scoreline | failed | -0.0171 | 0.0002 | 0.0014 | -0.0062 |
| ah2_away_m15 | scoreline | failed | -0.0141 | -0.0009 | -0.0008 | 0.0071 |
| ah2_home_p15 | scoreline | failed | -0.0141 | -0.0009 | -0.0008 | 0.0071 |
| eh3_0_1_home | scoreline | failed | -0.0171 | 0.0002 | 0.0014 | -0.0062 |
| eh3_0_1_draw | scoreline | failed | -0.0040 | -0.0031 | -0.0068 | -0.0005 |
| eh3_0_1_away | scoreline | failed | -0.0157 | 0.0016 | 0.0032 | -0.0124 |
| eh3_1_0_home | scoreline | failed | -0.0187 | 0.0028 | 0.0069 | -0.0041 |
| eh3_1_0_draw | scoreline | failed | -0.0232 | 0.0026 | 0.0071 | -0.0108 |
| eh3_1_0_away | scoreline | failed | -0.0141 | -0.0009 | -0.0008 | 0.0071 |
| o15 | scoreline | passed | 0.0097 | -0.0007 | -0.0026 | 0.0091 |
| u35 | scoreline | passed | 0.0020 | -0.0065 | -0.0140 | -0.0014 |
| mg_0 | scoreline | passed | 0.0181 | -0.0008 | -0.0037 | -0.0031 |
| mg_1_2 | scoreline | passed | 0.0152 | -0.0000 | -0.0006 | -0.0057 |
| mg_1_3 | scoreline | failed | -0.0043 | -0.0042 | -0.0087 | -0.0032 |
| mg_1_4 | scoreline | failed | -0.0094 | -0.0019 | -0.0041 | -0.0015 |
| mg_1_5 | scoreline | failed | 0.0130 | 0.0009 | 0.0025 | 0.0006 |
| mg_1_6 | scoreline | failed | -0.0150 | -0.0006 | -0.0013 | 0.0062 |
| mg_2_3 | scoreline | failed | -0.0132 | 0.0012 | 0.0023 | 0.0157 |
| mg_2_4 | scoreline | passed | 0.0036 | -0.0004 | -0.0009 | 0.0090 |
| mg_2_5 | scoreline | failed | -0.0004 | 0.0009 | 0.0019 | 0.0073 |
| mg_2_6 | scoreline | passed | 0.0007 | -0.0000 | -0.0001 | 0.0087 |
| mg_3_4 | scoreline | passed | 0.0222 | -0.0036 | -0.0075 | -0.0060 |
| mg_3_5 | scoreline | passed | 0.0148 | -0.0028 | -0.0055 | -0.0059 |
| mg_3_6 | scoreline | passed | 0.0140 | -0.0024 | -0.0049 | -0.0034 |
| mg_4_5 | scoreline | failed | -0.0031 | -0.0080 | -0.0179 | -0.0069 |
| mg_4_6 | scoreline | failed | -0.0049 | -0.0064 | -0.0136 | -0.0057 |
| mg_5_6 | scoreline | failed | -0.0135 | -0.0016 | -0.0041 | -0.0077 |
| mg_7p | scoreline | passed | 0.0809 | -0.0001 | -0.0023 | 0.0007 |
| hmg_0 | scoreline | failed | -0.0258 | 0.0000 | 0.0003 | 0.0086 |
| hmg_1_2 | scoreline | failed | -0.0071 | -0.0024 | -0.0050 | 0.0130 |
| hmg_1_3 | scoreline | failed | -0.0244 | -0.0024 | -0.0049 | 0.0147 |
| hmg_2_3 | scoreline | failed | -0.0101 | 0.0007 | 0.0015 | 0.0109 |
| hmg_4p | scoreline | passed | 0.0106 | -0.0054 | -0.0177 | 0.0002 |
| amg_0 | scoreline | failed | 0.0008 | 0.0019 | 0.0037 | 0.0017 |
| amg_1_2 | scoreline | failed | 0.0034 | 0.0013 | 0.0025 | -0.0084 |
| amg_1_3 | scoreline | failed | -0.0010 | 0.0016 | 0.0034 | -0.0074 |
| amg_2_3 | scoreline | passed | 0.0075 | -0.0004 | -0.0005 | 0.0003 |
| amg_4p | scoreline | passed | 0.0153 | -0.0006 | -0.0026 | 0.0004 |
| ms_h_1_0_2_0_3_0 | scoreline | failed | -0.0143 | 0.0065 | 0.0150 | -0.0112 |
| ms_a_0_1_0_2_0_3 | scoreline | failed | -0.0308 | 0.0007 | 0.0026 | 0.0000 |
| ms_h_4_0_5_0_6_0 | scoreline | passed | 0.0446 | -0.0049 | -0.0195 | 0.0000 |
| ms_a_0_4_0_5_0_6 | scoreline | passed | 0.0380 | -0.0017 | -0.0083 | -0.0018 |
| ms_h_2_1_3_1_4_1 | scoreline | passed | 0.0045 | -0.0084 | -0.0210 | -0.0078 |
| ms_h_1_2_1_3_1_4 | scoreline | failed | 0.0077 | 0.0034 | 0.0091 | 0.0016 |
| ms_h_3_2_4_2_5_1 | scoreline | failed | -0.0550 | -0.0010 | -0.0025 | 0.0001 |
| ms_a_2_3_2_4_1_5 | scoreline | failed | 0.0055 | 0.0001 | 0.0006 | 0.0006 |
| ms_other_homewin | scoreline | failed | 0.0128 | 0.0014 | 0.0053 | -0.0002 |
| ms_other_awaywin | scoreline | passed | 0.0292 | -0.0014 | -0.0075 | 0.0014 |
| ms_draw | scoreline | failed | 0.0254 | 0.0022 | 0.0043 | 0.0062 |
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
| h_1up | anytime | failed | -0.0279 | 0.0020 | 0.0044 | -0.0189 |
| a_1up | anytime | failed | -0.0131 | 0.0007 | 0.0015 | 0.0046 |
| h_2up | anytime | failed | -0.0125 | 0.0021 | 0.0047 | -0.0021 |
| a_2up | anytime | failed | -0.0053 | -0.0001 | 0.0002 | 0.0033 |
