# Shadow comparison report

- Generated at: 2026-03-16T03:37:33.048881+00:00
- Decision: passed
- Recommended scope: required_markets_only
- Required markets passed: 22/22

## Artifact versions

- anytime: baseline `anytime_direct_monotone_v1_candidate_20260308` vs challenger `markov_head_v1`
- corners: baseline `distribution_head_v1` vs challenger `live_verification_corners_20260306`
- scoreline: baseline `poisson_head_v1` vs challenger `scoreline_external_context_v1_candidate_20260316`

## Market deltas

| Market | Family | Status | ΔAUC | ΔBrier | ΔLogLoss | ΔECE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1x2_h | scoreline | passed | 0.0789 | -0.0211 | -0.0465 | 0.0088 |
| 1x2_d | scoreline | passed | 0.0297 | -0.0011 | -0.0037 | 0.0117 |
| 1x2_a | scoreline | passed | 0.0902 | -0.0177 | -0.0439 | 0.0117 |
| dc_1x | scoreline | passed | 0.0902 | -0.0177 | -0.0439 | 0.0117 |
| dc_x2 | scoreline | passed | 0.0789 | -0.0211 | -0.0465 | 0.0088 |
| dc_12 | scoreline | passed | 0.0297 | -0.0011 | -0.0037 | 0.0117 |
| ah2_home_m05 | scoreline | passed | 0.0789 | -0.0211 | -0.0465 | 0.0088 |
| ah2_away_p05 | scoreline | passed | 0.0789 | -0.0211 | -0.0465 | 0.0088 |
| ah2_away_m05 | scoreline | passed | 0.0902 | -0.0177 | -0.0439 | 0.0117 |
| ah2_home_p05 | scoreline | passed | 0.0902 | -0.0177 | -0.0439 | 0.0117 |
| ah2_home_m15 | scoreline | passed | 0.0573 | -0.0057 | -0.0179 | 0.0022 |
| ah2_away_p15 | scoreline | passed | 0.0573 | -0.0057 | -0.0179 | 0.0022 |
| ah2_away_m15 | scoreline | passed | 0.0798 | -0.0074 | -0.0276 | 0.0121 |
| ah2_home_p15 | scoreline | passed | 0.0798 | -0.0074 | -0.0276 | 0.0121 |
| eh3_0_1_home | scoreline | passed | 0.0573 | -0.0057 | -0.0179 | 0.0022 |
| eh3_0_1_draw | scoreline | passed | 0.0532 | -0.0053 | -0.0146 | 0.0088 |
| eh3_0_1_away | scoreline | passed | 0.0789 | -0.0211 | -0.0465 | 0.0088 |
| eh3_1_0_home | scoreline | passed | 0.0902 | -0.0177 | -0.0439 | 0.0117 |
| eh3_1_0_draw | scoreline | passed | 0.0654 | -0.0020 | -0.0097 | 0.0131 |
| eh3_1_0_away | scoreline | passed | 0.0798 | -0.0074 | -0.0276 | 0.0121 |
| o15 | scoreline | passed | 0.0203 | -0.0010 | -0.0039 | 0.0058 |
| u35 | scoreline | passed | 0.0197 | -0.0080 | -0.0177 | -0.0028 |
| mg_0 | scoreline | passed | 0.0363 | -0.0007 | -0.0047 | -0.0029 |
| mg_1_2 | scoreline | passed | 0.0268 | -0.0013 | -0.0034 | -0.0066 |
| mg_1_3 | scoreline | passed | 0.0061 | -0.0045 | -0.0094 | -0.0004 |
| mg_1_4 | scoreline | failed | -0.0107 | -0.0025 | -0.0055 | -0.0030 |
| mg_1_5 | scoreline | failed | -0.0013 | -0.0001 | -0.0002 | 0.0031 |
| mg_1_6 | scoreline | failed | -0.0018 | -0.0001 | -0.0004 | 0.0040 |
| mg_2_3 | scoreline | failed | -0.0043 | 0.0010 | 0.0020 | 0.0150 |
| mg_2_4 | scoreline | passed | 0.0072 | -0.0005 | -0.0010 | 0.0094 |
| mg_2_5 | scoreline | failed | 0.0118 | 0.0002 | 0.0005 | 0.0073 |
| mg_2_6 | scoreline | failed | 0.0098 | 0.0001 | -0.0001 | 0.0060 |
| mg_3_4 | scoreline | passed | 0.0350 | -0.0032 | -0.0067 | -0.0031 |
| mg_3_5 | scoreline | passed | 0.0331 | -0.0037 | -0.0075 | -0.0029 |
| mg_3_6 | scoreline | passed | 0.0292 | -0.0040 | -0.0081 | -0.0033 |
| mg_4_5 | scoreline | passed | 0.0197 | -0.0078 | -0.0181 | -0.0081 |
| mg_4_6 | scoreline | passed | 0.0125 | -0.0074 | -0.0166 | -0.0096 |
| mg_5_6 | scoreline | failed | -0.0062 | -0.0032 | -0.0089 | -0.0112 |
| mg_7p | scoreline | failed | 0.0901 | 0.0001 | -0.0019 | 0.0011 |
| hmg_0 | scoreline | passed | 0.0693 | -0.0081 | -0.0229 | 0.0258 |
| hmg_1_2 | scoreline | passed | 0.0114 | -0.0021 | -0.0044 | 0.0196 |
| hmg_1_3 | scoreline | passed | 0.0498 | -0.0059 | -0.0127 | 0.0305 |
| hmg_2_3 | scoreline | passed | 0.0476 | -0.0036 | -0.0083 | 0.0211 |
| hmg_4p | scoreline | passed | 0.0552 | -0.0070 | -0.0272 | 0.0012 |
| amg_0 | scoreline | passed | 0.0616 | -0.0035 | -0.0103 | 0.0031 |
| amg_1_2 | scoreline | failed | 0.0282 | 0.0001 | 0.0002 | 0.0028 |
| amg_1_3 | scoreline | passed | 0.0456 | -0.0007 | -0.0018 | 0.0086 |
| amg_2_3 | scoreline | passed | 0.0695 | -0.0072 | -0.0178 | 0.0186 |
| amg_4p | scoreline | passed | 0.0840 | -0.0016 | -0.0118 | 0.0056 |
| ms_h_1_0_2_0_3_0 | scoreline | failed | 0.0682 | 0.0027 | 0.0013 | 0.0033 |
| ms_a_0_1_0_2_0_3 | scoreline | passed | 0.0917 | -0.0043 | -0.0177 | 0.0069 |
| ms_h_4_0_5_0_6_0 | scoreline | passed | 0.0730 | -0.0035 | -0.0169 | 0.0032 |
| ms_a_0_4_0_5_0_6 | scoreline | passed | 0.1131 | -0.0026 | -0.0156 | -0.0009 |
| ms_h_2_1_3_1_4_1 | scoreline | passed | 0.0620 | -0.0090 | -0.0254 | -0.0034 |
| ms_h_1_2_1_3_1_4 | scoreline | failed | 0.0660 | 0.0015 | -0.0015 | 0.0153 |
| ms_h_3_2_4_2_5_1 | scoreline | failed | -0.0264 | -0.0026 | -0.0090 | -0.0003 |
| ms_a_2_3_2_4_1_5 | scoreline | passed | 0.0918 | -0.0015 | -0.0081 | 0.0001 |
| ms_other_homewin | scoreline | failed | 0.0696 | 0.0016 | 0.0045 | -0.0006 |
| ms_other_awaywin | scoreline | passed | 0.0807 | -0.0015 | -0.0084 | 0.0019 |
| ms_draw | scoreline | passed | 0.0297 | -0.0011 | -0.0037 | 0.0117 |
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
