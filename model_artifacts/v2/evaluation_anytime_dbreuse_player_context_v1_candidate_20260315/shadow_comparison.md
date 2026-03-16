# Shadow comparison report

- Generated at: 2026-03-15T09:48:52.687174+00:00
- Decision: failed
- Recommended scope: none
- Required markets passed: 2/22

## Artifact versions

- anytime: baseline `anytime_direct_monotone_v1_candidate_20260308` vs challenger `anytime_dbreuse_player_context_v1_candidate_20260315`
- corners: baseline `distribution_head_v1` vs challenger `live_verification_corners_20260306`
- scoreline: baseline `poisson_head_v1` vs challenger `poisson_head_v1`

## Market deltas

| Market | Family | Status | ΔAUC | ΔBrier | ΔLogLoss | ΔECE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1x2_h | scoreline | failed | -0.0072 | 0.0003 | 0.0005 | -0.0016 |
| 1x2_d | scoreline | failed | 0.0303 | 0.0010 | 0.0015 | 0.0092 |
| 1x2_a | scoreline | failed | -0.0061 | -0.0003 | -0.0004 | 0.0041 |
| dc_1x | scoreline | failed | -0.0061 | -0.0003 | -0.0004 | 0.0041 |
| dc_x2 | scoreline | failed | -0.0072 | 0.0003 | 0.0005 | -0.0016 |
| dc_12 | scoreline | failed | 0.0303 | 0.0010 | 0.0015 | 0.0092 |
| ah2_home_m05 | scoreline | failed | -0.0072 | 0.0003 | 0.0005 | -0.0016 |
| ah2_away_p05 | scoreline | failed | -0.0072 | 0.0003 | 0.0005 | -0.0016 |
| ah2_away_m05 | scoreline | failed | -0.0061 | -0.0003 | -0.0004 | 0.0041 |
| ah2_home_p05 | scoreline | failed | -0.0061 | -0.0003 | -0.0004 | 0.0041 |
| ah2_home_m15 | scoreline | failed | -0.0055 | -0.0022 | -0.0044 | -0.0046 |
| ah2_away_p15 | scoreline | failed | -0.0055 | -0.0022 | -0.0044 | -0.0046 |
| ah2_away_m15 | scoreline | failed | -0.0035 | -0.0015 | -0.0030 | 0.0078 |
| ah2_home_p15 | scoreline | failed | -0.0035 | -0.0015 | -0.0030 | 0.0078 |
| eh3_0_1_home | scoreline | failed | -0.0055 | -0.0022 | -0.0044 | -0.0046 |
| eh3_0_1_draw | scoreline | failed | -0.0057 | 0.0016 | 0.0040 | 0.0039 |
| eh3_0_1_away | scoreline | failed | -0.0072 | 0.0003 | 0.0005 | -0.0016 |
| eh3_1_0_home | scoreline | failed | -0.0061 | -0.0003 | -0.0004 | 0.0041 |
| eh3_1_0_draw | scoreline | failed | -0.0119 | -0.0006 | -0.0013 | -0.0050 |
| eh3_1_0_away | scoreline | failed | -0.0035 | -0.0015 | -0.0030 | 0.0078 |
| o15 | scoreline | passed | 0.0175 | -0.0007 | -0.0029 | 0.0066 |
| u35 | scoreline | passed | 0.0127 | -0.0030 | -0.0064 | 0.0028 |
| mg_0 | scoreline | passed | 0.0222 | -0.0008 | -0.0038 | -0.0034 |
| mg_1_2 | scoreline | passed | 0.0258 | -0.0023 | -0.0054 | 0.0007 |
| mg_1_3 | scoreline | passed | 0.0050 | -0.0018 | -0.0037 | 0.0018 |
| mg_1_4 | scoreline | failed | -0.0072 | 0.0004 | 0.0011 | 0.0033 |
| mg_1_5 | scoreline | failed | 0.0033 | 0.0018 | 0.0049 | 0.0006 |
| mg_1_6 | scoreline | failed | -0.0136 | -0.0007 | -0.0018 | 0.0046 |
| mg_2_3 | scoreline | failed | -0.0035 | 0.0001 | 0.0002 | 0.0047 |
| mg_2_4 | scoreline | failed | 0.0071 | 0.0006 | 0.0013 | 0.0042 |
| mg_2_5 | scoreline | failed | 0.0097 | 0.0013 | 0.0027 | 0.0046 |
| mg_2_6 | scoreline | passed | 0.0092 | -0.0001 | -0.0005 | 0.0070 |
| mg_3_4 | scoreline | passed | 0.0296 | -0.0023 | -0.0048 | -0.0011 |
| mg_3_5 | scoreline | passed | 0.0282 | -0.0028 | -0.0056 | 0.0015 |
| mg_3_6 | scoreline | passed | 0.0255 | -0.0032 | -0.0064 | 0.0049 |
| mg_4_5 | scoreline | passed | 0.0133 | -0.0036 | -0.0082 | -0.0006 |
| mg_4_6 | scoreline | passed | 0.0071 | -0.0021 | -0.0044 | 0.0015 |
| mg_5_6 | scoreline | failed | -0.0080 | 0.0013 | 0.0034 | -0.0047 |
| mg_7p | scoreline | passed | 0.0745 | -0.0004 | -0.0031 | 0.0013 |
| hmg_0 | scoreline | failed | -0.0134 | -0.0015 | -0.0037 | 0.0095 |
| hmg_1_2 | scoreline | passed | 0.0021 | -0.0021 | -0.0043 | 0.0092 |
| hmg_1_3 | scoreline | failed | -0.0096 | -0.0026 | -0.0054 | 0.0147 |
| hmg_2_3 | scoreline | failed | 0.0028 | 0.0014 | 0.0031 | 0.0137 |
| hmg_4p | scoreline | passed | 0.0036 | -0.0039 | -0.0126 | -0.0019 |
| amg_0 | scoreline | failed | 0.0009 | 0.0002 | 0.0000 | -0.0002 |
| amg_1_2 | scoreline | failed | -0.0051 | 0.0008 | 0.0017 | -0.0040 |
| amg_1_3 | scoreline | failed | -0.0036 | 0.0005 | 0.0011 | -0.0044 |
| amg_2_3 | scoreline | failed | 0.0092 | 0.0002 | 0.0006 | 0.0011 |
| amg_4p | scoreline | passed | 0.0295 | -0.0007 | -0.0036 | 0.0011 |
| ms_h_1_0_2_0_3_0 | scoreline | failed | -0.0126 | 0.0035 | 0.0081 | -0.0094 |
| ms_a_0_1_0_2_0_3 | scoreline | failed | -0.0182 | -0.0008 | -0.0017 | 0.0018 |
| ms_h_4_0_5_0_6_0 | scoreline | passed | 0.0146 | -0.0041 | -0.0160 | -0.0008 |
| ms_a_0_4_0_5_0_6 | scoreline | passed | 0.0470 | -0.0016 | -0.0080 | -0.0018 |
| ms_h_2_1_3_1_4_1 | scoreline | passed | 0.0112 | -0.0021 | -0.0051 | 0.0016 |
| ms_h_1_2_1_3_1_4 | scoreline | failed | 0.0186 | 0.0012 | 0.0025 | 0.0034 |
| ms_h_3_2_4_2_5_1 | scoreline | failed | -0.0468 | 0.0002 | 0.0013 | 0.0013 |
| ms_a_2_3_2_4_1_5 | scoreline | passed | 0.0157 | -0.0005 | -0.0016 | -0.0003 |
| ms_other_homewin | scoreline | failed | 0.0169 | 0.0010 | 0.0036 | -0.0003 |
| ms_other_awaywin | scoreline | passed | 0.0294 | -0.0014 | -0.0070 | 0.0015 |
| ms_draw | scoreline | failed | 0.0303 | 0.0010 | 0.0015 | 0.0092 |
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
| h_1up | anytime | failed | -0.0304 | 0.0043 | 0.0092 | 0.0102 |
| a_1up | anytime | failed | -0.0152 | 0.0008 | 0.0016 | 0.0059 |
| h_2up | anytime | failed | -0.0145 | 0.0022 | 0.0050 | 0.0032 |
| a_2up | anytime | failed | -0.0120 | 0.0010 | 0.0038 | 0.0164 |
