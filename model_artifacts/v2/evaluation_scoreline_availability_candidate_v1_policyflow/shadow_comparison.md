# Shadow comparison report

- Generated at: 2026-03-06T13:31:26.756702+00:00
- Decision: passed
- Recommended scope: required_markets_only
- Required markets passed: 14/14

## Artifact versions

- anytime: baseline `markov_head_v1` vs challenger `markov_head_v1`
- corners: baseline `distribution_head_v1` vs challenger `distribution_head_v1`
- scoreline: baseline `poisson_head_v1` vs challenger `scoreline_availability_candidate_v1`

## Market deltas

| Market | Family | Status | ΔAUC | ΔBrier | ΔLogLoss | ΔECE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1x2_h | scoreline | passed | 0.0180 | -0.0056 | -0.0141 | -0.0179 |
| 1x2_d | scoreline | passed | 0.0118 | -0.0018 | -0.0049 | -0.0281 |
| 1x2_a | scoreline | passed | 0.0237 | -0.0085 | -0.0203 | -0.0321 |
| dc_1x | scoreline | passed | 0.0237 | -0.0085 | -0.0203 | -0.0321 |
| dc_x2 | scoreline | passed | 0.0180 | -0.0056 | -0.0141 | -0.0179 |
| dc_12 | scoreline | passed | 0.0118 | -0.0018 | -0.0049 | -0.0281 |
| ah_h05 | scoreline | passed | 0.0180 | -0.0056 | -0.0141 | -0.0179 |
| ah_a05 | scoreline | passed | 0.0237 | -0.0085 | -0.0203 | -0.0321 |
| ah_h15 | scoreline | passed | 0.0069 | -0.0045 | -0.0129 | -0.0296 |
| ah_a15 | scoreline | passed | 0.0219 | -0.0062 | -0.0172 | -0.0301 |
| eh_h1 | scoreline | passed | 0.0069 | -0.0045 | -0.0129 | -0.0296 |
| eh_a1 | scoreline | passed | 0.0219 | -0.0062 | -0.0172 | -0.0301 |
| o15 | scoreline | passed | 0.0026 | -0.0057 | -0.0143 | -0.0262 |
| u35 | scoreline | passed | 0.0201 | -0.0081 | -0.0244 | -0.0516 |
| mg_0 | scoreline | passed | 0.0054 | -0.0009 | -0.0046 | -0.0132 |
| mg_1_2 | scoreline | passed | 0.0161 | -0.0043 | -0.0089 | -0.0071 |
| mg_1_3 | scoreline | passed | 0.0170 | -0.0042 | -0.0092 | -0.0304 |
| mg_1_4 | scoreline | passed | 0.0327 | -0.0024 | -0.0060 | -0.0234 |
| mg_1_5 | scoreline | failed | -0.0033 | -0.0019 | -0.0058 | -0.0205 |
| mg_1_6 | scoreline | passed | 0.0387 | -0.0015 | -0.0058 | -0.0199 |
| mg_2_3 | scoreline | passed | 0.0095 | -0.0015 | -0.0030 | -0.0222 |
| mg_2_4 | scoreline | failed | -0.0146 | -0.0026 | -0.0052 | -0.0282 |
| mg_2_5 | scoreline | failed | -0.0091 | -0.0037 | -0.0076 | -0.0268 |
| mg_2_6 | scoreline | failed | -0.0007 | -0.0044 | -0.0088 | -0.0216 |
| mg_3_4 | scoreline | passed | 0.0023 | -0.0052 | -0.0141 | -0.0172 |
| mg_3_5 | scoreline | passed | 0.0108 | -0.0075 | -0.0185 | -0.0248 |
| mg_3_6 | scoreline | passed | 0.0155 | -0.0086 | -0.0205 | -0.0148 |
| mg_4_5 | scoreline | passed | 0.0157 | -0.0030 | -0.0130 | -0.0216 |
| mg_4_6 | scoreline | passed | 0.0195 | -0.0045 | -0.0167 | -0.0250 |
| mg_5_6 | scoreline | passed | 0.0316 | -0.0011 | -0.0076 | -0.0105 |
| mg_7p | scoreline | passed | 0.0156 | -0.0013 | -0.0087 | -0.0087 |
| hmg_0 | scoreline | failed | -0.0049 | -0.0017 | -0.0030 | -0.0268 |
| hmg_1_2 | scoreline | failed | -0.0148 | -0.0005 | -0.0011 | -0.0288 |
| hmg_1_3 | scoreline | failed | -0.0125 | -0.0014 | -0.0029 | -0.0346 |
| hmg_2_3 | scoreline | failed | -0.0031 | -0.0027 | -0.0102 | -0.0228 |
| hmg_4p | scoreline | passed | 0.0283 | -0.0015 | -0.0101 | -0.0097 |
| amg_0 | scoreline | passed | 0.0368 | -0.0135 | -0.0315 | -0.0483 |
| amg_1_2 | scoreline | passed | 0.0018 | -0.0070 | -0.0147 | -0.0380 |
| amg_1_3 | scoreline | passed | 0.0225 | -0.0089 | -0.0190 | -0.0318 |
| amg_2_3 | scoreline | passed | 0.0116 | -0.0063 | -0.0200 | -0.0322 |
| amg_4p | scoreline | passed | 0.1085 | -0.0024 | -0.0198 | -0.0138 |
| ms_h_1_0_2_0_3_0 | scoreline | passed | 0.0277 | -0.0050 | -0.0124 | -0.0242 |
| ms_a_0_1_0_2_0_3 | scoreline | passed | 0.0019 | -0.0014 | -0.0028 | -0.0044 |
| ms_h_4_0_5_0_6_0 | scoreline | passed | 0.0552 | -0.0004 | -0.0051 | -0.0029 |
| ms_a_0_4_0_5_0_6 | scoreline | passed | 0.0631 | -0.0001 | -0.0024 | 0.0009 |
| ms_h_2_1_3_1_4_1 | scoreline | passed | 0.0015 | -0.0014 | -0.0110 | -0.0159 |
| ms_h_1_2_1_3_1_4 | scoreline | passed | 0.0287 | -0.0010 | -0.0068 | -0.0104 |
| ms_h_3_2_4_2_5_1 | scoreline | passed | 0.0794 | -0.0003 | -0.0054 | -0.0025 |
| ms_a_2_3_2_4_1_5 | scoreline | passed | 0.0438 | -0.0004 | -0.0074 | -0.0010 |
| ms_other_homewin | scoreline | passed | 0.0957 | -0.0006 | -0.0065 | -0.0043 |
| ms_other_awaywin | scoreline | failed | -0.0152 | -0.0002 | -0.0020 | -0.0041 |
| ms_draw | scoreline | passed | 0.0118 | -0.0018 | -0.0049 | -0.0281 |
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
| h_1up | anytime | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| a_1up | anytime | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| h_2up | anytime | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| a_2up | anytime | passed | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
