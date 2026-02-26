# Layer 1 Hyperparameter Sweep (xi fixed at 0.001)

Generated: 2026-02-26T09:25:52.244321+00:00

- Best config: `regB_xi001_depth3_mcw25_lr025`

| Config | Home RMSE | Away RMSE | Mean RMSE | Home Bias | Away Bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| regB_xi001_depth3_mcw25_lr025 | 1.1797 | 1.0795 | 1.1296 | 0.0311 | 0.0617 |
| regA_xi001_depth3_mcw20 | 1.1801 | 1.0818 | 1.1309 | 0.0322 | 0.0625 |
| regC_xi001_depth4_mcw25 | 1.1834 | 1.0828 | 1.1331 | 0.0351 | 0.0632 |
| baseline_xi001_depth4_mcw15 | 1.1828 | 1.0835 | 1.1332 | 0.0361 | 0.0645 |

## League Diagnostics (best config, n>=40)
| League | n | Home RMSE | Away RMSE | Mean RMSE | Home Bias | Away Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EC | 60 | 1.2877 | 1.1924 | 1.2401 | 0.2957 | 0.2672 |
| E2 | 52 | 1.3660 | 1.0581 | 1.2120 | 0.3624 | 0.1424 |
| E1 | 51 | 1.2294 | 1.0960 | 1.1627 | -0.0018 | 0.1598 |
| E3 | 56 | 1.1557 | 1.0033 | 1.0795 | -0.1955 | -0.0702 |
| I2 | 41 | 0.9380 | 1.0566 | 0.9973 | -0.0903 | 0.1840 |