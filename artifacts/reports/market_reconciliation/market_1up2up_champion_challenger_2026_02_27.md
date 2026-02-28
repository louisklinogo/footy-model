# 1UP/2UP Champion-Challenger (2026-02-27)

- Scope: challenger=trained GBM, baseline=fallback Markov, same 365-day scored fixtures.
- Decision: **PROMOTE**

| Market | N | AUC (GBM) | AUC (Fallback) | dAUC | Brier (GBM) | Brier (Fallback) | dBrier | Hit (GBM) | Hit (Fallback) | dHit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h_1up | 7028 | 0.7637 | 0.6565 | 0.1072 | 0.2021 | 0.2241 | -0.0220 | 0.7011 | 0.6345 | 0.0666 |
| a_1up | 7028 | 0.7534 | 0.6468 | 0.1066 | 0.2117 | 0.2336 | -0.0218 | 0.6831 | 0.6057 | 0.0774 |
| h_2up | 7028 | 0.7845 | 0.6617 | 0.1228 | 0.1640 | 0.1850 | -0.0210 | 0.7614 | 0.7356 | 0.0258 |
| a_2up | 7028 | 0.7870 | 0.6610 | 0.1260 | 0.1340 | 0.1490 | -0.0149 | 0.8147 | 0.8038 | 0.0110 |
