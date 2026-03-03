# Ablation Results

## Pass 1: Additive
| Stage | Features | Home RMSE | Away RMSE | Home Lift | Away Lift |
| --- | --- | --- | --- | --- | --- |
| baseline | 0 | 1.2031 | 1.0803 | 0.00% | 0.00% |
| existing_full | 22 | 1.2034 | 1.0789 | -0.02% | 0.13% |
| existing_full+tactical | 25 | 1.2000 | 1.0840 | 0.26% | -0.34% |
| existing_full+tactical+h1h2 | 35 | 1.1993 | 1.0855 | 0.32% | -0.48% |
| existing_full+tactical+h1h2+formations | 42 | 1.1986 | 1.0877 | 0.38% | -0.68% |
| existing_full+tactical+h1h2+formations+defensive | 46 | 1.1965 | 1.0861 | 0.56% | -0.54% |
| all_candidates | 46 | 1.1965 | 1.0861 | 0.56% | -0.54% |

## Pass 2: Drop-one
| Stage | Features | Home RMSE | Away RMSE | Verdict |
| --- | --- | --- | --- | --- |
| all_candidates | 46 | 1.1965 | 1.0861 | anchor |
| drop_tactical | 43 | 1.1997 | 1.0822 | VALUABLE |
| drop_h1h2 | 36 | 1.1994 | 1.0832 | VALUABLE |
| drop_formations | 39 | 1.1999 | 1.0861 | VALUABLE |
| drop_defensive | 42 | 1.1986 | 1.0877 | VALUABLE |

## Promotion Decisions
| Group | Additive | Drop-one | Coverage | Decision |
| --- | --- | --- | --- | --- |
| tactical | FAIL | VALUABLE | N/A | **REJECT** |
| h1h2 | FAIL | VALUABLE | SIGNAL | **REJECT** |
| formations | FAIL | VALUABLE | N/A | **REJECT** |
| defensive | PASS | VALUABLE | N/A | **PROMOTE** |