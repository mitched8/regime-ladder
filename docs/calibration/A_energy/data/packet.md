# Review packet · stage `data` · calibration A_energy (synthetic)

| key | value |
|---|---|
| pair | EURUSD |
| data | 2015-01-01 to 2024-08-28 (2520 days) |
| finder / leading target | expected forward 5d straddle P&L given the true state path (synthetic, noiseless); diagnostic outcome at 21d |
| git commit | 556ee10 |
| config hash | ba38712377 |
| generated | 2026-10-04 17:40 |

## Definitions
- Units: P&L is cash per unit of standard notional (per unit of vega at inception once D1 is closed). Horizons h are trading days from entry, capped at the option tenor.
- Point-in-time: every feature, label and statistic stamped on date t uses only data available at the close of t. Labels are assigned from trailing data; forward quantities appear only as targets.
- Costs are excluded everywhere by design.
- States: carry, rising, agitated, stressed, normalising, settling (+ extreme if it has >= 5 episodes, else merged into stressed). Stress rank low to high (used for 'moves up' and by the transition tilt): carry < settling < rising < agitated < normalising < stressed < extreme.
- One row per trade per day of life. `age` counts trading days from entry starting at 1. `pnl` = sum of components (trade, delta hedge, vega hedge) where present.
- Base legs: ATM straddle, 25d put/call, 10d put/call, each hedged leg by leg. Packages (RR, fly) are weight vectors over legs; if the source also runs them, the package should regress onto its legs with R² ~ 1 and stable coefficients.
- Expected: no gaps in age; components sum to pnl; long-option legs have positive vega at age 1; the largest daily P&L rows fall on dates of known market stress.

## 1. Integrity checks
```
[PASS] required_columns: ok
[PASS] no_duplicate_trade_date: 0 duplicate rows
[PASS] no_nan_pnl: 0 NaN pnl rows
[PASS] ages_consecutive_from_1: 0 trades not starting at age 1, 0 with gaps
[PASS] age_within_tenor: 0 rows with age > tenor
[PASS] dates_increase_with_age: 0 trades with non-increasing dates
[PASS] date_after_entry: 0 rows dated on/before entry
[PASS] components_sum_to_pnl: 0 rows off by > 1e-06; max gap 3.55e-15
[PASS] info_trades_not_yet_expired: 0 trades shorter than tenor (live or truncated)
```
## 2. Coverage
| pair | archetype | tenor_days | trades | rows | first_entry | last_entry | max_age |
|---|---|---|---|---|---|---|---|
| EURUSD | fly_10d | 21 | 2499 | 52479 | 2015-01-01 | 2024-07-30 | 21 |
| EURUSD | rr_25d | 21 | 2499 | 52479 | 2015-01-01 | 2024-07-30 | 21 |
| EURUSD | straddle_atm | 21 | 2499 | 52479 | 2015-01-01 | 2024-07-30 | 21 |

Months with no entries: none

## 3. Daily P&L distribution per leg
| pair | archetype | tenor_days | count | mean | std | min | 1% | 50% | 99% | max |
|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD | fly_10d | 21 | 52479.000 | -0.462 | 3.775 | -20.974 | -11.601 | -0.188 | 8.223 | 27.306 |
| EURUSD | rr_25d | 21 | 52479.000 | -0.581 | 3.811 | -31.685 | -11.380 | -0.434 | 8.647 | 20.926 |
| EURUSD | straddle_atm | 21 | 52479.000 | -0.045 | 3.880 | -17.679 | -9.101 | -0.345 | 11.062 | 25.160 |

## 4. Cumulative P&L from entry (all entries pooled)
| archetype | tenor_days | h | to_expiry | count | mean | std | min | max |
|---|---|---|---|---|---|---|---|---|
| fly_10d | 21 | 1 | False | 2499 | -0.396 | 3.782 | -18.896 | 23.475 |
| fly_10d | 21 | 5 | False | 2499 | -1.998 | 10.127 | -68.962 | 24.744 |
| fly_10d | 21 | 10 | False | 2499 | -3.986 | 16.283 | -120.379 | 39.088 |
| fly_10d | 21 | 21 | True | 2499 | -9.710 | 26.493 | -221.781 | 51.873 |
| rr_25d | 21 | 1 | False | 2499 | -0.440 | 3.764 | -29.149 | 18.874 |
| rr_25d | 21 | 5 | False | 2499 | -2.293 | 9.570 | -68.659 | 37.282 |
| rr_25d | 21 | 10 | False | 2499 | -4.730 | 14.832 | -99.766 | 64.887 |
| rr_25d | 21 | 21 | True | 2499 | -12.210 | 23.026 | -128.730 | 70.462 |
| straddle_atm | 21 | 1 | False | 2499 | -0.034 | 3.846 | -16.377 | 22.949 |
| straddle_atm | 21 | 5 | False | 2499 | -0.220 | 10.001 | -33.756 | 54.785 |
| straddle_atm | 21 | 10 | False | 2499 | -0.495 | 15.578 | -48.834 | 89.019 |
| straddle_atm | 21 | 21 | True | 2499 | -0.948 | 25.247 | -63.463 | 165.256 |

## 5. Twenty largest market days: mean daily P&L across live trades of a leg (check the dates against known stress)
| date | pair | archetype | mean_pnl | live_trades |
|---|---|---|---|---|
| 2019-12-27 | EURUSD | rr_25d | -29.821 | 21 |
| 2023-04-17 | EURUSD | fly_10d | 25.622 | 21 |
| 2021-03-16 | EURUSD | straddle_atm | 23.015 | 21 |
| 2019-12-25 | EURUSD | rr_25d | -21.188 | 21 |
| 2018-04-19 | EURUSD | straddle_atm | 20.299 | 21 |
| 2023-05-31 | EURUSD | rr_25d | -20.226 | 21 |
| 2021-03-03 | EURUSD | rr_25d | -19.693 | 21 |
| 2019-12-16 | EURUSD | straddle_atm | 19.548 | 21 |
| 2023-04-13 | EURUSD | fly_10d | -19.404 | 21 |
| 2020-01-03 | EURUSD | straddle_atm | 19.231 | 21 |
| 2020-01-14 | EURUSD | rr_25d | 19.077 | 21 |
| 2019-12-24 | EURUSD | fly_10d | -18.686 | 21 |
| 2019-12-16 | EURUSD | fly_10d | -18.439 | 21 |
| 2021-03-16 | EURUSD | rr_25d | -18.095 | 21 |
| 2019-12-12 | EURUSD | rr_25d | -17.823 | 21 |
| 2019-12-23 | EURUSD | straddle_atm | 17.788 | 21 |
| 2023-06-02 | EURUSD | straddle_atm | 17.538 | 21 |
| 2019-12-13 | EURUSD | fly_10d | -17.232 | 21 |
| 2017-12-06 | EURUSD | rr_25d | 16.793 | 21 |
| 2016-01-19 | EURUSD | straddle_atm | 16.637 | 21 |

## 6. Components: variance of each component / variance of total (can exceed 1 when components offset) and largest residual
| archetype | pnl_trade | pnl_delta_hedge | pnl_vega_hedge | max_abs_residual |
|---|---|---|---|---|
| fly_10d | 1.9600 | 0.0900 | 0.0100 | 0.0000 |
| rr_25d | 1.9600 | 0.0900 | 0.0100 | 0.0000 |
| straddle_atm | 1.9600 | 0.0900 | 0.0100 | 0.0000 |

## 7. Vega at age 1
| archetype | tenor_days | median | min | max | n_nonpositive |
|---|---|---|---|---|---|
| fly_10d | 21 | 0.4000 | 0.4000 | 0.4000 | 0 |
| rr_25d | 21 | 0.4000 | 0.4000 | 0.4000 | 0 |
| straddle_atm | 21 | 0.4000 | 0.4000 | 0.4000 | 0 |

## 8. Package reconciliation (package daily P&L regressed on its legs)
no package with all its legs present

## 9. Market frame
2520 days 2015-01-01 to 2024-08-28; trade dates missing from the market frame: 0

| index | count | mean | min | max |
|---|---|---|---|---|
| pressure_true | 2520.000 | 0.030 | 0.000 | 2.330 |
| external_true | 2520.000 | 0.047 | -3.019 | 3.221 |
| stress | 2520.000 | 52.279 | 5.905 | 106.099 |
| atm_1m | 2520.000 | 10.325 | 5.789 | 22.312 |
| atm_1y | 2520.000 | 9.596 | 6.851 | 16.857 |
| rr25_1m | 2520.000 | -1.243 | -3.421 | -0.005 |
| fly25_1m | 2520.000 | 0.401 | 0.174 | 0.989 |
| spot | 2520.000 | 1.023 | 0.751 | 1.235 |
| rv_1w | 2520.000 | 9.732 | 0.793 | 42.755 |
| rv_1m | 2520.000 | 10.296 | 3.950 | 26.186 |
| rv_3m | 2520.000 | 10.522 | 5.375 | 18.191 |
| g10_1 | 2520.000 | 0.832 | 0.683 | 1.162 |
| g10_2 | 2520.000 | 0.987 | 0.812 | 1.159 |
| g10_3 | 2520.000 | 0.724 | 0.547 | 1.040 |
| g10_4 | 2520.000 | 0.965 | 0.690 | 1.187 |
| g10_5 | 2520.000 | 0.808 | 0.632 | 1.117 |
| g10_6 | 2520.000 | 0.827 | 0.604 | 1.078 |
| eq | 2520.000 | 63.039 | 39.658 | 100.279 |
| eq2 | 2520.000 | 91.540 | 64.994 | 115.628 |
| rate2y | 2520.000 | 1.679 | 0.676 | 2.710 |
| rate10y | 2520.000 | 1.586 | -0.973 | 3.368 |
| oil | 2520.000 | 102.571 | 71.495 | 121.806 |
| gold | 2520.000 | 126.452 | 88.477 | 167.175 |
| dxy | 2520.000 | 94.527 | 70.910 | 119.420 |
| credit | 2520.000 | 103.555 | 80.145 | 129.834 |
| vix | 2520.000 | 105.378 | 80.869 | 141.920 |
| move | 2520.000 | 98.001 | 72.224 | 135.425 |

