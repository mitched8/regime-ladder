# Review packet · stage `states` · calibration A_energy (synthetic)

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
- Composite stress score = mean of trailing percentiles of ATM vol level, term-structure slope and risk-reversal stress (0..100), smoothed by an EWMA.
- Level bands (low / mid / high) are step-fitted on forward ATM 5 days ahead over the whole history (a surface fact). Partition, direction thresholds, smoothing and hysteresis are chosen out of sample on the finder target under a switch budget.
- Episode = one uninterrupted run of a state. A state or tagged cell with < 5 episodes is not reported separately.
- Transition matrix: daily, with a sticky prior; implied duration 1/(1-p_ii) should be close to the observed mean run (ratio 0.6-1.5) or the Markov approximation is poor for that state.
- Profile `d` = standardised difference of a characteristic's mean in the state versus all days.

## 1. Chosen spec
| key | value |
|---|---|
| halflife | 2.000 |
| delta | 2.000 |
| partition | 6 |
| dir_k | 1.000 |
| bounds | [35.71747225968421, 81.54409870040847] |
| b1 | 35.717 |
| b2 | 81.544 |
| d_up | 2.620 |
| d_down | -2.620 |
| delta_d | 0.524 |
| dir_window | 5.000 |
| oos_r2 | 0.259 |
| switches | 249.000 |
| switches_per_year | 24.900 |
| min_episodes | 5.000 |
| extreme_merged | False |
| extreme_episodes | 0.000 |
| extreme_days | 0.000 |

Best out-of-sample R² per partition: {"3": 0.24750226959775035, "6": 0.2586434692510159, "6x": 0.2586434692510159}

## 2. Finder: top 5 candidates per partition
| partition | halflife | delta | dir_k | b1 | b2 | oos_r2 | switches_per_year | min_episodes | extreme_merged |
|---|---|---|---|---|---|---|---|---|---|
| 6x | 2 | 0 | 1.000 | 35.717 | 81.544 | 0.286 | 29.000 | 2 | False |
| 6x | 2 | 0 | 0.500 | 35.717 | 81.544 | 0.284 | 36.800 | 12 | False |
| 6x | 2 | 0 | 0.750 | 35.717 | 81.544 | 0.276 | 33.300 | 5 | False |
| 6 | 2 | 0 | 1.000 | 35.717 | 81.544 | 0.272 | 26.600 | 2 | False |
| 6 | 2 | 0 | 0.500 | 35.717 | 81.544 | 0.270 | 34.500 | 12 | False |
| 6x | 3 | 0 | 0.500 | 36.854 | 79.828 | 0.265 | 33.000 | 10 | False |
| 6 | 3 | 5 | 0.500 | 36.854 | 79.828 | 0.262 | 29.000 | 13 | False |
| 6 | 2 | 5 | 0.500 | 35.717 | 81.544 | 0.262 | 31.900 | 14 | False |
| 6x | 2 | 2 | 0.500 | 35.717 | 81.544 | 0.262 | 33.000 | 13 | True |
| 6 | 2 | 2 | 0.500 | 35.717 | 81.544 | 0.262 | 33.000 | 13 | False |
| 3 | 2 | 0 | 0.000 | 35.717 | 81.544 | 0.248 | 14.000 | 24 | False |
| 3 | 2 | 2 | 0.000 | 35.717 | 81.544 | 0.231 | 11.000 | 19 | False |
| 3 | 5 | 0 | 0.000 | 37.795 | 82.435 | 0.230 | 6.800 | 11 | False |
| 3 | 3 | 0 | 0.000 | 36.854 | 79.828 | 0.225 | 10.600 | 18 | False |
| 3 | 2 | 5 | 0.000 | 35.717 | 81.544 | 0.216 | 7.800 | 13 | False |

## 3. States: share, episodes, durations, composite range
| state | implied_days | observed_mean_days | episodes | share_of_days | composite_mean | composite_min | composite_max | ratio_implied_over_observed |
|---|---|---|---|---|---|---|---|---|
| carry | 19.808 | 19.806 | 36 | 0.283 | 18.890 | 0.195 | 37.474 | 1.000 |
| rising | 5.369 | 5.236 | 55 | 0.114 | 61.864 | 23.170 | 83.435 | 1.025 |
| agitated | 9.918 | 9.866 | 97 | 0.380 | 54.221 | 33.734 | 83.447 | 1.005 |
| stressed | 15.946 | 15.053 | 19 | 0.113 | 88.813 | 79.836 | 98.978 | 1.059 |
| normalising | 7.273 | 6.000 | 5 | 0.012 | 63.605 | 36.376 | 80.369 | 1.212 |
| settling | 6.649 | 6.474 | 38 | 0.098 | 40.695 | 5.848 | 75.460 | 1.027 |

Features warm up until 2015-06-19: 121 earlier days carry the default label (carry) and must not be read as calm. The trade table should start after this date.

## 4. Share of days per state by year
| row_0 | agitated | carry | normalising | rising | settling | stressed |
|---|---|---|---|---|---|---|
| 2015 | 0.04 | 0.83 | 0.00 | 0.07 | 0.03 | 0.03 |
| 2016 | 0.39 | 0.18 | 0.00 | 0.11 | 0.15 | 0.18 |
| 2017 | 0.67 | 0.05 | 0.05 | 0.09 | 0.02 | 0.12 |
| 2018 | 0.29 | 0.16 | 0.02 | 0.17 | 0.14 | 0.22 |
| 2019 | 0.39 | 0.25 | 0.00 | 0.15 | 0.10 | 0.10 |
| 2020 | 0.32 | 0.22 | 0.04 | 0.16 | 0.18 | 0.08 |
| 2021 | 0.26 | 0.46 | 0.00 | 0.09 | 0.10 | 0.09 |
| 2022 | 0.50 | 0.38 | 0.00 | 0.05 | 0.01 | 0.05 |
| 2023 | 0.47 | 0.02 | 0.00 | 0.15 | 0.17 | 0.20 |
| 2024 | 0.52 | 0.27 | 0.00 | 0.09 | 0.08 | 0.04 |

## 5. Every run of every state
| start | end | state | days |
|---|---|---|---|
| 2015-01-01 | 2015-09-23 | carry | 190 |
| 2015-09-24 | 2015-09-30 | rising | 5 |
| 2015-10-01 | 2015-10-02 | agitated | 2 |
| 2015-10-05 | 2015-11-02 | carry | 21 |
| 2015-11-03 | 2015-11-03 | agitated | 1 |
| 2015-11-04 | 2015-11-11 | rising | 6 |
| 2015-11-12 | 2015-11-13 | agitated | 2 |
| 2015-11-16 | 2015-11-20 | settling | 5 |
| 2015-11-23 | 2015-11-24 | agitated | 2 |
| 2015-11-25 | 2015-12-02 | carry | 6 |
| 2015-12-03 | 2015-12-03 | agitated | 1 |
| 2015-12-04 | 2015-12-14 | rising | 7 |
| 2015-12-15 | 2015-12-23 | stressed | 7 |
| 2015-12-24 | 2015-12-28 | agitated | 3 |
| 2015-12-29 | 2016-01-05 | settling | 6 |
| 2016-01-06 | 2016-01-18 | agitated | 9 |
| 2016-01-19 | 2016-01-26 | rising | 6 |
| 2016-01-27 | 2016-02-02 | stressed | 5 |
| 2016-02-03 | 2016-02-04 | agitated | 2 |
| 2016-02-05 | 2016-02-17 | settling | 9 |
| 2016-02-18 | 2016-03-01 | agitated | 9 |
| 2016-03-02 | 2016-03-08 | carry | 5 |
| 2016-03-09 | 2016-03-11 | agitated | 3 |
| 2016-03-14 | 2016-03-15 | rising | 2 |
| 2016-03-16 | 2016-03-23 | agitated | 6 |
| 2016-03-24 | 2016-03-29 | rising | 4 |
| 2016-03-30 | 2016-05-26 | stressed | 42 |
| 2016-05-27 | 2016-05-30 | agitated | 2 |
| 2016-05-31 | 2016-06-10 | settling | 9 |
| 2016-06-13 | 2016-08-04 | agitated | 39 |
| 2016-08-05 | 2016-08-15 | carry | 7 |
| 2016-08-16 | 2016-08-24 | rising | 7 |
| 2016-08-25 | 2016-08-31 | agitated | 5 |
| 2016-09-01 | 2016-09-14 | settling | 10 |
| 2016-09-15 | 2016-09-29 | carry | 11 |
| 2016-09-30 | 2016-10-12 | rising | 9 |
| 2016-10-13 | 2016-11-17 | agitated | 26 |
| 2016-11-18 | 2016-11-29 | settling | 8 |
| 2016-11-30 | 2017-01-09 | carry | 29 |
| 2017-01-10 | 2017-01-19 | rising | 8 |
| 2017-01-20 | 2017-01-24 | agitated | 3 |
| 2017-01-25 | 2017-02-03 | carry | 8 |
| 2017-02-06 | 2017-02-21 | rising | 12 |
| 2017-02-22 | 2017-03-23 | agitated | 22 |
| 2017-03-24 | 2017-03-27 | rising | 2 |
| 2017-03-28 | 2017-04-20 | stressed | 18 |
| 2017-04-21 | 2017-04-28 | normalising | 6 |
| 2017-05-01 | 2017-05-19 | agitated | 15 |
| 2017-05-22 | 2017-05-24 | settling | 3 |
| 2017-05-25 | 2017-11-27 | agitated | 133 |
| 2017-11-28 | 2017-11-29 | rising | 2 |
| 2017-11-30 | 2017-12-18 | stressed | 13 |
| 2017-12-19 | 2017-12-27 | normalising | 7 |
| 2017-12-28 | 2018-01-05 | settling | 7 |
| 2018-01-08 | 2018-01-11 | carry | 4 |
| 2018-01-12 | 2018-01-19 | rising | 6 |
| 2018-01-22 | 2018-02-08 | agitated | 14 |
| 2018-02-09 | 2018-02-15 | rising | 5 |
| 2018-02-16 | 2018-04-26 | stressed | 50 |
| 2018-04-27 | 2018-05-04 | normalising | 6 |
| 2018-05-07 | 2018-05-09 | agitated | 3 |
| 2018-05-10 | 2018-05-16 | rising | 5 |
| 2018-05-17 | 2018-05-18 | stressed | 2 |
| 2018-05-21 | 2018-05-22 | agitated | 2 |
| 2018-05-23 | 2018-06-05 | settling | 10 |
| 2018-06-06 | 2018-06-08 | agitated | 3 |
| 2018-06-11 | 2018-06-18 | rising | 6 |
| 2018-06-19 | 2018-06-28 | agitated | 8 |
| 2018-06-29 | 2018-07-02 | settling | 2 |
| 2018-07-03 | 2018-07-23 | agitated | 15 |
| 2018-07-24 | 2018-07-27 | settling | 4 |
| 2018-07-30 | 2018-07-30 | carry | 1 |
| 2018-07-31 | 2018-08-20 | agitated | 15 |
| 2018-08-21 | 2018-09-11 | carry | 16 |
| 2018-09-12 | 2018-09-14 | rising | 3 |
| 2018-09-17 | 2018-09-25 | carry | 7 |
| 2018-09-26 | 2018-09-26 | agitated | 1 |
| 2018-09-27 | 2018-10-08 | rising | 8 |
| 2018-10-09 | 2018-10-15 | stressed | 5 |
| 2018-10-16 | 2018-10-16 | agitated | 1 |
| 2018-10-17 | 2018-10-26 | settling | 8 |
| 2018-10-29 | 2018-10-29 | carry | 1 |
| 2018-10-30 | 2018-11-01 | agitated | 3 |
| 2018-11-02 | 2018-11-06 | rising | 3 |
| 2018-11-07 | 2018-11-12 | agitated | 4 |
| 2018-11-13 | 2018-11-19 | settling | 5 |
| 2018-11-20 | 2018-11-21 | carry | 2 |
| 2018-11-22 | 2018-11-26 | rising | 3 |
| 2018-11-27 | 2018-11-28 | agitated | 2 |
| 2018-11-29 | 2018-12-13 | carry | 11 |
| 2018-12-14 | 2018-12-18 | agitated | 3 |
| 2018-12-19 | 2018-12-25 | rising | 5 |
| 2018-12-26 | 2018-12-26 | agitated | 1 |
| 2018-12-27 | 2019-01-03 | settling | 6 |
| 2019-01-04 | 2019-01-07 | carry | 2 |
| 2019-01-08 | 2019-01-23 | rising | 12 |
| 2019-01-24 | 2019-01-28 | agitated | 3 |
| 2019-01-29 | 2019-02-06 | settling | 7 |
| 2019-02-07 | 2019-04-01 | agitated | 38 |
| 2019-04-02 | 2019-04-05 | rising | 4 |
| 2019-04-08 | 2019-04-12 | agitated | 5 |
| 2019-04-15 | 2019-04-16 | settling | 2 |
| 2019-04-17 | 2019-05-03 | agitated | 13 |
| 2019-05-06 | 2019-05-09 | settling | 4 |
| 2019-05-10 | 2019-05-10 | carry | 1 |
| 2019-05-13 | 2019-05-14 | agitated | 2 |
| 2019-05-15 | 2019-05-16 | rising | 2 |
| 2019-05-17 | 2019-05-17 | agitated | 1 |
| 2019-05-20 | 2019-08-09 | carry | 60 |
| 2019-08-12 | 2019-08-20 | rising | 7 |
| 2019-08-21 | 2019-09-04 | agitated | 11 |
| 2019-09-05 | 2019-09-11 | settling | 5 |
| 2019-09-12 | 2019-09-12 | agitated | 1 |
| 2019-09-13 | 2019-09-19 | rising | 5 |
| 2019-09-20 | 2019-09-30 | stressed | 7 |
| 2019-10-01 | 2019-10-01 | agitated | 1 |
| 2019-10-02 | 2019-10-09 | settling | 6 |
| 2019-10-10 | 2019-10-10 | carry | 1 |
| 2019-10-11 | 2019-10-21 | agitated | 7 |
| 2019-10-22 | 2019-10-25 | rising | 4 |
| 2019-10-28 | 2019-11-19 | agitated | 17 |
| 2019-11-20 | 2019-11-26 | rising | 5 |
| 2019-11-27 | 2019-12-02 | agitated | 4 |
| 2019-12-03 | 2019-12-03 | rising | 1 |
| 2019-12-04 | 2020-01-17 | stressed | 33 |
| 2020-01-20 | 2020-01-31 | normalising | 10 |
| 2020-02-03 | 2020-02-06 | settling | 4 |
| 2020-02-07 | 2020-02-07 | carry | 1 |
| 2020-02-10 | 2020-02-10 | agitated | 1 |
| 2020-02-11 | 2020-02-21 | rising | 9 |
| 2020-02-24 | 2020-03-02 | agitated | 6 |
| 2020-03-03 | 2020-03-12 | settling | 8 |
| 2020-03-13 | 2020-03-13 | agitated | 1 |
| 2020-03-16 | 2020-03-24 | rising | 7 |
| 2020-03-25 | 2020-04-15 | agitated | 16 |
| 2020-04-16 | 2020-04-28 | stressed | 9 |
| 2020-04-29 | 2020-05-01 | agitated | 3 |
| 2020-05-04 | 2020-05-14 | settling | 9 |
| 2020-05-15 | 2020-06-05 | agitated | 16 |
| 2020-06-08 | 2020-06-12 | rising | 5 |
| 2020-06-15 | 2020-06-16 | agitated | 2 |
| 2020-06-17 | 2020-06-25 | settling | 7 |
| 2020-06-26 | 2020-06-26 | agitated | 1 |
| 2020-06-29 | 2020-07-08 | rising | 8 |
| 2020-07-09 | 2020-07-16 | agitated | 6 |
| 2020-07-17 | 2020-08-06 | settling | 15 |
| 2020-08-07 | 2020-10-23 | carry | 56 |
| 2020-10-26 | 2020-10-30 | rising | 5 |
| 2020-11-02 | 2020-11-02 | carry | 1 |
| 2020-11-03 | 2020-11-12 | rising | 8 |
| 2020-11-13 | 2020-11-18 | agitated | 4 |
| 2020-11-19 | 2020-11-23 | settling | 3 |
| 2020-11-24 | 2021-01-18 | agitated | 40 |
| 2021-01-19 | 2021-02-04 | carry | 13 |
| 2021-02-05 | 2021-02-22 | rising | 12 |
| 2021-02-23 | 2021-02-23 | agitated | 1 |
| 2021-02-24 | 2021-03-26 | stressed | 23 |
| 2021-03-29 | 2021-03-29 | agitated | 1 |
| 2021-03-30 | 2021-04-12 | settling | 10 |
| 2021-04-13 | 2021-04-20 | agitated | 6 |
| 2021-04-21 | 2021-04-28 | rising | 6 |
| 2021-04-29 | 2021-04-30 | agitated | 2 |
| 2021-05-03 | 2021-05-12 | settling | 8 |
| 2021-05-13 | 2021-05-13 | carry | 1 |
| 2021-05-14 | 2021-05-14 | agitated | 1 |
| 2021-05-17 | 2021-05-21 | rising | 5 |
| 2021-05-24 | 2021-06-14 | agitated | 16 |
| 2021-06-15 | 2021-06-15 | rising | 1 |
| 2021-06-16 | 2021-07-23 | agitated | 28 |
| 2021-07-26 | 2021-08-04 | settling | 8 |
| 2021-08-05 | 2022-04-04 | carry | 173 |
| 2022-04-05 | 2022-04-06 | agitated | 2 |
| 2022-04-07 | 2022-04-28 | carry | 16 |
| 2022-04-29 | 2022-05-03 | agitated | 3 |
| 2022-05-04 | 2022-05-05 | rising | 2 |
| 2022-05-06 | 2022-05-11 | agitated | 4 |
| 2022-05-12 | 2022-05-20 | carry | 7 |
| 2022-05-23 | 2022-05-24 | agitated | 2 |
| 2022-05-25 | 2022-05-27 | carry | 3 |
| 2022-05-30 | 2022-05-31 | agitated | 2 |
| 2022-06-01 | 2022-06-07 | carry | 5 |
| 2022-06-08 | 2022-06-09 | agitated | 2 |
| 2022-06-10 | 2022-06-14 | carry | 3 |
| 2022-06-15 | 2022-06-16 | agitated | 2 |
| 2022-06-17 | 2022-06-23 | rising | 5 |
| 2022-06-24 | 2022-08-16 | agitated | 38 |
| 2022-08-17 | 2022-08-17 | rising | 1 |
| 2022-08-18 | 2022-08-25 | stressed | 6 |
| 2022-08-26 | 2022-09-16 | agitated | 16 |
| 2022-09-19 | 2022-09-28 | stressed | 8 |
| 2022-09-29 | 2022-09-30 | agitated | 2 |
| 2022-10-03 | 2022-10-04 | settling | 2 |
| 2022-10-05 | 2022-11-10 | agitated | 27 |
| 2022-11-11 | 2022-11-15 | rising | 3 |
| 2022-11-16 | 2022-12-14 | agitated | 21 |
| 2022-12-15 | 2022-12-19 | rising | 3 |
| 2022-12-20 | 2023-02-21 | agitated | 46 |
| 2023-02-22 | 2023-02-27 | settling | 4 |
| 2023-02-28 | 2023-03-01 | agitated | 2 |
| 2023-03-02 | 2023-03-10 | rising | 7 |
| 2023-03-13 | 2023-03-14 | agitated | 2 |
| 2023-03-15 | 2023-03-16 | settling | 2 |
| 2023-03-17 | 2023-04-12 | agitated | 19 |
| 2023-04-13 | 2023-04-14 | rising | 2 |
| 2023-04-17 | 2023-05-04 | stressed | 14 |
| 2023-05-05 | 2023-05-10 | agitated | 4 |
| 2023-05-11 | 2023-06-09 | stressed | 22 |
| 2023-06-12 | 2023-06-12 | normalising | 1 |
| 2023-06-13 | 2023-06-20 | agitated | 6 |
| 2023-06-21 | 2023-06-30 | settling | 8 |
| 2023-07-03 | 2023-07-05 | carry | 3 |
| 2023-07-06 | 2023-07-25 | agitated | 14 |
| 2023-07-26 | 2023-08-02 | rising | 6 |
| 2023-08-03 | 2023-08-08 | agitated | 4 |
| 2023-08-09 | 2023-08-18 | settling | 8 |
| 2023-08-21 | 2023-09-05 | agitated | 12 |
| 2023-09-06 | 2023-09-14 | rising | 7 |
| 2023-09-15 | 2023-10-05 | stressed | 15 |
| 2023-10-06 | 2023-10-09 | agitated | 2 |
| 2023-10-10 | 2023-10-18 | settling | 7 |
| 2023-10-19 | 2023-10-23 | agitated | 3 |
| 2023-10-24 | 2023-11-01 | rising | 7 |
| 2023-11-02 | 2023-11-09 | agitated | 6 |
| 2023-11-10 | 2023-11-20 | settling | 7 |
| 2023-11-21 | 2023-11-21 | carry | 1 |
| 2023-11-22 | 2023-11-22 | agitated | 1 |
| 2023-11-23 | 2023-11-30 | rising | 6 |
| 2023-12-01 | 2023-12-11 | agitated | 7 |
| 2023-12-12 | 2023-12-14 | rising | 3 |
| 2023-12-15 | 2023-12-19 | agitated | 3 |
| 2023-12-20 | 2023-12-28 | settling | 7 |
| 2023-12-29 | 2024-01-16 | agitated | 13 |
| 2024-01-17 | 2024-01-23 | settling | 5 |
| 2024-01-24 | 2024-02-16 | carry | 18 |
| 2024-02-19 | 2024-02-20 | rising | 2 |
| 2024-02-21 | 2024-02-29 | carry | 7 |
| 2024-03-01 | 2024-03-26 | agitated | 18 |
| 2024-03-27 | 2024-04-01 | rising | 4 |
| 2024-04-02 | 2024-04-23 | agitated | 16 |
| 2024-04-24 | 2024-05-08 | carry | 11 |
| 2024-05-09 | 2024-05-20 | agitated | 8 |
| 2024-05-21 | 2024-06-04 | carry | 11 |
| 2024-06-05 | 2024-06-05 | agitated | 1 |
| 2024-06-06 | 2024-06-17 | rising | 8 |
| 2024-06-18 | 2024-06-18 | stressed | 1 |
| 2024-06-19 | 2024-06-21 | agitated | 3 |
| 2024-06-24 | 2024-07-03 | settling | 8 |
| 2024-07-04 | 2024-08-16 | agitated | 32 |
| 2024-08-19 | 2024-08-20 | rising | 2 |
| 2024-08-21 | 2024-08-28 | stressed | 6 |

## 6. Transition matrix with sticky prior (row = from)
| index | carry | rising | agitated | stressed | normalising | settling |
|---|---|---|---|---|---|---|
| carry | 0.950 | 0.020 | 0.031 | 0.000 | 0.000 | 0.000 |
| rising | 0.010 | 0.814 | 0.124 | 0.051 | 0.000 | 0.000 |
| agitated | 0.017 | 0.043 | 0.899 | 0.004 | 0.000 | 0.037 |
| stressed | 0.000 | 0.000 | 0.044 | 0.937 | 0.017 | 0.000 |
| normalising | 0.003 | 0.003 | 0.077 | 0.003 | 0.863 | 0.052 |
| settling | 0.063 | 0.000 | 0.086 | 0.000 | 0.000 | 0.850 |

Raw counts-based matrix (no prior):
| index | carry | rising | agitated | stressed | normalising | settling |
|---|---|---|---|---|---|---|
| carry | 0.950 | 0.020 | 0.031 | 0.000 | 0.000 | 0.000 |
| rising | 0.010 | 0.809 | 0.128 | 0.052 | 0.000 | 0.000 |
| agitated | 0.017 | 0.043 | 0.899 | 0.004 | 0.000 | 0.038 |
| stressed | 0.000 | 0.000 | 0.046 | 0.937 | 0.018 | 0.000 |
| normalising | 0.000 | 0.000 | 0.100 | 0.000 | 0.833 | 0.067 |
| settling | 0.065 | 0.000 | 0.089 | 0.000 | 0.000 | 0.846 |

## 7. What distinguishes each state (top 4 characteristics by |d|)
| state | characteristic | d | state_mean | all_mean |
|---|---|---|---|---|
| agitated | corr_vol_rate2y | 0.198 | 0.008 | -0.037 |
| agitated | spotvol_beta | 0.181 | -1.765 | -6.923 |
| agitated | spotvol_corr | 0.160 | -0.034 | -0.092 |
| agitated | corr_vol_vix | -0.123 | -0.049 | -0.018 |
| carry | term_slope | -0.880 | -0.139 | 0.729 |
| carry | fly_level | -0.715 | 0.305 | 0.401 |
| carry | rr_level | 0.680 | -0.858 | -1.243 |
| carry | volofvol | -0.383 | 0.439 | 0.497 |
| normalising | vrp | -1.622 | -4.034 | 0.029 |
| normalising | volofvol | 0.972 | 0.643 | 0.497 |
| normalising | corr_vol_move | -0.924 | -0.211 | 0.011 |
| normalising | corr_spot_oil | -0.868 | 0.022 | 0.215 |
| rising | rr_level | -0.885 | -1.744 | -1.243 |
| rising | fly_level | 0.787 | 0.506 | 0.401 |
| rising | vrp | 0.738 | 1.878 | 0.029 |
| rising | term_slope | 0.519 | 1.242 | 0.729 |
| settling | vrp | -0.912 | -2.255 | 0.029 |
| settling | volofvol | 0.625 | 0.590 | 0.497 |
| settling | fly_level | -0.493 | 0.335 | 0.401 |
| settling | rr_level | 0.474 | -0.974 | -1.243 |
| stressed | term_slope | 1.697 | 2.405 | 0.729 |
| stressed | fly_level | 1.619 | 0.618 | 0.401 |
| stressed | rr_level | -1.388 | -2.029 | -1.243 |
| stressed | jump_freq | 0.879 | 0.057 | 0.021 |

## 8. Sub-state discovery (kept only if stability >= 0.7 and every cluster >= 5 episodes)
| state | k_chosen | best_stability | names |
|---|---|---|---|
| agitated | 2 | 0.907 | agitated-2 · spotvol_corr ↓ · spotvol_beta ↓, agitated-1 · spotvol_corr ↑ · spotvol_beta ↑ |
| carry | 1 | 0.680 | carry |
| normalising | 1 |  | normalising |
| rising | 1 | 0.573 | rising |
| settling | 1 | 0.597 | settling |
| stressed | 1 | 0.596 | stressed |

## 9. Tag cells
| tag | cell | episodes | kept | days |
|---|---|---|---|---|
| corr_sign | agitated·neg | 60 | True | 372 |
| corr_sign | stressed·neg | 14 | True | 138 |
| corr_sign | normalising·pos | 1 | False | 5 |
| corr_sign | normalising·neg | 2 | False | 9 |
| corr_sign | carry·pos | 10 | True | 89 |
| corr_sign | settling·pos | 9 | True | 43 |
| corr_sign | settling·neg | 20 | True | 90 |
| corr_sign | rising·pos | 14 | True | 43 |
| corr_sign | stressed·pos | 6 | True | 49 |
| corr_sign | agitated·pos | 24 | True | 285 |
| corr_sign | carry·neg | 26 | True | 326 |
| corr_sign | rising·neg | 25 | True | 110 |
| pinned | stressed·pinned | 2 | False | 3 |
| pinned | carry·pinned | 4 | False | 7 |
| pinned | agitated·pinned | 1 | False | 2 |
| pinned | rising·pinned | 2 | False | 8 |

