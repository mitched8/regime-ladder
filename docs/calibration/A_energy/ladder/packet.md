# Review packet · stage `ladder` · calibration A_energy (synthetic)

| key | value |
|---|---|
| pair | EURUSD |
| data | 2015-01-01 to 2024-08-28 (2520 days) |
| finder / leading target | expected forward 5d straddle P&L given the true state path (synthetic, noiseless) |
| git commit | d87fd28 |
| config hash | 3667c35aac |
| generated | 2026-10-04 15:09 |

## Definitions
- Units: P&L is cash per unit of standard notional (per unit of vega at inception once D1 is closed). Horizons h are trading days from entry, capped at the option tenor.
- Point-in-time: every feature, label and statistic stamped on date t uses only data available at the close of t. Labels are assigned from trailing data; forward quantities appear only as targets.
- Costs are excluded everywhere by design.
- States: carry, rising, agitated, stressed, normalising, settling (+ extreme if it has >= 5 episodes, else merged into stressed). Stress rank low to high (used for 'moves up' and by the transition tilt): carry < settling < rising < agitated < normalising < stressed < extreme.
- EV(h | state at entry) = mean of cumulative P&L from entry to the close of entry+h, over all entries whose entry-date label is that state. No daily rate is rolled forward; transitions after entry are inside the outcome.
- Interval: 90%, the wider of a block bootstrap and an episode-cluster bootstrap. `mean_shrunk` shrinks toward the ALL row with strength kappa. `n_eff` corrects for overlapping horizons; `episodes` counts independent runs of the entry state.
- Gate 2: in >= 75% of (archetype, tenor, h<=10) cells the best-minus-worst state spread exceeds 2 x the larger SE. Gate 3: out of sample (walk-forward), state ladder beats unconditional mean at every h<=10, positive rank IC, calibration slope in 0.5-1.5.

## 1. Gates
Gate 2: {"fraction_separated": 1.0, "pass": true}

Gate 3: {"improvement_min": 0.01829650138428296, "rank_ic_min": 0.10141801185213595, "calib_slope_range": [0.9424187682674636, 1.6503227441901023], "pass": false}

## 2. Ladder at h = 1, 5, 10
| archetype | tenor_days | regime | h | mean | mean_shrunk | ci_lo | ci_hi | p_profit | es05 | n_eff | episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fly_10d | 21 | ALL | 1 | -0.396 | -0.396 | -0.531 | -0.260 | 0.486 | -9.780 | 2499.000 | 250 |
| fly_10d | 21 | ALL | 5 | -1.998 | -1.998 | -2.788 | -1.208 | 0.465 | -28.915 | 499.800 | 250 |
| fly_10d | 21 | ALL | 10 | -3.986 | -3.986 | -5.670 | -2.303 | 0.447 | -48.307 | 249.900 | 250 |
| fly_10d | 21 | agitated | 1 | -0.331 | -0.332 | -0.538 | -0.124 | 0.468 | -8.187 | 944.000 | 97 |
| fly_10d | 21 | agitated | 5 | -2.068 | -2.061 | -2.957 | -1.179 | 0.415 | -21.512 | 188.800 | 97 |
| fly_10d | 21 | agitated | 10 | -4.300 | -4.246 | -6.070 | -2.530 | 0.381 | -37.503 | 94.400 | 97 |
| fly_10d | 21 | carry | 1 | 0.189 | 0.173 | -0.027 | 0.404 | 0.554 | -5.911 | 713.000 | 36 |
| fly_10d | 21 | carry | 5 | 0.997 | 0.629 | -0.194 | 2.188 | 0.590 | -16.166 | 142.600 | 36 |
| fly_10d | 21 | carry | 10 | 1.178 | 0.047 | -1.245 | 3.601 | 0.609 | -26.491 | 71.300 | 36 |
| fly_10d | 21 | normalising | 1 | 1.617 | 0.812 | 0.410 | 2.823 | 0.767 | -3.730 | 30.000 | 5 |
| fly_10d | 21 | normalising | 5 | 7.476 | 0.188 | 5.147 | 9.805 | 0.933 | -1.391 | 6.000 | 5 |
| fly_10d | 21 | normalising | 10 | 13.105 | -1.757 | 8.731 | 17.479 | 0.933 | -0.587 | 3.000 | 5 |
| fly_10d | 21 | rising | 1 | -1.023 | -0.982 | -1.537 | -0.509 | 0.437 | -10.520 | 286.000 | 55 |
| fly_10d | 21 | rising | 5 | -4.459 | -3.821 | -6.544 | -2.374 | 0.430 | -31.008 | 57.200 | 55 |
| fly_10d | 21 | rising | 10 | -6.848 | -5.670 | -10.681 | -3.014 | 0.360 | -47.173 | 28.600 | 55 |
| fly_10d | 21 | settling | 1 | -0.202 | -0.217 | -0.545 | 0.141 | 0.496 | -6.978 | 246.000 | 38 |
| fly_10d | 21 | settling | 5 | -1.611 | -1.723 | -3.168 | -0.054 | 0.484 | -20.397 | 49.200 | 38 |
| fly_10d | 21 | settling | 10 | -5.096 | -4.598 | -8.517 | -1.674 | 0.443 | -39.148 | 24.600 | 38 |
| fly_10d | 21 | stressed | 1 | -1.846 | -1.749 | -2.894 | -0.798 | 0.386 | -14.468 | 280.000 | 19 |
| fly_10d | 21 | stressed | 5 | -8.229 | -6.589 | -13.847 | -2.612 | 0.286 | -55.493 | 56.000 | 19 |
| fly_10d | 21 | stressed | 10 | -14.013 | -9.835 | -24.313 | -3.712 | 0.296 | -101.930 | 28.000 | 19 |
| rr_25d | 21 | ALL | 1 | -0.440 | -0.440 | -0.566 | -0.314 | 0.466 | -9.515 | 2499.000 | 250 |
| rr_25d | 21 | ALL | 5 | -2.293 | -2.293 | -3.008 | -1.578 | 0.423 | -26.576 | 499.800 | 250 |
| rr_25d | 21 | ALL | 10 | -4.730 | -4.730 | -6.209 | -3.251 | 0.401 | -43.132 | 249.900 | 250 |
| rr_25d | 21 | agitated | 1 | -0.317 | -0.320 | -0.531 | -0.104 | 0.465 | -7.986 | 944.000 | 97 |
| rr_25d | 21 | agitated | 5 | -2.095 | -2.114 | -3.173 | -1.017 | 0.432 | -23.924 | 188.800 | 97 |
| rr_25d | 21 | agitated | 10 | -5.046 | -4.991 | -7.096 | -2.996 | 0.373 | -37.271 | 94.400 | 97 |
| rr_25d | 21 | carry | 1 | 0.014 | 0.001 | -0.149 | 0.177 | 0.523 | -5.949 | 713.000 | 36 |
| rr_25d | 21 | carry | 5 | -0.129 | -0.395 | -0.809 | 0.551 | 0.527 | -14.503 | 142.600 | 36 |
| rr_25d | 21 | carry | 10 | -0.547 | -1.463 | -1.966 | 0.872 | 0.502 | -23.121 | 71.300 | 36 |
| rr_25d | 21 | normalising | 1 | 0.372 | 0.047 | -0.683 | 1.427 | 0.500 | -6.782 | 30.000 | 5 |
| rr_25d | 21 | normalising | 5 | 1.854 | -1.336 | -1.893 | 5.601 | 0.500 | -9.314 | 6.000 | 5 |
| rr_25d | 21 | normalising | 10 | 1.338 | -3.938 | -4.065 | 6.742 | 0.600 | -22.227 | 3.000 | 5 |
| rr_25d | 21 | rising | 1 | -1.132 | -1.086 | -1.619 | -0.645 | 0.371 | -10.602 | 286.000 | 55 |
| rr_25d | 21 | rising | 5 | -4.560 | -3.972 | -6.089 | -3.030 | 0.255 | -23.593 | 57.200 | 55 |
| rr_25d | 21 | rising | 10 | -7.542 | -6.385 | -10.533 | -4.551 | 0.315 | -35.203 | 28.600 | 55 |
| rr_25d | 21 | settling | 1 | -0.414 | -0.416 | -0.744 | -0.084 | 0.443 | -6.452 | 246.000 | 38 |
| rr_25d | 21 | settling | 5 | -2.636 | -2.537 | -3.933 | -1.339 | 0.366 | -16.991 | 49.200 | 38 |
| rr_25d | 21 | settling | 10 | -6.044 | -5.455 | -8.534 | -3.555 | 0.333 | -31.056 | 24.600 | 38 |
| rr_25d | 21 | stressed | 1 | -1.411 | -1.346 | -2.108 | -0.715 | 0.436 | -17.074 | 280.000 | 19 |
| rr_25d | 21 | stressed | 5 | -6.300 | -5.246 | -9.974 | -2.626 | 0.343 | -49.359 | 56.000 | 19 |
| rr_25d | 21 | stressed | 10 | -10.939 | -8.352 | -18.613 | -3.265 | 0.361 | -72.444 | 28.000 | 19 |
| straddle_atm | 21 | ALL | 1 | -0.034 | -0.034 | -0.161 | 0.093 | 0.460 | -7.772 | 2499.000 | 250 |
| straddle_atm | 21 | ALL | 5 | -0.220 | -0.220 | -0.957 | 0.518 | 0.437 | -18.032 | 499.800 | 250 |
| straddle_atm | 21 | ALL | 10 | -0.495 | -0.495 | -2.083 | 1.093 | 0.434 | -26.441 | 249.900 | 250 |
| straddle_atm | 21 | agitated | 1 | -0.029 | -0.029 | -0.215 | 0.158 | 0.485 | -7.456 | 944.000 | 97 |
| straddle_atm | 21 | agitated | 5 | 0.300 | 0.250 | -0.514 | 1.113 | 0.513 | -17.757 | 188.800 | 97 |
| straddle_atm | 21 | agitated | 10 | 0.448 | 0.283 | -1.180 | 2.075 | 0.523 | -25.960 | 94.400 | 97 |
| straddle_atm | 21 | carry | 1 | -0.711 | -0.693 | -0.894 | -0.528 | 0.363 | -5.595 | 713.000 | 36 |
| straddle_atm | 21 | carry | 5 | -3.603 | -3.187 | -4.751 | -2.454 | 0.244 | -15.848 | 142.600 | 36 |
| straddle_atm | 21 | carry | 10 | -6.555 | -5.227 | -8.839 | -4.271 | 0.209 | -24.169 | 71.300 | 36 |
| straddle_atm | 21 | normalising | 1 | -0.966 | -0.593 | -2.050 | 0.118 | 0.300 | -8.698 | 30.000 | 5 |
| straddle_atm | 21 | normalising | 5 | -6.072 | -1.570 | -8.972 | -3.172 | 0.267 | -18.859 | 6.000 | 5 |
| straddle_atm | 21 | normalising | 10 | -11.931 | -1.987 | -19.330 | -4.532 | 0.233 | -31.148 | 3.000 | 5 |
| straddle_atm | 21 | rising | 1 | 0.637 | 0.593 | 0.186 | 1.088 | 0.545 | -8.007 | 286.000 | 55 |
| straddle_atm | 21 | rising | 5 | 1.280 | 0.891 | -0.579 | 3.138 | 0.521 | -17.681 | 57.200 | 55 |
| straddle_atm | 21 | rising | 10 | 1.322 | 0.574 | -1.144 | 3.789 | 0.510 | -22.704 | 28.600 | 55 |
| straddle_atm | 21 | settling | 1 | -0.448 | -0.417 | -0.867 | -0.029 | 0.423 | -6.775 | 246.000 | 38 |
| straddle_atm | 21 | settling | 5 | -2.000 | -1.485 | -3.754 | -0.245 | 0.382 | -18.803 | 49.200 | 38 |
| straddle_atm | 21 | settling | 10 | -2.017 | -1.334 | -5.086 | 1.053 | 0.390 | -23.437 | 24.600 | 38 |
| straddle_atm | 21 | stressed | 1 | 1.450 | 1.351 | 0.459 | 2.440 | 0.586 | -11.176 | 280.000 | 19 |
| straddle_atm | 21 | stressed | 5 | 7.304 | 5.324 | 2.028 | 12.579 | 0.650 | -20.771 | 56.000 | 19 |
| straddle_atm | 21 | stressed | 10 | 12.461 | 7.063 | 3.114 | 21.809 | 0.689 | -34.006 | 28.000 | 19 |

## 3. Thin cells (episodes < 5 or n_eff < 30)
| archetype | tenor_days | regime | h | n_eff | episodes |
|---|---|---|---|---|---|
| fly_10d | 21 | normalising | 3 | 10.000 | 5 |
| fly_10d | 21 | normalising | 5 | 6.000 | 5 |
| fly_10d | 21 | normalising | 10 | 3.000 | 5 |
| fly_10d | 21 | normalising | 20 | 1.500 | 5 |
| fly_10d | 21 | normalising | 21 | 1.400 | 5 |
| fly_10d | 21 | rising | 10 | 28.600 | 55 |
| fly_10d | 21 | rising | 20 | 14.300 | 55 |
| fly_10d | 21 | rising | 21 | 13.600 | 55 |
| fly_10d | 21 | settling | 10 | 24.600 | 38 |
| fly_10d | 21 | settling | 20 | 12.300 | 38 |
| fly_10d | 21 | settling | 21 | 11.700 | 38 |
| fly_10d | 21 | stressed | 10 | 28.000 | 19 |
| fly_10d | 21 | stressed | 20 | 14.000 | 19 |
| fly_10d | 21 | stressed | 21 | 13.300 | 19 |
| rr_25d | 21 | normalising | 3 | 10.000 | 5 |
| rr_25d | 21 | normalising | 5 | 6.000 | 5 |
| rr_25d | 21 | normalising | 10 | 3.000 | 5 |
| rr_25d | 21 | normalising | 20 | 1.500 | 5 |
| rr_25d | 21 | normalising | 21 | 1.400 | 5 |
| rr_25d | 21 | rising | 10 | 28.600 | 55 |
| rr_25d | 21 | rising | 20 | 14.300 | 55 |
| rr_25d | 21 | rising | 21 | 13.600 | 55 |
| rr_25d | 21 | settling | 10 | 24.600 | 38 |
| rr_25d | 21 | settling | 20 | 12.300 | 38 |
| rr_25d | 21 | settling | 21 | 11.700 | 38 |
| rr_25d | 21 | stressed | 10 | 28.000 | 19 |
| rr_25d | 21 | stressed | 20 | 14.000 | 19 |
| rr_25d | 21 | stressed | 21 | 13.300 | 19 |
| straddle_atm | 21 | normalising | 3 | 10.000 | 5 |
| straddle_atm | 21 | normalising | 5 | 6.000 | 5 |
| straddle_atm | 21 | normalising | 10 | 3.000 | 5 |
| straddle_atm | 21 | normalising | 20 | 1.500 | 5 |
| straddle_atm | 21 | normalising | 21 | 1.400 | 5 |
| straddle_atm | 21 | rising | 10 | 28.600 | 55 |
| straddle_atm | 21 | rising | 20 | 14.300 | 55 |
| straddle_atm | 21 | rising | 21 | 13.600 | 55 |
| straddle_atm | 21 | settling | 10 | 24.600 | 38 |
| straddle_atm | 21 | settling | 20 | 12.300 | 38 |
| straddle_atm | 21 | settling | 21 | 11.700 | 38 |
| straddle_atm | 21 | stressed | 10 | 28.000 | 19 |
| straddle_atm | 21 | stressed | 20 | 14.000 | 19 |
| straddle_atm | 21 | stressed | 21 | 13.300 | 19 |

## 4. Cells whose mean changes sign across horizons
fly_10d/21/carry, rr_25d/21/carry, rr_25d/21/normalising, straddle_atm/21/agitated

## 5. Walk-forward: state ladder vs unconditional (out of sample)
| pair | archetype | tenor_days | h | n_test | mse_uncond | mse_regime | improvement | rank_ic | calib_slope |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD | fly_10d | 21 | 1 | 1499 | 14.981 | 14.620 | 0.024 | 0.157 | 0.942 |
| EURUSD | fly_10d | 21 | 3 | 1499 | 55.062 | 52.022 | 0.055 | 0.281 | 0.984 |
| EURUSD | fly_10d | 21 | 5 | 1499 | 109.401 | 102.030 | 0.067 | 0.325 | 0.966 |
| EURUSD | fly_10d | 21 | 10 | 1499 | 304.237 | 285.372 | 0.062 | 0.350 | 0.949 |
| EURUSD | fly_10d | 21 | 20 | 1499 | 812.904 | 778.274 | 0.043 | 0.261 | 0.837 |
| EURUSD | fly_10d | 21 | 21 | 1499 | 866.764 | 830.060 | 0.042 | 0.250 | 0.830 |
| EURUSD | rr_25d | 21 | 1 | 1499 | 14.777 | 14.507 | 0.018 | 0.101 | 1.518 |
| EURUSD | rr_25d | 21 | 3 | 1499 | 55.994 | 53.843 | 0.038 | 0.151 | 1.650 |
| EURUSD | rr_25d | 21 | 5 | 1499 | 105.286 | 100.286 | 0.047 | 0.171 | 1.618 |
| EURUSD | rr_25d | 21 | 10 | 1499 | 263.364 | 250.365 | 0.049 | 0.142 | 1.416 |
| EURUSD | rr_25d | 21 | 20 | 1499 | 619.304 | 586.831 | 0.052 | 0.139 | 1.171 |
| EURUSD | rr_25d | 21 | 21 | 1499 | 658.691 | 624.170 | 0.052 | 0.141 | 1.151 |
| EURUSD | straddle_atm | 21 | 1 | 1499 | 15.993 | 15.493 | 0.031 | 0.144 | 1.171 |
| EURUSD | straddle_atm | 21 | 3 | 1499 | 60.467 | 56.222 | 0.070 | 0.224 | 1.198 |
| EURUSD | straddle_atm | 21 | 5 | 1499 | 115.090 | 102.910 | 0.106 | 0.281 | 1.270 |
| EURUSD | straddle_atm | 21 | 10 | 1499 | 283.532 | 247.534 | 0.127 | 0.326 | 1.326 |
| EURUSD | straddle_atm | 21 | 20 | 1499 | 726.331 | 621.308 | 0.145 | 0.364 | 1.346 |
| EURUSD | straddle_atm | 21 | 21 | 1499 | 778.134 | 664.491 | 0.146 | 0.368 | 1.333 |

