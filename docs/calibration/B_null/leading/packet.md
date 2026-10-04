# Review packet · stage `leading` · calibration B_null (synthetic)

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
- Leading features (0..100; all trailing percentiles except stored_energy, which is a product of two percentiles divided by 100, so its typical level is ~20-25, not 50):
  - gap_pct: how unusual today's distance of spot from its 120-day anchor is, in implied-vol units.
  - complacency: realised under implied, low vol-of-vol, cheap front end, combined.
  - stored_energy = gap_pct x complacency / 100 (a stretched price held by a complacent market; the hypothesis is the product).
  - jump_cluster_pct: clustering of large returns. coupling_shift_pct: change in short vs long spot-vol coupling. coherence_pct: cross-pair co-movement.
  - Any other name is a desk series supplied through the adapter; it is claimed trailing-only.
- Screen, one feature at a time, against a baseline that already knows the named state AND the continuous composite, 4 walk-forward folds after a 40% initial training block:
  - (1) outcome test: `delta_r2` = pooled out-of-sample R² gain on the forward outcome; `folds_r2_up` = folds in which the gain is > 0; `n` = days with feature, label, composite and outcome; `r2_base` = the baseline's own OOS R² on that feature's sample (samples differ by feature because warm-ups differ).
  - (2) transition test: `transition_gain` = out-of-sample 5-step transition log-likelihood gain per transition (nats) through a tilt, with the composite already in the tilt; `folds_transition_up` = folds with gain > 0; `n_transitions` = days with feature and label (no outcome needed, so it can exceed `n`). `beta_mean` = the feature's tilt coefficient per standard deviation, averaged over folds.
  - Retain only if delta_r2 >= 0.005 AND folds_r2_up >= 3 AND transition_gain >= 0.002 AND folds_transition_up >= 3. `delta_r2_lagged` repeats test (1) with the feature delayed one day; it is noisy at the 0.005 scale and is informative only when far above or below delta_r2.
- Shock score (0..100) = mean of CUSUM pressure on squared surprises (alarm when it reaches 1), BOCD probability of a short run, and recent |surprise|; surprise = return / implied daily vol marked the day before.
- Gate 4: at least one feature retained AND the pressure index of the retained features (their mean, centred at 50, /25) passes test (2) on its own: gain >= 0.002 per transition, > 0 in >= 3/4 folds.
- Pressure index = mean of the RETAINED features, centred at 50 and divided by 25 (0 = normal). If nothing is retained, no pressure index exists and only the constant fan is meaningful.
- High band = stressed and extreme. Transition tilt: P_ij(psi) proportional to P0_ij x exp(beta x psi x (rank_j - rank_i)), with the composite as a second covariate (gamma); likelihood scored k = 5 days ahead. The fan horizon (21 days) is the transition forecast's horizon, not a ladder horizon.

## 1. Gate 4
```
{
 "retained": [],
 "pressure_transition_gain": NaN,
 "pressure_folds_positive": 0,
 "pass": false
}
```
## 2. Screen (one feature at a time vs state + composite baseline)
| feature | n | n_transitions | r2_base | delta_r2 | folds_r2_up | transition_gain | folds_transition_up | beta_mean | retain | delta_r2_lagged |
|---|---|---|---|---|---|---|---|---|---|---|
| coherence_pct | 2306 | 2310 | 0.3731 | 0.0019 | 2 | -0.0004 | 1 | 0.0026 | False | 0.0000 |
| jump_cluster_pct | 2307 | 2311 | 0.3718 | -0.0005 | 1 | -0.0017 | 1 | 0.0744 | False | -0.0015 |
| gap_pct | 2267 | 2271 | 0.3778 | -0.0017 | 2 | 0.0003 | 2 | 0.0102 | False | -0.0021 |
| stored_energy | 2267 | 2271 | 0.3778 | -0.0025 | 2 | 0.0009 | 3 | 0.0586 | False | -0.0027 |
| coupling_shift_pct | 2201 | 2205 | 0.3903 | -0.0026 | 1 | 0.0005 | 2 | 0.0473 | False | -0.0022 |
| complacency | 2306 | 2310 | 0.3731 | -0.0065 | 1 | -0.0002 | 2 | 0.0412 | False | -0.0063 |
| desk_flow | 2394 | 2459 | 0.3467 | -0.0093 | 1 | -0.0014 | 2 | 0.0403 | False | -0.0078 |

## 3. Feature distributions
| index | count | mean | std | min | 5% | 50% | 95% | max | first_valid | last_valid | share_at_mode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | 2272.0 | 26.1 | 20.0 | 0.0 | 1.9 | 22.0 | 68.1 | 88.0 | 2015-12-15 | 2024-08-28 | 0.0 |
| gap_pct | 2272.0 | 48.7 | 29.2 | 0.0 | 4.9 | 47.8 | 97.4 | 100.0 | 2015-12-15 | 2024-08-28 | 0.0 |
| complacency | 2311.0 | 51.5 | 17.7 | 8.4 | 23.1 | 50.8 | 83.4 | 93.8 | 2015-10-21 | 2024-08-28 | 0.0 |
| jump_cluster_pct | 2312.0 | 34.5 | 31.0 | 0.0 | 0.0 | 29.8 | 90.5 | 100.0 | 2015-10-20 | 2024-08-28 | 0.3 |
| coupling_shift_pct | 2206.0 | 45.3 | 27.4 | 0.0 | 4.1 | 44.5 | 91.7 | 100.0 | 2016-03-16 | 2024-08-28 | 0.0 |
| coherence_pct | 2311.0 | 49.8 | 29.3 | 0.0 | 4.0 | 49.7 | 94.1 | 100.0 | 2015-10-21 | 2024-08-28 | 0.0 |
| desk_flow | 2460.0 | 52.3 | 6.4 | 50.0 | 50.0 | 50.0 | 66.1 | 96.6 | 2015-03-26 | 2024-08-28 | 0.8 |

## 4. Mean feature value by state (0..100)
| index | agitated | carry | normalising | rising | settling | stressed |
|---|---|---|---|---|---|---|
| stored_energy | 16.5 | 38.9 | 17.7 | 23.3 | 27.9 | 18.4 |
| gap_pct | 37.5 | 57.0 | 64.5 | 43.2 | 61.1 | 46.2 |
| complacency | 43.3 | 67.4 | 29.2 | 52.4 | 46.8 | 37.4 |
| jump_cluster_pct | 39.8 | 27.3 | 21.5 | 45.5 | 35.9 | 30.2 |
| coupling_shift_pct | 50.8 | 41.8 | 30.5 | 41.8 | 43.7 | 47.2 |
| coherence_pct | 56.3 | 47.4 | 48.1 | 53.5 | 40.3 | 46.6 |
| desk_flow | 50.1 | 54.3 | 53.0 | 50.2 | 56.1 | 50.0 |

## 5. Correlations (features and the composite)
| index | stored_energy | gap_pct | complacency | jump_cluster_pct | coupling_shift_pct | coherence_pct | desk_flow | composite |
|---|---|---|---|---|---|---|---|---|
| stored_energy | 1.00 | 0.86 | 0.59 | -0.24 | -0.20 | -0.15 | 0.59 | -0.46 |
| gap_pct | 0.86 | 1.00 | 0.19 | -0.12 | -0.22 | -0.14 | 0.56 | -0.22 |
| complacency | 0.59 | 0.19 | 1.00 | -0.44 | -0.12 | -0.12 | 0.22 | -0.67 |
| jump_cluster_pct | -0.24 | -0.12 | -0.44 | 1.00 | 0.08 | 0.28 | -0.05 | 0.08 |
| coupling_shift_pct | -0.20 | -0.22 | -0.12 | 0.08 | 1.00 | 0.09 | -0.11 | 0.06 |
| coherence_pct | -0.15 | -0.14 | -0.12 | 0.28 | 0.09 | 1.00 | -0.05 | 0.05 |
| desk_flow | 0.59 | 0.56 | 0.22 | -0.05 | -0.11 | -0.05 | 1.00 | -0.33 |
| composite | -0.46 | -0.22 | -0.67 | 0.08 | 0.06 | 0.05 | -0.33 | 1.00 |

## 6. Lead profile: mean value in the days before ANY move up the state ranking (e.g. carry to rising), before moves down, and on all days. Moves skip ranks (a rising-to-stressed move counts once), so up and down counts need not match
| feature | window_days | all_days | before_escalation | before_de_escalation | n_escalations | n_de_escalations |
|---|---|---|---|---|---|---|
| stored_energy | 5 | 26.1 | 26.3 | 21.7 | 70.0 | 81.0 |
| stored_energy | 20 | 26.1 | 29.7 | 21.6 | 70.0 | 81.0 |
| gap_pct | 5 | 48.7 | 47.5 | 50.4 | 70.0 | 81.0 |
| gap_pct | 20 | 48.7 | 51.7 | 49.6 | 70.0 | 81.0 |
| complacency | 5 | 51.5 | 53.1 | 43.7 | 70.0 | 81.0 |
| complacency | 20 | 51.5 | 56.5 | 43.2 | 70.0 | 81.0 |
| jump_cluster_pct | 5 | 34.5 | 40.6 | 32.7 | 70.0 | 81.0 |
| jump_cluster_pct | 20 | 34.5 | 35.5 | 33.9 | 70.0 | 81.0 |
| coupling_shift_pct | 5 | 45.3 | 43.7 | 41.8 | 70.0 | 81.0 |
| coupling_shift_pct | 20 | 45.3 | 40.9 | 42.1 | 70.0 | 81.0 |
| coherence_pct | 5 | 49.8 | 52.9 | 44.1 | 70.0 | 81.0 |
| coherence_pct | 20 | 49.8 | 51.8 | 49.1 | 70.0 | 81.0 |
| desk_flow | 5 | 52.3 | 50.9 | 52.3 | 70.0 | 81.0 |
| desk_flow | 20 | 52.3 | 51.6 | 51.3 | 70.0 | 81.0 |

## 7. Every move into the high band (19): values before it
| date | from | to | stored_energy_t-20 | stored_energy_t-5 | stored_energy_t-1 | desk_flow_t-5 | desk_flow_t-1 | shock_max_20d | days_since_cusum_alarm |
|---|---|---|---|---|---|---|---|---|---|
| 2015-11-06 | rising | stressed |  |  |  | 50.0 | 50.0 | 68.9 | 172 |
| 2016-06-07 | rising | stressed | 15.9 | 7.7 | 6.4 | 50.0 | 50.0 | 58.0 | 386 |
| 2018-09-27 | rising | stressed | 66.0 | 45.7 | 39.5 | 50.0 | 50.0 | 33.0 | 1228 |
| 2018-11-28 | agitated | stressed | 24.9 | 46.7 | 42.9 | 50.0 | 50.0 | 60.0 | 1290 |
| 2019-09-06 | agitated | stressed | 0.5 | 22.8 | 26.5 | 50.0 | 50.0 | 35.0 | 1572 |
| 2020-03-16 | agitated | stressed | 31.4 | 28.6 | 31.4 | 50.0 | 50.0 | 35.6 | 1764 |
| 2020-03-27 | agitated | stressed | 36.0 | 48.5 | 44.4 | 50.0 | 50.0 | 44.8 | 1775 |
| 2020-06-15 | agitated | stressed | 34.4 | 5.5 | 7.6 | 50.0 | 50.0 | 55.1 | 1855 |
| 2020-10-29 | agitated | stressed | 26.8 | 21.5 | 18.2 | 50.0 | 50.0 | 66.2 | 1991 |
| 2020-12-17 | rising | stressed | 21.9 | 11.7 | 6.2 | 50.0 | 50.0 | 49.7 | 2040 |
| 2021-01-28 | rising | stressed | 2.2 | 2.9 | 9.1 | 50.0 | 50.0 | 30.9 | 2082 |
| 2021-05-31 | rising | stressed | 16.5 | 35.1 | 29.2 | 50.0 | 50.0 | 33.5 | 2205 |
| 2022-04-28 | agitated | stressed | 9.8 | 28.5 | 27.7 | 50.0 | 50.0 | 37.9 | 2537 |
| 2022-05-24 | agitated | stressed | 41.1 | 17.3 | 1.4 | 50.0 | 50.0 | 52.6 | 2563 |
| 2022-05-26 | agitated | stressed | 22.5 | 16.7 | 0.2 | 50.0 | 50.0 | 53.2 | 2565 |
| 2022-08-19 | agitated | stressed | 34.0 | 13.7 | 13.4 | 50.0 | 50.0 | 35.8 | 2650 |
| 2023-04-05 | rising | stressed | 39.4 | 25.0 | 31.6 | 50.0 | 50.0 | 52.6 | 2879 |
| 2023-06-15 | agitated | stressed | 36.3 | 46.6 | 43.7 | 50.0 | 50.0 | 29.3 | 2950 |
| 2024-05-21 | rising | stressed | 18.8 | 7.9 | 5.1 | 50.0 | 50.0 | 46.2 | 3291 |

## 8. Shock detector against the labels
CUSUM alarms: 1; followed by a move into the high band within 30 calendar days: 0; moves into the high band preceded by an alarm within 30 days: 0 of 19.

Surprise: mean +0.013, sd 1.008 (should be ~1 if implied matches realised on average), 1%/99% -2.41/+2.27; max CUSUM pressure 1.0000 on 2015-05-18 (alarm when it reaches 0.999).

Shock score lead profile (5 days before):
| all | before_up | before_up_n | before_down | before_down_n |
|---|---|---|---|---|
| 29.52 | 32.35 | 70 | 27.54 | 81 |

Alarm dates (first 60): 2015-05-18

## 8b. Alignment check: Spearman correlation of each feature on day t with trailing 1-week realised vol on day t+k (k < 0 past, k > 0 future)
| index | k=-10 | k=-5 | k=-1 | k=+0 | k=+1 | k=+2 | k=+5 | k=+10 |
|---|---|---|---|---|---|---|---|---|
| stored_energy | -0.27 | -0.31 | -0.32 | -0.32 | -0.29 | -0.27 | -0.21 | -0.14 |
| gap_pct | -0.13 | -0.17 | -0.17 | -0.17 | -0.16 | -0.15 | -0.15 | -0.12 |
| complacency | -0.45 | -0.45 | -0.46 | -0.46 | -0.41 | -0.37 | -0.23 | -0.16 |
| jump_cluster_pct | 0.21 | 0.25 | 0.29 | 0.30 | 0.26 | 0.21 | 0.08 | 0.01 |
| coupling_shift_pct | 0.03 | 0.04 | 0.08 | 0.08 | 0.07 | 0.06 | 0.05 | 0.06 |
| coherence_pct | 0.11 | 0.10 | 0.13 | 0.13 | 0.11 | 0.09 | 0.02 | 0.00 |
| desk_flow | -0.14 | -0.22 | -0.26 | -0.28 | -0.28 | -0.27 | -0.25 | -0.20 |

## 9. Fan from today's state (agitated, 2024-08-28): constant matrix only (nothing retained, so no pressure index)
| h | P(high band) constant |
|---|---|
| 5 | 0.068 |
| 10 | 0.106 |
| 21 | 0.135 |

