# Review packet · stage `leading` · calibration A_energy (synthetic)

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
 "retained": [
  "desk_flow"
 ],
 "pressure_transition_gain": 0.004594130996537837,
 "pressure_folds_positive": 3,
 "pass": true
}
```
## 2. Screen (one feature at a time vs state + composite baseline)
| feature | n | n_transitions | r2_base | delta_r2 | folds_r2_up | transition_gain | folds_transition_up | beta_mean | retain | delta_r2_lagged |
|---|---|---|---|---|---|---|---|---|---|---|
| desk_flow | 2394 | 2459 | 0.3242 | 0.0139 | 4 | 0.0053 | 3 | 0.1230 | True | 0.0183 |
| gap_pct | 2267 | 2271 | 0.3510 | -0.0025 | 1 | -0.0010 | 1 | -0.0236 | False | 0.0004 |
| coupling_shift_pct | 2201 | 2205 | 0.3505 | -0.0033 | 3 | 0.0007 | 3 | 0.0704 | False | -0.0010 |
| stored_energy | 2267 | 2271 | 0.3510 | -0.0041 | 2 | -0.0004 | 2 | 0.0365 | False | -0.0035 |
| coherence_pct | 2306 | 2310 | 0.3419 | -0.0064 | 1 | -0.0035 | 1 | -0.1066 | False | -0.0065 |
| jump_cluster_pct | 2307 | 2311 | 0.3424 | -0.0071 | 1 | 0.0011 | 2 | -0.0958 | False | -0.0095 |
| complacency | 2306 | 2310 | 0.3419 | -0.0130 | 1 | -0.0098 | 1 | 0.0966 | False | -0.0157 |

## 3. Feature distributions
| index | count | mean | std | min | 5% | 50% | 95% | max | first_valid | last_valid | share_at_mode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | 2272.0 | 21.2 | 15.6 | 0.0 | 1.4 | 18.6 | 49.5 | 79.1 | 2015-12-15 | 2024-08-28 | 0.0 |
| gap_pct | 2272.0 | 44.3 | 29.8 | 0.0 | 3.0 | 41.0 | 96.0 | 100.0 | 2015-12-15 | 2024-08-28 | 0.0 |
| complacency | 2311.0 | 50.1 | 17.3 | 2.1 | 24.0 | 49.7 | 79.4 | 97.2 | 2015-10-21 | 2024-08-28 | 0.0 |
| jump_cluster_pct | 2312.0 | 30.9 | 30.2 | 0.0 | 0.0 | 25.0 | 90.9 | 100.0 | 2015-10-20 | 2024-08-28 | 0.3 |
| coupling_shift_pct | 2206.0 | 46.8 | 27.9 | 0.0 | 4.9 | 44.9 | 94.1 | 100.0 | 2016-03-16 | 2024-08-28 | 0.0 |
| coherence_pct | 2311.0 | 48.5 | 29.0 | 0.0 | 4.1 | 46.8 | 94.3 | 100.0 | 2015-10-21 | 2024-08-28 | 0.0 |
| desk_flow | 2460.0 | 50.8 | 3.8 | 50.0 | 50.0 | 50.0 | 54.3 | 100.0 | 2015-03-26 | 2024-08-28 | 0.9 |

## 4. Mean feature value by state (0..100)
| index | agitated | carry | normalising | rising | settling | stressed |
|---|---|---|---|---|---|---|
| stored_energy | 21.0 | 25.2 | 15.7 | 20.5 | 22.7 | 14.9 |
| gap_pct | 43.7 | 39.0 | 65.7 | 46.6 | 55.8 | 40.8 |
| complacency | 48.6 | 67.5 | 28.4 | 45.6 | 41.0 | 38.4 |
| jump_cluster_pct | 30.1 | 33.6 | 33.5 | 30.8 | 33.8 | 26.4 |
| coupling_shift_pct | 53.0 | 44.4 | 21.0 | 44.0 | 43.1 | 38.3 |
| coherence_pct | 46.9 | 48.9 | 48.1 | 53.3 | 55.7 | 42.0 |
| desk_flow | 50.1 | 51.6 | 50.0 | 50.1 | 53.0 | 50.0 |

## 5. Correlations (features and the composite)
| index | stored_energy | gap_pct | complacency | jump_cluster_pct | coupling_shift_pct | coherence_pct | desk_flow | composite |
|---|---|---|---|---|---|---|---|---|
| stored_energy | 1.00 | 0.83 | 0.28 | -0.01 | 0.01 | 0.07 | 0.24 | -0.17 |
| gap_pct | 0.83 | 1.00 | -0.20 | 0.13 | 0.01 | 0.22 | 0.26 | 0.08 |
| complacency | 0.28 | -0.20 | 1.00 | -0.30 | -0.00 | -0.24 | -0.01 | -0.57 |
| jump_cluster_pct | -0.01 | 0.13 | -0.30 | 1.00 | 0.17 | 0.29 | 0.03 | -0.05 |
| coupling_shift_pct | 0.01 | 0.01 | -0.00 | 0.17 | 1.00 | 0.01 | 0.01 | -0.03 |
| coherence_pct | 0.07 | 0.22 | -0.24 | 0.29 | 0.01 | 1.00 | 0.11 | -0.06 |
| desk_flow | 0.24 | 0.26 | -0.01 | 0.03 | 0.01 | 0.11 | 1.00 | -0.18 |
| composite | -0.17 | 0.08 | -0.57 | -0.05 | -0.03 | -0.06 | -0.18 | 1.00 |

## 6. Lead profile: mean value in the days before ANY move up the state ranking (e.g. carry to rising), before moves down, and on all days. Moves skip ranks (a rising-to-stressed move counts once), so up and down counts need not match
| feature | window_days | all_days | before_escalation | before_de_escalation | n_escalations | n_de_escalations |
|---|---|---|---|---|---|---|
| stored_energy | 5 | 21.2 | 24.1 | 22.7 | 114.0 | 135.0 |
| stored_energy | 20 | 21.2 | 23.6 | 22.8 | 114.0 | 135.0 |
| gap_pct | 5 | 44.3 | 51.2 | 50.8 | 114.0 | 135.0 |
| gap_pct | 20 | 44.3 | 50.4 | 51.7 | 114.0 | 135.0 |
| complacency | 5 | 50.1 | 47.8 | 45.0 | 114.0 | 135.0 |
| complacency | 20 | 50.1 | 49.3 | 45.3 | 114.0 | 135.0 |
| jump_cluster_pct | 5 | 30.9 | 30.9 | 32.8 | 114.0 | 135.0 |
| jump_cluster_pct | 20 | 30.9 | 30.2 | 30.3 | 114.0 | 135.0 |
| coupling_shift_pct | 5 | 46.8 | 43.3 | 42.6 | 114.0 | 135.0 |
| coupling_shift_pct | 20 | 46.8 | 42.6 | 41.5 | 114.0 | 135.0 |
| coherence_pct | 5 | 48.5 | 54.0 | 51.7 | 114.0 | 135.0 |
| coherence_pct | 20 | 48.5 | 51.7 | 50.1 | 114.0 | 135.0 |
| desk_flow | 5 | 50.8 | 52.0 | 51.5 | 114.0 | 135.0 |
| desk_flow | 20 | 50.8 | 51.3 | 51.1 | 114.0 | 135.0 |

## 7. Every move into the high band (19): values before it
| date | from | to | stored_energy_t-20 | stored_energy_t-5 | stored_energy_t-1 | desk_flow_t-5 | desk_flow_t-1 | pressure_t-1 | shock_max_20d | days_since_cusum_alarm |
|---|---|---|---|---|---|---|---|---|---|---|
| 2015-12-15 | rising | stressed |  |  |  | 50.0 | 50.0 | 0.0 | 56.4 |  |
| 2016-01-27 | rising | stressed | 1.9 | 4.8 | 4.4 | 50.0 | 50.0 | 0.0 | 35.3 |  |
| 2016-03-30 | rising | stressed | 48.3 | 18.9 | 21.5 | 50.0 | 50.0 | 0.0 | 36.4 |  |
| 2017-03-28 | rising | stressed | 0.3 | 6.6 | 0.0 | 50.0 | 50.0 | 0.0 | 46.4 |  |
| 2017-11-30 | rising | stressed | 1.3 | 4.5 | 11.1 | 50.0 | 50.0 | 0.0 | 81.9 |  |
| 2018-02-16 | rising | stressed | 2.2 | 15.3 | 1.0 | 50.0 | 50.0 | 0.0 | 44.9 |  |
| 2018-05-17 | rising | stressed | 11.4 | 7.9 | 7.3 | 50.0 | 50.0 | 0.0 | 29.7 |  |
| 2018-10-09 | rising | stressed | 34.0 | 45.3 | 43.1 | 50.0 | 50.0 | 0.0 | 33.8 |  |
| 2019-09-20 | rising | stressed | 48.4 | 30.8 | 23.9 | 50.0 | 50.0 | 0.0 | 36.5 |  |
| 2019-12-04 | rising | stressed | 23.2 | 49.5 | 40.6 | 50.0 | 50.0 | 0.0 | 50.4 |  |
| 2020-04-16 | agitated | stressed | 24.7 | 32.5 | 57.8 | 50.0 | 50.0 | 0.0 | 51.8 |  |
| 2021-02-24 | agitated | stressed | 18.6 | 16.7 | 15.3 | 50.0 | 50.0 | 0.0 | 51.9 |  |
| 2022-08-18 | rising | stressed | 24.6 | 16.5 | 20.3 | 50.0 | 50.0 | 0.0 | 53.8 |  |
| 2022-09-19 | agitated | stressed | 15.7 | 36.3 | 25.2 | 50.0 | 50.0 | 0.0 | 58.6 |  |
| 2023-04-17 | rising | stressed | 3.5 | 14.5 | 7.4 | 50.0 | 50.0 | 0.0 | 30.6 |  |
| 2023-05-11 | agitated | stressed | 3.1 | 4.7 | 1.4 | 50.0 | 50.0 | 0.0 | 48.5 |  |
| 2023-09-15 | rising | stressed | 48.8 | 34.4 | 20.4 | 50.0 | 50.0 | 0.0 | 46.9 |  |
| 2024-06-18 | rising | stressed | 1.0 | 18.8 | 2.6 | 50.0 | 50.0 | 0.0 | 31.0 |  |
| 2024-08-21 | rising | stressed | 22.2 | 6.4 | 0.5 | 50.0 | 50.0 | 0.0 | 47.9 |  |

## 8. Shock detector against the labels
CUSUM alarms: 0; followed by a move into the high band within 30 calendar days: 0; moves into the high band preceded by an alarm within 30 days: 0 of 19.

Surprise: mean -0.011, sd 1.009 (should be ~1 if implied matches realised on average), 1%/99% -2.39/+2.32; max CUSUM pressure 0.9945 on 2021-04-20 (alarm when it reaches 0.999).

Shock score lead profile (5 days before):
| all | before_up | before_up_n | before_down | before_down_n |
|---|---|---|---|---|
| 30.07 | 30.54 | 114 | 30.71 | 135 |

Alarm dates (first 60): 

## 8b. Alignment check: Spearman correlation of each feature on day t with trailing 1-week realised vol on day t+k (k < 0 past, k > 0 future)
| index | k=-10 | k=-5 | k=-1 | k=+0 | k=+1 | k=+2 | k=+5 | k=+10 |
|---|---|---|---|---|---|---|---|---|
| stored_energy | -0.12 | -0.10 | -0.11 | -0.11 | -0.09 | -0.07 | -0.03 | -0.02 |
| gap_pct | 0.10 | 0.12 | 0.10 | 0.10 | 0.10 | 0.11 | 0.10 | 0.10 |
| complacency | -0.45 | -0.47 | -0.49 | -0.49 | -0.45 | -0.41 | -0.28 | -0.24 |
| jump_cluster_pct | 0.19 | 0.25 | 0.26 | 0.26 | 0.22 | 0.18 | 0.05 | 0.05 |
| coupling_shift_pct | 0.05 | 0.07 | 0.04 | 0.03 | 0.03 | 0.02 | 0.02 | -0.03 |
| coherence_pct | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.08 | -0.01 | -0.01 |
| desk_flow | -0.04 | -0.04 | -0.05 | -0.06 | -0.08 | -0.09 | -0.08 | -0.05 |

## 9. Fan from today's state (stressed, 2024-08-28); pressure from ['desk_flow']; pressure today +0.00 (0 = normal; the tilt moves the fan only when pressure is away from 0); whole-history tilt fit on the raw pressure scale (β per unit of pressure, not per sd as in section 2): {"beta": 0.8234, "gamma": 0.3365, "loglik": -2476.9221, "loglik0": -2489.1842, "gain_per_transition": 0.0049, "n": 2515, "k": 5}
| h | P(high band) constant | P(high band) tilted |
|---|---|---|
| 5 | 0.726 | 0.726 |
| 10 | 0.536 | 0.536 |
| 21 | 0.302 | 0.302 |

