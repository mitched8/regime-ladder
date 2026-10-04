# Review packet · stage `leading` · calibration D_external (synthetic)

| key | value |
|---|---|
| pair | EURUSD |
| data | 2015-01-01 to 2024-08-28 (2520 days) |
| finder / leading target | expected forward 5d straddle P&L given the true state path (synthetic, noiseless); diagnostic outcome at 21d |
| git commit | 556ee10 |
| config hash | ba38712377 |
| generated | 2026-10-04 17:45 |

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
  - gap_z_signed: the signed stretch itself (a z, not a percentile); drift_t: t-statistic of the 60-day drift (signed). Signed features are judged on the RR outcome (`target` column), unsigned ones on the straddle.
  - Any other name is a desk series supplied through the adapter; it is claimed trailing-only.
- Screen, one feature at a time, against a baseline that already knows the named state AND the continuous composite, 4 walk-forward folds after a 40% initial training block (`n_pairs` > 1 means the test is pooled across pairs with pair dummies, folds cut at common dates, one tilt coefficient across pairs):
  - (1) outcome test: `delta_r2` = pooled out-of-sample R² gain on the forward outcome named in `target`; `folds_r2_up` = folds in which the gain is > 0; `n` = days with feature, label, composite and outcome; `r2_base` = the baseline's own OOS R² on that feature's sample (samples differ by feature because warm-ups differ). `delta_r2_low` is the gain when the feature is allowed to act only in the calm states (carry, settling) as well as overall: the conditional form of the stored-energy hypothesis. `delta_r2_lagged` repeats the test with the feature delayed one day; differences under 0.005 are noise.
  - (2) transition test: `transition_gain` = out-of-sample k-step transition log-likelihood gain per transition (nats) through a tilt, with the composite already in the tilt, both z-scored per fold on the training part; the declared k is 5. `transition_gain_k10` and `_k21` are the same at longer horizons, diagnostics only: a slow build-up that leads by a week can show at 21 before it shows at 5. `folds_transition_up` = folds with gain > 0 at the declared k; `n_transitions` = days with feature and label (no outcome needed, so it can exceed `n`). `beta_mean` = tilt coefficient per standard deviation of the feature, averaged over folds.
  - `fail_reason` names the first rule a non-retained feature failed, on unrounded values (the table rounds). `target_note` says when a feature's configured outcome was unavailable and the default was used.
  - `p_perm`: share of 100 circular shifts of the feature (>= 63 days) whose outcome gain matches or beats the observed one; the selection control for a screen over many candidates. A real feature has p_perm near 0.01 (the floor); a noise feature anywhere.
  - Retention, two stages. Walk-forward on the first 80% of the history: delta_r2 >= 0.005 AND folds_r2_up >= 3 AND transition_gain >= 0.002 AND folds_transition_up >= 3 AND p_perm <= 0.05. Then only the top 3 candidates by delta_r2 that pass are tested once on the final 20% (`holdout_tested` = True; trained on everything before it) and keep the flag unless the hold-out CONTRADICTS them: `delta_r2_holdout` must be > -0.005 and `transition_gain_holdout` > -0.002. The hold-out holds three or four episodes, so it is asked not to reverse the finding, not to re-prove it; read a hold-out value near zero as neutral, a clearly negative one as the warning it is. A candidate that passed the walk-forward but was not in the top 3 shows `holdout_tested` = False and is not retained.
- Lift (event study): on days when the feature is at or above its trailing 90th percentile, the probability that the state enters the high band within 20 days, divided by the same probability on all eligible days (days not already in the high band). `n_signal_runs` counts separate signal episodes — the effective sample. 90% block-bootstrap interval. Not a gate; it is the more powerful test for a sparse feature and the number a trader can read directly.
- Shock score (0..100) = mean of CUSUM pressure on squared surprises (an alarm is a day on which the pressure reaches 0.999, i.e. the statistic crosses its calibrated threshold), BOCD probability of a short run, and recent |surprise|; surprise = return / implied daily vol marked the day before.
- Gate 4: at least one feature retained AND the pressure index of the retained features passes test (2) on its own: gain >= 0.002 per transition, > 0 in >= 3/4 folds.
- Pressure index = mean of the RETAINED features after each is z-scored against its own trailing 3-year window (0 = normal for that feature; percentiles, products and signed series on one footing). If nothing is retained, no pressure index exists and only the constant fan is meaningful.
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
Target substitutions: gap_z_signed: rr_25d unavailable, judged on straddle_atm; drift_t: rr_25d unavailable, judged on straddle_atm

`delta_r2_diag` is the outcome test on straddle_atm_h21 (the same archetype at the diagnostic horizon); it does not decide retention.

| feature | target | n | n_pairs | r2_base | delta_r2 | p_perm | delta_r2_low | delta_r2_diag | delta_r2_lagged | folds_r2_up | transition_gain | folds_transition_up | holdout_tested | delta_r2_holdout | transition_gain_holdout | retain | fail_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complacency | straddle_atm | 1807 | 1 | 0.4129 | 0.0131 | 0.0198 | 0.0054 | 0.0370 | 0.0119 | 3 | 0.0020 | 3 | False |  |  | False | transition_gain |
| coherence_pct | straddle_atm | 1807 | 1 | 0.4129 | 0.0073 | 0.1188 | 0.0068 | 0.0034 | 0.0032 | 2 | 0.0000 | 2 | False |  |  | False | folds_r2_up;transition_gain;folds_transition_up;p_perm |
| jump_cluster_pct | straddle_atm | 1808 | 1 | 0.4126 | 0.0064 | 0.0693 | 0.0021 | 0.0290 | 0.0105 | 3 | -0.0026 | 1 | False |  |  | False | transition_gain;folds_transition_up;p_perm |
| desk_flow | straddle_atm | 1895 | 1 | 0.4045 | 0.0012 | 0.1386 | -0.0033 | 0.1059 | 0.0064 | 2 | 0.0083 | 4 | False |  |  | False | delta_r2;folds_r2_up;p_perm |
| coupling_shift_pct | straddle_atm | 1702 | 1 | 0.4015 | -0.0004 | 0.5050 | -0.0055 | 0.0040 | -0.0056 | 2 | -0.0072 | 3 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;p_perm |
| gap_pct | straddle_atm | 1768 | 1 | 0.4209 | -0.0016 | 0.2871 | -0.0182 | 0.0018 | 0.0020 | 2 | 0.0034 | 3 | False |  |  | False | delta_r2;folds_r2_up;p_perm |
| stored_energy | straddle_atm | 1768 | 1 | 0.4209 | -0.0044 | 0.4455 | -0.0922 | -0.0056 | -0.0005 | 0 | 0.0073 | 3 | False |  |  | False | delta_r2;folds_r2_up;p_perm |
| gap_z_signed | straddle_atm | 1895 | 1 | 0.4045 | -0.0071 | 0.4653 | -0.0103 | 0.0035 | -0.0086 | 1 | 0.0015 | 3 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;p_perm |
| drift_t | straddle_atm | 1895 | 1 | 0.4045 | -0.0073 | 0.5545 | -0.0186 | -0.0069 | -0.0037 | 3 | 0.0011 | 2 | False |  |  | False | delta_r2;transition_gain;folds_transition_up;p_perm |

Transition test by horizon (k = 5 decides; 10 and 21 are diagnostics):
| feature | n_transitions | transition_gain | transition_gain_k10 | transition_gain_k21 | beta_mean |
|---|---|---|---|---|---|
| complacency | 1806 | 0.0020 | -0.0040 | -0.0135 | -0.0502 |
| coherence_pct | 1806 | 0.0000 | -0.0004 | -0.0048 | -0.0160 |
| jump_cluster_pct | 1807 | -0.0026 | -0.0035 | -0.0076 | -0.0161 |
| desk_flow | 2015 | 0.0083 | 0.0437 | 0.1005 | 0.1444 |
| coupling_shift_pct | 1701 | -0.0072 | -0.0067 | -0.0004 | 0.0359 |
| gap_pct | 1767 | 0.0034 | -0.0078 | -0.0012 | -0.0763 |
| stored_energy | 1767 | 0.0073 | -0.0032 | -0.0125 | -0.1145 |
| gap_z_signed | 1955 | 0.0015 | -0.0017 | -0.0083 | -0.0564 |
| drift_t | 1955 | 0.0011 | -0.0026 | -0.0101 | -0.0463 |

## 2b. Lift: P(high band within 20 days | feature in its top decile) vs base rate
| feature | p_event_signal | p_event_all | lift | lift_lo | lift_hi | n_signal | n_signal_runs | n_events_after_signal |
|---|---|---|---|---|---|---|---|---|
| stored_energy | 0.173 | 0.279 | 0.622 | 0.297 | 0.987 | 277 | 30 | 48 |
| gap_pct | 0.131 | 0.279 | 0.471 | 0.207 | 0.860 | 259 | 24 | 34 |
| complacency | 0.275 | 0.278 | 0.990 | 0.612 | 1.366 | 280 | 40 | 77 |
| jump_cluster_pct | 0.125 | 0.278 | 0.450 | 0.143 | 0.749 | 192 | 24 | 24 |
| coupling_shift_pct | 0.240 | 0.281 | 0.853 | 0.533 | 1.203 | 288 | 57 | 69 |
| coherence_pct | 0.098 | 0.278 | 0.352 | 0.128 | 0.600 | 174 | 51 | 17 |
| gap_z_signed | 0.105 | 0.256 | 0.409 | 0.100 | 0.789 | 258 | 26 | 27 |
| drift_t | 0.156 | 0.256 | 0.610 | 0.136 | 1.161 | 224 | 26 | 35 |
| desk_flow | 0.412 | 0.248 | 1.660 | 1.197 | 2.167 | 182 | 70 | 75 |

## 3. Feature distributions
| index | count | mean | std | min | 5% | 50% | 95% | max | first_valid | last_valid | share_at_mode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | 2272.0 | 24.3 | 19.3 | 0.0 | 1.9 | 19.9 | 65.0 | 89.8 | 2015-12-15 | 2024-08-28 | 0.0 |
| gap_pct | 2272.0 | 47.9 | 29.7 | 0.0 | 4.4 | 45.3 | 96.7 | 100.0 | 2015-12-15 | 2024-08-28 | 0.0 |
| complacency | 2311.0 | 49.5 | 16.7 | 6.0 | 23.4 | 49.5 | 75.9 | 93.9 | 2015-10-21 | 2024-08-28 | 0.0 |
| jump_cluster_pct | 2312.0 | 31.3 | 30.9 | 0.0 | 0.0 | 27.9 | 90.2 | 99.6 | 2015-10-20 | 2024-08-28 | 0.4 |
| coupling_shift_pct | 2206.0 | 47.6 | 28.0 | 0.0 | 4.8 | 46.4 | 94.4 | 100.0 | 2016-03-16 | 2024-08-28 | 0.0 |
| coherence_pct | 2311.0 | 50.2 | 28.9 | 0.0 | 4.4 | 51.1 | 93.8 | 100.0 | 2015-10-21 | 2024-08-28 | 0.0 |
| gap_z_signed | 2460.0 | -0.1 | 1.0 | -3.2 | -1.7 | -0.1 | 1.5 | 3.3 | 2015-03-26 | 2024-08-28 | 0.0 |
| drift_t | 2460.0 | -0.1 | 1.0 | -2.3 | -1.7 | -0.2 | 1.7 | 3.0 | 2015-03-26 | 2024-08-28 | 0.0 |
| desk_flow | 2520.0 | 51.0 | 18.4 | 0.0 | 19.9 | 51.4 | 80.3 | 100.0 | 2015-01-01 | 2024-08-28 | 0.0 |

## 4. Mean feature value by state (0..100)
| index | agitated | carry | normalising | rising | settling | stressed |
|---|---|---|---|---|---|---|
| stored_energy | 19.0 | 44.8 | 19.6 | 23.9 | 28.9 | 16.1 |
| gap_pct | 43.7 | 65.3 | 47.9 | 41.7 | 54.4 | 41.2 |
| complacency | 45.3 | 67.0 | 40.6 | 55.5 | 52.2 | 39.8 |
| jump_cluster_pct | 31.5 | 28.1 | 26.1 | 33.4 | 37.2 | 30.6 |
| coupling_shift_pct | 51.7 | 47.6 | 44.7 | 47.0 | 55.4 | 41.9 |
| coherence_pct | 51.5 | 53.3 | 59.6 | 45.1 | 49.6 | 46.8 |
| gap_z_signed | -0.3 | 0.1 | -0.2 | -0.0 | 0.0 | -0.3 |
| drift_t | -0.2 | 0.1 | -0.2 | 0.1 | 0.1 | -0.3 |
| desk_flow | 50.0 | 49.8 | 49.2 | 52.3 | 46.4 | 55.4 |

## 5. Correlations (features and the composite)
| index | stored_energy | gap_pct | complacency | jump_cluster_pct | coupling_shift_pct | coherence_pct | gap_z_signed | drift_t | desk_flow | composite |
|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | 1.00 | 0.85 | 0.56 | -0.03 | 0.10 | -0.04 | -0.01 | -0.02 | -0.08 | -0.51 |
| gap_pct | 0.85 | 1.00 | 0.14 | 0.11 | 0.10 | 0.07 | -0.07 | -0.08 | -0.09 | -0.28 |
| complacency | 0.56 | 0.14 | 1.00 | -0.26 | -0.02 | -0.20 | 0.06 | 0.08 | -0.01 | -0.61 |
| jump_cluster_pct | -0.03 | 0.11 | -0.26 | 1.00 | 0.04 | 0.25 | 0.17 | 0.10 | 0.00 | -0.04 |
| coupling_shift_pct | 0.10 | 0.10 | -0.02 | 0.04 | 1.00 | 0.06 | 0.15 | 0.15 | -0.07 | -0.09 |
| coherence_pct | -0.04 | 0.07 | -0.20 | 0.25 | 0.06 | 1.00 | 0.04 | 0.01 | -0.07 | -0.03 |
| gap_z_signed | -0.01 | -0.07 | 0.06 | 0.17 | 0.15 | 0.04 | 1.00 | 0.88 | -0.14 | -0.15 |
| drift_t | -0.02 | -0.08 | 0.08 | 0.10 | 0.15 | 0.01 | 0.88 | 1.00 | -0.13 | -0.13 |
| desk_flow | -0.08 | -0.09 | -0.01 | 0.00 | -0.07 | -0.07 | -0.14 | -0.13 | 1.00 | 0.10 |
| composite | -0.51 | -0.28 | -0.61 | -0.04 | -0.09 | -0.03 | -0.15 | -0.13 | 0.10 | 1.00 |

## 6. Lead profile: mean value in the days before ANY move up the state ranking (e.g. carry to rising), before moves down, and on all days. Moves skip ranks (a rising-to-stressed move counts once), so up and down counts need not match
| feature | window_days | all_days | before_escalation | before_de_escalation | n_escalations | n_de_escalations |
|---|---|---|---|---|---|---|
| stored_energy | 5 | 24.3 | 24.1 | 23.1 | 89.0 | 121.0 |
| stored_energy | 20 | 24.3 | 26.4 | 21.6 | 89.0 | 121.0 |
| gap_pct | 5 | 47.9 | 43.3 | 47.0 | 89.0 | 121.0 |
| gap_pct | 20 | 47.9 | 46.6 | 45.1 | 89.0 | 121.0 |
| complacency | 5 | 49.5 | 53.7 | 48.1 | 89.0 | 121.0 |
| complacency | 20 | 49.5 | 55.2 | 47.5 | 89.0 | 121.0 |
| jump_cluster_pct | 5 | 31.3 | 29.7 | 29.0 | 89.0 | 121.0 |
| jump_cluster_pct | 20 | 31.3 | 27.9 | 28.9 | 89.0 | 121.0 |
| coupling_shift_pct | 5 | 47.6 | 46.3 | 48.4 | 89.0 | 121.0 |
| coupling_shift_pct | 20 | 47.6 | 46.9 | 47.6 | 89.0 | 121.0 |
| coherence_pct | 5 | 50.2 | 48.1 | 53.2 | 89.0 | 121.0 |
| coherence_pct | 20 | 50.2 | 52.0 | 52.9 | 89.0 | 121.0 |
| gap_z_signed | 5 | -0.1 | -0.1 | -0.2 | 89.0 | 121.0 |
| gap_z_signed | 20 | -0.1 | -0.1 | -0.1 | 89.0 | 121.0 |
| drift_t | 5 | -0.1 | -0.1 | -0.1 | 89.0 | 121.0 |
| drift_t | 20 | -0.1 | -0.1 | -0.1 | 89.0 | 121.0 |
| desk_flow | 5 | 51.0 | 56.4 | 50.8 | 89.0 | 121.0 |
| desk_flow | 20 | 51.0 | 55.1 | 50.2 | 89.0 | 121.0 |

## 7. Every move into the high band (30): values before it
| date | from | to | stored_energy_t-20 | stored_energy_t-5 | stored_energy_t-1 | gap_z_signed_t-5 | gap_z_signed_t-1 | drift_t_t-5 | drift_t_t-1 | desk_flow_t-5 | desk_flow_t-1 | shock_max_20d | days_since_cusum_alarm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015-12-30 | rising | stressed |  | 19.6 | 11.7 | -0.8 | -0.5 | -0.7 | -0.7 | 80.0 | 55.7 | 57.6 |  |
| 2016-06-21 | rising | stressed | 13.1 | 29.3 | 36.2 | -0.9 | -1.0 | -1.0 | -1.1 | 83.2 | 69.1 | 42.0 |  |
| 2016-08-24 | rising | stressed | 38.9 | 29.6 | 24.2 | -1.0 | -0.9 | -1.6 | -1.3 | 50.4 | 18.2 | 64.7 |  |
| 2017-01-20 | rising | stressed | 17.4 | 13.2 | 20.4 | 0.6 | 1.0 | 0.2 | 0.4 | 37.4 | 59.6 | 54.9 |  |
| 2017-04-28 | rising | stressed | 2.5 | 31.2 | 8.1 | -0.9 | -0.4 | -1.6 | -1.0 | 76.2 | 63.8 | 56.9 |  |
| 2017-08-14 | rising | stressed | 8.6 | 10.0 | 3.2 | 0.3 | 0.1 | 1.0 | 0.3 | 52.2 | 63.1 | 39.3 |  |
| 2017-11-21 | rising | stressed | 11.6 | 26.0 | 1.2 | -0.6 | 0.0 | -0.6 | 0.1 | 52.4 | 66.9 | 45.5 |  |
| 2017-12-06 | normalising | stressed | 16.8 | 10.3 | 10.0 | 0.6 | 0.6 | 0.2 | 0.3 | 78.9 | 76.1 | 81.2 |  |
| 2018-02-05 | rising | stressed | 23.5 | 19.0 | 17.8 | -0.5 | -0.5 | -0.2 | 0.1 | 73.6 | 42.2 | 35.8 |  |
| 2018-04-11 | normalising | stressed | 13.3 | 5.5 | 2.3 | -0.1 | 0.1 | 0.2 | 0.2 | 65.6 | 65.6 | 42.5 |  |
| 2019-01-17 | rising | stressed | 77.5 | 31.0 | 7.2 | -0.6 | -0.2 | -0.6 | 0.1 | 47.3 | 73.4 | 41.0 |  |
| 2019-02-19 | normalising | stressed | 1.2 | 23.1 | 36.1 | -0.6 | -1.2 | -0.6 | -1.1 | 100.0 | 74.1 | 55.0 |  |
| 2019-03-21 | normalising | stressed | 21.8 | 1.6 | 4.4 | 0.1 | 0.3 | 0.1 | 0.4 | 59.8 | 41.6 | 75.9 |  |
| 2019-05-29 | rising | stressed | 82.2 | 14.4 | 6.4 | 0.4 | 0.2 | 2.4 | 2.0 | 61.8 | 56.7 | 44.6 |  |
| 2019-08-29 | rising | stressed | 4.3 | 37.2 | 33.6 | -0.7 | -0.7 | -1.2 | -1.2 | 77.5 | 64.3 | 36.8 |  |
| 2020-07-01 | agitated | stressed | 45.8 | 23.6 | 20.8 | 0.9 | 0.7 | 0.6 | 0.2 | 47.4 | 40.0 | 40.1 |  |
| 2020-09-29 | rising | stressed | 9.6 | 17.3 | 17.1 | 0.3 | 0.3 | 0.3 | 0.2 | 55.5 | 70.8 | 32.6 |  |
| 2020-10-29 | rising | stressed | 13.1 | 8.0 | 5.3 | -0.4 | -0.3 | -0.5 | -0.4 | 43.5 | 49.2 | 49.5 |  |
| 2021-09-28 | agitated | stressed | 22.4 | 30.5 | 16.8 | -1.0 | -0.6 | -1.1 | -0.6 | 39.2 | 63.2 | 35.3 |  |
| 2022-02-16 | rising | stressed | 32.8 | 8.5 | 6.9 | 0.2 | 0.2 | 1.1 | 0.8 | 56.1 | 52.7 | 41.0 |  |
| 2022-04-25 | rising | stressed | 5.1 | 3.2 | 10.9 | -0.1 | -0.4 | -0.4 | -0.7 | 51.3 | 70.1 | 40.2 |  |
| 2022-08-25 | rising | stressed | 27.0 | 26.5 | 26.1 | -1.6 | -1.0 | -1.8 | -1.7 | 64.0 | 54.9 | 38.5 |  |
| 2022-10-05 | normalising | stressed | 23.6 | 35.3 | 29.8 | -1.9 | -1.4 | -1.6 | -1.3 | 68.6 | 48.9 | 56.9 |  |
| 2022-11-24 | rising | stressed | 18.6 | 4.9 | 15.2 | 0.1 | 0.4 | -0.4 | 0.5 | 36.6 | 60.4 | 33.2 |  |
| 2022-12-20 | rising | stressed | 8.7 | 9.7 | 9.2 | 0.3 | 0.2 | 1.4 | 1.1 | 58.4 | 36.9 | 32.8 |  |
| 2023-03-28 | rising | stressed | 7.1 | 1.4 | 14.7 | -0.0 | -0.3 | 0.2 | -0.7 | 72.8 | 65.8 | 36.6 |  |
| 2023-05-10 | rising | stressed | 3.5 | 5.6 | 6.6 | 0.1 | 0.2 | -0.2 | 0.3 | 44.7 | 73.4 | 40.1 |  |
| 2023-08-23 | rising | stressed | 5.0 | 42.2 | 37.8 | 0.9 | 0.8 | 1.4 | 1.5 | 44.6 | 54.7 | 49.2 |  |
| 2024-01-12 | rising | stressed | 47.7 | 10.3 | 1.3 | 0.3 | 0.1 | 0.6 | 0.6 | 58.7 | 51.8 | 50.2 |  |
| 2024-08-13 | rising | stressed | 29.9 | 35.6 | 22.5 | -1.0 | -0.6 | -2.0 | -1.6 | 71.2 | 51.7 | 39.5 |  |

## 8. Shock detector against the labels
CUSUM alarms: 0; followed by a move into the high band within 30 calendar days: 0; moves into the high band preceded by an alarm within 30 days: 0 of 30.

Surprise: mean -0.020, sd 1.005 (should be ~1 if implied matches realised on average), 1%/99% -2.39/+2.26; max CUSUM pressure 0.9836 on 2017-12-06 (alarm when it reaches 0.999).

Shock score lead profile (5 days before):
| all | before_up | before_up_n | before_down | before_down_n |
|---|---|---|---|---|
| 29.92 | 30.01 | 89 | 29.56 | 121 |

Alarm dates (first 60): 

## 8b. Alignment check: Spearman correlation of each feature on day t with trailing 1-week realised vol on day t+k (k < 0 past, k > 0 future). A stamping error is a sharp peak at one future k; a genuine slow lead rises gently and plateaus
| index | k=-21 | k=-10 | k=-5 | k=-1 | k=+0 | k=+1 | k=+2 | k=+5 | k=+10 | k=+15 | k=+21 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | -0.05 | -0.22 | -0.24 | -0.26 | -0.25 | -0.23 | -0.21 | -0.16 | -0.12 | -0.07 | -0.09 |
| gap_pct | 0.01 | -0.03 | -0.04 | -0.08 | -0.08 | -0.09 | -0.09 | -0.09 | -0.10 | -0.08 | -0.11 |
| complacency | -0.14 | -0.50 | -0.53 | -0.50 | -0.49 | -0.44 | -0.39 | -0.24 | -0.11 | -0.02 | 0.02 |
| jump_cluster_pct | 0.03 | 0.16 | 0.19 | 0.22 | 0.23 | 0.20 | 0.17 | 0.07 | 0.07 | 0.02 | 0.01 |
| coupling_shift_pct | 0.00 | 0.05 | 0.04 | 0.02 | 0.01 | -0.00 | -0.00 | -0.04 | -0.07 | -0.05 | -0.05 |
| coherence_pct | 0.01 | 0.19 | 0.15 | 0.10 | 0.09 | 0.06 | 0.03 | -0.07 | -0.09 | -0.06 | -0.03 |
| gap_z_signed | -0.12 | -0.08 | -0.07 | -0.07 | -0.07 | -0.07 | -0.07 | -0.06 | -0.06 | -0.05 | -0.02 |
| drift_t | -0.14 | -0.08 | -0.07 | -0.07 | -0.07 | -0.07 | -0.07 | -0.07 | -0.07 | -0.08 | -0.05 |
| desk_flow | -0.06 | -0.01 | -0.01 | 0.02 | 0.03 | 0.05 | 0.07 | 0.12 | 0.21 | 0.27 | 0.27 |

## 9. Fan from today's state (stressed, 2024-08-28): constant matrix only (nothing retained, so no pressure index)
| h | P(high band) constant |
|---|---|
| 5 | 0.767 |
| 10 | 0.601 |
| 21 | 0.385 |

