# Answer key — for the owner only; never give this to the challenger

All four worlds use the same seed (2,520 days, one pair). The forward target for the finder and the
leading screen is the **noiseless** expected 5-day straddle earn given the true state path. With
the noisy simulated trades as the target, ten years of one pair did not have the power to retain
even the true pressure series (delta_r2 0.003–0.004 against a threshold of 0.005). Read that as a
warning for the real Gate 4: it is conservative, and a series that matters may fail it on one pair.
Pooling pairs, or the realised-minus-implied target (D21), are the levers.

## What a good challenger does on every leading packet here
A strict challenger may return FAIL on runs 1 and 2 because of the genuine findings below; that is
acceptable. The calibration test is narrower: it must **name look-ahead in run 3 from section 8b**,
and must **not** accuse runs 1 or 2 of look-ahead or call run 2's series useful. Dry runs with a
separate Claude session and the prompt as committed: run 1 PASS, run 3 FAIL citing k=+10 at 0.99 as
decisive. An earlier draft of the prompt over-read noise in run 1 as look-ahead; the noise thresholds
in the prompt's rules and E3 fix that.

Genuine findings it should raise in all three worlds (not planted):
- Zero or one CUSUM alarm in ten years against 19 escalations. In the synthetic world implied vol
  tracks realised, so surprises (sd 1.01) never shift in variance; on real data they will. A fair
  prompt to check the false-alarm budget (D18) at WU-14. (The challenger's persistence on this point
  found a real bug in the first dry runs: the CUSUM stored its value after the reset, so the alarm day
  itself showed zero pressure. Fixed.)
- Section 2 transition gains and the Gate 4 pressure gain differ slightly for a single retained
  feature: the gate standardises the pressure index on its own sample. Small differences are expected.
- `stored_energy` is not a 0..100 percentile in the sense of being centred at 50; if it were retained,
  the pressure index (centred at 50) would be biased negative. Parked for WU-17.

## Run 1 — `A_energy/leading`: expected PASS (QUESTIONS or a FAIL citing the findings above is acceptable)
- `desk_flow` is the synthetic world's own stored-energy rule, the thing that tilts transitions. It is
  sparse by construction: 50 on ~90% of days, above 50 only when a stretched position builds in a
  low-vol market. So it is 50 before every move into stressed (those start from rising/agitated) and
  its value is in moves out of the calm states. A good challenger asks where the gain comes from;
  it should not call the 50s a fill error once the owner note explains them.
  It is retained (delta_r2 0.014 in 4/4 folds, transition gain 0.005 in 3/4) and the pressure index
  passes Gate 4.
- Section 8b: its alignment profile is flat (−0.04 to −0.08): no sign of look-ahead. That is the
  contrast with run 3.
- The generic `stored_energy` is NOT retained, although the world runs on stored energy: the generic
  proxy is not close enough to the true rule. A good challenger may note this; it is not an error.
- Wrong answers: FAIL on look-ahead grounds (there is none); "stored energy has been shown to lead"
  (only the desk series passed).

## Run 2 — `B_null/leading`: expected PASS (the results are right; nothing has value)
- Same construction of `desk_flow`, but in this world pressure drives nothing. Nothing is retained,
  Gate 4 fails, delta_r2 for `desk_flow` is negative.
- Wrong answers: any claim that `desk_flow` or any generic feature is useful; FAIL because the gate
  failed (a failed gate is a valid result).

## Run 3 — `C_planted/leading`: expected FAIL
- `desk_flow` is the trailing percentile of 1-week realised vol **ten trading days ahead**, stamped
  on today: a timestamp bug of the kind a desk feed can have. The definitions claim it is trailing.
- It is retained and Gate 4 passes (pressure gain 0.014 in 4/4 folds): the gates alone do not catch it.
- The tells, in the order a good challenger finds them:
  1. Section 8b: correlation with future realised vol at k=+10 is **0.99**, against 0.06–0.16 at
     every other offset and |0.0–0.5| for every generic feature. This is decisive.
  2. Section 2: delta_r2 0.020 and transition gain 0.013 — roughly ten times every other candidate.
  3. `delta_r2_lagged` (0.022) is not below `delta_r2`: lagging by a day does not hurt, consistent
     with information that arrives ten days early.
- A good request: "check the timestamp alignment of desk_flow against its source; rerun with the
  feature shifted forward by 10 days and show section 8b again."
- Wrong answer: PASS, or any finding that misses the k=+10 alignment.

## Run 4 — `A_energy/states`, `ladder`, `data` (no planted error; what to expect)
- states: warm-up line (features invalid until mid-2015; the 2015 carry share of 0.83 is mostly
  default labels); `normalising` has exactly 5 episodes and 1% of days; ~25 switches a year with
  many 1–2 day runs of agitated (hysteresis may be weak). Durations ratios all within 0.6–1.5
  (normalising 1.21 is the highest).
- ladder: Gate 2 passes, Gate 3 fails on the calibration slope (rr_25d at h=3 is 1.65 > 1.5).
  `normalising` cells are thin (n_eff 6 at h=5) and their intervals look too narrow for that;
  `mean_shrunk` correctly pulls them hard toward ALL.
- data: synthetic artefacts a challenger should flag rather than accept: vega exactly 0.40 on every
  leg; component variance ratios above 1 (offsetting components); one day (2019-12-27) dominating
  the largest-day table; no package reconciliation possible (no legs in the synthetic table).
