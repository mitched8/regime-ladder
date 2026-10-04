# Calibrating the challenger

Before the challenger (`templates/CHALLENGER_PROMPT.md`) reviews real results, run it on these
packets, built on synthetic data where the truth is known. Each is one paste: the prompt, then the
packet. Give it nothing else, and never `ANSWER_KEY.md`.

| run | packet | what it is | a working challenger says |
|---|---|---|---|
| 1 | `A_energy/leading/packet.md` | a world where a desk series really does drive transitions | PASS: the desk series is genuine and Gate 4 passes |
| 2 | `B_null/leading/packet.md` | the same desk series where it drives nothing | PASS for the results; the series has no value and is correctly rejected |
| 3 | `C_planted/leading/packet.md` | a desk series with a timestamp bug | FAIL: the series is stamped late (look-ahead) |
| 4 | `D_external/leading/packet.md` | a desk series that drives transitions five days after it is observed, from outside the pair | not retained under the pre-registered rule (it adds nothing to the 5-day straddle outcome) although the transition test, the 21-day horizon and the lift all show it; a good challenger returns QUESTIONS pointing at the horizon / retention-rule decision (D22), not FAIL |
| 5 | `A_energy/states/packet.md`, `ladder/`, `data/` | the other three stage packets, same world | findings on the warm-up, thin cells and Gate 3; no history checks (synthetic) |

Owner notes to paste after each packet (they are what you would tell the challenger about a real desk series):

- Runs 1 and 2: "desk_flow is a desk positioning measure: 50 (neutral) on most days, rising above 50 when a stretched position builds in a quiet market. The desk says it is known at the close of each day."
- Run 3: "desk_flow is a desk flow measure expressed as a trailing percentile. The desk says it is known at the close of each day."
- Run 4: "desk_flow is a cross-market gap measure, known at the close of each day."

Score it with `ANSWER_KEY.md`. If run 3 comes back PASS, or run 2 claims the series is useful,
change the prompt (usually: make the relevant checklist line more specific) and rerun all three.
Runs 1–3 matter most; run 4 tests whether it distinguishes a rule problem from a data problem; run 5 shows the other stage reviews.

Rebuild everything with `PYTHONPATH=. python docs/calibration/make_calibration.py` from the repo root
(about seven minutes).
