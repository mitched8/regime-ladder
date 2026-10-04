# regime-ladder

Research scaffold for **multi-horizon, entry-conditional trade statistics** on option archetypes:
given a backtester that reports daily P&L for every historical entry of a structure, estimate what
a fresh unit of that structure earns over the next 1, 3, 5, 10 and 20 days conditional on the
market regime on the entry date — with intervals that respect overlap, the outcome distribution,
and an honest count of the independent episodes behind each number. States are six named
regions of level × direction (carry, rising, agitated, stressed, normalising, settling) plus a rare
extreme band, calibrated by a state finder against forward outcomes; each state carries a profile
(spot-vol behaviour, dollar-factor share, cross-asset correlations), may be qualified by tags
(spot-vol correlation sign, event window, pinned, intervention risk) and may contain discovered,
stability-tested sub-states. Around the ladder sit the fast path and the forward look: shock
detection (surprise, CUSUM, BOCD), time-varying transitions (a multinomial-logit tilt fitted on the
k-step likelihood) and leading features built on the stored-energy hypothesis, each retained only
where it adds out-of-sample value beyond the state.

It is deliberately small (about 3,479 lines including tests), has no dependency on any particular
data source, and is validated on synthetic data with analytic ground truth before any real data is
touched — including a synthetic world in which stored energy is causal and one in which it is not.
Site-specific adapters are written locally and never committed.

## Run

```
pip install -e ".[test]"
pytest -q                          # 95 tests, ~75 s
python -m regime_ladder demo       # end to end on synthetic data; writes out/demo/card.md
python -m regime_ladder demo --energy-beta 3    # the world where stored energy drives escalations
python -m regime_ladder validate   # intervals recover the analytic truth across seeds
python -m regime_ladder shock|transitions|leading ...   # the Phase 4 pieces on saved frames
python -m regime_ladder inspect --market market.parquet [--td trade_days.parquet]   # every derived input, one file each, with a README
```

## Read

| File | Purpose |
|---|---|
| `SPEC.md` | the estimand, assumptions, falsification tests — the contract the agent builds from |
| `PLAN.md` | work units through the Phase 3 go/no-go and the Phase 4 leading-feature gate, with acceptance criteria |
| `docs/MODEL_GUIDE.md` | method, module map, what the synthetic validation shows |
| `docs/LOCALISATION.md` | how to connect real data through untracked adapters |
| `docs/COMPONENTS.md` | how to compute and look at any single component — features, finder, matrix, tags, profiles, shocks, leading features, stored energy — on its own, in a notebook, without the backtester |
| `docs/framework.html` | the complete framework for a human reader — data, features, six states and the state finder, tags, profiles, sub-state discovery, shock detection, transition probabilities, leading features and stored energy, the ladder, the path model, validation and gates — with synthetic screenshots and two interactive demos; open in a browser |
| `docs/states.html` · `transitions.html` · `leading.html` · `profiles.html` · `shock.html` · `ladder.html` | six deep dives, one per component, each with its own figures and live panels that run the same code in the browser; regenerated from `docs/src/build_*.py` (point `docs/src/common.py › world()` at a real frame to rebuild them on real data) |
| `templates/` | standing rules for the coding agent, builder and reviewer prompts, handoff and decisions templates |
| `configs/gates.yaml` | pre-registered pass/fail thresholds; owner-only |

## Working method

Plan outside, execute inside. A coding agent builds one work unit per session from `SPEC.md` and
the unit's card; a separate reviewer session that cannot run code passes or fails it; the owner
reviews only at gates. `HANDOFF.md` carries state between sessions; `DECISIONS.md` carries the
choices only the owner makes.

## Status

Scaffold complete through Phase 4 (shocks, transitions, leading features); no real data connected. Start at WU-00 in `PLAN.md`.
