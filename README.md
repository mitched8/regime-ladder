# regime-ladder

Research scaffold for **multi-horizon, entry-conditional trade statistics** on option archetypes:
given a backtester that reports daily P&L for every historical entry of a structure, estimate what
a fresh unit of that structure earns over the next 1, 3, 5, 10 and 20 days conditional on the
market regime on the entry date — with intervals that respect overlap, the outcome distribution,
and an honest count of the independent episodes behind each number. States are five named
regions of level × direction (carry, rising, crisis, normalising, settling), calibrated by a state
finder against forward outcomes; each state carries a profile (spot-vol behaviour, dollar-factor
share, cross-asset correlations) and may contain discovered, stability-tested sub-states.

It is deliberately small (about 1,900 lines including tests), has no dependency on any particular
data source, and is validated on synthetic data with analytic ground truth before any real data is
touched. Site-specific adapters are written locally and never committed.

## Run

```
pip install -e ".[test]"
pytest -q                          # 49 tests, ~40 s
python -m regime_ladder demo       # end to end on synthetic data; writes out/demo/card.md
python -m regime_ladder validate   # intervals recover the analytic truth across seeds
```

## Read

| File | Purpose |
|---|---|
| `SPEC.md` | the estimand, assumptions, falsification tests — the contract the agent builds from |
| `PLAN.md` | work units through the Phase 3 go/no-go, with acceptance criteria and gates |
| `docs/MODEL_GUIDE.md` | method, module map, what the synthetic validation shows |
| `docs/LOCALISATION.md` | how to connect real data through untracked adapters |
| `docs/framework.html` | the complete framework for a human reader — data, features, five states and the state finder, profiles, sub-state discovery, shock detection, transition probabilities, leading features, the ladder, the path model, validation and gates — with synthetic screenshots and two interactive demos; open in a browser |
| `templates/` | standing rules for the coding agent, builder and reviewer prompts, handoff and decisions templates |
| `configs/gates.yaml` | pre-registered pass/fail thresholds; owner-only |

## Working method

Plan outside, execute inside. A coding agent builds one work unit per session from `SPEC.md` and
the unit's card; a separate reviewer session that cannot run code passes or fails it; the owner
reviews only at gates. `HANDOFF.md` carries state between sessions; `DECISIONS.md` carries the
choices only the owner makes.

## Status

Scaffold complete; no real data connected. Start at WU-00 in `PLAN.md`.
