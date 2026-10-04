# DECISIONS

Choices only the research owner makes. The agent stops when it hits an open one. Each line is
dated when closed. Gate-threshold changes are recorded here with the reason.

| # | Decision | Options | Status | Closed on | Note |
|---|---|---|---|---|---|
| D1 | Standard notional unit per archetype (for per-unit P&L) | per unit ATM vega; per unit RR notional in vega-neutral ratio; per unit fly notional; fixed cash notional per leg | open | | see `configs/archetypes.yaml` |
| D2 | P&L currency | source currency; converted to a single reporting currency | open | | |
| D3 | Horizons | {1,3,5,10,20,T} (default) | closed | | from SPEC |
| D4 | Primary horizon for the vertical slice | 5d (default, market-making book) | closed | | |
| D5 | Labeller smoothing half-life and hysteresis band | 3 / 3 (defaults); chosen inside training folds | open | | WU-08 |
| D6 | Scoring target for the state finder | forward 5d archetype P&L where the backtester covers (default); forward surface change on the 10-year history | open | | WU-08 |
| D7 | Shrinkage κ method | leave-one-fold-out MSE (default); fixed | open | | WU-10 |
| D8 | Pairs in Phase 1–3 | one liquid pair first, second pair at WU-05 | open | | |
| D9 | Data as-of date for the Phase 3 snapshot | | open | | pinned in `configs/local.yaml` |
| D10 | Interval method | block bootstrap (default); HAC | closed | | validated on synthetic data |
| D11 | Number of named states | 5 (level × direction, default); 3 (level only) | open | | decided by the finder on forward P&L at WU-08, confirmed at Gate 2 |
| D12 | Cross-asset series in the profile | defaults in `configs/profile.yaml`; mapped to source tickers in `configs/local.yaml` | open | | expand or change freely; nothing else depends on the list |
| D13 | Override of any `configs/gates.yaml` threshold | | — | | record each change as a new line below |

## Gate threshold changes
| Date | Key | Old | New | Reason |
|---|---|---|---|---|
