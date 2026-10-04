# Localisation — connecting the scaffold to real data

Everything site-specific lives in two untracked places: `adapters/` and `configs/local.yaml`.
Nothing in them is committed, and nothing in the tracked repository references them by name.

## 1. The trade-day adapter (`adapters/trade_days.py`)

Produce a frame matching `regime_ladder/schema.py` from whatever the backtest results API returns.

| Schema column | Typical source | Notes |
|---|---|---|
| `trade_id` | source identifier, or `f"{pair}-{archetype}-{entry_date}"` | must be unique |
| `pair` | source | |
| `archetype` | map source names → the names in `configs/archetypes.yaml` | |
| `tenor_days` | source tenor | trading days |
| `entry_date` | source | the conditioning date |
| `date` | source | one row per trading day of life |
| `age` | derived: trading-day count from entry, starting at 1 | `checks.py` enforces consecutiveness |
| `pnl` | sum of the source's P&L components | cash per unit standard notional |
| `pnl_trade`, `pnl_delta_hedge`, `pnl_vega_hedge` | source components if available | `checks.py` verifies they sum to `pnl` |
| `vega` | per-day vega of the live position per unit standard notional | needed for vega-unit scaling (`combine.to_vega_units`); without it, legs stay in notional units |
| `gamma_cash`, `vanna`, `volga` | source Greeks if available | needed for the age-dependence diagnostics, not for the ladder |

Pattern:

```python
def load_trade_days(pair, archetype, start, end, asof=None) -> pd.DataFrame:
    key = f"{pair}_{archetype}_{start}_{end}_{asof or 'latest'}.parquet"
    if cached(key): return read(key)
    raw = <source API call>(pair=..., strategy=..., start=start, end=end)
    td = <rename / derive columns>
    td = schema.coerce(td)
    checks.assert_clean({k: v for k, v in checks.check_trade_days(td).items() if not k.startswith("info_")})
    write(key, td); return td
```

Cache to parquet under a directory named in `configs/local.yaml`, keyed by the arguments and the
data as-of date. Every later unit reads the cache, never the API, so results are reproducible.

## 2. The market adapter (`adapters/market.py`)

Produce a daily frame indexed by date with at least: `atm_1m, atm_1y, rr25_1m, fly25_1m, spot,
rv_1w, rv_1m, rv_3m`. Realised vols come from the hourly spot series. The frame must be
point-in-time: the value stored for date t is the value that was available at the close of t.

For the state profile (WU-06b) the same frame also carries the other G10 pairs' spot levels
(`g10_1 … g10_6`, same side versus the dollar) and the cross-asset series named in
`configs/profile.yaml` under the generic names used there (`eq, eq2, rate2y, rate10y, oil, gold,
dxy, credit, vix, move`); map source tickers to those names in `configs/local.yaml`.

Cross-check: the trade-day table carries the marks the backtester used at the strategy tenor.
Join on (pair, date) and tabulate differences against the market adapter's values. Non-zero
differences mean a cut-time or source mismatch and must be explained before Phase 2.

## 3. `configs/local.yaml`

```yaml
cache_dir: <path>
pairs: [<pair>, <pair>]
asof: <YYYY-MM-DD>        # the data snapshot date; put it in output filenames
archetype_map:            # source name -> scaffold base-leg name (combinations are built from legs)
  <source>: straddle_atm
  <source>: put_25d
  <source>: call_25d
  <source>: put_10d
  <source>: call_10d
  # if the source also runs packages: map them too and reconcile against combine() at WU-03
  <source>: rr_25d
  <source>: fly_10d
tenors: [5, 21, 63]       # trading days; any of 1W-1Y
```

## 4. Archetype conventions (`configs/archetypes.yaml`, tracked, generic)

For each archetype: legs, ratio rule (e.g. "smile-vega-neutral notional ratio at inception"),
hedge rule as the source applies it, standard notional unit for the per-unit P&L, P&L currency.
Write them as the source defines them; this file documents, it does not choose.

## 5. Golden trades

The owner names 4–5 trade IDs covering calm, pre-stress entry, through-stress, near-the-money
expiry and a far-drifted risk reversal. Extract their rows with the adapter to `tests/golden/`,
and write `tests/test_golden.py` asserting the adapter reproduces them to `golden_abs_tol` in
`configs/gates.yaml`. Once the owner has eyeballed the CSVs, they are fixtures for the life of
the project.

## 6. Running order on a new machine

```
pytest -q                               # the scaffold is intact
python -m regime_ladder demo            # end to end on synthetic data
python -m regime_ladder validate        # intervals recover analytic truth
# then WU-01 onwards per PLAN.md
```
