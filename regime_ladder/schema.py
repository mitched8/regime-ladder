"""Trade-day table schema.

One row per trade per calendar day of its life. This is the only interface
between the backtester (whatever it is) and the rest of the package. The
adapter written at the data owner's site produces a frame with these columns;
nothing downstream knows anything else about the source.

Required columns
----------------
trade_id      str     unique per trade
pair          str     e.g. "EURUSD"
archetype     str     e.g. "straddle_atm", "rr_25d", "fly_10d"
tenor_days    int     option tenor in trading days
entry_date    date    entry date (the conditioning date)
date          date    calendar day of this row
age           int     trading days since entry, 1..tenor_days
pnl           float   net daily cash P&L per unit of standard notional

Optional columns (used by checks and diagnostics when present)
--------------------------------------------------------------
pnl_<name>                                   any P&L attribution column; a SET of them sums to pnl. Two sets are
                                             known by name: legs (pnl_trade, pnl_delta_hedge, pnl_vega_hedge) and
                                             factors (pnl_spot, pnl_vol, pnl_rr, pnl_fly, pnl_rates, pnl_theta,
                                             pnl_residual). A table may carry both; the adapter maps the source's
                                             attribution onto these names (or names its own set in the config).
vega, gamma_cash, vanna, volga               daily Greeks of the live position
remaining                                    tenor_days - age (derived if absent)
regime_day                                   label on `date` (for persistence split)
"""
from __future__ import annotations

import pandas as pd

REQUIRED = ["trade_id", "pair", "archetype", "tenor_days", "entry_date", "date", "age", "pnl"]
COMPONENTS = ["pnl_trade", "pnl_delta_hedge", "pnl_vega_hedge"]                      # the legs set (kept for callers that name it)
COMPONENT_SETS = {"legs": COMPONENTS,
                  "factors": ["pnl_spot", "pnl_vol", "pnl_rr", "pnl_fly", "pnl_rates", "pnl_theta", "pnl_residual"]}
GREEKS = ["vega", "gamma_cash", "vanna", "volga"]


def component_columns(td: pd.DataFrame) -> list[str]:
    """Every attribution column in the table: `pnl_<name>`."""
    return [c for c in td.columns if c.startswith("pnl_")]


def component_sets(td: pd.DataFrame, cfg: dict | None = None) -> dict[str, list[str]]:
    """The attribution sets present in the table: the known sets (where every column of the set is there), any set
    named in the config (`components: {name: [cols]}`), and `other` for pnl_ columns in no set. Each set should sum to
    pnl on its own; `checks.check_trade_days` verifies that per set."""
    sets = {**COMPONENT_SETS, **{k: list(v) for k, v in (cfg or {}).items()}}
    present = {name: [c for c in cols if c in td.columns] for name, cols in sets.items()}
    out = {name: cols for name, cols in present.items() if cols and len(cols) >= max(2, len(sets[name]) - 1)}   # allow one missing (e.g. no rates leg)
    used = {c for cols in out.values() for c in cols}
    other = [c for c in component_columns(td) if c not in used]
    if other:
        out["other"] = other
    return out


def coerce(td: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes and derive `remaining`. Does not validate; see checks.py."""
    td = td.copy()
    for c in ("entry_date", "date"):
        td[c] = pd.to_datetime(td[c]).dt.normalize()
    td["tenor_days"] = td["tenor_days"].astype(int)
    td["age"] = td["age"].astype(int)
    td["pnl"] = td["pnl"].astype(float)
    if "remaining" not in td.columns:
        td["remaining"] = td["tenor_days"] - td["age"]
    return td.sort_values(["pair", "archetype", "trade_id", "age"]).reset_index(drop=True)


def labels_frame(pair: str, dates, regimes) -> pd.DataFrame:
    """Build a (pair, date, regime) label frame — the second interface."""
    return pd.DataFrame({"pair": pair, "date": pd.to_datetime(dates).normalize(), "regime": list(regimes)})
