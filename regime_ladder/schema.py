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
pnl_trade, pnl_delta_hedge, pnl_vega_hedge   components summing to pnl
vega, gamma_cash, vanna, volga               daily Greeks of the live position
remaining                                    tenor_days - age (derived if absent)
regime_day                                   label on `date` (for persistence split)
"""
from __future__ import annotations

import pandas as pd

REQUIRED = ["trade_id", "pair", "archetype", "tenor_days", "entry_date", "date", "age", "pnl"]
COMPONENTS = ["pnl_trade", "pnl_delta_hedge", "pnl_vega_hedge"]
GREEKS = ["vega", "gamma_cash", "vanna", "volga"]


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
