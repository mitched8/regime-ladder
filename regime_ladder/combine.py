"""Combinations of base legs, and the scaling that makes them comparable.

The backtester hedges leg by leg and reports P&L per leg per day, so the daily P&L of any
combination of legs entered on the same date is exactly the weighted sum of the legs' daily
P&L, hedges included. A spread's ladder is therefore the ladder of that summed series: no new
backtest runs, ever.

Scaling: by default every base leg is normalised to ONE UNIT OF VEGA AT INCEPTION (per trade),
so that weights are in vega units. A weight vector summing to zero is then smile-vega-neutral by
construction (risk reversal = +1 call, -1 put; fly = +1/2 10d call, +1/2 10d put, -1 ATM) and
one that does not carries net vega — the market-making decomposition. When no `vega` column is
available the legs stay in the source's standard notional units and `configs/archetypes.yaml`
says what those are.

Means are linear, so the state-conditional EV of any combination is the weighted sum of the
legs' state-conditional EVs (`ev_linear`). Intervals, quantiles and ES are not, so for a specific
spread build the combined trade-day rows (`combine`) and run the ladder on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COMPONENTS

BASE_LEGS = ("straddle_atm", "put_25d", "call_25d", "put_10d", "call_10d")
STANDARD_COMBOS = {  # weights in vega units
    "rr_25d": {"call_25d": 1.0, "put_25d": -1.0},
    "rr_10d": {"call_10d": 1.0, "put_10d": -1.0},
    "fly_10d": {"call_10d": 0.5, "put_10d": 0.5, "straddle_atm": -1.0},
    "fly_25d": {"call_25d": 0.5, "put_25d": 0.5, "straddle_atm": -1.0},
}
KEY = ["pair", "tenor_days", "entry_date", "age"]


def to_vega_units(td: pd.DataFrame, vega_col: str = "vega") -> pd.DataFrame:
    """Rescale each trade's P&L (and components) so that its vega at age 1 is one unit.

    Requires a per-day `vega` column in the trade-day table (vega of the live position per unit of
    standard notional). Trades with zero or missing inception vega are dropped and counted.
    """
    if vega_col not in td.columns:
        raise ValueError("to_vega_units needs a per-day vega column; see configs/archetypes.yaml for notional units otherwise")
    v0 = td[td["age"] == 1].set_index("trade_id")[vega_col].abs()
    out = td.merge(v0.rename("_v0"), left_on="trade_id", right_index=True, how="left")
    bad = out["_v0"].isna() | (out["_v0"] == 0)
    out = out[~bad].copy()
    for c in ["pnl"] + [c for c in COMPONENTS if c in out.columns]:
        out[c] = out[c] / out["_v0"]
    out.attrs["dropped_trades_no_inception_vega"] = int(td.loc[bad.values, "trade_id"].nunique()) if bad.any() else 0
    return out.drop(columns="_v0")


def combine(td: pd.DataFrame, weights: dict, name: str) -> pd.DataFrame:
    """Trade-day rows for a weighted combination of base legs entered on the same date.

    Rows are matched on (pair, tenor, entry_date, age); a combination row exists only where every
    leg has a row that day, so partial legs never leak in. Components are summed with the same
    weights. Returns a frame in the trade-day schema with archetype = `name`.
    """
    legs = []
    for leg, w in weights.items():
        g = td[td["archetype"] == leg]
        if g.empty:
            raise ValueError(f"leg {leg!r} not in table")
        cols = ["pnl"] + [c for c in COMPONENTS if c in g.columns]
        h = g[KEY + ["date"] + cols].copy()
        for c in cols:
            h[c] = h[c] * w
        legs.append(h.set_index(KEY + ["date"]).rename(columns={c: f"{c}__{leg}" for c in cols}))
    joined = pd.concat(legs, axis=1, join="inner").reset_index()
    out = joined[KEY + ["date"]].copy()
    for c in ["pnl"] + [c for c in COMPONENTS if any(col.startswith(c + "__") for col in joined.columns)]:
        out[c] = joined[[col for col in joined.columns if col.startswith(c + "__")]].sum(axis=1)
    out["archetype"] = name
    out["trade_id"] = out["pair"] + "-" + name + "-" + out["entry_date"].dt.strftime("%Y-%m-%d") + "-" + out["tenor_days"].astype(str)
    out["remaining"] = out["tenor_days"] - out["age"]
    return out[["trade_id", "pair", "archetype", "tenor_days", "entry_date", "date", "age", "remaining", "pnl"]
               + [c for c in COMPONENTS if c in out.columns]].sort_values(["trade_id", "age"]).reset_index(drop=True)


def ev_linear(lad: pd.DataFrame, weights: dict, name: str) -> pd.DataFrame:
    """EV ladder of a combination by linearity of means: sum_i w_i * EV_i per (pair, tenor, regime, h).

    Exact for the mean; carries no interval, quantile or ES (those need `combine` + `ladder`).
    Use it to screen many candidate combinations cheaply before computing distributions for a shortlist.
    """
    idx = ["pair", "tenor_days", "regime", "h"]
    means, meta = [], None
    for leg, w in weights.items():
        g = lad[lad["archetype"] == leg].set_index(idx)
        if g.empty:
            raise ValueError(f"leg {leg!r} not in ladder")
        means.append((g["mean"] * w).rename(leg))
        meta = g[["to_expiry", "n", "episodes"]] if meta is None else meta
    ev = pd.concat(means, axis=1, join="inner")
    out = meta.loc[ev.index].copy()
    out["mean"] = ev.sum(axis=1)
    out["archetype"] = name
    return out.reset_index()[["pair", "archetype", "tenor_days", "regime", "h", "to_expiry", "n", "mean", "episodes"]]


def screen(lad: pd.DataFrame, candidates: dict, regime: str, h: int) -> pd.DataFrame:
    """Rank candidate combinations by linear EV at one (regime, h). `candidates` = {name: weights}."""
    rows = []
    for name, w in candidates.items():
        e = ev_linear(lad, w, name)
        e = e[(e["regime"] == regime) & (e["h"] == h)]
        for _, r in e.iterrows():
            rows.append((name, r["pair"], int(r["tenor_days"]), float(r["mean"]), int(r["episodes"])))
    return pd.DataFrame(rows, columns=["combination", "pair", "tenor_days", "ev_linear", "episodes"]).sort_values("ev_linear", ascending=False).reset_index(drop=True)
