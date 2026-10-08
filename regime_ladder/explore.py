"""The explorer's two tables, and the bridge from a saved filter back into the rigorous pipeline.

The explorer (the Explore tab of `view`) rests on one object and one operation. The object is the
per-entry table: for every fresh unit entered on a date, its cumulative P&L at each horizon, by
component, normalised by the position's own vega at inception (or by notional). The operation is to
condition on a daily statistic the trader picks: "entries made when the ATM z-score was above 2".
The named states, the tags and the finder are all instances of that operation; here the user is the
finder.

`inspect` writes `entries.csv`, `statistics.csv` and `statistics_defs.json`. A filter the owner saves in
the page is a few lines of YAML under `filters:` in the config; `filters_to_tags` turns it into a tag
column, so it splits the ladder's cells, appears in the packets and is tested like any other tag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import leading, schema

HORIZONS = (1, 3, 5, 10, 20)
GROUP = ["pair", "archetype", "tenor_days"]


# ----------------------------------------------------------------------------- per-entry earn
def entries(td: pd.DataFrame, horizons=HORIZONS, normalise: str = "vega", include_expiry: bool = True) -> pd.DataFrame:
    """One row per (pair, archetype, tenor, entry date, horizon): cumulative P&L of a fresh unit, total and by
    component, divided by the unit's |vega| at age 1 (`normalise="vega"`, falling back to notional where vega is
    missing or zero) or left in cash per unit notional (`"notional"`). Several trades entered the same day are averaged."""
    td = td.sort_values(["trade_id", "age"])
    comps = [c for c in schema.COMPONENTS if c in td.columns]
    cols = ["pnl"] + comps
    cum = td.groupby(GROUP + ["trade_id"])[cols].cumsum().add_prefix("cum_")      # a trade id is unique within its group; never trust it across groups
    t = pd.concat([td[GROUP + ["trade_id", "entry_date", "age"]], cum], axis=1)
    if normalise == "vega" and "vega" in td.columns:
        v0 = td.groupby(GROUP + ["trade_id"])["vega"].first().abs()          # td is sorted by age, so first = vega at inception
        scale = pd.MultiIndex.from_frame(t[GROUP + ["trade_id"]]).map(v0).to_series(index=t.index).astype(float).replace(0, np.nan)
        t["scale"] = scale.fillna(1.0)
        t["normalised_by"] = np.where(scale.notna(), "vega", "notional")
    else:
        t["scale"] = 1.0
        t["normalised_by"] = "notional"
    keep = t["age"].isin(horizons)
    if include_expiry:
        keep |= t["age"] == t["tenor_days"]
    t = t[keep].copy()
    t["h"] = t["age"]
    t["to_expiry"] = t["h"] == t["tenor_days"]
    for c in cols:
        t[c.replace("pnl", "earn")] = t[f"cum_{c}"] / t["scale"]
    earn_cols = [c.replace("pnl", "earn") for c in cols]
    out = (t.groupby(GROUP + ["entry_date", "h", "to_expiry"], observed=True)
             .agg(**{c: (c, "mean") for c in earn_cols}, n_trades=("trade_id", "nunique"), vega0=("scale", "mean"))
             .reset_index())
    out.attrs["normalised_by"] = "vega" if (t["normalised_by"] == "vega").all() else ("notional" if (t["normalised_by"] == "notional").all() else "mixed")
    out.attrs["components"] = [c.replace("pnl", "earn") for c in comps]
    return out


def trailing_earn(ent: pd.DataFrame, index: pd.DatetimeIndex, h: int = 5, window: int = 20, zwindow: int = 252) -> pd.DataFrame:
    """Per archetype (smallest tenor), the strategy's own recent performance as a point-in-time statistic: the
    realised h-day earn of each entry is stamped on the day it is known (entry + h trading days), averaged over
    the trailing `window` days, then z-scored against its trailing `zwindow`. Column `earn{window}_{archetype}_z`."""
    out = pd.DataFrame(index=index)
    pos = pd.Series(np.arange(len(index)), index=index)
    for arch, g in ent[(ent["h"] == h) & (~ent["to_expiry"])].groupby("archetype"):
        g = g[g["tenor_days"] == g["tenor_days"].min()]
        p = pos.reindex(g["entry_date"]).values + h
        ok = np.isfinite(p) & (p < len(index))
        s = pd.Series(np.nan, index=index)
        s.iloc[p[ok].astype(int)] = g["earn"].values[ok]
        r = s.rolling(window, min_periods=max(5, window // 2)).mean()
        z = (r - r.rolling(zwindow, min_periods=zwindow // 2).mean().shift(1)) / r.rolling(zwindow, min_periods=zwindow // 2).std().shift(1)
        out[f"earn{window}_{arch}_z"] = z
    return out


# ----------------------------------------------------------------------------- daily statistics
def _z(s: pd.Series, window: int = 252) -> pd.Series:
    mp = max(60, window // 4)
    return (s - s.rolling(window, min_periods=mp).mean().shift(1)) / s.rolling(window, min_periods=mp).std().shift(1)


def raw_statistics(market: pd.DataFrame, spot="spot", iv="atm_1m", iv_long="atm_1y", rr="rr25_1m", fly="fly25_1m",
                   rv="rv_1m", rv_short="rv_1w") -> tuple[pd.DataFrame, dict]:
    """Plain, trader-readable statistics straight from the market frame, all trailing. Returns (frame, definitions)."""
    m = market
    out, defs = pd.DataFrame(index=m.index), {}

    def add(name, s, d):
        out[name] = s; defs[name] = d

    if iv in m:
        add("atm_1m", m[iv], "1m ATM implied vol, vol points")
        add("atm_1m_z252", _z(m[iv]), "1m ATM vol: z-score against its trailing 252 days (window ends yesterday)")
        add("atm_1m_chg5", m[iv].diff(5), "1m ATM vol: change over the last 5 trading days, vol points")
        add("atm_1m_chg20", m[iv].diff(20), "1m ATM vol: change over the last 20 trading days, vol points")
    if iv in m and iv_long in m:
        add("term_1y_1m", m[iv_long] - m[iv], "term structure: 1y ATM minus 1m ATM, vol points (positive = upward sloping)")
    if rr in m:
        add("rr25_1m", m[rr], "1m 25-delta risk reversal, vol points")
        add("rr25_1m_z252", _z(m[rr]), "1m 25d risk reversal: z-score against its trailing 252 days")
    if fly in m:
        add("fly25_1m", m[fly], "1m 25-delta butterfly, vol points")
    if rv in m and iv in m:
        add("rv_iv_1m", m[rv] / m[iv], "realised vol (1m) / implied vol (1m): below 1 = implied rich")
    if rv_short in m and rv in m:
        add("rv_1w_1m", m[rv_short] / m[rv], "short realised (1w) / long realised (1m): above 1 = realised picking up")
    if spot in m:
        ls = np.log(m[spot])
        for w in (20, 60, 120):
            s = m[iv] / 100 * np.sqrt(w / 252) if iv in m else ls.diff().rolling(w).std() * np.sqrt(w)
            add(f"spot_vs_ma{w}", (ls - ls.rolling(w, min_periods=w // 2).mean().shift(1)) / s,
                f"spot's distance from its {w}-day moving average, in units of the move implied vol allows over {w} days (signed)")
        if iv in m:
            add("spot_ret20_z", ls.diff(20) / (m[iv] / 100 * np.sqrt(20 / 252)), "20-day spot return divided by the implied 20-day move: a trend in vol units (signed)")
        add("drift_t60", leading.drift_t(m, window=60), "t-statistic of the mean daily spot return over 60 days: one-way market (signed)")
    return out, defs


# ----------------------------------------------------------------------------- adding a statistic
# Three ways, in order of effort:
#   1. a raw market column: list it under `explore: passthrough:` in the config ({column: "one-line definition"});
#   2. a function of the market frame: add it to EXTRA below, `name: (fn(market) -> Series, definition, group)`,
#      trailing data only (the truncation test in tests/test_explore.py runs over every entry);
#   3. site-specific series (flow, positioning): an untracked `adapters/statistics.py` with
#      `def extra(market) -> tuple[pd.DataFrame, dict[str, str]]` returning the columns and their definitions;
#      it is imported if present and never committed.
EXTRA: dict[str, tuple] = {
    "atm_1m_pct252": (lambda m: features_pct(m["atm_1m"], 252), "1m ATM vol: trailing-252-day percentile (0–100)", "surface & spot"),
    "atm_1m_minus_rv_1m": (lambda m: m["atm_1m"] - m["rv_1m"], "implied minus realised (1m), vol points: the carry on offer", "surface & spot"),
    "spot_range20_z": (lambda m: (np.log(m["spot"]).rolling(20).max() - np.log(m["spot"]).rolling(20).min()) / (m["atm_1m"] / 100 * np.sqrt(20 / 252)),
                       "20-day spot high–low range divided by the implied 20-day move: how much of the implied range was used", "surface & spot"),
}


def features_pct(s: pd.Series, window: int = 252) -> pd.Series:
    from .features import rolling_percentile
    return rolling_percentile(s, window)


def extra_statistics(market: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """EXTRA, the config's passthrough columns, and the adapter hook. Returns (frame, defs, groups)."""
    out, defs, groups = pd.DataFrame(index=market.index), {}, {}
    for name, (fn, d, g) in EXTRA.items():
        try:
            out[name] = fn(market); defs[name] = d; groups[name] = g
        except KeyError:
            continue                                                 # the frame lacks a column this one needs
    for col, d in ((cfg or {}).get("passthrough") or {}).items():
        if col in market:
            out[col] = market[col]; defs[col] = d or f"{col}: market column"; groups[col] = "market columns"
    try:
        from adapters import statistics as site                      # untracked, site-specific
        ex, exd = site.extra(market)
        for c in ex.columns:
            out[c] = ex[c].reindex(market.index); defs[c] = exd.get(c, c); groups[c] = "site series"
    except ImportError:
        pass
    return out, defs, groups


def statistics(market: pd.DataFrame, features: pd.DataFrame | None = None, composite: pd.DataFrame | None = None,
               leading_feats: pd.DataFrame | None = None, shock: pd.DataFrame | None = None,
               characteristics: pd.DataFrame | None = None, ent: pd.DataFrame | None = None, cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Everything a trader can condition on, one column per statistic on the market index, with one-line definitions
    and a group per statistic. Categorical columns (the named state, the level and direction bands) are strings."""
    raw, defs = raw_statistics(market)
    groups = {k: "surface & spot" for k in raw.columns}
    xtra, xd, xg = extra_statistics(market, cfg)
    defs.update(xd); groups.update(xg)
    parts = [raw, xtra]

    def take(df, cols, group, deftext):
        if df is None:
            return
        sub = df[[c for c in cols if c in df.columns]].copy()
        for c in sub.columns:
            defs[c] = deftext(c); groups[c] = group
        parts.append(sub)

    take(features, list(features.columns) if features is not None else [], "state features (trailing percentiles 0–100)",
         lambda c: f"{c}: trailing-percentile state feature (0–100), one of the composite's inputs" if c in ("vol_pct", "term_slope_pct", "rr_stress_pct") else f"{c}: trailing-percentile state feature (0–100), registered but not in the composite")
    if composite is not None:
        take(composite, ["composite", "smoothed", "direction_score"], "state score",
             lambda c: {"composite": "mean of the state features' percentiles (0–100)", "smoothed": "EWMA of the composite: the level the state bounds are applied to",
                        "direction_score": "standardised slope of the smoothed composite: the direction axis"}[c])
        take(composite, ["regime", "level", "direction"], "named state (categorical)",
             lambda c: {"regime": "the named state in use (carry, settling, rising, agitated, normalising, stressed, extreme)", "level": "level band of the smoothed composite (low / mid / high / extreme)",
                        "direction": "direction band (down / flat / up)"}[c])
    take(leading_feats, list(leading_feats.columns) if leading_feats is not None else [], "leading candidates",
         lambda c: f"{c}: leading-feature candidate as screened in Phase 4" + (" (0–100)" if c.endswith("_pct") or c in ("complacency", "stored_energy") else " (signed)"))
    take(shock, ["score", "cusum_pressure", "surprise", "har_vol_forecast"], "shock detectors",
         lambda c: {"score": "shock score 0–1: mean of CUSUM pressure, BOCD P(short run) and recent |surprise|", "cusum_pressure": "CUSUM statistic / its alarm threshold (1 = alarm)",
                    "surprise": "today's realised move against the HAR forecast, standardised", "har_vol_forecast": "HAR forecast of realised vol"}[c])
    take(characteristics, list(characteristics.columns) if characteristics is not None else [], "characteristics (profile inputs)",
         lambda c: f"{c}: daily characteristic from the profile frame (own-pair, cross-pair or cross-asset), trailing window")
    if ent is not None and len(ent):
        te = trailing_earn(ent, market.index)
        for c in te.columns:
            defs[c] = f"{c}: the strategy's own recent performance — realised 5d earn of its entries, known at entry+5, trailing 20-day mean, z-scored vs 252 days"
            groups[c] = "strategy's own recent earn"
        parts.append(te)
    df = pd.concat(parts, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df, {"defs": defs, "groups": groups}


# ----------------------------------------------------------------------------- a saved filter as a tag
def apply_filter(stats: pd.DataFrame, spec: dict) -> pd.Series:
    """Boolean series for one saved filter: {stat, op, value} with op in > < >= <= between in, and optional
    `and: [...]` of further clauses. `between` takes value: [lo, hi]; `in` takes a list of categories."""
    s = stats[spec["stat"]]
    op, v = spec.get("op", ">"), spec.get("value")
    if op == ">":
        m = s > v
    elif op == ">=":
        m = s >= v
    elif op == "<":
        m = s < v
    elif op == "<=":
        m = s <= v
    elif op == "between":
        m = (s >= v[0]) & (s <= v[1])
    elif op == "in":
        m = s.isin(list(v))
    else:
        raise ValueError(f"unknown op {op}")
    m = m.fillna(False)
    for clause in spec.get("and", []) or []:
        m &= apply_filter(stats, clause)
    return m.rename(spec.get("name", spec["stat"]))


def filters_to_tags(stats: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """{name: spec} -> one tag column per filter with values 'on' / 'none' (so `tags.tag_split` adds a cell only
    for the condition, as for any other tag)."""
    out = pd.DataFrame(index=stats.index)
    for name, spec in (filters or {}).items():
        m = apply_filter(stats, {**spec, "name": name})
        out[name] = np.where(m, "on", "none")
    return out


def filter_yaml(name: str, spec: dict) -> str:
    """The YAML the page offers for copying into `filters:` of the config."""
    lines = [f"  {name}:", f"    stat: {spec['stat']}", f"    op: '{spec['op']}'", f"    value: {spec['value']}"]
    if spec.get("and"):
        lines.append("    and:")
        for c in spec["and"]:
            lines.append(f"      - {{stat: {c['stat']}, op: '{c['op']}', value: {c['value']}}}")
    return "\n".join(lines)
