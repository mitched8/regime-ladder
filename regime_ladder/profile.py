"""State profile: what a state looks like, in the trader's terms.

`characteristics` builds a daily, trailing, point-in-time vector describing the market: the
pair's own spot-vol behaviour, realised vs implied, skew and wings; how much of its move is the
dollar factor versus idiosyncratic; and how spot and vol co-move with other assets. `state_profile`
summarises that vector per state against the unconditional distribution and ranks what makes
each state different. `describe` turns the top differences into a phrase.

Column names are generic; the adapter maps the source's series onto them (configs/profile.yaml).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PROFILE = {
    "own": {"spot": "spot", "iv": "atm_1m", "iv_long": "atm_1y", "rr": "rr25_1m", "fly": "fly25_1m", "rv": "rv_1m"},
    "cross_pair": ["g10_1", "g10_2", "g10_3", "g10_4", "g10_5", "g10_6"],
    "cross_asset": {"eq": "pct", "eq2": "pct", "rate2y": "diff", "rate10y": "diff", "oil": "pct", "gold": "pct",
                    "dxy": "pct", "credit": "pct", "vix": "pct", "move": "pct"},
    "window": 20,
}


def _chg(s: pd.Series, how: str) -> pd.Series:
    return s.pct_change() if how == "pct" else s.diff()


def _pc1_share(R: np.ndarray) -> float:
    C = np.corrcoef(R.T)
    if not np.all(np.isfinite(C)):
        return np.nan
    w = np.linalg.eigvalsh(C)
    return float(w[-1] / w.sum())


def characteristics(market: pd.DataFrame, cfg: dict = DEFAULT_PROFILE, asof=None) -> pd.DataFrame:
    m = market if asof is None else market.loc[: pd.Timestamp(asof)]
    W, own = cfg["window"], cfg["own"]
    r = np.log(m[own["spot"]]).diff()
    div = m[own["iv"]].diff()
    out = pd.DataFrame(index=m.index)
    rv_w = r.rolling(W)
    out["spotvol_beta"] = div.rolling(W).cov(r) / rv_w.var()
    out["spotvol_corr"] = div.rolling(W).corr(r)
    out["vrp"] = m[own["iv"]] - m[own["rv"]]
    out["term_slope"] = m[own["iv"]] - m[own["iv_long"]]
    out["rr_level"] = m[own["rr"]]
    out["fly_level"] = m[own["fly"]]
    out["volofvol"] = div.rolling(W).std()
    sd = r.rolling(63).std()
    out["jump_freq"] = (r.abs() > 2.5 * sd.shift(1)).astype(float).rolling(W).mean()
    # cross-pair: dollar-factor share and average pairwise correlation of the G10 panel
    cols = [c for c in cfg["cross_pair"] if c in m.columns]
    if cols:
        R = pd.concat([r.rename("pair")] + [np.log(m[c]).diff().rename(c) for c in cols], axis=1)
        share, avgc = np.full(len(m), np.nan), np.full(len(m), np.nan)
        vals = R.values
        for t in range(W, len(m)):
            win = vals[t - W + 1: t + 1]
            if np.isnan(win).any():
                continue
            C = np.corrcoef(win.T)
            share[t] = _pc1_share(win)
            avgc[t] = (C.sum() - len(C)) / (len(C) * (len(C) - 1))
        out["dollar_share"] = share
        out["avg_pair_corr"] = avgc
    # cross-asset: correlation of the pair's spot return and of its vol change with each asset
    for col, how in cfg["cross_asset"].items():
        if col in m.columns:
            dx = _chg(m[col], how)
            out[f"corr_spot_{col}"] = r.rolling(W).corr(dx)
            out[f"corr_vol_{col}"] = div.rolling(W).corr(dx)
    return out


def state_profile(chars: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Per-characteristic: unconditional mean/sd, each state's mean, and the standardised difference d."""
    df = chars.join(labels.rename("state"), how="inner")
    mu, sd = df.drop(columns="state").mean(), df.drop(columns="state").std()
    out = pd.DataFrame({"all_mean": mu, "all_sd": sd})
    for s, g in df.groupby("state"):
        out[f"{s}_mean"] = g.drop(columns="state").mean()
        out[f"{s}_d"] = (out[f"{s}_mean"] - mu) / sd
    out["n_days"] = len(df)
    return out


def distinguishing(profile: pd.DataFrame, state: str, top: int = 3) -> pd.DataFrame:
    d = profile[f"{state}_d"].dropna()
    idx = d.abs().sort_values(ascending=False).index[:top]
    return pd.DataFrame({"d": d[idx], "state_mean": profile.loc[idx, f"{state}_mean"], "all_mean": profile.loc[idx, "all_mean"]})


def describe(profile: pd.DataFrame, state: str, top: int = 3) -> str:
    parts = [f"{c} {'high' if r.d > 0 else 'low'} ({r.d:+.1f}σ)" for c, r in distinguishing(profile, state, top).iterrows()]
    return f"{state} · " + " · ".join(parts)


def today_placement(chars: pd.DataFrame, labels: pd.Series, date=None) -> pd.DataFrame:
    """Where today's characteristics sit within today's state's historical distribution (percentiles)."""
    date = pd.Timestamp(date) if date is not None else chars.index[-1]
    state = labels.loc[date]
    hist = chars.loc[labels.index[labels == state]].dropna(how="all")
    today = chars.loc[date]
    pct = {c: float((hist[c].dropna() < today[c]).mean()) for c in chars.columns if pd.notna(today[c])}
    return pd.DataFrame({"today": today, "pct_within_state": pd.Series(pct), "state": state})
