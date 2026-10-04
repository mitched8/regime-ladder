"""State features as plain functions on a daily market frame.

Contract (the only one in this package):
    f(market, asof=None, **params) -> pd.Series indexed by date
* Uses only rows with date <= asof when `asof` is given.
* Trailing normalisation only. No centred windows, no full-sample statistics.
* Deterministic.

`tests/test_pit_truncation.py` applies the truncation test to every entry in
FEATURES: computing with history cut at T must agree with the full computation
on every date <= T. A feature that fails cannot be registered.

The market frame columns used here (atm_1m, atm_1y, rr25_1m, fly25_1m, spot,
rv_1w, rv_1m, rv_3m) are names, not a requirement: the adapter maps whatever
the source calls them onto these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cut(market: pd.DataFrame, asof) -> pd.DataFrame:
    return market if asof is None else market.loc[: pd.Timestamp(asof)]


def rolling_percentile(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing percentile rank of the latest value within the window, in [0, 100]."""
    mp = min_periods or max(20, window // 4)
    return s.rolling(window, min_periods=mp).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)


def vol_percentile(market, asof=None, col="atm_1m", window=756):
    return rolling_percentile(_cut(market, asof)[col], window).rename("vol_pct")


def term_slope(market, asof=None, short="atm_1m", long="atm_1y", window=756):
    m = _cut(market, asof)
    return rolling_percentile(m[short] - m[long], window).rename("term_slope_pct")


def rr_z(market, asof=None, col="rr25_1m", window=252):
    s = _cut(market, asof)[col]
    z = (s - s.rolling(window, min_periods=60).mean()) / s.rolling(window, min_periods=60).std()
    return rolling_percentile(-z, window).rename("rr_stress_pct")  # more negative RR = more stress


def vrp(market, asof=None, iv="atm_1m", rv="rv_1m", window=756):
    m = _cut(market, asof)
    return rolling_percentile(-(m[iv] - m[rv]), window).rename("vrp_stress_pct")  # thin VRP = stress


def spot_vol_beta(market, asof=None, spot="spot", iv="atm_1m", window=63, rank_window=756):
    m = _cut(market, asof)
    dr = np.log(m[spot]).diff() * 100
    dv = m[iv].diff()
    beta = dv.rolling(window, min_periods=30).cov(dr) / dr.rolling(window, min_periods=30).var()
    return rolling_percentile(beta.abs(), rank_window).rename("spotvol_beta_pct")


def realised_term_ratio(market, asof=None, short="rv_1w", long="rv_3m", window=756):
    m = _cut(market, asof)
    return rolling_percentile(m[short] / m[long], window).rename("rv_term_pct")


FEATURES = {
    "vol_pct": vol_percentile,
    "term_slope_pct": term_slope,
    "rr_stress_pct": rr_z,
    "vrp_stress_pct": vrp,
    "spotvol_beta_pct": spot_vol_beta,
    "rv_term_pct": realised_term_ratio,
}


def build(market: pd.DataFrame, names=None, asof=None, **kw) -> pd.DataFrame:
    names = names or list(FEATURES)
    return pd.concat([FEATURES[n](market, asof=asof, **kw.get(n, {})) for n in names], axis=1)


def truncation_agrees(fn, market: pd.DataFrame, cut_at, atol: float = 1e-9, **params) -> bool:
    """The point-in-time test: cut history at `cut_at`, compare with full history on dates <= cut_at."""
    full = fn(market, **params).loc[: pd.Timestamp(cut_at)]
    cut = fn(market, asof=cut_at, **params)
    a, b = full.align(cut, join="outer")
    both = a.notna() & b.notna()
    if pd.api.types.is_numeric_dtype(a):
        same = np.allclose(a[both].astype(float), b[both].astype(float), atol=atol)
    else:  # categorical outputs (tags) must agree exactly
        same = bool((a[both] == b[both]).all())
    return bool(same and (a.notna() == b.notna()).all())
