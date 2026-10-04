"""Multi-horizon, entry-conditional statistics: the horizon ladder.

Inputs: a trade-day table (schema.py) and a label frame (pair, date, regime).
Output: for each (pair, archetype, tenor) x entry regime x horizon h —
mean cumulative P&L, an interval that respects overlap, the outcome
distribution, effective N and the number of independent regime episodes.

Reference statistic: hold-for-h from entry, no exit rule, net daily P&L per
unit of standard notional. Within-horizon regime transitions are in the data;
nothing is added for them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10, 20)
GROUP = ["pair", "archetype", "tenor_days"]


# ----------------------------------------------------------------------------- building blocks
def cumulative(td: pd.DataFrame, horizons=HORIZONS, include_expiry: bool = True) -> pd.DataFrame:
    """Cumulative P&L of each trade at each horizon h <= tenor (plus to-expiry)."""
    td = td.sort_values(["trade_id", "age"])
    td = td.assign(cum_pnl=td.groupby("trade_id")["pnl"].cumsum())
    keep = td["age"].isin(horizons)
    if include_expiry:
        keep |= td["age"] == td["tenor_days"]
    cum = td.loc[keep, GROUP + ["trade_id", "entry_date", "age", "cum_pnl"]].rename(columns={"age": "h"})
    cum["to_expiry"] = cum["h"] == cum["tenor_days"]
    return cum.reset_index(drop=True)


def attach_entry_labels(cum: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join the point-in-time label on the entry date. Rows without a label are dropped and counted."""
    lab = labels.rename(columns={"date": "entry_date"})[["pair", "entry_date", "regime"]]
    out = cum.merge(lab, on=["pair", "entry_date"], how="left")
    out.attrs["unlabelled_entries"] = int(out["regime"].isna().sum())
    return out.dropna(subset=["regime"])


def hac_se(x: np.ndarray, bandwidth: int) -> float:
    """Newey-West (Bartlett) standard error of the mean of an ordered series."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    L = int(min(bandwidth, n - 1))
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        var += 2.0 * w * (e[l:] @ e[:-l]) / n
    return float(np.sqrt(max(var, 0.0) / n))


def block_bootstrap_ci(x: np.ndarray, block_len: int, n_boot: int = 2000, alpha: float = 0.10,
                       rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Circular block bootstrap interval for the mean of an ordered series."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    L = max(1, min(block_len, n))
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    means = x[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def count_episodes(labels: pd.DataFrame) -> dict:
    """Number of contiguous runs of each regime in the (date-ordered) label series, per pair."""
    out = {}
    for pair, g in labels.sort_values("date").groupby("pair"):
        r = g["regime"].values
        runs = {}
        for i, v in enumerate(r):
            if i == 0 or r[i - 1] != v:
                runs[v] = runs.get(v, 0) + 1
        out[pair] = runs
    return out


def entry_spacing_days(entry_dates: pd.Series) -> float:
    d = pd.Series(sorted(pd.to_datetime(entry_dates).unique()))
    if len(d) < 2:
        return 1.0
    return float(np.median(np.busday_count(d.values[:-1].astype("datetime64[D]"), d.values[1:].astype("datetime64[D]"))))


# ----------------------------------------------------------------------------- the ladder
def ladder(cum_lab: pd.DataFrame, labels: pd.DataFrame, alpha: float = 0.10, ci: str = "hac",
           n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Entry-conditional ladder. One row per group x regime x h, plus an 'ALL' (unconditional) regime row."""
    rng = np.random.default_rng(seed)
    eps = count_episodes(labels)
    rows = []
    z = 1.6448536 if abs(alpha - 0.10) < 1e-9 else float(abs(np.quantile(np.random.default_rng(1).standard_normal(400000), alpha / 2)))
    for gkey, g in cum_lab.groupby(GROUP):
        spacing = entry_spacing_days(g["entry_date"])
        regimes = ["ALL"] + sorted(g["regime"].unique())
        for reg in regimes:
            gr = g if reg == "ALL" else g[g["regime"] == reg]
            for h, gh in gr.groupby("h"):
                x = gh.sort_values("entry_date")["cum_pnl"].values
                n = len(x)
                mean = float(x.mean()) if n else np.nan
                n_eff = n * min(1.0, spacing / h)
                if ci == "boot":
                    lo, hi = block_bootstrap_ci(x, block_len=2 * h, n_boot=n_boot, alpha=alpha, rng=rng)
                    se = (hi - lo) / (2 * z) if np.isfinite(lo) else np.nan
                else:
                    se = hac_se(x, bandwidth=h)
                    lo, hi = mean - z * se, mean + z * se
                q05, q50, q95 = (np.quantile(x, [0.05, 0.5, 0.95]) if n else (np.nan,) * 3)
                es05 = float(x[x <= q05].mean()) if n else np.nan
                pair = gkey[0]
                n_ep = sum(eps.get(pair, {}).values()) if reg == "ALL" else eps.get(pair, {}).get(reg, 0)
                rows.append((*gkey, reg, int(h), bool(gh["to_expiry"].iloc[0]), n, round(n_eff, 1), mean, se,
                             lo, hi, float((x > 0).mean()), q05, q50, q95, es05, n_ep))
    cols = GROUP + ["regime", "h", "to_expiry", "n", "n_eff", "mean", "se", "ci_lo", "ci_hi",
                    "p_profit", "q05", "q50", "q95", "es05", "episodes"]
    return pd.DataFrame(rows, columns=cols).sort_values(GROUP + ["regime", "h"]).reset_index(drop=True)


def increments(lad: pd.DataFrame) -> pd.DataFrame:
    """Expected earn per day between consecutive horizons: the aged-position rate."""
    out = []
    for key, g in lad.groupby(GROUP + ["regime"]):
        g = g.sort_values("h")
        prev_h, prev_m = 0, 0.0
        for _, r in g.iterrows():
            dh = r["h"] - prev_h
            out.append((*key, f"d{prev_h + 1}-{int(r['h'])}", prev_h + 1, int(r["h"]), (r["mean"] - prev_m) / dh))
            prev_h, prev_m = r["h"], r["mean"]
    return pd.DataFrame(out, columns=GROUP + ["regime", "bucket", "h_from", "h_to", "earn_per_day"])


def shrink(lad: pd.DataFrame, kappa: float = 20.0) -> pd.DataFrame:
    """Shrink each regime mean toward the unconditional mean at the same horizon. Tails are left unshrunk."""
    lad = lad.copy()
    all_mean = lad[lad["regime"] == "ALL"].set_index(GROUP + ["h"])["mean"]
    key = list(zip(*[lad[c] for c in GROUP + ["h"]]))
    m_all = np.array([all_mean.get(k, np.nan) for k in key])
    w = lad["n_eff"] / (lad["n_eff"] + kappa)
    lad["w_shrink"] = np.where(lad["regime"] == "ALL", 1.0, w)
    lad["mean_shrunk"] = lad["w_shrink"] * lad["mean"] + (1 - lad["w_shrink"]) * m_all
    return lad


def persistence_split(td: pd.DataFrame, labels: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """For each group x entry regime x h: EV | entry regime persisted, EV | it broke, and q = P(broke).

    Needs the label on every day of each trade's life. If `regime_day` is not in td it is joined from
    `labels` on (pair, date). The identity (1-q)*EV_stay + q*EV_broke == EV holds exactly in sample.
    """
    td = td.copy()
    if "regime_day" not in td.columns:
        td = td.merge(labels, on=["pair", "date"], how="left").rename(columns={"regime": "regime_day"})
    ent = labels.rename(columns={"date": "entry_date", "regime": "regime_entry"})
    td = td.merge(ent, on=["pair", "entry_date"], how="inner")
    td = td.sort_values(["trade_id", "age"])
    td["cum_pnl"] = td.groupby("trade_id")["pnl"].cumsum()
    td["left"] = (td["regime_day"] != td["regime_entry"]).astype(int)
    td["left_by"] = td.groupby("trade_id")["left"].cummax()
    rows = []
    hs = sorted(set(horizons) | set(td["tenor_days"].unique()))
    for h in hs:
        at = td[td["age"] == h]
        for key, g in at.groupby(GROUP + ["regime_entry"]):
            q = g["left_by"].mean()
            stay, broke = g[g["left_by"] == 0]["cum_pnl"], g[g["left_by"] == 1]["cum_pnl"]
            rows.append((*key, int(h), len(g), float(q), float(stay.mean()) if len(stay) else np.nan,
                         float(broke.mean()) if len(broke) else np.nan, float(g["cum_pnl"].mean())))
    return pd.DataFrame(rows, columns=GROUP + ["regime", "h", "n", "q_broke", "ev_stayed", "ev_broke", "ev"])
