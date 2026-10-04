"""Synthetic market and trade-day table with known ground truth.

Purpose: validate the statistics (HAC intervals, block bootstrap, shrinkage,
labeller, persistence split, evaluation) before any real data is touched.
Everything here is deliberately simple and fully specified so that the true
horizon EV can be computed analytically and the estimators checked against it.

Generating process
------------------
* Daily regime path: 3-state Markov chain (carry, transition, crisis).
* Market series: ATM vol level set by regime plus AR(1) noise; RR and fly
  regime-dependent; spot a random walk with the ATM vol; realised vols from
  spot returns; a continuous `stress` score with regime-dependent mean (the
  labeller's input).
* Daily P&L of a trade aged `a` on a day in regime `r` for archetype `A`:

      pnl = base[A][r] + slope[A] * exp(-remaining / decay)        (deterministic earn)
            + common_shock[date, A]  (shared by every live trade of A on that day)
            + idiosyncratic AR(1) noise

  The remaining-tenor term embeds an age-dependence of the kind observed in
  real data, so the ladder's increments have something real to find.

Ground truth
------------
Noise has zero mean, so  E[cum pnl to h | entry regime k]  =
      sum_{d=1..h} ( sum_j (P^d)[k, j] * base[A][j]  +  slope[A] * exp(-(T-d)/decay) ).
`true_ladder` returns exactly that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ("carry", "transition", "crisis")

# Illustrative parameters. Units: cash per unit standard notional per day.
DEFAULT_P = np.array([[0.970, 0.0255, 0.0045],
                      [0.055, 0.900, 0.045],
                      [0.0175, 0.0525, 0.930]])
DEFAULT_BASE = {  # earn per day by regime
    "straddle_atm": {"carry": -1.2, "transition": 0.3, "crisis": 1.5},
    "rr_25d": {"carry": 0.2, "transition": -0.4, "crisis": -1.0},
    "fly_10d": {"carry": 0.5, "transition": -0.2, "crisis": -1.8},
}
DEFAULT_SLOPE = {"straddle_atm": 0.0, "rr_25d": -0.6, "fly_10d": -0.3}  # remaining-tenor term
DEFAULT_SIGMA_COMMON = {"carry": 1.5, "transition": 3.0, "crisis": 5.0}  # shared daily shock
DEFAULT_SIGMA_IDIO = 1.0
DEFAULT_AR = 0.3
DEFAULT_DECAY = 4.0


def _markov_path(n: int, P: np.ndarray, rng: np.random.Generator, start: int = 0) -> np.ndarray:
    s = np.empty(n, dtype=int)
    s[0] = start
    for t in range(1, n):
        s[t] = rng.choice(3, p=P[s[t - 1]])
    return s


def simulate_market(n_days: int = 1260, seed: int = 0, P: np.ndarray = DEFAULT_P) -> pd.DataFrame:
    """Daily market frame indexed by business date with a known regime path."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    reg = _markov_path(n_days, P, rng)
    atm_level = np.array([7.0, 9.5, 14.0])[reg]
    noise = np.zeros(n_days)
    for t in range(1, n_days):
        noise[t] = 0.9 * noise[t - 1] + rng.normal(0, 0.35)
    atm_1m = atm_level + noise
    atm_1y = 0.6 * atm_1m + 0.4 * 8.5 + rng.normal(0, 0.1, n_days)
    rr25 = np.array([-0.4, -1.2, -2.5])[reg] + 0.3 * noise + rng.normal(0, 0.15, n_days)
    fly25 = np.array([0.25, 0.35, 0.6])[reg] + rng.normal(0, 0.03, n_days)
    rets = rng.normal(0, 1, n_days) * atm_1m / 100 / np.sqrt(252)
    spot = 1.10 * np.exp(np.cumsum(rets))
    r = pd.Series(rets, index=dates)
    rv = {w: (r.rolling(w).std() * np.sqrt(252) * 100).bfill().values for w in (5, 21, 63)}
    stress = np.array([25.0, 50.0, 80.0])[reg] + 8.0 * noise + rng.normal(0, 6.0, n_days)
    return pd.DataFrame({
        "regime_true": [REGIMES[i] for i in reg], "stress": stress,
        "atm_1m": atm_1m, "atm_1y": atm_1y, "rr25_1m": rr25, "fly25_1m": fly25,
        "spot": spot, "rv_1w": rv[5], "rv_1m": rv[21], "rv_3m": rv[63],
    }, index=dates).rename_axis("date")


def earn(archetype: str, regime: str, remaining: int, base=DEFAULT_BASE, slope=DEFAULT_SLOPE,
         decay: float = DEFAULT_DECAY) -> float:
    return base[archetype][regime] + slope[archetype] * np.exp(-remaining / decay)


def simulate_trades(market: pd.DataFrame, archetypes=tuple(DEFAULT_BASE), tenor_days: int = 21,
                    pair: str = "EURUSD", seed: int = 1, sigma_common=DEFAULT_SIGMA_COMMON,
                    sigma_idio: float = DEFAULT_SIGMA_IDIO, ar: float = DEFAULT_AR,
                    base=DEFAULT_BASE, slope=DEFAULT_SLOPE, decay: float = DEFAULT_DECAY) -> pd.DataFrame:
    """Trade-day table: one trade per archetype per business day, held to expiry."""
    rng = np.random.default_rng(seed)
    dates = market.index
    reg = market["regime_true"].values
    n = len(dates)
    common = {a: rng.normal(0, 1, n) * np.array([sigma_common[r] for r in reg]) for a in archetypes}
    rows = []
    for a in archetypes:
        for i in range(n - tenor_days):
            tid = f"{pair}-{a}-{dates[i].date()}"
            eps = 0.0
            for age in range(1, tenor_days + 1):
                d = i + age
                eps = ar * eps + rng.normal(0, sigma_idio)
                mu = earn(a, reg[d], tenor_days - age, base, slope, decay)
                pnl = mu + common[a][d] + eps
                rows.append((tid, pair, a, tenor_days, dates[i], dates[d], age, pnl, reg[d]))
    td = pd.DataFrame(rows, columns=["trade_id", "pair", "archetype", "tenor_days", "entry_date",
                                     "date", "age", "pnl", "regime_day"])
    # components that sum exactly to pnl, so the integrity checks have something to check
    td["pnl_trade"] = td["pnl"] * 1.4
    td["pnl_delta_hedge"] = -td["pnl"] * 0.3
    td["pnl_vega_hedge"] = td["pnl"] - td["pnl_trade"] - td["pnl_delta_hedge"]
    return td


def true_ladder(archetypes=tuple(DEFAULT_BASE), tenor_days: int = 21, horizons=(1, 3, 5, 10, 20),
                P: np.ndarray = DEFAULT_P, base=DEFAULT_BASE, slope=DEFAULT_SLOPE,
                decay: float = DEFAULT_DECAY) -> pd.DataFrame:
    """Analytic E[cum pnl to h | entry regime] under the generating process."""
    hs = sorted(set(h for h in horizons if h <= tenor_days) | {tenor_days})
    out = []
    for a in archetypes:
        b = np.array([base[a][r] for r in REGIMES])
        for k, kname in enumerate(REGIMES):
            cum, Pd = 0.0, np.eye(3)
            for d in range(1, tenor_days + 1):
                Pd = Pd @ P
                cum += Pd[k] @ b + slope[a] * np.exp(-(tenor_days - d) / decay)
                if d in hs:
                    out.append((a, kname, d, cum))
    return pd.DataFrame(out, columns=["archetype", "regime", "h", "true_ev"])
