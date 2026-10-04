"""Synthetic market and trade-day table with known ground truth.

Purpose: validate the statistics (intervals, shrinkage, labeller, calibration, discovery, profile,
tags, persistence split, evaluation) before any real data is touched. Everything is simple and
fully specified so the true horizon EV is analytic and the estimators can be checked.

Generating process
------------------
* Daily state path: a Markov chain over carry, rising, agitated, stressed, normalising, settling
  and a RARE `extreme` state (a few episodes in ten years) — which is level x direction:
  carry (low, flat), rising (low->mid, up), agitated (mid, flat, can persist for weeks),
  stressed (high), normalising (high->mid, down), settling (low, down), extreme (tail).
* Hidden sub-types. Rising: "usd_led" (pair returns strongly correlated with equities and
  front-end rates, dollar factor dominant) or "idiosyncratic". Agitated: "corr_neg" or "corr_pos"
  — the sign of spot-vol correlation flips between episodes. Discovery and the correlation tag are
  expected to recover them; nothing else uses them.
* Stored energy (optional, `energy_beta` > 0): the daily matrix is tilted toward escalation by a
  pressure built from the simulated path itself — a large gap between spot and its trailing anchor
  (in implied-vol units) while implied is low. That is the "gap held by something temporary"
  hypothesis made causal, so the leading-feature machinery has a world in which it must find
  incremental value, and (`energy_beta` = 0, the default) one in which it must find none.
* Market: ATM vol mean-reverts toward the state's target (so direction is informative) with a
  spot-vol coupling whose sign depends on the agitated sub-type; RR and fly state-dependent; spot a
  random walk driven by a dollar factor plus idiosyncratic noise with sub-type-dependent factor
  share; a G10 panel loading on the same factor; cross-asset series with sub-type-dependent
  correlation to the pair.
* Daily P&L of a trade aged `a` on a day in state `s` for archetype `A`:
      pnl = base[A][s] + slope[A] * exp(-remaining / decay) + common_shock[date, A] + AR(1) noise

Ground truth: E[cum pnl to h | entry state k] = sum_d ( sum_j (P^d)[k,j] base[A][j] + slope[A] exp(-(T-d)/decay) ).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ("carry", "rising", "agitated", "stressed", "normalising", "settling", "extreme")
RISING_SUBTYPES = ("usd_led", "idiosyncratic")
AGITATED_SUBTYPES = ("corr_neg", "corr_pos")

DEFAULT_P = np.array([
    # carry  rising  agit   stress normal settle extreme
    [0.962, 0.030, 0.003, 0.000, 0.000, 0.005, 0.000],  # carry
    [0.010, 0.860, 0.080, 0.030, 0.000, 0.020, 0.000],  # rising
    [0.000, 0.030, 0.920, 0.025, 0.000, 0.025, 0.000],  # agitated
    [0.000, 0.010, 0.030, 0.890, 0.060, 0.000, 0.010],  # stressed
    [0.020, 0.010, 0.060, 0.040, 0.830, 0.040, 0.000],  # normalising
    [0.110, 0.030, 0.020, 0.000, 0.010, 0.830, 0.000],  # settling
    [0.000, 0.000, 0.000, 0.150, 0.050, 0.000, 0.800],  # extreme (rare, short)
])
ATM_TARGET = {"carry": 7.0, "rising": 10.0, "agitated": 10.5, "stressed": 13.5, "normalising": 10.0, "settling": 7.5, "extreme": 22.0}
RR_LEVEL = {"carry": -0.4, "rising": -1.2, "agitated": -1.3, "stressed": -2.4, "normalising": -1.6, "settling": -0.8, "extreme": -4.0}
FLY_LEVEL = {"carry": 0.25, "rising": 0.35, "agitated": 0.38, "stressed": 0.58, "normalising": 0.45, "settling": 0.30, "extreme": 0.90}
STRESS_MEAN = {"carry": 25.0, "rising": 48.0, "agitated": 52.0, "stressed": 80.0, "normalising": 60.0, "settling": 35.0, "extreme": 97.0}

DEFAULT_BASE = {  # earn per day by state, cash per unit vega at inception for the legs
    "straddle_atm": {"carry": -1.2, "rising": 0.8, "agitated": 0.1, "stressed": 1.4, "normalising": -2.0, "settling": -1.0, "extreme": 4.0},
    "put_25d": {"carry": -0.5, "rising": 0.7, "agitated": 0.2, "stressed": 1.2, "normalising": -1.2, "settling": -0.6, "extreme": 3.5},
    "call_25d": {"carry": -0.3, "rising": 0.1, "agitated": -0.1, "stressed": 0.2, "normalising": -0.7, "settling": -0.3, "extreme": 0.5},
    "put_10d": {"carry": -0.4, "rising": 0.6, "agitated": 0.1, "stressed": 1.6, "normalising": -1.4, "settling": -0.5, "extreme": 5.0},
    "call_10d": {"carry": -0.3, "rising": 0.0, "agitated": -0.2, "stressed": 0.1, "normalising": -0.6, "settling": -0.3, "extreme": 0.3},
    "rr_25d": {"carry": 0.2, "rising": -0.8, "agitated": -0.4, "stressed": -1.4, "normalising": 0.7, "settling": 0.3, "extreme": -3.0},
    "fly_10d": {"carry": 0.5, "rising": -0.8, "agitated": -0.4, "stressed": -2.0, "normalising": 1.4, "settling": 0.6, "extreme": -5.0},
}
DEFAULT_SLOPE = {"straddle_atm": 0.0, "put_25d": -0.4, "call_25d": -0.2, "put_10d": -0.5, "call_10d": -0.2, "rr_25d": -0.6, "fly_10d": -0.3}
# Within agitated, the hidden spot-vol correlation sign moves the skew legs' earn symmetrically (the state mean
# is unchanged, so the analytic ladder still holds): with vol bid on rallies the long-call side earns.
SUBTYPE_ADJ = {"rr_25d": {"corr_neg": -0.6, "corr_pos": 0.6}, "call_25d": {"corr_neg": -0.3, "corr_pos": 0.3}, "put_25d": {"corr_neg": 0.3, "corr_pos": -0.3},
               "call_10d": {"corr_neg": -0.25, "corr_pos": 0.25}, "put_10d": {"corr_neg": 0.25, "corr_pos": -0.25}}
DEMO_ARCHETYPES = ("straddle_atm", "rr_25d", "fly_10d")
DEFAULT_SIGMA_COMMON = {"carry": 1.5, "rising": 3.0, "agitated": 3.0, "stressed": 5.0, "normalising": 3.5, "settling": 2.0, "extreme": 9.0}
DEFAULT_SIGMA_IDIO = 1.0
DEFAULT_AR = 0.3
DEFAULT_DECAY = 4.0

CA_BETA = {  # (none, usd_led, idiosyncratic)
    "eq": (0.3, 0.9, 0.0), "eq2": (0.25, 0.8, 0.0), "rate2y": (0.2, 0.7, 0.0), "rate10y": (0.15, 0.5, 0.0),
    "oil": (0.2, 0.4, 0.0), "gold": (-0.1, -0.3, 0.0), "dxy": (-0.6, -0.9, -0.3), "credit": (-0.2, -0.7, 0.0),
    "vix": (-0.3, -0.8, 0.0), "move": (-0.1, -0.5, 0.0),
}
FACTOR_SHARE = {"none": 0.6, "usd_led": 0.9, "idiosyncratic": 0.15}
SPOTVOL_COUPLING = {"none": -0.25, "corr_neg": -0.9, "corr_pos": 0.9}  # vol change per standardised spot return, in vol points
G10 = ("g10_1", "g10_2", "g10_3", "g10_4", "g10_5", "g10_6")


def _markov_path(n, P, rng, start=0):
    s = np.empty(n, dtype=int)
    s[0] = start
    for t in range(1, n):
        s[t] = rng.choice(len(P), p=P[s[t - 1]])
    return s


ENERGY_ANCHOR, ENERGY_LOWVOL, ENERGY_THRESHOLD = 120, 9.0, 1.0


def energy_pressure(gap_z: float, atm: float) -> float:
    """The synthetic world's own stored-energy rule: pressure = excess of |gap| (in implied units) over a
    threshold, counted in full when implied is low and at 30% otherwise. Zero most of the time."""
    weak = 1.0 if atm < ENERGY_LOWVOL else 0.3
    return float(np.clip((abs(gap_z) * weak - ENERGY_THRESHOLD), 0.0, 3.0))


def _simulate_with_energy(n_days, P, rng, energy_beta):
    """Sequential simulation: today's pressure (from the path so far) tilts tonight's transition."""
    from .transitions import _tilt_array, ranks
    r = ranks(STATES)
    s = np.empty(n_days, dtype=int); s[0] = 0
    sub = np.array(["none"] * n_days, dtype=object); cur = "none"
    z_pair = np.empty(n_days); atm = np.empty(n_days); atm[0] = 7.0; logspot = np.empty(n_days); logspot[0] = np.log(1.10)
    psi = np.zeros(n_days)
    w0 = FACTOR_SHARE["none"]
    f_all, u_all, n_all = rng.normal(0, 1, n_days), rng.normal(0, 1, n_days), rng.normal(0, 0.3, n_days)
    z_pair[0] = np.sqrt(w0) * f_all[0] + np.sqrt(1 - w0) * u_all[0]
    for t in range(1, n_days):
        T = _tilt_array(P, r, np.array([psi[t - 1]]), energy_beta)[0]
        s[t] = rng.choice(len(P), p=T[s[t - 1]])
        st = STATES[s[t]]
        if st in ("rising", "agitated"):
            if STATES[s[t - 1]] != st:
                cur = (RISING_SUBTYPES if st == "rising" else AGITATED_SUBTYPES)[rng.integers(2)]
            sub[t] = cur
        w = FACTOR_SHARE.get(sub[t], 0.6)
        z_pair[t] = np.sqrt(w) * f_all[t] + np.sqrt(1 - w) * u_all[t]
        coup = SPOTVOL_COUPLING.get(sub[t], SPOTVOL_COUPLING["none"])
        atm[t] = max(3.0, atm[t - 1] + 0.25 * (ATM_TARGET[st] - atm[t - 1]) + coup * z_pair[t] * 0.35 + n_all[t])
        logspot[t] = logspot[t - 1] + z_pair[t] * atm[t] / 100 / np.sqrt(252)
        a0 = max(0, t - ENERGY_ANCHOR)
        anchor = logspot[a0:t].mean()
        gap_z = (logspot[t] - anchor) / (atm[t] / 100 * np.sqrt(min(t, ENERGY_ANCHOR) / 2 / 252))
        psi[t] = energy_pressure(gap_z, atm[t])
    return s, sub, z_pair, atm, np.exp(logspot), f_all, psi


def simulate_market(n_days: int = 1260, seed: int = 0, P: np.ndarray = DEFAULT_P, energy_beta: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    if energy_beta > 0:
        s, sub, z_pair, atm, spot, f, psi_true = _simulate_with_energy(n_days, P, rng, energy_beta)
        state = np.array(STATES)[s]
    else:
        psi_true = np.zeros(n_days)
        s = _markov_path(n_days, P, rng)
        state = np.array(STATES)[s]
        sub = np.array(["none"] * n_days, dtype=object)
        cur = "none"
        for t in range(n_days):
            if state[t] in ("rising", "agitated"):
                if t == 0 or state[t - 1] != state[t]:
                    cur = (RISING_SUBTYPES if state[t] == "rising" else AGITATED_SUBTYPES)[rng.integers(2)]
                sub[t] = cur
        # returns first (spot-vol coupling needs them), then ATM with coupling
        w = np.array([FACTOR_SHARE.get(x, 0.6) for x in sub])
        f, u = rng.normal(0, 1, n_days), rng.normal(0, 1, n_days)
        z_pair = np.sqrt(w) * f + np.sqrt(1 - w) * u
        atm = np.empty(n_days); atm[0] = 7.0
        for t in range(1, n_days):
            coup = SPOTVOL_COUPLING.get(sub[t], SPOTVOL_COUPLING["none"])
            atm[t] = max(3.0, atm[t - 1] + 0.25 * (ATM_TARGET[state[t]] - atm[t - 1]) + coup * z_pair[t] * 0.35 + rng.normal(0, 0.3))
        daily_sd = atm / 100 / np.sqrt(252)
        spot = 1.10 * np.exp(np.cumsum(z_pair * daily_sd))
    daily_sd = atm / 100 / np.sqrt(252)
    rets = z_pair * daily_sd
    r = pd.Series(rets, index=dates)
    rv = {k: (r.rolling(k).std() * np.sqrt(252) * 100).bfill().values for k in (5, 21, 63)}
    atm_1y = 0.6 * atm + 0.4 * 8.5 + rng.normal(0, 0.1, n_days)
    rr25 = np.array([RR_LEVEL[x] for x in state]) + 0.3 * (atm - 9) / 3 + rng.normal(0, 0.15, n_days)
    fly25 = np.array([FLY_LEVEL[x] for x in state]) + rng.normal(0, 0.03, n_days)
    stress = np.array([STRESS_MEAN[x] for x in state]) + 3.0 * (atm - np.array([ATM_TARGET[x] for x in state])) + rng.normal(0, 6.0, n_days)
    out = {"state_true": state, "subtype_true": sub, "pressure_true": psi_true, "stress": stress, "atm_1m": atm, "atm_1y": atm_1y,
           "rr25_1m": rr25, "fly25_1m": fly25, "spot": spot, "rv_1w": rv[5], "rv_1m": rv[21], "rv_3m": rv[63]}
    for i, g in enumerate(G10):
        lam = 0.5 + 0.1 * i
        out[g] = np.exp(np.cumsum((lam * f + np.sqrt(max(0.0, 1 - lam ** 2)) * rng.normal(0, 1, n_days)) * 0.0065))
    idx = np.array([1 if x == "usd_led" else (2 if x == "idiosyncratic" else 0) for x in sub])
    for col, betas in CA_BETA.items():
        b = np.array(betas)[idx]
        dz = b * z_pair + np.sqrt(np.clip(1 - b ** 2, 0.05, 1)) * rng.normal(0, 1, n_days)
        out[col] = 100 * np.exp(np.cumsum(dz * 0.008)) if col not in ("rate2y", "rate10y") else 2.0 + np.cumsum(dz * 0.03)
    return pd.DataFrame(out, index=dates).rename_axis("date")


def earn(archetype, state, remaining, base=DEFAULT_BASE, slope=DEFAULT_SLOPE, decay=DEFAULT_DECAY, sub="none"):
    return base[archetype][state] + slope[archetype] * np.exp(-remaining / decay) + SUBTYPE_ADJ.get(archetype, {}).get(sub, 0.0)


def simulate_trades(market: pd.DataFrame, archetypes=DEMO_ARCHETYPES, tenor_days: int = 21, pair: str = "EURUSD",
                    seed: int = 1, sigma_common=DEFAULT_SIGMA_COMMON, sigma_idio: float = DEFAULT_SIGMA_IDIO,
                    ar: float = DEFAULT_AR, base=DEFAULT_BASE, slope=DEFAULT_SLOPE, decay: float = DEFAULT_DECAY) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates, st, n = market.index, market["state_true"].values, len(market)
    sub = market["subtype_true"].values if "subtype_true" in market else np.array(["none"] * n)
    common = {a: rng.normal(0, 1, n) * np.array([sigma_common[x] for x in st]) for a in archetypes}
    rows = []
    for a in archetypes:
        for i in range(n - tenor_days):
            tid, eps = f"{pair}-{a}-{dates[i].date()}", 0.0
            for age in range(1, tenor_days + 1):
                d = i + age
                eps = ar * eps + rng.normal(0, sigma_idio)
                pnl = earn(a, st[d], tenor_days - age, base, slope, decay, sub[d]) + common[a][d] + eps
                rows.append((tid, pair, a, tenor_days, dates[i], dates[d], age, pnl, st[d]))
    td = pd.DataFrame(rows, columns=["trade_id", "pair", "archetype", "tenor_days", "entry_date", "date", "age", "pnl", "regime_day"])
    td["vega"] = 0.4 * np.sqrt((td["tenor_days"] - td["age"] + 1) / td["tenor_days"])
    td["pnl_trade"] = td["pnl"] * 1.4
    td["pnl_delta_hedge"] = -td["pnl"] * 0.3
    td["pnl_vega_hedge"] = td["pnl"] - td["pnl_trade"] - td["pnl_delta_hedge"]
    return td


def true_ladder(archetypes=DEMO_ARCHETYPES, tenor_days: int = 21, horizons=(1, 3, 5, 10, 20), P: np.ndarray = DEFAULT_P,
                base=DEFAULT_BASE, slope=DEFAULT_SLOPE, decay: float = DEFAULT_DECAY) -> pd.DataFrame:
    hs = sorted(set(h for h in horizons if h <= tenor_days) | {tenor_days})
    out = []
    for a in archetypes:
        b = np.array([base[a][s] for s in STATES])
        for k, kname in enumerate(STATES):
            cum, Pd = 0.0, np.eye(len(STATES))
            for d in range(1, tenor_days + 1):
                Pd = Pd @ P
                cum += Pd[k] @ b + slope[a] * np.exp(-(tenor_days - d) / decay)
                if d in hs:
                    out.append((a, kname, d, cum))
    return pd.DataFrame(out, columns=["archetype", "regime", "h", "true_ev"])
