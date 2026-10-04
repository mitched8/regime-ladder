"""Shock detection: the fast path next to a deliberately slow labeller.

The labeller is smoothed and hysteretic, so it is late by construction. Shock detectors do not
define the state; they move probability mass toward the next one (`blend_pi`) and, in the path
model, feed the regime probability vector. Two quantities are kept apart because they answer
different questions:

* economic surprise   the day's return standardised by the implied daily vol marked BEFORE it —
                      the move relative to what the market priced, which is what option P&L is
                      realised against;
* physical forecast   a HAR model of realised variance with its own innovation, for comparing a
                      forward-RV path with the traded surface.

Detectors on squared surprises:
* `cusum`  one-sided CUSUM with allowance k and threshold h. The null mean is EMPIRICAL (trailing
           window), because implied-standardised squared returns do not have mean one in general.
           h is set on an average-run-length basis by resampling the null (`calibrate_cusum`).
* `bocd`   Bayesian online changepoint detection (Adams–MacKay) with a Normal–Inverse-Gamma
           observation model on the surprises: supplies P(run length < m) at every date.
A single-day 3-sigma rule is kept only as the thing to beat (`three_sigma`): it raises alarms on
isolated big days and misses a sustained rise in realised that never produces one.

All functions obey the `asof` contract and pass the truncation test. The blend into the state
probability vector is a heuristic until the calibration test (Section 14 of the framework) says
otherwise; the economic threshold — how the ladder responds to recent shock size — decides what
counts as a shock, not the statistics here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .labels import ALL_STATES

ESCALATION = {"carry": "rising", "settling": "rising", "rising": "stressed", "agitated": "stressed",
              "normalising": "stressed", "stressed": "extreme", "extreme": "extreme",
              "transition": "crisis", "crisis": "crisis"}


def _cut(market: pd.DataFrame, asof) -> pd.DataFrame:
    return market if asof is None else market.loc[: pd.Timestamp(asof)]


# ----------------------------------------------------------------------------- surprise and forecast
def surprise(market, asof=None, spot="spot", iv="atm_1m") -> pd.Series:
    """Economic surprise z_t = log return_t / (implied daily vol marked at t-1)."""
    m = _cut(market, asof)
    r = np.log(m[spot]).diff()
    sd = m[iv].shift(1) / 100 / np.sqrt(252)
    return (r / sd).rename("surprise")


def three_sigma(z: pd.Series, k: float = 3.0) -> pd.Series:
    return (z.abs() > k).rename("alarm_3sigma")


def har_fit(X: np.ndarray, y: np.ndarray, prior: np.ndarray | None = None, ridge: float = 0.0) -> np.ndarray:
    """OLS with intercept, optionally ridge-shrunk toward `prior` (a coefficient vector incl. intercept) with
    weight `ridge` expressed as a fraction of the sample size. Returns NaNs when there are too few rows."""
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n = int(ok.sum())
    if n < 60:
        return np.full(X.shape[1] + 1, np.nan)
    A = np.column_stack([np.ones(n), X[ok]])
    yy = y[ok]
    if prior is not None and ridge > 0:
        lam = ridge * n
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yy + lam * prior)
    else:
        beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    return beta


def har_series(market, asof=None, regressors=("rv_1w", "rv_1m", "rv_3m"), target="rv_1m", horizon: int = 21,
               window: int = 756, refit: int = 63, warmup: int = 252, ridge: float = 0.5) -> pd.Series:
    """Physical forecast of realised vol over the next `horizon` days: a HAR regression of the forward
    realised vol on today's short / medium / long realised vols, refitted every `refit` days on the trailing
    `window`, ridge-shrunk toward persistence (forecast = today's `target` vol) with weight `ridge` x n.
    The shrinkage is what keeps a short, regime-switching sample from fitting a confident mean-reversion
    it cannot forecast out of sample. Refit dates are fixed positions from the start of the history and
    only rows whose forward target is already realised enter a fit, so the series is point-in-time."""
    m = _cut(market, asof)
    X = m[list(regressors)].values.astype(float)
    y_all = m[target].shift(-horizon).values.astype(float)  # forward realised vol, known only `horizon` days later
    prior = np.zeros(len(regressors) + 1)
    if target in regressors:
        prior[1 + list(regressors).index(target)] = 1.0
    out = np.full(len(m), np.nan)
    beta = None
    for t in range(len(m)):
        if t % refit == 0 and t >= warmup:
            lo = max(0, t - window)
            beta = har_fit(X[lo: t - horizon], y_all[lo: t - horizon], prior, ridge)  # targets realised by t
        if beta is not None and np.isfinite(beta).all() and np.isfinite(X[t]).all():
            out[t] = max(0.0, beta[0] + X[t] @ beta[1:])
    return pd.Series(out, index=m.index, name="har_vol_forecast")


# ----------------------------------------------------------------------------- CUSUM
def cusum(x2: pd.Series, k: float, h: float, null_window: int = 252, reset: bool = True, clip: float = 9.0) -> pd.DataFrame:
    """One-sided CUSUM on squared surprises against a trailing empirical null mean.
    S_t = max(0, S_{t-1} + min(x2_t, clip) - mu0_t - k); alarm when S_t > h; S resets after an alarm.
    Squared surprises are capped at `clip` (default 3 sigma squared): the CUSUM is the detector for a
    SUSTAINED rise in realised relative to implied, and a single isolated big day must not be able to
    trip it on its own — that day belongs to the jump channel (`three_sigma`), kept separate on purpose."""
    x2 = x2.clip(upper=clip)
    mu0 = x2.rolling(null_window, min_periods=max(40, null_window // 4)).mean().shift(1)
    S, alarm, s = np.zeros(len(x2)), np.zeros(len(x2), dtype=bool), 0.0
    xv, mv = x2.values, mu0.values
    for t in range(len(xv)):
        if np.isnan(xv[t]) or np.isnan(mv[t]):
            S[t] = s
            continue
        s = max(0.0, s + xv[t] - mv[t] - k)
        if s > h:
            alarm[t] = True
            if reset:
                s = 0.0
        S[t] = s
    return pd.DataFrame({"S": S, "alarm": alarm, "mu0": mu0.values}, index=x2.index)


def calibrate_cusum(x2_null: np.ndarray, k: float, arl0: float = 1000.0, n_paths: int = 200, max_len: int = 3000,
                    seed: int = 0, clip: float = 9.0) -> float:
    """Threshold h giving average run length `arl0` under the null, by resampling the null squared surprises."""
    rng = np.random.default_rng(seed)
    x = np.minimum(np.asarray(x2_null)[~np.isnan(np.asarray(x2_null))], clip)
    mu = x.mean()
    paths = rng.choice(x, size=(n_paths, max_len), replace=True) - mu - k

    def arl(h):  # all paths at once; run length = first crossing (max_len if none)
        s = np.zeros(n_paths); hit = np.full(n_paths, max_len); alive = np.ones(n_paths, dtype=bool)
        for i in range(max_len):
            s = np.maximum(0.0, s + paths[:, i])
            cross = alive & (s > h)
            hit[cross] = i + 1
            alive &= ~cross
            if not alive.any():
                break
        return float(hit.mean())
    lo, hi = 0.5, 50.0
    for _ in range(18):  # bisection on a monotone function
        mid = 0.5 * (lo + hi)
        if arl(mid) < arl0:
            lo = mid
        else:
            hi = mid
    return float(hi)


def cusum_series(market, asof=None, spot="spot", iv="atm_1m", k: float = 0.3, h: float | None = None,
                 null_window: int = 252, arl0: float = 1000.0, clip: float = 9.0) -> pd.Series:
    """CUSUM statistic on squared surprises as a 0..1 pressure (S / h, capped), point-in-time. If h is None it
    is calibrated once on the first `null_window` days of squared surprises (a fixed position, so truncation-safe)."""
    z = surprise(market, asof, spot, iv)
    x2 = z ** 2
    if h is None:
        first = x2.dropna().iloc[:null_window]
        h = calibrate_cusum(first.values, k, arl0, clip=clip) if len(first) >= 60 else 10.0
    c = cusum(x2, k, h, null_window, clip=clip)
    return (c["S"] / h).clip(upper=1.0).rename("cusum_pressure")


# ----------------------------------------------------------------------------- BOCD
def bocd(x: pd.Series, hazard: float = 1 / 100, mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0,
         beta0: float = 1.0, max_run: int = 400, m: int = 10) -> pd.DataFrame:
    """Adams–MacKay BOCD with a Normal–Inverse-Gamma predictive (unknown mean and variance).
    Returns P(run length < m) and the MAP run length per date. NaNs are skipped (no update)."""
    xv = x.values
    T = len(xv)
    R = np.zeros(max_run + 1); R[0] = 1.0
    mu = np.array([mu0]); kap = np.array([kappa0]); al = np.array([alpha0]); be = np.array([beta0])
    p_short, map_run = np.full(T, np.nan), np.full(T, np.nan)
    for t in range(T):
        v = xv[t]
        if np.isnan(v):
            if t > 0:
                p_short[t], map_run[t] = p_short[t - 1], map_run[t - 1]
            continue
        # Student-t predictive for each run length hypothesis
        df_ = 2 * al
        scale2 = be * (kap + 1) / (al * kap)
        tt2 = (v - mu) ** 2 / scale2
        pred = np.exp(gammaln((df_ + 1) / 2) - gammaln(df_ / 2) - 0.5 * np.log(df_ * np.pi * scale2)
                      - (df_ + 1) / 2 * np.log1p(tt2 / df_))
        n = len(mu)
        growth = R[:n] * pred * (1 - hazard)
        cp = float((R[:n] * pred * hazard).sum())
        newR = np.zeros(max_run + 1)
        newR[0] = cp
        newR[1: n + 1] = growth[: max_run]
        newR /= max(newR.sum(), 1e-300)
        R = newR
        # posterior updates (prepend the prior for run length 0)
        mu_n = (kap * mu + v) / (kap + 1)
        be_n = be + kap * (v - mu) ** 2 / (2 * (kap + 1))
        mu = np.concatenate([[mu0], mu_n])[: max_run + 1]
        kap = np.concatenate([[kappa0], kap + 1])[: max_run + 1]
        al = np.concatenate([[alpha0], al + 0.5])[: max_run + 1]
        be = np.concatenate([[beta0], be_n])[: max_run + 1]
        p_short[t] = R[:m].sum()
        map_run[t] = int(np.argmax(R))
    return pd.DataFrame({"p_run_short": p_short, "map_run": map_run}, index=x.index)


def bocd_series(market, asof=None, spot="spot", iv="atm_1m", hazard: float = 1 / 100, m: int = 10) -> pd.Series:
    """P(run length < m) of the surprise process, point-in-time."""
    return bocd(surprise(market, asof, spot, iv), hazard=hazard, m=m)["p_run_short"].rename("bocd_p_short")


# ----------------------------------------------------------------------------- score and blend
def shock_score(market, asof=None, cfg: dict | None = None) -> pd.DataFrame:
    """Components and a 0..100 score: CUSUM pressure, BOCD P(short run), and the EWMA of |surprise| scaled
    by its trailing 95th percentile. The score is the mean of the three, each in 0..1."""
    cfg = cfg or {}
    z = surprise(market, asof, cfg.get("spot", "spot"), cfg.get("iv", "atm_1m"))
    cus = cusum_series(market, asof, k=cfg.get("cusum_k", 0.3), h=cfg.get("cusum_h"), null_window=cfg.get("null_window", 252),
                       arl0=cfg.get("arl0", 1000.0))
    boc = bocd_series(market, asof, hazard=cfg.get("hazard", 1 / 100), m=cfg.get("bocd_m", 10))
    az = z.abs().ewm(halflife=cfg.get("abs_halflife", 3), adjust=False).mean()
    q = az.rolling(cfg.get("null_window", 252), min_periods=60).quantile(0.95).shift(1)
    recent = (az / q).clip(upper=1.0)
    out = pd.concat([z, cus, boc, recent.rename("recent_abs")], axis=1)
    out["score"] = 100 * out[["cusum_pressure", "bocd_p_short", "recent_abs"]].mean(axis=1, skipna=False)
    return out


def blend_pi(pi: pd.Series, score: float, w_max: float = 0.5, escalation: dict = ESCALATION) -> pd.Series:
    """Heuristic: move a fraction w = w_max * score/100 of each state's probability to its escalation state.
    `pi` is a probability vector over state names; returns a new one that still sums to one."""
    w = float(np.clip(w_max * score / 100.0, 0.0, 1.0))
    out = pd.Series(0.0, index=pi.index)
    for s, p in pi.items():
        nxt = escalation.get(s, s)
        if nxt in out.index and nxt != s:
            out[s] += p * (1 - w); out[nxt] += p * w
        else:
            out[s] += p
    return out / out.sum()


def lead_profile(score: pd.Series, labels: pd.Series, before: int = 5, rank=ALL_STATES) -> pd.DataFrame:
    """Does the score rise before escalations? Mean score on the `before` days preceding each move to a
    higher-ranked state, versus before moves down and versus all days. A detector that is merely
    coincident shows nothing here; one with lead shows the 'up' row above 'all'."""
    order = {s: i for i, s in enumerate(rank)}
    lv = labels.map(order)
    chg = lv.diff()
    rows = {"all": score.mean()}
    for name, mask in (("before_up", chg > 0), ("before_down", chg < 0)):
        idx = np.where(mask.values)[0]
        vals = [score.iloc[max(0, i - before): i].mean() for i in idx if i > 0]
        rows[name] = float(np.nanmean(vals)) if vals else np.nan
        rows[name + "_n"] = len(vals)
    return pd.DataFrame([rows])


def figure7_path(n_calm: int = 250, n_shift: int = 60, iv: float = 8.0, seed: int = 0) -> pd.DataFrame:
    """The scenario from the framework page: a calm path with three isolated 3-sigma days, then realised
    running 50% above implied with no single day above 3 sigma. Returns a market frame (spot, atm_1m)."""
    rng = np.random.default_rng(seed)
    sd = iv / 100 / np.sqrt(252)
    z = rng.normal(0, 0.8, n_calm)
    for i in (60, 140, 210):
        z[i] = 3.4 * np.sign(rng.normal())
    z2 = rng.normal(0, 1.5, n_shift)
    z2 = np.clip(z2, -2.9, 2.9)
    r = np.concatenate([z, z2]) * sd
    idx = pd.bdate_range("2020-01-01", periods=len(r))
    return pd.DataFrame({"spot": 1.1 * np.exp(np.cumsum(r)), "atm_1m": iv}, index=idx).rename_axis("date")
