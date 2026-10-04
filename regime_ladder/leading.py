"""Leading features and stored energy.

State features say where we are; leading features say what might push us out. They are organised
around the anatomy of a vol event — a SPARK (catalyst, calendar) becomes a vol event only if it
hits FUEL (positioning that gets hurt) and an AMPLIFIER (short gamma, thin liquidity, coupled
markets); DAMPERS absorb it. Two hypotheses are kept as desk language and made measurable here:

* stored energy  — a gap between a price and its anchor, held by something temporary. Measured as
                   the product of (how far spot has travelled from its anchor, in implied-vol units)
                   and (how complacent the holder looks: realised far under implied, vol-of-vol low,
                   front end cheap). Either alone is unremarkable; the product is the hypothesis.
* feedback       — whether a shock would spread: jump clustering, a shift in spot-vol coupling,
                   cross-pair coherence (a dollar move hits everything at once).

Every leading feature is a plain function `f(market, asof=None, **params)` on trailing data only,
registered in `LEADING` so the truncation test covers it, and expressed as a trailing percentile
(0..100) so features from different sources are comparable and a `pressure` index can average them.
The scheduled-event calendar is the one covariate known in advance (`event_proximity`).

The rule that keeps this honest: option P&L is realised relative to implied, so a signal the surface
already prices adds nothing. A leading feature is retained only if, out of sample, it (1) improves the
transition log-likelihood through the tilt and (2) improves the forward-outcome fit beyond a baseline
that already uses the named state AND the continuous state score (`incremental_value`). Each
feature is tested one at a time against that baseline; the ones that survive are averaged into the
pressure index that enters the transition tilt (`transitions.tilt`) with a declared future path.
The first draft's closure weight and Hawkes fragility gauge stay parked for the reasons given in the
framework; nothing here depends on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import rolling_percentile
from .transitions import oos_gain


def _cut(market: pd.DataFrame, asof) -> pd.DataFrame:
    return market if asof is None else market.loc[: pd.Timestamp(asof)]


# ----------------------------------------------------------------------------- stored energy
def gap_z(market, asof=None, spot="spot", iv="atm_1m", anchor_window=120) -> pd.Series:
    """Spot's distance from its trailing anchor (mean log level over `anchor_window` days, excluding today)
    in units of the move implied vol would allow over half that window. Signed."""
    m = _cut(market, asof)
    ls = np.log(m[spot])
    anchor = ls.rolling(anchor_window, min_periods=anchor_window // 2).mean().shift(1)
    scale = m[iv] / 100 * np.sqrt(anchor_window / 2 / 252)
    return ((ls - anchor) / scale).rename("gap_z")


def gap_pct(market, asof=None, window=756, **kw) -> pd.Series:
    """Trailing percentile of |gap_z|: how unusual today's stretch from the anchor is."""
    return rolling_percentile(gap_z(market, asof, **kw).abs(), window).rename("gap_pct")


def drift_t(market, asof=None, spot="spot", window=60) -> pd.Series:
    """t-statistic of the mean daily return over `window`: one-way markets build positioning."""
    m = _cut(market, asof)
    r = np.log(m[spot]).diff()
    return (r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(window)).rename("drift_t")


def complacency(market, asof=None, iv="atm_1m", iv_long="atm_1y", rv="rv_1m", window=756, vov_window=21) -> pd.Series:
    """How temporary the holder of the gap looks, 0..100: the mean of three trailing percentiles —
    realised/implied LOW, vol-of-vol LOW, front end cheap vs the back (term slope LOW). High = a quiet,
    well-supplied surface that has not been asked a question yet."""
    m = _cut(market, asof)
    rv_iv = rolling_percentile(-(m[rv] / m[iv]), window)
    vov = rolling_percentile(-m[iv].diff().rolling(vov_window).std(), window)
    slope = rolling_percentile(-(m[iv] - m[iv_long]), window)
    return pd.concat([rv_iv, vov, slope], axis=1).mean(axis=1, skipna=False).rename("complacency")


def stored_energy(market, asof=None, window=756, **kw) -> pd.Series:
    """gap_pct x complacency / 100: a stretched spot held by a complacent surface. 0..100."""
    g = gap_pct(market, asof, window=window, **{k: v for k, v in kw.items() if k in ("spot", "iv", "anchor_window")})
    c = complacency(market, asof, window=window, **{k: v for k, v in kw.items() if k in ("iv", "iv_long", "rv", "vov_window")})
    return (g * c / 100).rename("stored_energy")


# ----------------------------------------------------------------------------- feedback / amplifiers
def jump_clustering(market, asof=None, spot="spot", iv="atm_1m", window=21, k=2.0, rank_window=756) -> pd.Series:
    """Share of the last `window` days with |surprise| > k, as a trailing percentile."""
    m = _cut(market, asof)
    z = np.log(m[spot]).diff() / (m[iv].shift(1) / 100 / np.sqrt(252))
    return rolling_percentile((z.abs() > k).rolling(window).mean(), rank_window).rename("jump_cluster_pct")


def coupling_shift(market, asof=None, spot="spot", iv="atm_1m", short=21, long=126, rank_window=756) -> pd.Series:
    """|corr_short - corr_long| of spot returns with implied changes: the spot-vol coupling moving away from
    its recent norm, whichever way. Trailing percentile."""
    m = _cut(market, asof)
    dr, dv = np.log(m[spot]).diff(), m[iv].diff()
    c = (dr.rolling(short).corr(dv) - dr.rolling(long).corr(dv)).abs()
    return rolling_percentile(c, rank_window).rename("coupling_shift_pct")


def cross_pair_coherence(market, asof=None, pairs=("g10_1", "g10_2", "g10_3", "g10_4", "g10_5", "g10_6"), window=21, rank_window=756) -> pd.Series:
    """Mean pairwise correlation of other pairs' returns over `window`: when everything moves together a
    shock in one place reaches all of them. Trailing percentile."""
    m = _cut(market, asof)
    cols = [p for p in pairs if p in m]
    r = np.log(m[cols]).diff()
    k = len(cols)
    if k < 2:
        return pd.Series(np.nan, index=m.index, name="coherence_pct")
    vals = np.full(len(m), np.nan)
    R = r.values
    for t in range(window, len(m)):
        X = R[t - window + 1: t + 1]
        if np.isnan(X).any():
            continue
        C = np.corrcoef(X.T)
        vals[t] = (C.sum() - k) / (k * (k - 1))
    return rolling_percentile(pd.Series(vals, index=m.index), rank_window).rename("coherence_pct")


# ----------------------------------------------------------------------------- the calendar covariate
def event_proximity(index, events, horizon: int = 10) -> pd.Series:
    """1 on an event day, falling linearly to 0 `horizon` business days before it; 0 otherwise. Known in
    advance, so its future path is 'calendar' in `transitions.covariate_path`."""
    idx = pd.DatetimeIndex(index)
    out = np.zeros(len(idx))
    for e in pd.to_datetime(list(events)):
        i = idx.searchsorted(e)
        if i >= len(idx):
            continue
        for d in range(horizon + 1):
            j = i - d
            if j >= 0:
                out[j] = max(out[j], 1 - d / horizon)
    return pd.Series(out, index=idx, name="event_proximity")


LEADING = {
    "stored_energy": stored_energy, "gap_pct": gap_pct, "complacency": complacency,
    "jump_cluster_pct": jump_clustering, "coupling_shift_pct": coupling_shift, "coherence_pct": cross_pair_coherence,
}


def build(market: pd.DataFrame, names=None, asof=None, events=None, **kw) -> pd.DataFrame:
    names = names or list(LEADING)
    out = [LEADING[n](market, asof=asof, **kw.get(n, {})) for n in names]
    if events:
        out.append(event_proximity(_cut(market, asof).index, events) * 100)
    return pd.concat(out, axis=1)


# ----------------------------------------------------------------------------- pressure
def pressure(leading: pd.DataFrame, weights: dict | None = None, centre: float = 50.0, scale: float = 25.0) -> pd.Series:
    """Pressure psi for the transition tilt: the weighted mean of the retained leading features (each 0..100),
    centred and scaled so that psi ~ 0 is normal and psi ~ +2 is the top of the range. Only the features
    in `weights` enter; the default is an equal-weight mean of every column."""
    cols = list(weights) if weights else list(leading.columns)
    w = pd.Series(weights) if weights else pd.Series(1.0, index=cols)
    w = w / w.sum()
    return ((leading[cols] * w).sum(axis=1, min_count=len(cols)) - centre).div(scale).rename("pressure")


# ----------------------------------------------------------------------------- the retention test
def _oos_r2(X: np.ndarray, y: np.ndarray, folds: int, min_train_frac: float):
    n = len(y)
    cuts = np.linspace(int(min_train_frac * n), n, folds + 1).astype(int)
    sse, sst, per_fold = 0.0, 0.0, []
    for a, b in zip(cuts[:-1], cuts[1:]):
        Xtr, ytr, Xte, yte = X[:a], y[:a], X[a:b], y[a:b]
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        pred = Xte @ beta
        s1, s0 = float(((yte - pred) ** 2).sum()), float(((yte - ytr.mean()) ** 2).sum())
        sse += s1; sst += s0
        per_fold.append(1 - s1 / s0 if s0 > 0 else np.nan)
    return 1 - sse / sst if sst > 0 else np.nan, per_fold


def incremental_value(feature: pd.Series, labels: pd.Series, state_score: pd.Series, target: pd.Series,
                      folds: int = 4, min_train_frac: float = 0.4, lag: int = 0, k: int = 5) -> dict:
    """One feature against the baseline that already knows the state.

    Outcome test: out-of-sample R^2 of `target` (a forward outcome aligned to the decision date) on
    [state dummies + continuous state score] versus the same plus the feature; reports the R^2 gain and
    how many folds improve. Transition test: `transitions.oos_gain` with the standardised feature as the
    pressure, the standardised state score as the baseline covariate already in the tilt, and the k-step
    likelihood (k = 5 by default: where the state is a week out, not tonight's move — a smoothed
    labeller's one-step moves lag the market by days and would hide any lead). `lag` shifts the feature
    back by that many days, to show whether value survives a delay in getting the data. Everything is
    time-ordered; nothing is fitted on the future.
    """
    f = feature.shift(lag)
    # transition test on every day with a feature and a label (the whole history); outcome test where the target exists too
    dt = pd.concat([f.rename("f"), labels.rename("s")], axis=1).dropna()
    df = pd.concat([f.rename("f"), labels.rename("s"), state_score.rename("c"), target.rename("y")], axis=1).dropna()
    if len(df) < 200 or f.std() == 0:
        return {"n": len(df), "retain": False}
    mu, sd = dt["f"].mean(), dt["f"].std()
    D = pd.get_dummies(df["s"]).astype(float).values
    base = np.column_stack([D, df["c"].values, df["c"].values ** 2])
    fz = ((df["f"] - mu) / sd).values
    r2_base, pf_base = _oos_r2(base, df["y"].values, folds, min_train_frac)
    r2_feat, pf_feat = _oos_r2(np.column_stack([base, fz]), df["y"].values, folds, min_train_frac)
    folds_up = int(sum(1 for a, b in zip(pf_base, pf_feat) if b > a))
    psi = (dt["f"] - mu) / sd
    cz = (state_score - state_score.mean()) / state_score.std()
    tg = oos_gain(dt["s"], psi, folds=folds, min_train_frac=min_train_frac, k=k, base=cz)
    return {"n": int(len(df)), "n_transitions": int(len(dt) - 1), "r2_base": r2_base, "r2_with": r2_feat, "delta_r2": r2_feat - r2_base,
            "folds_r2_up": folds_up, "transition_gain": tg.attrs["total_gain_per_transition"], "folds_transition_up": tg.attrs["folds_positive"],
            "beta_mean": float(tg["beta"].mean()), "gamma_mean": float(tg["gamma"].mean()), "folds": folds, "k": k}


def retain(res: dict, cfg: dict) -> bool:
    """Gate 4 rule: both tests positive, in most folds."""
    if not res or not res.get("n") or "delta_r2" not in res:
        return False
    return bool(res["delta_r2"] >= cfg["min_delta_r2"] and res["folds_r2_up"] >= cfg["min_folds_up"]
                and res["transition_gain"] >= cfg["min_transition_gain"] and res["folds_transition_up"] >= cfg["min_folds_up"])


def screen(candidates: pd.DataFrame, labels: pd.Series, state_score: pd.Series, target: pd.Series, cfg: dict,
           folds: int = 4, lag: int = 0, k: int = 5) -> pd.DataFrame:
    """Every candidate one at a time against the same baseline; a table sorted by R^2 gain with the retain flag."""
    rows = []
    for c in candidates.columns:
        r = incremental_value(candidates[c], labels, state_score, target, folds=folds, lag=lag, k=k)
        r["feature"] = c
        r["retain"] = retain(r, cfg)
        rows.append(r)
    cols = ["feature", "n", "n_transitions", "r2_base", "delta_r2", "folds_r2_up", "transition_gain", "folds_transition_up", "beta_mean", "retain"]
    return pd.DataFrame(rows).reindex(columns=cols).sort_values("delta_r2", ascending=False).reset_index(drop=True)
