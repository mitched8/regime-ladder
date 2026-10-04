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
from .transitions import fold_cuts, oos_gain


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


def gap_z_signed(market, asof=None, **kw) -> pd.Series:
    """The signed gap itself (not a percentile): a one-way stretch has a direction, and the RR's earn has a
    sign. Screen it against the RR target; the unsigned `gap_pct` belongs with the straddle."""
    return gap_z(market, asof, **kw).rename("gap_z_signed")


LEADING = {
    "stored_energy": stored_energy, "gap_pct": gap_pct, "complacency": complacency,
    "jump_cluster_pct": jump_clustering, "coupling_shift_pct": coupling_shift, "coherence_pct": cross_pair_coherence,
    "gap_z_signed": gap_z_signed, "drift_t": drift_t,
}
SIGNED = ("gap_z_signed", "drift_t")          # not 0..100; the pressure index z-scores every column anyway


def build(market: pd.DataFrame, names=None, asof=None, events=None, **kw) -> pd.DataFrame:
    names = names or list(LEADING)
    out = [LEADING[n](market, asof=asof, **kw.get(n, {})) for n in names]
    if events:
        out.append(event_proximity(_cut(market, asof).index, events) * 100)
    return pd.concat(out, axis=1)


# ----------------------------------------------------------------------------- pressure
def pressure(leading: pd.DataFrame, weights: dict | None = None, centre: float = 50.0, scale: float = 25.0,
             zscore_window: int | None = None) -> pd.Series:
    """Pressure psi for the transition tilt: the weighted mean of the retained leading features, scaled so
    that psi ~ 0 is normal and psi ~ +2 is the top of the range. By default each feature is read as a
    0..100 percentile (centre 50, scale 25). With `zscore_window`, each column is instead z-scored against
    its own TRAILING window first, which puts percentiles, products of percentiles (`stored_energy` sits
    near 20, not 50) and signed series on one footing without touching the future. Only the features in
    `weights` enter; the default is an equal-weight mean of every column."""
    cols = list(weights) if weights else list(leading.columns)
    w = pd.Series(weights) if weights else pd.Series(1.0, index=cols)
    w = w / w.sum()
    X = leading[cols]
    if zscore_window:
        mp = max(60, zscore_window // 4)
        X = (X - X.rolling(zscore_window, min_periods=mp).mean().shift(1)) / X.rolling(zscore_window, min_periods=mp).std().shift(1)
        return (X * w).sum(axis=1, min_count=len(cols)).rename("pressure")
    return ((X * w).sum(axis=1, min_count=len(cols)) - centre).div(scale).rename("pressure")


# ----------------------------------------------------------------------------- the retention test
LOW_BAND = ("carry", "settling")
HIGH_BAND = ("stressed", "extreme", "crisis")


def _oos_r2(X: np.ndarray, y: np.ndarray, dates, folds: int, min_train_frac: float):
    """Walk-forward R^2 with folds cut at DATES (so pooled pairs are cut at the same point in time). With an
    intercept in X, predictions are invariant to affine rescaling of the columns, so standardising with
    full-sample or training-fold statistics gives identical results here; only the tilt (no intercept)
    needs the fold-local version, which `transitions.oos_gain(standardise=True)` provides."""
    d = pd.DatetimeIndex(dates)
    cuts = fold_cuts([d], folds, min_train_frac)
    sse, sst, per_fold = 0.0, 0.0, []
    for ca, cb in zip(cuts[:-1], cuts[1:]):
        tr, te = (d < ca), (d >= ca) & (d < cb)
        if tr.sum() < 50 or te.sum() == 0:
            per_fold.append(np.nan); continue
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred = X[te] @ beta
        s1, s0 = float(((y[te] - pred) ** 2).sum()), float(((y[te] - y[tr].mean()) ** 2).sum())
        sse += s1; sst += s0
        per_fold.append(1 - s1 / s0 if s0 > 0 else np.nan)
    return 1 - sse / sst if sst > 0 else np.nan, per_fold


def _as_dict(x) -> dict:
    return dict(x) if isinstance(x, dict) else {"_": x}


def incremental_value(feature, labels, state_score, target, folds: int = 4, min_train_frac: float = 0.4,
                      lag: int = 0, k: int = 5, ks=(), interact=LOW_BAND, prior_strength: float = 10.0,
                      outcome_only: bool = False, n_perm: int = 0, perm_min_shift: int = 63, seed: int = 0) -> dict:
    """One feature against the baseline that already knows the state.

    Outcome test: out-of-sample R^2 of `target` (a forward outcome aligned to the decision date) on
    [intercept + state dummies (reference coding) + continuous state score and its square (+ pair dummies
    when pooled)] versus the same plus the feature; reports the R^2 gain and how many folds improve.
    `delta_r2_low` adds instead the feature AND the feature × 1[state in `interact`] — the conditional form
    of the stored-energy hypothesis (energy matters in the calm states, not once the move is on).

    Transition test: `transitions.oos_gain` with the feature as the pressure, the state score as the
    baseline covariate already in the tilt (both z-scored per fold on the training part), and the k-step
    likelihood (k = 5 by default: where the state is a week out, not tonight's move — a smoothed
    labeller's one-step moves lag the market by days and would hide any lead). `ks` adds the same gain
    at other horizons as diagnostics; retention uses `k` only.

    Each of `feature`, `labels`, `state_score`, `target` may be a dict keyed by pair: the test is then
    pooled — stacked rows with pair dummies, folds cut at common dates, one beta across pairs — which is
    how a feature gets more than one pair's dozen episodes to be judged on. `lag` shifts the feature
    back by that many days, to show whether value survives a delay in getting the data.

    `n_perm` > 0 adds `p_perm`: the share of circular shifts of the feature (by at least `perm_min_shift`
    days, within each pair) whose out-of-sample R^2 gain matches or beats the observed one. The shift keeps
    the feature's own persistence and the target's, and breaks only their alignment, so it is the null a
    screen over many candidates needs. `outcome_only` skips the transition tests (used for diagnostic targets).
    """
    F, Lb, C, Y = (_as_dict(x) for x in (feature, labels, state_score, target))
    pairs = [p for p in F if p in Lb and p in C and p in Y]
    frames, dts, psis, labs, bases = [], [], [], [], []
    for p in pairs:
        f = F[p].shift(lag)
        dt = pd.concat([f.rename("f"), Lb[p].rename("s")], axis=1).dropna()
        df = pd.concat([f.rename("f"), Lb[p].rename("s"), C[p].rename("c"), Y[p].rename("y")], axis=1).dropna()
        if len(df) < 100 or not np.isfinite(f.std()) or f.std() == 0:
            continue
        df["pair"] = p; frames.append(df); dts.append(dt); psis.append(dt["f"]); labs.append(dt["s"]); bases.append(C[p])
    if not frames or sum(len(d) for d in frames) < 200:
        return {"n": int(sum(len(d) for d in frames)), "retain": False}
    df = pd.concat(frames).sort_index(kind="stable")
    ref = df["s"].value_counts().idxmax()
    D = pd.get_dummies(df["s"]).astype(float).drop(columns=[ref]).values
    cz_ = ((df["c"] - df["c"].mean()) / df["c"].std()).values
    cols = [np.ones(len(df)), D, cz_, cz_ ** 2]
    if len(frames) > 1:
        cols.append(pd.get_dummies(df["pair"]).astype(float).iloc[:, 1:].values)
    base = np.column_stack(cols)
    fz = ((df["f"] - df["f"].mean()) / df["f"].std()).values
    y, dates = df["y"].values, df.index
    r2_base, pf_base = _oos_r2(base, y, dates, folds, min_train_frac)
    r2_feat, pf_feat = _oos_r2(np.column_stack([base, fz]), y, dates, folds, min_train_frac)
    low = df["s"].isin(interact).values.astype(float) if interact else np.zeros(len(df))
    r2_low, _ = _oos_r2(np.column_stack([base, fz, fz * low]), y, dates, folds, min_train_frac) if interact else (np.nan, None)
    folds_up = int(sum(1 for a, b in zip(pf_base, pf_feat) if np.isfinite(a) and np.isfinite(b) and b > a))
    res = {"n": int(len(df)), "n_pairs": len(frames), "n_transitions": int(sum(len(d) for d in dts) - len(dts)),
           "r2_base": r2_base, "r2_with": r2_feat, "delta_r2": r2_feat - r2_base, "delta_r2_low": r2_low - r2_base,
           "folds_r2_up": folds_up, "folds": folds, "k": k}
    if n_perm:
        rng = np.random.default_rng(seed); beats = 0
        sizes = [len(fr) for fr in frames]
        for _ in range(n_perm):
            parts = [np.roll(fr["f"].values, int(rng.integers(perm_min_shift, max(perm_min_shift + 1, n_ - perm_min_shift)))) for fr, n_ in zip(frames, sizes)]
            fp = pd.Series(np.concatenate(parts), index=pd.concat(frames).index).sort_index(kind="stable").values
            fp = (fp - fp.mean()) / fp.std()
            r2p, _ = _oos_r2(np.column_stack([base, fp]), y, dates, folds, min_train_frac)
            beats += int(r2p - r2_base >= r2_feat - r2_base)
        res["p_perm"] = (1 + beats) / (1 + n_perm)
    if outcome_only:
        return res
    for kk in sorted(set([k, *ks])):
        tg = oos_gain(labs, psis, folds=folds, min_train_frac=min_train_frac, k=kk, base=bases, standardise=True, prior_strength=prior_strength)
        if kk == k:
            res.update({"transition_gain": tg.attrs["total_gain_per_transition"], "folds_transition_up": tg.attrs["folds_positive"],
                        "beta_mean": float(tg["beta"].mean()) if len(tg) else np.nan, "gamma_mean": float(tg["gamma"].mean()) if len(tg) else np.nan})
        res[f"transition_gain_k{kk}"] = tg.attrs["total_gain_per_transition"]
    return res


def lift(feature: pd.Series, labels: pd.Series, horizon: int = 20, quantile: float = 0.9, window: int = 504,
         into=HIGH_BAND, block: int = 21, n_boot: int = 300, seed: int = 0) -> dict:
    """Event-study form of the same question, for a sparse feature: P(a move into the high band within
    `horizon` days | the feature is at or above its trailing q-quantile today) against the same probability
    on all eligible days (days not already in the high band). Returns the two rates, their ratio and a
    90% block-bootstrap interval, with the number of signal days and of separate signal runs (the
    effective sample)."""
    df = pd.concat([feature.rename("f"), labels.rename("s")], axis=1).dropna()
    if len(df) < window:
        return {"n_signal": 0, "lift": np.nan}
    thr = df["f"].rolling(window, min_periods=max(120, window // 4)).quantile(quantile).shift(1)
    hi = df["s"].isin(into).values
    fut = np.array([hi[i + 1: i + 1 + horizon].any() if i + 1 < len(hi) else False for i in range(len(hi))])
    elig = ~hi & (np.arange(len(hi)) + horizon < len(hi))
    sig = (df["f"] >= thr).values & elig
    e_all, e_sig = fut[elig], fut[sig]
    if sig.sum() == 0 or e_all.sum() == 0:
        return {"n_signal": int(sig.sum()), "lift": np.nan}
    rng = np.random.default_rng(seed)
    idx = np.where(elig)[0]; n = len(idx)
    ratios = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        pick = np.concatenate([idx[s: s + block] for s in starts])[:n]
        fs, fa = fut[pick][sig[pick]], fut[pick]
        if fs.size and fa.mean() > 0:
            ratios.append(fs.mean() / fa.mean())
    runs = int(((sig[1:] & ~sig[:-1]).sum() + (1 if sig[0] else 0)))
    return {"p_event_signal": float(e_sig.mean()), "p_event_all": float(e_all.mean()), "lift": float(e_sig.mean() / e_all.mean()),
            "lift_lo": float(np.nanpercentile(ratios, 5)) if ratios else np.nan, "lift_hi": float(np.nanpercentile(ratios, 95)) if ratios else np.nan,
            "n_signal": int(sig.sum()), "n_signal_runs": runs, "n_events_after_signal": int(e_sig.sum()), "horizon": horizon, "quantile": quantile}


def fail_reason(res: dict, cfg: dict) -> str:
    """Which rule a non-retained feature failed first (unrounded values), for the packet."""
    if not res or not res.get("n") or "delta_r2" not in res:
        return "too few observations"
    checks = [("delta_r2", res["delta_r2"] >= cfg["min_delta_r2"]), ("folds_r2_up", res["folds_r2_up"] >= cfg["min_folds_up"]),
              ("transition_gain", res["transition_gain"] >= cfg["min_transition_gain"]), ("folds_transition_up", res["folds_transition_up"] >= cfg["min_folds_up"])]
    if np.isfinite(res.get("p_perm", np.nan)):
        checks.append(("p_perm", res["p_perm"] <= cfg.get("max_p_perm", 1.0)))
    failed = [n for n, ok in checks if not ok]
    if failed:
        return ";".join(failed)
    if res.get("holdout_tested"):
        if not (res.get("delta_r2_holdout", -1) > -cfg["min_delta_r2"]):
            return "holdout_contradicts_delta_r2"
        if not (res.get("transition_gain_holdout", -1) > -cfg["min_transition_gain"]):
            return "holdout_contradicts_transition_gain"
        return ""
    return "not_in_holdout_top_n" if res.get("_holdout_run") else ""


def retain(res: dict, cfg: dict) -> bool:
    """Gate 4 rule: both tests clear their thresholds in most folds; the outcome gain beats its permutation
    null (`max_p_perm`, when a p-value was computed); and, when a hold-out was run, neither test is
    CONTRADICTED there (each hold-out value above minus its threshold). The hold-out window is short — two
    years holds three or four episodes — so it is asked to not reverse the finding, not to re-prove it; the
    permutation p-value is what controls the selection over many candidates."""
    if not res or not res.get("n") or "delta_r2" not in res:
        return False
    ok = bool(res["delta_r2"] >= cfg["min_delta_r2"] and res["folds_r2_up"] >= cfg["min_folds_up"]
              and res["transition_gain"] >= cfg["min_transition_gain"] and res["folds_transition_up"] >= cfg["min_folds_up"])
    if ok and np.isfinite(res.get("p_perm", np.nan)):
        ok = bool(res["p_perm"] <= cfg.get("max_p_perm", 1.0))
    if ok and res.get("holdout_tested"):
        ok = bool(res.get("delta_r2_holdout", -1) > -cfg["min_delta_r2"] and res.get("transition_gain_holdout", -1) > -cfg["min_transition_gain"])
    return ok


SCREEN_COLS = ["feature", "target", "n", "n_pairs", "n_transitions", "r2_base", "delta_r2", "p_perm", "delta_r2_low", "delta_r2_diag", "diag_target", "folds_r2_up",
               "transition_gain", "folds_transition_up", "transition_gain_k10", "transition_gain_k21", "beta_mean",
               "holdout_tested", "delta_r2_holdout", "transition_gain_holdout", "retain", "fail_reason", "target_note"]


def screen(candidates, labels, state_score, target, cfg: dict, folds: int = 4, lag: int = 0, k: int = 5, ks=(10, 21),
           feature_targets: dict | None = None, holdout_frac: float | None = None, max_to_holdout: int | None = None,
           interact=LOW_BAND, prior_strength: float = 10.0, diag_targets: dict | None = None, n_perm: int | None = None) -> pd.DataFrame:
    """Every candidate one at a time against the same baseline; a table sorted by R^2 gain with the retain flag.

    `candidates` is a frame (or a dict pair -> frame for the pooled screen); `labels`, `state_score` and
    `target` likewise Series or dicts. `target` may also be a dict of NAMED outcomes ({"straddle_atm": y,
    "rr_25d": y2} per pair or plain); `feature_targets` maps a feature to the outcome it is judged on
    (default: the first), so a signed feature is tested against a signed earn.

    False-discovery control (`holdout_frac`, `max_to_holdout`, both from `cfg` when None): the walk-forward
    runs on the first 1 - holdout_frac of the history; only the `max_to_holdout` candidates that retain
    there, ranked by delta_r2, are then tested once on the final window (train on everything before it),
    and must be positive on both tests there to keep the flag. Everything else is reported but not retained.
    With holdout_frac 0 the old behaviour (walk-forward on the whole history) applies. `n_perm` (from `cfg`
    when None) adds the permutation p-value of the outcome gain; `diag_targets` maps an outcome name to a
    second, diagnostic outcome (the same archetype at a longer horizon) reported as `delta_r2_diag` only."""
    hf = cfg.get("holdout_frac", 0.0) if holdout_frac is None else holdout_frac
    mx = cfg.get("max_to_holdout", 3) if max_to_holdout is None else max_to_holdout
    npm = int(cfg.get("n_perm", 0)) if n_perm is None else int(n_perm)
    diag_targets = diag_targets or {}
    Cd, Lb, Cs = _as_dict(candidates), _as_dict(labels), _as_dict(state_score)
    pairs = list(Cd)
    # targets: {pair: {name: series}}
    T = {}
    for p in pairs:
        tp = target[p] if isinstance(target, dict) and p in target else target
        T[p] = dict(tp) if isinstance(tp, dict) else {"default": tp}
    names = list(T[pairs[0]])
    feature_targets = feature_targets or {}
    end = max(Cd[p].index.max() for p in pairs); start = min(Cd[p].index.min() for p in pairs)
    cut = start + (end - start) * (1 - hf) if hf else None
    rows = []
    for c in Cd[pairs[0]].columns:
        want = feature_targets.get(c, names[0]); tname = want if want in names else names[0]
        note = None if want == tname else f"{want} unavailable, judged on {tname}"
        feat = {p: Cd[p][c] for p in pairs}; y = {p: T[p][tname] for p in pairs}
        dev = {p: feat[p].loc[:cut] for p in pairs} if cut is not None else feat
        r = incremental_value(dev, Lb, Cs, y, folds=folds, lag=lag, k=k, ks=ks, interact=interact, prior_strength=prior_strength, n_perm=npm)
        dname = diag_targets.get(tname)
        if dname and all(dname in T[p] for p in pairs):
            rd = incremental_value(dev, Lb, Cs, {p: T[p][dname] for p in pairs}, folds=folds, lag=lag, interact=None, outcome_only=True)
            r.update({"delta_r2_diag": rd.get("delta_r2", np.nan), "diag_target": dname})
        r.update({"feature": c, "target": tname, "target_note": note, "holdout_tested": False, "_holdout_run": cut is not None, "_feat": feat, "_y": y})
        rows.append(r)
    for r in rows:
        r["retain"] = retain(r, cfg)
    if cut is not None:
        cands = sorted([r for r in rows if r["retain"]], key=lambda r: -r["delta_r2"])[:mx]
        for r in cands:
            h = incremental_value(r["_feat"], Lb, Cs, r["_y"], folds=1, min_train_frac=1 - hf, lag=lag, k=k, ks=(), interact=None, prior_strength=prior_strength)
            r.update({"holdout_tested": True, "delta_r2_holdout": h.get("delta_r2", np.nan), "transition_gain_holdout": h.get("transition_gain", np.nan)})
            r["retain"] = retain(r, cfg)
        for r in rows:
            if not r["holdout_tested"]:
                r["retain"] = False
    for r in rows:
        r["fail_reason"] = "" if r["retain"] else fail_reason(r, cfg)
        r.pop("_feat", None); r.pop("_y", None); r.pop("_holdout_run", None)
    return pd.DataFrame(rows).reindex(columns=SCREEN_COLS).sort_values("delta_r2", ascending=False).reset_index(drop=True)


def pressure_gain(psi, labels, state_score, folds: int = 4, k: int = 5, prior_strength: float = 10.0) -> dict:
    """Gate 4's second test, on the same footing as the screen's transition test: the pressure index as the
    tilt covariate, the state score as the baseline already in the tilt (both z-scored per fold on the
    training part), k-step likelihood. Accepts lists for the pooled case. Without the baseline the pressure
    is confounded with distance to the state boundary."""
    return oos_gain(labels, psi, folds=folds, k=k, base=state_score, prior_strength=prior_strength, standardise=True).attrs
