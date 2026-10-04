"""Tags: qualifiers on a named state for the kinds of market a six-state labeller cannot tell apart.

A tag is a daily categorical computed point-in-time (same `asof` contract as `features.py`, same
truncation test). Tags never change the state. They split a state's ladder — the regime becomes
"state·tag" — and the split survives only where the (state, tag) cell has enough episodes to
condition on; otherwise the cell collapses back to its parent state (`tag_split`). That gate is
what keeps tags from multiplying cells until nothing is estimable.

Four tags, chosen for the agitated markets a desk actually distinguishes:
* corr_sign          sign of trailing spot-vol correlation: "neg" (risk-off pair behaviour), "pos"
                     (flipped: vol bid on rallies), "flat". A dead band and hysteresis stop flicker.
* event_window       inside a known window before/after a scheduled event (central bank, data,
                     election) from a calendar supplied by the adapter. Known in advance, so
                     point-in-time by construction.
* pinned             realised running well below implied with spot in a narrow range: the market
                     paying for movement it is not getting. Long-gamma P&L dies here first.
* intervention_risk  pair-specific: a fast one-way move, a level breached, or an official-comment
                     flag — the BoJ-style configuration for USDJPY. Direction and thresholds live in
                     config per pair; the function is generic.

Two things tags do NOT do: they are not sub-states (discovery finds those from the characteristics
vector without being told what to look for; tags are declared by the desk), and they are not
leading features (tags describe today; leading features are scored on what they say about tomorrow).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import episodes, ordered_labels

SEP = "·"


def _cut(market: pd.DataFrame, asof) -> pd.DataFrame:
    return market if asof is None else market.loc[: pd.Timestamp(asof)]


def corr_sign_tag(market, asof=None, spot="spot", iv="atm_1m", window=21, band=0.2, delta=0.05) -> pd.Series:
    """Sign of the trailing correlation between spot returns and implied-vol changes, with a dead band
    [-band, band] labelled "flat" and hysteresis `delta` on both edges."""
    m = _cut(market, asof)
    dr, dv = np.log(m[spot]).diff(), m[iv].diff()
    c = dr.rolling(window, min_periods=max(10, window // 2)).corr(dv)
    return ordered_labels(c, [-band, band], delta, ("neg", "flat", "pos"), start=1).rename("corr_sign")


def event_window_tag(index: pd.DatetimeIndex, events, before: int = 3, after: int = 1) -> pd.Series:
    """'pre' on the `before` business days up to and including an event date, 'post' on the `after` days
    after it, else 'none'. `events` is any iterable of dates the desk knows in advance."""
    idx = pd.DatetimeIndex(index)
    out = pd.Series("none", index=idx, name="event_window")
    pos = {d: i for i, d in enumerate(idx)}
    for e in pd.to_datetime(list(events)):
        i = idx.searchsorted(e)  # first index >= event (events on non-business days roll forward)
        if i >= len(idx):
            continue
        out.iloc[max(0, i - before + 1): i + 1] = "pre"
        out.iloc[i + 1: i + 1 + after] = np.where(out.iloc[i + 1: i + 1 + after] == "pre", "pre", "post")
    return out


def pinned_tag(market, asof=None, spot="spot", iv="atm_1m", rv="rv_1m", window=21, rv_ratio=0.6, range_ratio=0.6) -> pd.Series:
    """'pinned' when realised/implied < rv_ratio AND the trailing high-low range is < range_ratio of the
    range implied vol would suggest for the window (spot * iv * sqrt(window/252)); else 'none'."""
    m = _cut(market, asof)
    s = m[spot]
    rng = (s.rolling(window).max() - s.rolling(window).min()) / s
    implied_range = 2 * m[iv] / 100 * np.sqrt(window / 252)  # ~ +/- one sigma band width
    quiet = (m[rv] / m[iv] < rv_ratio) & (rng / implied_range < range_ratio)
    return pd.Series(np.where(quiet, "pinned", "none"), index=m.index, name="pinned")


def intervention_risk_tag(market, asof=None, spot="spot", direction: int = 1, level: float | None = None,
                          move_window: int = 21, move_threshold: float = 0.05, flag_col: str | None = None) -> pd.Series:
    """'risk' when spot has moved more than `move_threshold` in `direction` over `move_window` days, or sits
    beyond `level` in that direction, or the adapter's `flag_col` (official comment, rate check) is set.
    direction=+1 means the authority leans against a RISING spot (USDJPY: yen weakness)."""
    m = _cut(market, asof)
    s = m[spot]
    risk = direction * (s / s.shift(move_window) - 1) > move_threshold
    if level is not None:
        risk |= direction * (s - level) > 0
    if flag_col and flag_col in m:
        risk |= m[flag_col].fillna(False).astype(bool)
    return pd.Series(np.where(risk.fillna(False), "risk", "none"), index=m.index, name="intervention_risk")


TAGS = {"corr_sign": corr_sign_tag, "pinned": pinned_tag, "intervention_risk": intervention_risk_tag}
# event_window_tag takes a calendar, not a market frame, so it is applied by the caller (see `build`).


def build(market: pd.DataFrame, cfg: dict, pair: str, asof=None) -> pd.DataFrame:
    """All configured tags for one pair. `cfg` is the `tags:` block of configs/archetypes.yaml or local.yaml:
    {name: {params}} plus optional events: [dates] and per-pair overrides under pairs: {PAIR: {name: {params}}}."""
    out = {}
    per_pair = (cfg.get("pairs") or {}).get(pair, {})
    for name, fn in TAGS.items():
        if name in cfg or name in per_pair:
            params = {**(cfg.get(name) or {}), **(per_pair.get(name) or {})}
            out[name] = fn(market, asof=asof, **params)
    if cfg.get("events") or per_pair.get("events"):
        ev = list(cfg.get("events") or []) + list(per_pair.get("events") or [])
        w = cfg.get("event_window") or {}
        out["event_window"] = event_window_tag(_cut(market, asof).index, ev, **w)
    return pd.DataFrame(out, index=_cut(market, asof).index)


def tag_split(labels: pd.Series, tag: pd.Series, min_episodes: int = 5, neutral=("none", "flat")) -> tuple[pd.Series, dict]:
    """Regime = state·tag where the cell has >= min_episodes episodes, else the parent state.

    Neutral tag values (`none`, `flat`) never split: the untagged days keep the plain state name, so a
    tag adds cells only for the condition it names. Returns (regime series, info per cell).
    """
    t = tag.reindex(labels.index)
    cand = pd.Series(np.where(t.isin(neutral) | t.isna(), labels.values, labels.astype(str) + SEP + t.astype(str)), index=labels.index)
    ep = episodes(cand)
    keep = {c for c, n in ep.items() if SEP not in c or n >= min_episodes}
    out = cand.where(cand.isin(keep), labels).rename("regime")
    info = {c: {"episodes": n, "kept": c in keep, "days": int((cand == c).sum())} for c, n in ep.items() if SEP in c}
    return out, info


def split_labels_frame(labels_frame: pd.DataFrame, tag: pd.Series, min_episodes: int = 5) -> tuple[pd.DataFrame, dict]:
    """Same as `tag_split` on a (pair, date, regime) label frame; the result feeds `ladder.ladder` unchanged."""
    out, infos = [], {}
    for pair, g in labels_frame.groupby("pair"):
        s = g.set_index("date")["regime"]
        reg, info = tag_split(s, tag, min_episodes)
        infos[pair] = info
        out.append(pd.DataFrame({"pair": pair, "date": reg.index, "regime": reg.values}))
    return pd.concat(out, ignore_index=True), infos


def parent(regime: str) -> str:
    return regime.split(SEP)[0]
