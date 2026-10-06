"""Review packets: one self-describing markdown file per stage, for a reviewer that cannot see the code.

`python -m regime_ladder pack --stage {data,states,ladder,leading} ...` reads what `inspect` wrote (or the
trade-day table, for `data`) and writes `packet.md`: a header (pair, dates, git commit, config hash), the
definitions the reviewer needs, and compact tables chosen so that a reader can check them against what
they know — dated runs of states rather than daily series, values before each escalation rather than
2,500 rows. Nothing here computes a new statistic; it formats and summarises. The challenger prompt in
`templates/CHALLENGER_PROMPT.md` is written against these sections.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import checks, labels, ladder, schema, shock

HIGH = ("stressed", "extreme", "crisis")
GENERIC = ("stored_energy", "gap_pct", "complacency", "jump_cluster_pct", "coupling_shift_pct", "coherence_pct", "event_proximity")
CAP = 250

COMMON = """- Units: P&L is cash per unit of standard notional (per unit of vega at inception once D1 is closed). Horizons h are trading days from entry, capped at the option tenor.
- Point-in-time: every feature, label and statistic stamped on date t uses only data available at the close of t. Labels are assigned from trailing data; forward quantities appear only as targets.
- Costs are excluded everywhere by design.
- States: carry, rising, agitated, stressed, normalising, settling (+ extreme if it has >= 5 episodes, else merged into stressed). Stress rank low to high (used for 'moves up' and by the transition tilt): carry < settling < rising < agitated < normalising < stressed < extreme."""

DEFS = {
    "sweep": """- A sweep holds the trade table, the horizons and Gate 2 fixed and varies only the labeller: feature set, partition (3 = level only; 6 = level x direction; 6x = plus an extreme band), finder grid (half-lives, hysteresis deltas, direction k), direction window, finder target.
- Each variant's finder is calibrated on the first part of the history (dates up to the fit end in section 1); `gate2_fit` is Gate 2's fraction of separated (group, h <= 10) cells on entries in that window, `gate2_confirm` on the entries after it, which the finder never saw. Threshold 0.75 for both.
- `finder_oos_r2` is the finder's own walk-forward R² of the state means on the daily forward target inside the fit window; it is a selection score, not a gate, and low values (0.02-0.10) are normal for a daily 5-day P&L target.
- `on_grid_edge`: the chosen half-life or hysteresis is the smallest / largest value tried. `fell_back`: the asked-for partition had too few episodes and a coarser one was used. `straddle_high_minus_low_h5`: mean 5-day earn of straddle entries in the high band minus the low band (should be positive). `agreement_with_first`: share of days on which the variant's label equals the first variant's.
- Every variant reads the same forward P&L, so a sweep is a look at the test set: one sweep, one adoption, recorded in DECISIONS.md.""",
    "data": """- One row per trade per day of life. `age` counts trading days from entry starting at 1. `pnl` = sum of components (trade, delta hedge, vega hedge) where present.
- Base legs: ATM straddle, 25d put/call, 10d put/call, each hedged leg by leg. Packages (RR, fly) are weight vectors over legs; if the source also runs them, the package should regress onto its legs with R² ~ 1 and stable coefficients.
- Expected: no gaps in age; components sum to pnl; long-option legs have positive vega at age 1; the largest daily P&L rows fall on dates of known market stress.""",
    "states": """- Composite stress score = mean of trailing percentiles of ATM vol level, term-structure slope and risk-reversal stress (0..100), smoothed by an EWMA.
- Level bands (low / mid / high) are step-fitted on forward ATM 5 days ahead over the whole history (a surface fact). Partition, direction thresholds, smoothing and hysteresis are chosen out of sample on the finder target under a switch budget.
- Episode = one uninterrupted run of a state. A state or tagged cell with < 5 episodes is not reported separately.
- Transition matrix: daily, with a sticky prior; implied duration 1/(1-p_ii) should be close to the observed mean run (ratio 0.6-1.5) or the Markov approximation is poor for that state.
- Profile `d` = standardised difference of a characteristic's mean in the state versus all days.""",
    "ladder": """- EV(h | state at entry) = mean of cumulative P&L from entry to the close of entry+h, over all entries whose entry-date label is that state. No daily rate is rolled forward; transitions after entry are inside the outcome.
- Interval: 90%, the wider of a block bootstrap and an episode-cluster bootstrap. `mean_shrunk` shrinks toward the ALL row with strength kappa. `n_eff` corrects for overlapping horizons; `episodes` counts independent runs of the entry state.
- Gate 2: in >= 75% of (archetype, tenor, h<=10) cells the best-minus-worst state spread exceeds 2 x the larger SE. Gate 3: out of sample (walk-forward), state ladder beats unconditional mean at every h<=10, positive rank IC, calibration slope in 0.5-1.5.""",
    "leading": """- Leading features (0..100; all trailing percentiles except stored_energy, which is a product of two percentiles divided by 100, so its typical level is ~20-25, not 50):
  - gap_pct: how unusual today's distance of spot from its 120-day anchor is, in implied-vol units.
  - complacency: realised under implied, low vol-of-vol, cheap front end, combined.
  - stored_energy = gap_pct x complacency / 100 (a stretched price held by a complacent market; the hypothesis is the product).
  - jump_cluster_pct: clustering of large returns. coupling_shift_pct: change in short vs long spot-vol coupling. coherence_pct: cross-pair co-movement.
  - gap_z_signed: the signed stretch itself (a z, not a percentile); drift_t: t-statistic of the 60-day drift (signed). Signed features are judged on the RR outcome (`target` column), unsigned ones on the straddle.
  - Any other name is a desk series supplied through the adapter; it is claimed trailing-only.
- Screen, one feature at a time, against a baseline that already knows the named state AND the continuous composite, 4 walk-forward folds after a 40% initial training block (`n_pairs` > 1 means the test is pooled across pairs with pair dummies, folds cut at common dates, one tilt coefficient across pairs):
  - (1) outcome test: `delta_r2` = pooled out-of-sample R² gain on the forward outcome named in `target`; `folds_r2_up` = folds in which the gain is > 0; `n` = days with feature, label, composite and outcome; `r2_base` = the baseline's own OOS R² on that feature's sample (samples differ by feature because warm-ups differ). `delta_r2_low` is the gain when the feature is allowed to act only in the calm states (carry, settling) as well as overall: the conditional form of the stored-energy hypothesis. `delta_r2_lagged` repeats the test with the feature delayed one day; differences under 0.005 are noise.
  - (2) transition test: `transition_gain` = out-of-sample k-step transition log-likelihood gain per transition (nats) through a tilt, with the composite already in the tilt, both z-scored per fold on the training part; the declared k is 5. `transition_gain_k10` and `_k21` are the same at longer horizons, diagnostics only: a slow build-up that leads by a week can show at 21 before it shows at 5. `folds_transition_up` = folds with gain > 0 at the declared k; `n_transitions` = days with feature and label (no outcome needed, so it can exceed `n`). `beta_mean` = tilt coefficient per standard deviation of the feature, averaged over folds.
  - `fail_reason` names the first rule a non-retained feature failed, on unrounded values (the table rounds). `target_note` says when a feature's configured outcome was unavailable and the default was used.
  - `p_perm`: share of 100 circular shifts of the feature (>= 63 days) whose outcome gain matches or beats the observed one; the selection control for a screen over many candidates. A real feature has p_perm near 0.01 (the floor); a noise feature anywhere.
  - Retention, two stages. Walk-forward on the first 80% of the history: delta_r2 >= 0.005 AND folds_r2_up >= 3 AND transition_gain >= 0.002 AND folds_transition_up >= 3 AND p_perm <= 0.05. Then only the top 3 candidates by delta_r2 that pass are tested once on the final 20% (`holdout_tested` = True; trained on everything before it) and keep the flag unless the hold-out CONTRADICTS them: `delta_r2_holdout` must be > -0.005 and `transition_gain_holdout` > -0.002. The hold-out holds three or four episodes, so it is asked not to reverse the finding, not to re-prove it; read a hold-out value near zero as neutral, a clearly negative one as the warning it is. A candidate that passed the walk-forward but was not in the top 3 shows `holdout_tested` = False and is not retained.
- Lift (event study): on days when the feature is at or above its trailing 90th percentile, the probability that the state enters the high band within 20 days, divided by the same probability on all eligible days (days not already in the high band). `n_signal_runs` counts separate signal episodes — the effective sample. 90% block-bootstrap interval. Not a gate; it is the more powerful test for a sparse feature and the number a trader can read directly.
- Shock score (0..100) = mean of CUSUM pressure on squared surprises (an alarm is a day on which the pressure reaches 0.999, i.e. the statistic crosses its calibrated threshold), BOCD probability of a short run, and recent |surprise|; surprise = return / implied daily vol marked the day before.
- Gate 4: at least one feature retained AND the pressure index of the retained features passes test (2) on its own: gain >= 0.002 per transition, > 0 in >= 3/4 folds.
- Pressure index = mean of the RETAINED features after each is z-scored against its own trailing 3-year window (0 = normal for that feature; percentiles, products and signed series on one footing). If nothing is retained, no pressure index exists and only the constant fan is meaningful.
- High band = stressed and extreme. Transition tilt: P_ij(psi) proportional to P0_ij x exp(beta x psi x (rank_j - rank_i)), with the composite as a second covariate (gamma); likelihood scored k = 5 days ahead. The fan horizon (21 days) is the transition forecast's horizon, not a ladder horizon.""",
}


# ----------------------------------------------------------------------------- formatting
def _md(df: pd.DataFrame, cap: int = CAP, digits: int = 3, index: bool = False) -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    d = df.reset_index() if index else df
    note = f"\n_{len(d) - cap} more rows not shown_\n" if len(d) > cap else ""
    d = d.head(cap)

    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else (f"{v:.{digits}f}" if abs(v) < 1e5 else f"{v:.3g}")
        return str(v)
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    rows += ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows) + "\n" + note


def _kv(obj: dict) -> str:
    return _md(pd.DataFrame({"key": list(obj), "value": [json.dumps(v) if isinstance(v, (dict, list)) else v for v in obj.values()]}))


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _cfg_hash(paths) -> str:
    h = hashlib.sha1()
    for p in sorted(map(str, paths)):
        if Path(p).exists():
            h.update(Path(p).read_bytes())
    return h.hexdigest()[:10]


def _read(src: Path, name: str, **kw):
    p = src / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p, **kw)


def runs(lab: pd.Series) -> pd.DataFrame:
    """Uninterrupted runs of a label series: start, end, state, trading days."""
    lab = lab.dropna()
    grp = (lab != lab.shift()).cumsum()
    r = lab.groupby(grp).agg(["first", "size"]).rename(columns={"first": "state", "size": "days"})
    idx = lab.index.to_series().groupby(grp.values)
    r["start"] = idx.min().dt.date.values; r["end"] = idx.max().dt.date.values
    return r[["start", "end", "state", "days"]].reset_index(drop=True)


def escalations(lab: pd.Series, into=HIGH) -> pd.DataFrame:
    """Days the label moves into the high band from outside it."""
    prev = lab.shift()
    m = lab.isin(into) & ~prev.isin(into) & prev.notna()
    return pd.DataFrame({"date": lab.index[m], "from": prev[m].values, "to": lab[m].values})


# ----------------------------------------------------------------------------- stages
def stage_data(td: pd.DataFrame, market: pd.DataFrame | None, archetypes_cfg: dict) -> list[str]:
    td = schema.coerce(td); out = []
    rep = checks.check_trade_days(td)
    out += ["## 1. Integrity checks", "```", checks.format_report(rep), "```"]
    g = td.groupby(["pair", "archetype", "tenor_days"])
    cov = g.agg(trades=("trade_id", "nunique"), rows=("pnl", "size"), first_entry=("entry_date", "min"),
                last_entry=("entry_date", "max"), max_age=("age", "max")).reset_index()
    for c in ("first_entry", "last_entry"):
        cov[c] = cov[c].dt.date
    out += ["## 2. Coverage", _md(cov)]
    months = td.groupby(["pair", "archetype"])["entry_date"].apply(lambda s: pd.period_range(s.min(), s.max(), freq="M").difference(s.dt.to_period("M").unique()))
    gaps = [(p, a, ", ".join(map(str, v[:12]))) for (p, a), v in months.items() if len(v)]
    out += ["Months with no entries: " + ("; ".join(f"{p} {a}: {v}" for p, a, v in gaps) if gaps else "none") + "\n"]
    dsum = g["pnl"].describe(percentiles=[0.01, 0.5, 0.99]).reset_index()
    out += ["## 3. Daily P&L distribution per leg", _md(dsum)]
    cum = ladder.cumulative(td, horizons=(1, 5, 10), include_expiry=True)
    cs = cum.groupby(["archetype", "tenor_days", "h", "to_expiry"])["cum_pnl"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    out += ["## 4. Cumulative P&L from entry (all entries pooled)", _md(cs)]
    day = td.groupby(["date", "pair", "archetype"]).agg(mean_pnl=("pnl", "mean"), live_trades=("pnl", "size")).reset_index()
    big = day.reindex(day["mean_pnl"].abs().sort_values(ascending=False).index).head(20)
    big = big.assign(date=big["date"].dt.date)
    out += ["## 5. Twenty largest market days: mean daily P&L across live trades of a leg (check the dates against known stress)", _md(big)]
    if all(c in td for c in schema.COMPONENTS):
        comp = td.groupby("archetype").apply(lambda x: pd.Series({c: x[c].var() / x["pnl"].var() for c in schema.COMPONENTS}
                                                               | {"max_abs_residual": (x["pnl"] - x[schema.COMPONENTS].sum(axis=1)).abs().max()})).reset_index()
        out += ["## 6. Components: variance of each component / variance of total (can exceed 1 when components offset) and largest residual", _md(comp, digits=4)]
    else:
        out += ["## 6. Components", "not present in the trade table\n"]
    if "vega" in td:
        v1 = td[td["age"] == 1].groupby(["archetype", "tenor_days"])["vega"].agg(["median", "min", "max", lambda s: int((s <= 0).sum())]).reset_index()
        v1.columns = ["archetype", "tenor_days", "median", "min", "max", "n_nonpositive"]
        out += ["## 7. Vega at age 1", _md(v1, digits=4)]
    else:
        out += ["## 7. Vega at age 1", "`vega` not present: D1 falls back to source notional\n"]
    rec = []
    key = ["pair", "tenor_days", "entry_date", "age"]
    for name, w in (archetypes_cfg.get("combinations") or {}).items():
        if name in set(td["archetype"]) and all(l in set(td["archetype"]) for l in w):
            wide = td[td["archetype"].isin([name, *w])].pivot_table(index=key, columns="archetype", values="pnl").dropna()
            X, y = wide[list(w)].values, wide[name].values
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            std = X @ np.array(list(w.values()))
            rec.append({"package": name, **{f"coef_{l}": c for l, c in zip(w, b)}, "r2": r2, "corr_with_standard_weights": np.corrcoef(y, std)[0, 1], "n": len(y)})
    out += ["## 8. Package reconciliation (package daily P&L regressed on its legs)", _md(pd.DataFrame(rec)) if rec else "no package with all its legs present\n"]
    if market is not None:
        missing = sorted(set(td["date"]) - set(market.index.normalize()))
        out += ["## 9. Market frame", f"{len(market)} days {market.index[0].date()} to {market.index[-1].date()}; trade dates missing from the market frame: {len(missing)}" + (f" (first: {[str(d.date()) for d in missing[:5]]})" if missing else "") + "\n",
                _md(market.describe().T[["count", "mean", "min", "max"]], index=True)]
    return out


def stage_states(src: Path) -> list[str]:
    comp = _read(src, "composite.csv", index_col=0, parse_dates=True); lab = comp["regime"]
    out = ["## 1. Chosen spec", _kv(_read(src, "finder_spec.json")), "Best out-of-sample R² per partition: " + json.dumps(_read(src, "finder_r2.json")) + "\n"]
    rep = _read(src, "finder_report.csv", index_col=0)
    cols = [c for c in ["partition", "halflife", "delta", "dir_k", "b1", "b2", "oos_r2", "switches_per_year", "min_episodes", "extreme_merged"] if c in rep]
    top = rep.sort_values("oos_r2", ascending=False).groupby("partition").head(5)[cols]
    out += ["## 2. Finder: top 5 candidates per partition", _md(top)]
    fi = _read(src, "finder_inputs.csv", index_col=0, parse_dates=True)
    if fi is not None:
        fi = fi.join(comp[["smoothed", "regime", "level", "direction"]], how="inner")
        b = pd.cut(fi["smoothed"], bins=range(0, 101, 10), include_lowest=True)
        g = fi.groupby(b, observed=True).agg(n=("target", "count"), target_mean=("target", "mean"), target_sd=("target", "std"), level_target_mean=("level_target", "mean"))
        g["target_se_x_sqrt5"] = g["target_sd"] / np.sqrt(g["n"].clip(lower=1)) * np.sqrt(5)
        out += ["## 2b. What the finder saw: both targets by smoothed-composite decile",
                "`target` is the forward P&L the partition / direction / hysteresis are scored on; `level_target` the forward ATM the bounds are step-fitted on. "
                "The s.e. is inflated by sqrt(5) because consecutive 5-day targets overlap. In-sample, whole history.", _md(g.reset_index().rename(columns={"smoothed": "composite_bin"}))]
        by = {}
        for key in ("regime", "level", "direction"):
            t = fi.groupby(key)["target"].agg(n="count", mean="mean", sd="std"); t["se_x_sqrt5"] = t["sd"] / np.sqrt(t["n"].clip(lower=1)) * np.sqrt(5); by[key] = t
        out += ["## 2c. Mean target by named state, by level band alone, by direction band alone (in-sample)",
                "If the level bands separate and the direction bands do not, the direction axis is not earning its place.",
                _md(by["regime"], index=True), _md(by["level"], index=True), _md(by["direction"], index=True)]
    dur = _read(src, "durations.csv")
    share = lab.value_counts(normalize=True).rename("share_of_days")
    sm = comp.groupby("regime")["smoothed"].agg(["mean", "min", "max"]).add_prefix("composite_")
    summ = dur.set_index("state").join(share).join(sm)
    summ["ratio_implied_over_observed"] = summ["implied_days"] / summ["observed_mean_days"]
    out += ["## 3. States: share, episodes, durations, composite range", _md(summ, index=True)]
    warm = comp["smoothed"].first_valid_index()
    out += [f"Features warm up until {warm.date() if warm is not None else '?'}: {int((comp.index < warm).sum()) if warm is not None else '?'} earlier days carry the "
            "default label (carry) and must not be read as calm. The trade table should start after this date.\n"]
    yr = pd.crosstab(pd.Index(lab.index.year, name="year"), lab, normalize="index").round(2)
    out += ["## 4. Share of days per state by year", _md(yr, index=True, digits=2)]
    r = runs(lab)
    if len(r) > CAP:
        out += [f"## 5. Runs of agitated / stressed / extreme ({len(r)} runs in all; low-band runs omitted)", _md(r[r["state"].isin(("agitated",) + HIGH)])]
    else:
        out += ["## 5. Every run of every state", _md(r)]
    out += ["## 6. Transition matrix with sticky prior (row = from)", _md(_read(src, "transition_matrix.csv", index_col=0), index=True),
            "Raw counts-based matrix (no prior):", _md(_read(src, "transition_counts.csv", index_col=0), index=True)]
    prof = _read(src, "state_profile.csv", index_col=0)
    if prof is not None:
        rows = []
        for s in [c[:-2] for c in prof.columns if c.endswith("_d")]:
            d = prof[f"{s}_d"].dropna()
            for ch in d.abs().sort_values(ascending=False).index[:4]:
                rows.append({"state": s, "characteristic": ch, "d": d[ch], "state_mean": prof.loc[ch, f"{s}_mean"], "all_mean": prof.loc[ch, "all_mean"]})
        out += ["## 7. What distinguishes each state (top 4 characteristics by |d|)", _md(pd.DataFrame(rows))]
    disc = _read(src, "discovery.json")
    if disc:
        drows = [{"state": s, "k_chosen": v["k"], "best_stability": max(v["stability"].values()) if v["stability"] else np.nan,
                  "names": ", ".join(v["names"].values())} for s, v in disc.items()]
        out += ["## 8. Sub-state discovery (kept only if stability >= 0.7 and every cluster >= 5 episodes)", _md(pd.DataFrame(drows))]
    tc = _read(src, "tag_cells.json")
    if tc:
        trows = [{"tag": t, "cell": c, **info} for t, cells in tc.items() for c, info in cells.items() if isinstance(info, dict)]
        out += ["## 9. Tag cells", _md(pd.DataFrame(trows))]
    return out


def stage_sweep(src: Path) -> list[str]:
    """One sweep folder (regime_ladder/sweep.py): the variants table and, per variant, what its labels look like."""
    rows = _read(src, "sweep.csv")
    readme = (src / "README.md").read_text() if (src / "README.md").exists() else ""
    out = ["## 1. Variants", readme.strip(), "",
           _md(rows[[c for c in ["name", "features", "target", "partition", "fell_back", "halflife", "delta", "dir_k", "b1", "b2", "finder_oos_r2", "switches_per_year", "states",
                                 "min_episodes", "duration_ratio_min", "duration_ratio_max", "straddle_high_minus_low_h5", "gate2_fit", "gate2_confirm", "on_grid_edge", "agreement_with_first"] if c in rows]], cap=50)]
    yr = {}
    for r in rows.itertuples():
        f = src / f"labels_{r.name}.parquet"
        if f.exists():
            lab = pd.read_parquet(f).set_index("date")["regime"]
            yr[r.name] = pd.crosstab(pd.Index(lab.index.year, name="year"), lab, normalize="index").round(2)
    if yr:
        first = next(iter(yr))
        out += [f"## 2. Share of days per state by year — variant `{first}` (the reference)", _md(yr[first], index=True, digits=2)]
        diffs = []
        for n, t in yr.items():
            if n == first:
                continue
            a, b = yr[first].align(t, join="outer", fill_value=0)
            d = (a - b).abs().sum(axis=1) / 2          # total variation distance per year
            diffs.append({"variant": n, "mean_yearly_label_shift": d.mean(), "max_yearly_label_shift": d.max(), "year_of_max": int(d.idxmax())})
        out += ["## 3. How far each variant's labels move from the reference, per year (total variation of the state shares; 0 = same, 1 = disjoint)", _md(pd.DataFrame(diffs))]
    return out


def stage_ladder(src: Path) -> list[str]:
    lad = _read(src, "ladder.csv")
    if lad is None:
        return ["No ladder in the source folder: run `inspect` with `--td`."]
    out = ["## 1. Gates", "Gate 2: " + json.dumps({k: v for k, v in (_read(src, "gate_phase2.json") or {}).items() if k != "cells"}) + "\n",
           "Gate 3: " + json.dumps(_read(src, "gate_phase3.json")) + "\n"]
    cols = ["archetype", "tenor_days", "regime", "h", "mean", "mean_shrunk", "ci_lo", "ci_hi", "p_profit", "es05", "n_eff", "episodes"]
    show = lad[(~lad["to_expiry"]) & lad["h"].isin([1, 5, 10])][cols]
    out += ["## 2. Ladder at h = 1, 5, 10", _md(show, cap=400)]
    thin = lad[(lad["regime"] != "ALL") & ((lad["episodes"] < 5) | (lad["n_eff"] < 30))][["archetype", "tenor_days", "regime", "h", "n_eff", "episodes"]]
    out += ["## 3. Thin cells (episodes < 5 or n_eff < 30)", _md(thin)]
    sg = lad[(~lad["to_expiry"]) & (lad["regime"] != "ALL")].groupby(["archetype", "tenor_days", "regime"])["mean"].apply(lambda s: np.sign(s).nunique() > 1)
    out += ["## 4. Cells whose mean changes sign across horizons", ", ".join("/".join(map(str, k)) for k, v in sg.items() if v) or "none", ""]
    wf = _read(src, "walk_forward.csv")
    out += ["## 5. Walk-forward: state ladder vs unconditional (out of sample)", _md(wf)]
    return out


def alignment(L: pd.DataFrame, market: pd.DataFrame, col: str = "rv_1w", shifts=(-21, -10, -5, -1, 0, 1, 2, 5, 10, 15, 21)) -> pd.DataFrame:
    """Rank correlation of each feature on day t with `col` (a trailing realised vol) stamped on day t+k. For a
    trailing feature the profile is smooth and usually peaks at k <= 0; a sharp peak at one k > 0, far above its
    neighbours, is the signature of a future value stamped on today. A genuine slow leader rises gently into
    the future and plateaus well below that; the grid runs to +21 so the two shapes can be told apart."""
    rows = {}
    for c in L.columns:
        rows[c] = {f"k={k:+d}": L[c].corr(market[col].shift(-k).reindex(L.index), method="spearman") for k in shifts}
    return pd.DataFrame(rows).T


def stage_leading(src: Path, market: pd.DataFrame | None = None) -> list[str]:
    comp = _read(src, "composite.csv", index_col=0, parse_dates=True); lab = comp["regime"]
    L = _read(src, "leading_features.csv", index_col=0, parse_dates=True)
    S = _read(src, "shock.csv", index_col=0, parse_dates=True)
    psi = _read(src, "pressure.csv", index_col=0, parse_dates=True)
    out = ["## 1. Gate 4", "```", json.dumps(_read(src, "gate_phase4.json"), indent=1), "```",
           "## 2. Screen (one feature at a time vs state + composite baseline)"]
    sc = _read(src, "leading_screen.csv")
    main = [c for c in ["feature", "target", "n", "n_pairs", "r2_base", "delta_r2", "p_perm", "delta_r2_low", "delta_r2_diag", "delta_r2_lagged", "folds_r2_up", "transition_gain", "folds_transition_up",
                        "holdout_tested", "delta_r2_holdout", "transition_gain_holdout", "retain", "fail_reason"] if c in sc]
    if "target_note" in sc and sc["target_note"].notna().any():
        out += ["Target substitutions: " + "; ".join(f"{r.feature}: {r.target_note}" for r in sc.dropna(subset=["target_note"]).itertuples()) + "\n"]
    if "diag_target" in sc and sc["diag_target"].notna().any():
        out += [f"`delta_r2_diag` is the outcome test on {', '.join(sorted(set(sc['diag_target'].dropna())))} (the same archetype at the diagnostic horizon); it does not decide retention.\n"]
    out += [_md(sc[main], digits=4)]
    hz = [c for c in ["feature", "n_transitions", "transition_gain", "transition_gain_k10", "transition_gain_k21", "beta_mean"] if c in sc]
    out += ["Transition test by horizon (k = 5 decides; 10 and 21 are diagnostics):", _md(sc[hz], digits=4)]
    lf = _read(src, "leading_lift.csv")
    if lf is not None:
        lc = [c for c in ["feature", "p_event_signal", "p_event_all", "lift", "lift_lo", "lift_hi", "n_signal", "n_signal_runs", "n_events_after_signal"] if c in lf]
        out += ["## 2b. Lift: P(high band within 20 days | feature in its top decile) vs base rate", _md(lf[lc], digits=3)]
    desc = L.describe(percentiles=[0.05, 0.5, 0.95]).T
    desc["first_valid"] = [L[c].first_valid_index().date() if L[c].notna().any() else None for c in L]
    desc["last_valid"] = [L[c].last_valid_index().date() if L[c].notna().any() else None for c in L]
    desc["share_at_mode"] = [L[c].round(6).value_counts(normalize=True).iloc[0] if L[c].notna().any() else np.nan for c in L]
    out += ["## 3. Feature distributions", _md(desc, index=True, digits=1)]
    by = L.join(lab).groupby("regime").mean().T
    out += ["## 4. Mean feature value by state (0..100)", _md(by, index=True, digits=1)]
    cm = L.join(comp["smoothed"].rename("composite")).corr()
    out += ["## 5. Correlations (features and the composite)", _md(cm, index=True, digits=2)]
    lp = []
    for c in L.columns:
        for b in (5, 20):
            r = shock.lead_profile(L[c], lab, before=b).iloc[0]
            lp.append({"feature": c, "window_days": b, "all_days": r["all"], "before_escalation": r["before_up"], "before_de_escalation": r["before_down"], "n_escalations": r["before_up_n"], "n_de_escalations": r["before_down_n"]})
    out += ["## 6. Lead profile: mean value in the days before ANY move up the state ranking (e.g. carry to rising), before moves down, and on all days. Moves skip ranks (a rising-to-stressed move counts once), so up and down counts need not match", _md(pd.DataFrame(lp), digits=1)]
    esc = escalations(lab)
    alarms = S.index[S["cusum_pressure"] >= 0.999]
    rows = []
    pos = {d: i for i, d in enumerate(lab.index)}
    for e in esc.itertuples(index=False):
        i = pos[e.date]
        row = {"date": e.date.date(), "from": e[1], "to": e.to}
        if "stored_energy" in L:
            for k in (20, 5, 1):
                row[f"stored_energy_t-{k}"] = L["stored_energy"].iloc[i - k] if i >= k else np.nan
        for c in [c for c in L.columns if c not in GENERIC]:
            row[f"{c}_t-5"] = L[c].iloc[i - 5] if i >= 5 else np.nan
            row[f"{c}_t-1"] = L[c].iloc[i - 1] if i >= 1 else np.nan
        if psi is not None and (_read(src, "gate_phase4.json") or {}).get("retained"):
            row["pressure_t-1"] = psi.iloc[i - 1, 0] if i >= 1 else np.nan
        row["shock_max_20d"] = S["score"].iloc[max(0, i - 20): i].max()
        prior = alarms[alarms < e.date]
        row["days_since_cusum_alarm"] = (lab.index[i] - prior[-1]).days if len(prior) else np.nan
        rows.append(row)
    out += [f"## 7. Every move into the high band ({len(esc)}): values before it", _md(pd.DataFrame(rows), digits=1)]
    esc_d = set(esc["date"])
    hit = [any(d < e <= d + pd.Timedelta(days=30) for e in esc_d) for d in alarms]
    caught = [any(e - pd.Timedelta(days=30) <= a < e for a in alarms) for e in esc_d]
    out += ["## 8. Shock detector against the labels",
            f"CUSUM alarms: {len(alarms)}; followed by a move into the high band within 30 calendar days: {sum(hit)}; "
            f"moves into the high band preceded by an alarm within 30 days: {sum(caught)} of {len(esc_d)}.\n",
            f"Surprise: mean {S['surprise'].mean():+.3f}, sd {S['surprise'].std():.3f} (should be ~1 if implied matches realised on average), "
            f"1%/99% {S['surprise'].quantile(0.01):+.2f}/{S['surprise'].quantile(0.99):+.2f}; max CUSUM pressure {S['cusum_pressure'].max():.4f} on {S['cusum_pressure'].idxmax().date()} (alarm when it reaches 0.999).\n",
            "Shock score lead profile (5 days before):", _md(_read(src, "shock_lead_profile.csv"), digits=2),
            "Alarm dates (first 60): " + ", ".join(str(d.date()) for d in alarms[:60]) + "\n"]
    if market is not None and "rv_1w" in market:
        out += ["## 8b. Alignment check: Spearman correlation of each feature on day t with trailing 1-week realised vol on day t+k "
                "(k < 0 past, k > 0 future). A stamping error is a sharp peak at one future k; a genuine slow lead rises gently and plateaus", _md(alignment(L, market), index=True, digits=2)]
    f0, f1 = _read(src, "fan_constant.csv", index_col=0), _read(src, "fan_tilted.csv", index_col=0)
    if f0 is not None:
        hi = [c for c in f0.columns if c in HIGH]
        fan = pd.DataFrame({"h": [5, 10, 21], "P(high band) constant": [f0.loc[h, hi].sum() for h in (5, 10, 21)],
                            "P(high band) tilted": [f1.loc[h, hi].sum() for h in (5, 10, 21)]})
        g4 = _read(src, "gate_phase4.json") or {}
        if not g4.get("retained"):
            out += [f"## 9. Fan from today's state ({lab.iloc[-1]}, {lab.index[-1].date()}): constant matrix only (nothing retained, so no pressure index)",
                    _md(fan[["h", "P(high band) constant"]])]
            return out
        today_psi = float(psi.iloc[-1, 0]) if psi is not None else float("nan")
        tp = {k: round(v, 4) if isinstance(v, float) else v for k, v in (_read(src, 'tilt_pressure.json') or {}).items()}
        out += [f"## 9. Fan from today's state ({lab.iloc[-1]}, {lab.index[-1].date()}); pressure from {g4['retained']}; pressure today {today_psi:+.2f} (0 = normal; the tilt moves the fan only when pressure is away from 0); "
                f"whole-history tilt fit on the raw pressure scale (β per unit of pressure, not per sd as in section 2): " + json.dumps(tp), _md(fan)]
    return out


def build(stage: str, out: str | Path, src: str | Path | None = None, td: pd.DataFrame | None = None,
          market: pd.DataFrame | None = None, label: str = "", config_paths=(), archetypes_cfg: dict | None = None) -> Path:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    meta = _read(Path(src), "meta.json") if src and (Path(src) / "meta.json").exists() else None
    meta = meta or {}
    head = [f"# Review packet · stage `{stage}`" + (f" · {label}" if label else ""), "",
            _kv({"pair": meta.get("pair", "?"), "data": f"{meta.get('first_date', '?')} to {meta.get('last_date', '?')} ({meta.get('days', '?')} days)",
                 "finder / leading target": meta.get("finder_target", "n/a"), "git commit": _git(), "config hash": _cfg_hash(config_paths),
                 "generated": datetime.now().strftime("%Y-%m-%d %H:%M")}),
            "## Definitions", COMMON, DEFS[stage], ""]
    if stage == "data":
        body = stage_data(td, market, archetypes_cfg or {})
    else:
        body = stage_leading(Path(src), market) if stage == "leading" else {"states": stage_states, "ladder": stage_ladder, "sweep": stage_sweep}[stage](Path(src))
    p = out / "packet.md"
    p.write_text("\n".join(head + body) + "\n")
    return p


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())
