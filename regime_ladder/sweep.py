"""The states component on its own: one table over labeller variants, scored on a frozen trade table.

    python -m regime_ladder sweep --market m.parquet --td td.parquet --pair EURUSD [--sweep configs/sweep.yaml] [--out out/sweep]

Every variant changes only the labeller (feature set, partition, finder grid, direction window, finder
target). The trade table, the horizons and Gate 2 are the same for all of them, so differences in the
table are differences in the states. Each variant is calibrated on the first `1 - confirm_frac` of the
history and Gate 2 is reported separately on that fit window and on the final `confirm_frac`, which the
finder never saw: the confirm column is the one that stops the sweep from fitting the test set.

Writes `sweep.csv` (one row per variant) and `labels_<name>.parquet` for each, so any variant can go
straight into `ladder`, `inspect` and `view`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import features, gates, labels, ladder, schema, transitions

HORIZONS = (1, 3, 5, 10)
DEFAULT = {"features": ["vol_pct", "term_slope_pct", "rr_stress_pct"], "states": "6", "target": "forward_pnl", "target_archetype": "straddle_atm",
           "dir_window": 5, "halflives": [2, 3, 5], "deltas": [0, 2, 3, 5], "dir_ks": [0.5, 0.75, 1.0], "min_episodes": 5, "max_switches_per_year": 30,
           "level_target": "atm_1m", "level_horizon": 5}


def forward_pnl(td: pd.DataFrame, archetype: str, h: int = 5) -> pd.Series:
    cum = ladder.cumulative(td[td["archetype"] == archetype], horizons=(h,), include_expiry=False)
    return cum.groupby("entry_date")["cum_pnl"].mean()


def gate2(cum_lab: pd.DataFrame, lab: pd.DataFrame, gcfg: dict) -> float:
    if cum_lab.empty:
        return np.nan
    lad = ladder.ladder(cum_lab, lab, ci="hac")
    return gates.gate_phase2(lad, gcfg)["fraction_separated"]


def run_variant(v: dict, market: pd.DataFrame, td: pd.DataFrame, pair: str, cum: pd.DataFrame, gcfg: dict, fit_end) -> tuple[dict, pd.Series]:
    c = {**DEFAULT, **v}
    F = features.build(market, names=c["features"])
    comp = labels.composite(F)
    target = forward_pnl(td, c["target_archetype"]) if c["target"] == "forward_pnl" else market["rv_1m"].shift(-21)
    lvl = market[c["level_target"]].shift(-int(c["level_horizon"])) if c["level_target"] in market else None
    fit = comp.index <= fit_end
    cal = labels.calibrate_states(comp[fit], target.reindex(comp.index)[fit], level_target=None if lvl is None else lvl[fit],
                                  halflives=tuple(c["halflives"]), deltas=tuple(c["deltas"]), dir_ks=tuple(c["dir_ks"]), dir_window=int(c["dir_window"]),
                                  min_episodes=c["min_episodes"], max_switches_per_year=c["max_switches_per_year"])
    want = str(c["states"])
    spec = (cal["specs"].get(want) if want != "auto" else None) or cal["spec"]
    lab_s = labels.apply_spec(comp, spec)
    lab = schema.labels_frame(pair, lab_s.index, lab_s.values)
    cum_lab = ladder.attach_entry_labels(cum, lab)
    P = labels.transition_matrix(lab_s, prior_strength=10)
    dur = transitions.implied_vs_observed_duration(P, lab_s)
    ratio = (dur["implied_days"] / dur["observed_mean_days"]).replace([np.inf, -np.inf], np.nan)
    eps = labels.episodes(lab_s)
    # the straddle's sign: high-band entries (stressed, or crisis in the 3-state partition) should out-earn carry entries at h = 5
    s5 = cum_lab[(cum_lab["archetype"] == c["target_archetype"]) & (cum_lab["h"] == 5)].groupby("regime")["cum_pnl"].mean()
    row = {"name": v.get("name", "variant"), "features": "+".join(c["features"]), "target": c["target"], "partition": spec["partition"],
           "partition_requested": want, "fell_back": want != "auto" and str(spec["partition"]) != want,   # the asked-for partition had too few episodes
           "halflife": spec["halflife"], "delta": spec["delta"], "dir_k": spec.get("dir_k"), "b1": spec["bounds"][0], "b2": spec["bounds"][1],
           "finder_oos_r2": spec.get("oos_r2"), "switches_per_year": spec.get("switches_per_year"),
           "states": len(eps), "min_episodes": min(eps.values()) if eps else 0,
           "duration_ratio_min": ratio.min(), "duration_ratio_max": ratio.max(),
           "straddle_high_minus_low_h5": float(s5.get("stressed", s5.get("crisis", np.nan)) - s5.get("carry", np.nan)),
           "gate2_fit": gate2(cum_lab[cum_lab["entry_date"] <= fit_end], lab, gcfg),
           "gate2_confirm": gate2(cum_lab[cum_lab["entry_date"] > fit_end], lab, gcfg),
           "unlabelled_entries": cum_lab.attrs.get("unlabelled_entries", 0),
           "on_grid_edge": bool(spec["halflife"] in (min(c["halflives"]), max(c["halflives"])) or spec["delta"] == max(c["deltas"]))}
    return row, lab_s


def run(market: pd.DataFrame, td: pd.DataFrame, pair: str, sweep_cfg: dict, gcfg: dict, out: str | Path) -> pd.DataFrame:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = {**DEFAULT, **sweep_cfg.get("base", {})}
    variants = sweep_cfg.get("variants") or [{"name": "base"}]
    cum = ladder.cumulative(td, horizons=HORIZONS, include_expiry=False)
    dates = market.index
    fit_end = dates[int(len(dates) * (1 - float(sweep_cfg.get("confirm_frac", 0.3)))) - 1]
    rows, ref = [], None
    for v in variants:
        row, lab_s = run_variant({**base, **v}, market, td, pair, cum, gcfg, fit_end)
        if ref is None:
            ref = lab_s
        row["agreement_with_first"] = float((lab_s == ref).mean())
        rows.append(row)
        schema.labels_frame(pair, lab_s.index, lab_s.values).to_parquet(out / f"labels_{row['name']}.parquet")
    df = pd.DataFrame(rows)
    df.attrs["fit_end"] = str(pd.Timestamp(fit_end).date())
    df.to_csv(out / "sweep.csv", index=False)
    (out / "README.md").write_text(
        f"# States sweep\n\nPair {pair}. Finder calibrated on dates <= {df.attrs['fit_end']}; `gate2_fit` is Gate 2's fraction of separated cells "
        f"(h <= {gcfg['phase2']['max_h']}) on entries in that window, `gate2_confirm` on the entries after it, which the finder never saw. "
        f"Threshold {gcfg['phase2']['min_fraction_cells']}. `on_grid_edge` flags a spec at the edge of its own search grid. "
        f"Labels per variant in `labels_<name>.parquet`.\n")
    return df


def load(path: str | None) -> dict:
    return yaml.safe_load(open(path)) if path else {}
