"""The explorer's tables: per-entry normalised earn, daily statistics, and a saved filter as a tag."""
import numpy as np
import pandas as pd
import yaml

from regime_ladder import explore, features, gates, inspect as inspect_, synth


def _data():
    m = synth.simulate_market(900, seed=0)
    td = pd.concat([synth.simulate_trades(m.iloc[-500:], tenor_days=t, seed=1 + t) for t in (21, 42)], ignore_index=True)
    return m, td


def test_entries_are_normalised_per_vega_and_sum_components():
    m, td = _data()
    e = explore.entries(td)
    assert e.attrs["normalised_by"] == "vega" and set(e.attrs["components"]) == {"earn_trade", "earn_delta_hedge", "earn_vega_hedge"}
    assert set(e["h"].unique()) >= {1, 3, 5, 10, 20} and e["to_expiry"].any()
    # components sum to the total after normalisation too
    assert np.allclose(e["earn"], e[["earn_trade", "earn_delta_hedge", "earn_vega_hedge"]].sum(axis=1), atol=1e-6)
    # one straddle at h=5: cum pnl / vega at inception
    row = e[(e["archetype"] == "straddle_atm") & (e["tenor_days"] == 21) & (e["h"] == 5)].iloc[0]
    t = td[(td["archetype"] == "straddle_atm") & (td["tenor_days"] == 21) & (td["entry_date"] == row["entry_date"])].sort_values("age")
    assert abs(row["earn"] - t["pnl"].iloc[:5].sum() / abs(t["vega"].iloc[0])) < 1e-6
    n = explore.entries(td, normalise="notional")
    assert n.attrs["normalised_by"] == "notional" and abs(n[(n["archetype"] == "straddle_atm") & (n["tenor_days"] == 21) & (n["h"] == 5)].iloc[0]["earn"] - t["pnl"].iloc[:5].sum()) < 1e-6


def test_statistics_have_definitions_and_are_point_in_time():
    m, td = _data()
    e = explore.entries(td)
    st, meta = explore.statistics(m, features.build(m), None, None, None, None, e)
    assert set(st.columns) == set(meta["defs"]) == set(meta["groups"])
    assert {"atm_1m_z252", "rv_iv_1m", "spot_vs_ma60", "earn20_straddle_atm_z", "vol_pct"} <= set(st.columns)
    # truncation: the value on a date does not change when later data is removed
    cut = m.index[600]
    st2, _ = explore.statistics(m.loc[:cut], features.build(m.loc[:cut]), None, None, None, None, e[e["entry_date"] <= cut])
    for c in ("atm_1m_z252", "spot_vs_ma60", "rv_iv_1m"):
        a, b = st[c].loc[:cut].dropna(), st2[c].dropna()
        common = a.index.intersection(b.index)
        assert np.allclose(a.loc[common], b.loc[common], atol=1e-9), c
    # the strategy's own earn is known only at entry + h: the last 5 days of the window carry no new entry information
    assert st["earn20_straddle_atm_z"].notna().sum() > 100


def test_filters_become_tags_and_inspect_writes_them(tmp_path):
    m, td = _data()
    e = explore.entries(td)
    st, _ = explore.statistics(m, features.build(m), None, None, None, None, e)
    spec = {"hi_vol": {"stat": "atm_1m_z252", "op": ">", "value": 1.0}, "cheap": {"stat": "rv_iv_1m", "op": "<", "value": 0.8, "and": [{"stat": "vol_pct", "op": "between", "value": [0, 50]}]}}
    T = explore.filters_to_tags(st, spec)
    assert set(T.columns) == {"hi_vol", "cheap"} and set(T["hi_vol"].unique()) <= {"on", "none"}
    assert (T["hi_vol"] == "on").sum() == (st["atm_1m_z252"] > 1.0).sum()
    assert ((T["cheap"] == "on") <= ((st["rv_iv_1m"] < 0.8) & (st["vol_pct"] <= 50))).all()
    cfg, pcfg, lcfg = (yaml.safe_load(open(f"configs/{n}.yaml")) for n in ("default", "profile", "leading"))
    cfg = {**cfg, "filters": {"hi_vol": spec["hi_vol"]}}
    inspect_.dump(m, cfg, pcfg, lcfg, gates.load_gates("configs/gates.yaml"), "EURUSD", tmp_path, td=td)
    assert (tmp_path / "entries.csv").exists() and (tmp_path / "statistics.csv").exists() and (tmp_path / "filter_tags.csv").exists()
    import json
    cells = json.loads((tmp_path / "tag_cells.json").read_text())
    assert "hi_vol" in cells
    yml = explore.filter_yaml("hi_vol", spec["hi_vol"])
    assert yaml.safe_load("filters:\n" + yml)["filters"]["hi_vol"]["op"] == ">"


def test_extra_statistics_three_ways():
    m, td = _data()
    st, meta = explore.statistics(m, cfg={"passthrough": {"vix": "VIX level", "not_a_column": "ignored"}})
    assert {"atm_1m_pct252", "atm_1m_minus_rv_1m", "spot_range20_z"} <= set(st.columns)          # explore.EXTRA
    assert "vix" in st.columns and meta["groups"]["vix"] == "market columns" and "not_a_column" not in st.columns   # config passthrough
    assert (st["atm_1m_pct252"].dropna().between(0, 100)).all()
    # EXTRA entries are trailing: truncation leaves earlier values unchanged
    cut = m.index[500]
    st2, _ = explore.statistics(m.loc[:cut])
    for c in ("atm_1m_pct252", "atm_1m_minus_rv_1m", "spot_range20_z"):
        common = st[c].loc[:cut].dropna().index.intersection(st2[c].dropna().index)
        assert np.allclose(st[c].loc[common], st2[c].loc[common], atol=1e-9), c
