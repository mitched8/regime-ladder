import numpy as np
import pandas as pd
import pytest

from regime_ladder import combine, ladder, schema, synth


@pytest.fixture(scope="module")
def legs():
    m = synth.simulate_market(1260, seed=0)
    td = schema.coerce(synth.simulate_trades(m, archetypes=combine.BASE_LEGS, seed=100))
    return m, td, combine.to_vega_units(td)


def test_vega_units_rescale_by_inception_vega(legs):
    _, td, tdv = legs
    t = td["trade_id"].iloc[0]
    v0 = td[(td.trade_id == t) & (td.age == 1)]["vega"].iloc[0]
    assert np.allclose(tdv[tdv.trade_id == t]["pnl"].values, td[td.trade_id == t]["pnl"].values / v0)
    assert tdv.attrs["dropped_trades_no_inception_vega"] == 0


def test_combination_is_exact_aggregation(legs):
    _, _, tdv = legs
    rr = combine.combine(tdv, combine.STANDARD_COMBOS["rr_25d"], "rr_25d")
    piv = tdv[tdv.archetype.isin(["call_25d", "put_25d"])].pivot_table(index=["entry_date", "age"], columns="archetype", values="pnl")
    got = rr.set_index(["entry_date", "age"])["pnl"]
    assert np.allclose(got.values, (piv["call_25d"] - piv["put_25d"]).reindex(got.index).values)
    comp = rr[["pnl_trade", "pnl_delta_hedge", "pnl_vega_hedge"]].sum(axis=1)
    assert np.allclose(comp.values, rr["pnl"].values)


def test_partial_legs_do_not_leak(legs):
    _, _, tdv = legs
    t = tdv[tdv.archetype == "put_25d"]["trade_id"].iloc[0]
    cut = tdv[~((tdv.trade_id == t) & (tdv.age >= 10))]
    rr = combine.combine(cut, combine.STANDARD_COMBOS["rr_25d"], "rr_25d")
    ed = tdv[tdv.trade_id == t]["entry_date"].iloc[0]
    assert rr[(rr.entry_date == ed)]["age"].max() == 9
    with pytest.raises(ValueError):
        combine.combine(cut, {"not_a_leg": 1.0}, "x")


def test_ev_is_linear_in_legs(legs):
    m, _, tdv = legs
    rr = combine.combine(tdv, combine.STANDARD_COMBOS["rr_25d"], "rr_25d")
    lab = schema.labels_frame("EURUSD", m.index, m["state_true"].values)
    lad = ladder.ladder(ladder.attach_entry_labels(ladder.cumulative(pd.concat([tdv, rr])), lab), lab, ci="hac")
    lin = combine.ev_linear(lad, combine.STANDARD_COMBOS["rr_25d"], "x").set_index(["regime", "h"])["mean"]
    direct = lad[lad.archetype == "rr_25d"].set_index(["regime", "h"])["mean"]
    assert np.allclose(direct.values, lin.reindex(direct.index).values)
    sc = combine.screen(lad, {"rr": combine.STANDARD_COMBOS["rr_25d"], "atm": {"straddle_atm": 1.0}}, "crisis", 5)
    assert list(sc.columns) == ["combination", "pair", "tenor_days", "ev_linear", "episodes"] and len(sc) == 2


def test_horizons_are_capped_at_tenor():
    m = synth.simulate_market(600, seed=2)
    td = schema.coerce(synth.simulate_trades(m, archetypes=("straddle_atm",), tenor_days=5, seed=3))
    lab = schema.labels_frame("EURUSD", m.index, m["state_true"].values)
    lad = ladder.ladder(ladder.attach_entry_labels(ladder.cumulative(td, horizons=(1, 3, 5, 10, 20, 40, 60)), lab), lab, ci="hac")
    assert sorted(lad["h"].unique()) == [1, 3, 5]
    assert lad[lad.h == 5]["to_expiry"].all()
