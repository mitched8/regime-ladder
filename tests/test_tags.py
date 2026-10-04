import numpy as np
import pandas as pd
import pytest

from regime_ladder import features, ladder, schema, synth, tags


@pytest.fixture(scope="module")
def world():
    m = synth.simulate_market(2520, seed=0)
    return m, m["state_true"].rename("regime")


def test_corr_sign_recovers_the_hidden_agitated_subtypes(world):
    m, lab = world
    t = tags.corr_sign_tag(m)
    ag = m["state_true"] == "agitated"
    ct = pd.crosstab(m.loc[ag, "subtype_true"], t[ag])
    assert ct.loc["corr_neg", "neg"] > 2 * ct.loc["corr_neg", "pos"]
    assert ct.loc["corr_pos", "pos"] > 2 * ct.loc["corr_pos", "neg"]
    assert t.iloc[0] == "flat"  # warm-up is neutral, never a sign


def test_event_window_marks_the_days_around_each_event():
    idx = pd.bdate_range("2024-01-01", periods=20)
    t = tags.event_window_tag(idx, ["2024-01-10", "2024-01-13"], before=2, after=1)  # 13th is a Saturday -> rolls to Mon 15th
    assert list(t.loc["2024-01-08":"2024-01-11"]) == ["none", "pre", "pre", "post"]
    assert list(t.loc["2024-01-12":"2024-01-16"]) == ["pre", "pre", "post"]
    assert (t == "none").sum() == 20 - 6


def test_pinned_fires_only_in_a_quiet_range():
    idx = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(0)
    quiet = 1.0 + 0.0005 * np.sin(np.arange(60))              # tiny oscillation
    busy = quiet[-1] * np.exp(np.cumsum(rng.normal(0, 0.008, 60)))
    spot = np.concatenate([quiet, busy])
    m = pd.DataFrame({"spot": spot, "atm_1m": 8.0}, index=idx)
    r = pd.Series(np.log(m["spot"]).diff(), index=idx)
    m["rv_1m"] = (r.rolling(21).std() * np.sqrt(252) * 100).bfill()
    t = tags.pinned_tag(m)
    assert (t.iloc[25:60] == "pinned").mean() > 0.9 and (t.iloc[90:] == "pinned").mean() < 0.2


def test_intervention_risk_by_move_level_and_flag():
    idx = pd.bdate_range("2024-01-01", periods=60)
    spot = np.concatenate([np.full(30, 150.0), np.linspace(150, 162, 30)])  # +8% in the second month, ~5.7% over the last 21 days
    m = pd.DataFrame({"spot": spot}, index=idx)
    t = tags.intervention_risk_tag(m, direction=1, move_window=21, move_threshold=0.05)
    assert t.iloc[-1] == "risk" and t.iloc[:30].eq("none").all()
    t2 = tags.intervention_risk_tag(m, direction=1, level=155.0, move_threshold=9.0)
    assert (t2 == "risk").sum() == (spot > 155).sum()
    m["official_comment"] = False; m.loc[idx[5], "official_comment"] = True
    t3 = tags.intervention_risk_tag(m, direction=-1, move_threshold=9.0, flag_col="official_comment")
    assert t3.iloc[5] == "risk" and (t3 == "risk").sum() == 1


@pytest.mark.parametrize("name", list(tags.TAGS))
def test_tags_are_point_in_time(world, name):
    m, _ = world
    assert features.truncation_agrees(tags.TAGS[name], m, "2019-03-15"), f"{name} uses future data"


def test_tag_split_keeps_only_cells_with_enough_episodes(world):
    m, lab = world
    t = tags.corr_sign_tag(m)
    reg, info = tags.tag_split(lab, t, min_episodes=5)
    assert {tags.parent(r) for r in reg.unique()} <= set(lab.unique())
    assert all(v["episodes"] >= 5 for k, v in info.items() if v["kept"])
    assert any(k.startswith("agitated·") and v["kept"] for k, v in info.items())
    reg_strict, info_strict = tags.tag_split(lab, t, min_episodes=10_000)
    assert (reg_strict == lab).all() and not any(v["kept"] for v in info_strict.values())
    # neutral values never split
    assert not any(r.endswith("·flat") for r in reg.unique())


def test_split_frame_feeds_the_ladder(world):
    m, lab = world
    m5 = m.iloc[-760:]
    td = synth.simulate_trades(m5, archetypes=("straddle_atm",), seed=3)
    frame = schema.labels_frame("EURUSD", m5.index, m5["state_true"].values)
    split, info = tags.split_labels_frame(frame, tags.corr_sign_tag(m), min_episodes=3)
    cum = ladder.attach_entry_labels(ladder.cumulative(td, horizons=(5,)), split)
    lad = ladder.ladder(cum, split, ci="hac")
    assert any("·" in r for r in lad["regime"].unique())
    assert (lad[lad["regime"] != "ALL"]["episodes"] >= 1).all()


def test_build_applies_config_and_per_pair_overrides(world):
    m, _ = world
    cfg = {"corr_sign": {"window": 21}, "pinned": {}, "events": ["2020-03-18"], "event_window": {"before": 2, "after": 1},
           "pairs": {"USDJPY": {"intervention_risk": {"direction": 1, "move_threshold": 0.05}}}}
    a = tags.build(m, cfg, "EURUSD"); b = tags.build(m, cfg, "USDJPY")
    assert set(a.columns) == {"corr_sign", "pinned", "event_window"} and "intervention_risk" in b.columns
    assert (a["event_window"] == "pre").sum() == 2
