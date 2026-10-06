"""`pack` turns an inspect folder (or the trade table) into one self-describing packet per stage."""
import pandas as pd
import pytest
import yaml

from regime_ladder import gates, inspect as inspect_, pack, synth


@pytest.fixture(scope="module")
def dumped(tmp_path_factory):
    out = tmp_path_factory.mktemp("inspect")
    m = synth.simulate_market(1500, seed=0); td = synth.simulate_trades(m.iloc[-700:], seed=1)
    cfg, pcfg, lcfg = (yaml.safe_load(open(f"configs/{n}.yaml")) for n in ("default", "profile", "leading"))
    inspect_.dump(m, cfg, pcfg, lcfg, gates.load_gates("configs/gates.yaml"), "EURUSD", out, td=td)
    return out, m, td


@pytest.mark.parametrize("stage,sections", [
    ("states", ["## 1. Chosen spec", "## 2b. What the finder saw", "## 2c. Mean target by named state", "## 3. States", "## 5.", "## 6. Transition matrix", "Features warm up until"]),
    ("ladder", ["## 1. Gates", "Gate 3:", "## 2. Ladder at h = 1, 5, 10", "## 5. Walk-forward"]),
    ("leading", ["## 1. Gate 4", "## 2. Screen", "## 6. Lead profile", "## 7. Every move into the high band", "## 8b. Alignment check"]),
])
def test_stage_packets_have_their_sections(dumped, tmp_path, stage, sections):
    src, m, _ = dumped
    p = pack.build(stage, tmp_path / stage, src=src, market=m)
    text = p.read_text()
    assert "## Definitions" in text and "| git commit |" in text and "Costs are excluded" in text
    for s in sections:
        assert s in text, s
    assert "| nan |" not in text.lower()                         # missing values render as blanks
    assert len(text) < 60_000                                     # small enough to paste


def test_data_packet_from_the_trade_table(dumped, tmp_path):
    _, m, td = dumped
    text = pack.build("data", tmp_path, td=td, market=m, archetypes_cfg=yaml.safe_load(open("configs/archetypes.yaml"))).read_text()
    for s in ("[PASS] components_sum_to_pnl", "## 2. Coverage", "## 5. Twenty largest market days", "## 7. Vega at age 1"):
        assert s in text, s


def test_runs_and_escalations():
    idx = pd.bdate_range("2024-01-01", periods=8)
    lab = pd.Series(["carry", "carry", "rising", "stressed", "stressed", "agitated", "stressed", "carry"], index=idx)
    r = pack.runs(lab)
    assert list(r["state"]) == ["carry", "rising", "stressed", "agitated", "stressed", "carry"] and r["days"].sum() == 8
    e = pack.escalations(lab)
    assert list(e["date"]) == [idx[3], idx[6]] and list(e["from"]) == ["rising", "agitated"]


def test_alignment_flags_a_future_value():
    m = synth.simulate_market(800, seed=0)
    L = pd.DataFrame({"honest": m["rv_1w"].rolling(5).mean(), "late": m["rv_1w"].shift(-10)})
    a = pack.alignment(L, m)
    assert a.loc["late", "k=+10"] > 0.99 and a.loc["late", "k=+10"] > a.loc["late", "k=+0"] + 0.3
    assert a.loc["honest", "k=+10"] < a.loc["honest", "k=+0"]


def test_sweep_packet(dumped, tmp_path):
    from regime_ladder import sweep as sweep_
    _, m, td = dumped
    sw = tmp_path / "sweep"
    sweep_.run(m, td, "EURUSD", {"variants": [{"name": "base"}, {"name": "level_only", "states": "3"}]}, gates.load_gates("configs/gates.yaml"), sw)
    text = pack.build("sweep", tmp_path / "sweep_packet", src=sw).read_text()
    for s in ("## 1. Variants", "gate2_confirm", "## 2. Share of days per state by year", "## 3. How far each variant", "level_only"):
        assert s in text, s
