"""`sweep` scores labeller variants on one frozen trade table, with Gate 2 on a window the finder never saw."""
import pandas as pd

from regime_ladder import gates, sweep, synth


def test_sweep_table_and_labels(tmp_path):
    m = synth.simulate_market(2000, seed=0); td = synth.simulate_trades(m.iloc[-1000:], seed=1)
    cfg = {"confirm_frac": 0.3, "variants": [{"name": "base"}, {"name": "level_only", "states": "3"}, {"name": "rv", "target": "forward_rv"}]}
    df = sweep.run(m, td, "EURUSD", cfg, gates.load_gates("configs/gates.yaml"), tmp_path)
    assert list(df["name"]) == ["base", "level_only", "rv"]
    assert (df["gate2_fit"].between(0, 1) | df["gate2_fit"].isna()).all() and (df["gate2_confirm"].between(0, 1) | df["gate2_confirm"].isna()).all()
    assert df.loc[df["name"] == "level_only", "partition"].item() == "3" and not df.loc[0, "fell_back"]
    assert df.loc[0, "agreement_with_first"] == 1.0 and df.loc[1, "agreement_with_first"] < 1.0
    for n in df["name"]:
        lab = pd.read_parquet(tmp_path / f"labels_{n}.parquet"); assert len(lab) == 2000 and set(lab.columns) == {"pair", "date", "regime"}
    assert (tmp_path / "sweep.csv").exists() and "gate2_confirm" in (tmp_path / "README.md").read_text()
