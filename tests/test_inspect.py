"""`inspect` writes every derived quantity, with a README naming the function behind each file."""
import json
from pathlib import Path

import pandas as pd
import yaml

from regime_ladder import gates, inspect as inspect_, synth


def test_inspect_dump_writes_every_component(tmp_path):
    m = synth.simulate_market(1500, seed=0)
    cfg = yaml.safe_load(open("configs/default.yaml")); pcfg = yaml.safe_load(open("configs/profile.yaml")); lcfg = yaml.safe_load(open("configs/leading.yaml"))
    res = inspect_.dump(m, cfg, pcfg, lcfg, gates.load_gates("configs/gates.yaml"), "EURUSD", tmp_path)
    expected = {"features.csv", "finder_report.csv", "finder_spec.json", "composite.csv", "transition_matrix.csv", "transition_counts.csv", "durations.csv",
                "tags.csv", "tag_cells.json", "characteristics.csv", "state_profile.csv", "discovery.json", "shock.csv", "leading_features.csv",
                "leading_screen.csv", "tilt_fits.csv", "pressure.csv", "tilt_pressure.json", "fan_constant.csv", "fan_tilted.csv", "README.md"}
    present = {p.name for p in Path(tmp_path).iterdir()}
    assert expected <= present, expected - present
    readme = (tmp_path / "README.md").read_text()
    assert all(f"`{n}`" in readme for n in res["files"])
    P = pd.read_csv(tmp_path / "transition_matrix.csv", index_col=0)
    assert abs(P.sum(axis=1) - 1).max() < 1e-9 and list(P.index) == list(P.columns)
    comp = pd.read_csv(tmp_path / "composite.csv", index_col=0)
    assert set(comp.columns) == {"composite", "smoothed", "direction_score", "level", "direction", "regime"}
    spec = json.loads((tmp_path / "finder_spec.json").read_text())
    assert len(spec["bounds"]) in (2, 3) and spec["partition"] in ("3", "6", "6x")
