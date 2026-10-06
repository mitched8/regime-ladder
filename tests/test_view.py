"""`view` turns one or more inspect folders into one self-contained HTML page with the data embedded."""
import json
import re
import shutil

import pandas as pd
import pytest
import yaml

from regime_ladder import gates, inspect as inspect_, synth, view


@pytest.fixture(scope="module")
def dumped(tmp_path_factory):
    out = tmp_path_factory.mktemp("inspect") / "EURUSD"
    m = synth.simulate_market(1500, seed=0)
    td = pd.concat([synth.simulate_trades(m.iloc[-700:], tenor_days=t, seed=1 + t) for t in (21, 42)], ignore_index=True)
    cfg, pcfg, lcfg = (yaml.safe_load(open(f"configs/{n}.yaml")) for n in ("default", "profile", "leading"))
    inspect_.dump(m, cfg, pcfg, lcfg, gates.load_gates("configs/gates.yaml"), "EURUSD", out, td=td)
    return out


def _embedded(path):
    html = path.read_text()
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    return html, json.loads(m.group(1).replace("<\\/", "</"))


def test_single_folder_page(dumped, tmp_path):
    p = view.build([dumped], tmp_path / "view.html", label="unit")
    html, data = _embedded(p)
    assert html.startswith("<!doctype html>") and "unit · view" in html
    assert "<script src" not in html and "<link " not in html and "fetch(" not in html and "cdn" not in html.lower()   # no network dependency
    assert data["pairs"] == ["EURUSD"]
    s = data["series"]["EURUSD"]
    assert len(s["dates"]) == 1500 and len(s["smoothed"]) == 1500 and set(s["regime"]) <= {"carry", "settling", "rising", "agitated", "normalising", "stressed", "extreme"}
    assert "stored_energy" in s["_groups"]["leading_features.csv"]
    lad = pd.DataFrame(data["ladder"])
    assert set(lad["tenor_days"]) == {21, 42} and "ALL" in set(lad["regime"]) and "mean_shrunk" in lad.columns
    assert data["tables"]["EURUSD"]["today"]["state"] in set(s["regime"])
    assert data["tables"]["EURUSD"]["fan"]["tilted"] is not None and data["tables"]["EURUSD"]["matrix"]["states"]
    assert "NaN" not in json.dumps(data)                           # NaN became null


def test_parent_folder_with_two_pairs(dumped, tmp_path):
    parent = tmp_path / "inspect"; parent.mkdir()
    shutil.copytree(dumped, parent / "EURUSD")
    shutil.copytree(dumped, parent / "USDJPY")
    meta = json.loads((parent / "USDJPY" / "meta.json").read_text()); meta["pair"] = "USDJPY"
    (parent / "USDJPY" / "meta.json").write_text(json.dumps(meta))
    lad = pd.read_csv(parent / "USDJPY" / "ladder.csv"); lad["pair"] = "USDJPY"; lad.to_csv(parent / "USDJPY" / "ladder.csv", index=False)
    p = view.build([parent], tmp_path / "two.html")
    _, data = _embedded(p)
    assert data["pairs"] == ["EURUSD", "USDJPY"]
    assert set(pd.DataFrame(data["ladder"])["pair"]) == {"EURUSD", "USDJPY"}


def test_sweep_tab_data(dumped, tmp_path):
    from regime_ladder import sweep as sweep_
    m = synth.simulate_market(1500, seed=0)
    td = pd.concat([synth.simulate_trades(m.iloc[-700:], tenor_days=t, seed=1 + t) for t in (21, 42)], ignore_index=True)
    sw = tmp_path / "sweep"
    sweep_.run(m, td, "EURUSD", {"variants": [{"name": "base"}, {"name": "level_only", "states": "3"}]}, gates.load_gates("configs/gates.yaml"), sw)
    p = view.build([dumped], tmp_path / "with_sweep.html", sweep=sw)
    _, data = _embedded(p)
    assert [r["name"] for r in data["sweep"]["rows"]] == ["base", "level_only"]
    assert len(data["sweep"]["strips"]["base"]) == 1500 and data["sweep"]["fit_end"]
    assert "target" in data["series"]["EURUSD"] and "level" in data["series"]["EURUSD"]        # the Finder tab's inputs
    assert data["tables"]["EURUSD"]["finder_report"]


def test_duplicate_pair_is_refused(dumped, tmp_path):
    with pytest.raises(SystemExit, match="appears twice"):
        view.build([dumped, dumped], tmp_path / "dup.html")


def test_no_inspect_output_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no inspect output"):
        view.build([tmp_path], tmp_path / "none.html")
