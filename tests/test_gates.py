"""Gate code evaluates correctly: with true labels on synthetic data (where regimes are real by
construction) phases 2 and 3 pass; the gate functions read thresholds from configs/gates.yaml."""
import pandas as pd

from regime_ladder import checks, evaluate, gates, ladder


def test_phase1_gate(trade_days):
    cfg = gates.load_gates("configs/gates.yaml")
    rep = {k: v for k, v in checks.check_trade_days(trade_days).items() if not k.startswith("info_")}
    golden = pd.DataFrame({"trade_id": ["a", "b"], "abs_error": [0.0, 1e-8]})
    assert gates.gate_phase1(rep, golden, cfg)["pass"]
    golden.loc[1, "abs_error"] = 1e-3
    assert not gates.gate_phase1(rep, golden, cfg)["pass"]


def test_phase2_and_3_pass_with_true_labels(cum_true, true_labels):
    cfg = gates.load_gates("configs/gates.yaml")
    lad = ladder.ladder(cum_true, true_labels, ci="hac")
    assert gates.gate_phase2(lad, cfg)["pass"]
    wf = evaluate.walk_forward(cum_true, n_splits=4)
    g3 = gates.gate_phase3(wf, cfg)
    assert g3["pass"], g3


def test_phase3_fails_on_shuffled_labels(cum_true):
    """Labels carrying no information must not pass."""
    cfg = gates.load_gates("configs/gates.yaml")
    shuffled = cum_true.copy()
    shuffled["regime"] = shuffled["regime"].sample(frac=1, random_state=0).values
    wf = evaluate.walk_forward(shuffled, n_splits=4)
    assert not gates.gate_phase3(wf, cfg)["pass"]
