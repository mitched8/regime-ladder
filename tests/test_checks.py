import pandas as pd
import pytest

from regime_ladder import checks


def _ok(report):
    return {k: v for k, v in report.items() if not k.startswith("info_")}


def test_clean_table_passes(trade_days, true_labels):
    rep = checks.check_trade_days(trade_days)
    rep.update(checks.check_label_coverage(trade_days, true_labels))
    checks.assert_clean(_ok(rep))


def test_duplicate_row_is_caught(trade_days):
    td = pd.concat([trade_days, trade_days.iloc[[10]]])
    assert not checks.check_trade_days(td)["no_duplicate_trade_date"][0]


def test_missing_day_is_caught(trade_days):
    td = trade_days.drop(trade_days[(trade_days["trade_id"] == trade_days["trade_id"].iloc[0]) & (trade_days["age"] == 5)].index)
    assert not checks.check_trade_days(td)["ages_consecutive_from_1"][0]


def test_component_mismatch_is_caught(trade_days):
    td = trade_days.copy()
    td.loc[td.index[3], "pnl_trade"] += 0.01
    assert not checks.check_trade_days(td)["components_sum_to_pnl"][0]


def test_date_before_entry_is_caught(trade_days):
    td = trade_days.copy()
    td.loc[td.index[0], "date"] = td.loc[td.index[0], "entry_date"]
    assert not checks.check_trade_days(td)["date_after_entry"][0]


def test_missing_label_is_caught(trade_days, true_labels):
    lab = true_labels.iloc[50:]
    rep = checks.check_label_coverage(trade_days, lab)
    assert not rep["label_coverage"][0]
    with pytest.raises(AssertionError):
        checks.assert_clean(rep)
