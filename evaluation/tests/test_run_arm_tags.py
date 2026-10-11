"""024 T017 (contract C8, F2): family tag accounting from the backtest archive's trades.

The strategy object lives in freqtrade's process, so a counter on it cannot reach the run
record. `run_arm.tag_accounting` reads the trades' `enter_tag` instead.
"""

from __future__ import annotations

import pandas as pd
from regime.run_arm import tag_accounting


def _trades(tags, reasons):
    return pd.DataFrame({"enter_tag": tags, "exit_reason": reasons})


def test_family_counts_overlap_and_untagged():
    t = _trades(
        ["reversion:gbb", "reversion:gbb+ov", "momentum:breakout_gbb", "model_entry ", None],
        ["roi", "unclog_12h", "trailing_stop_loss", "roi", "roi"],
    )
    out = tag_accounting(t)
    assert out["family_trade_counts"] == {"momentum": 1, "reversion": 2}
    assert out["overlap_entries"] == {"reversion": 1}
    assert out["untagged_trades"] == 2
    assert out["family_exit_reason_counts"] == {
        "momentum": {"trailing_stop_loss": 1},
        "reversion": {"roi": 1, "unclog_12h": 1},
    }


def test_no_family_tags_means_no_fields():
    assert tag_accounting(_trades(["model_entry "], ["roi"])) == {}
    assert tag_accounting(pd.DataFrame({"exit_reason": ["roi"]})) == {}
    assert tag_accounting(None) == {}
