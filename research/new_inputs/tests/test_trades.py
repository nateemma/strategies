"""035 T013a: the early trade set reproduces 034's instrument and contains early folds only."""

from __future__ import annotations

import pandas as pd
import pytest
from research.new_inputs import trades as TR


def test_check_refuses_late_folds_and_a_drifted_instrument():
    late = pd.DataFrame({"fold": ["f00", "f08"], "pp": [1.0, 0.0],
                         "agree_ctl_share": [True, False]})
    with pytest.raises(RuntimeError, match="early half only"):
        TR.check(late)
    small = pd.DataFrame({"fold": ["f00", "f01"], "pp": [1.0, 0.0],
                          "agree_ctl_share": [True, False]})
    with pytest.raises(RuntimeError, match="does not reproduce"):
        TR.check(small)


def test_real_trade_set_reproduces():
    t = TR.trades()
    assert t.attrs["check"]["trades"] == 412
    assert set(t["fold"]) <= set(TR.EARLY)
    assert t["bb_width"].notna().all() and t["p_ctl"].notna().all()
