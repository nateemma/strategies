"""034 T007: the S0a audit's pure parts -- shares, liveness verdicts, the floor rule, overlap."""

from __future__ import annotations

import numpy as np

from Framework.BaseStrategy import TradingAction
from research.label_screen import audit as A


B, S, H = int(TradingAction.BUY), int(TradingAction.SELL), int(TradingAction.HOLD)


def test_shares_use_the_framework_codes_and_sum_to_one():
    r = A.shares(np.array([B, B, S, H, H, H, H, H, H, H]))
    assert (r["buy"], r["sell"]) == (0.2, 0.1)
    assert abs(r["buy"] + r["sell"] + r["hold"] - 1) < 1e-12 and r["rows"] == 10


def test_liveness_verdicts():
    assert A.liveness_status(0.04, 0.02, 15) == "live"
    assert A.liveness_status(0.02, 0.02, 0) == "equal_default"
    assert A.liveness_status(0.04, 0.02, 0) == "no_op"


def test_labeler_defaults_including_the_inline_forward_mae_sell():
    assert A.labeler_default(1, "sl", "sell") == 0.02          # labels_triple_barrier_sell
    assert A.labeler_default(1, "sl", "buy") == 0.02           # labels_triple_barrier
    assert A.labeler_default(0, "max_drawdown", "sell") == 0.02
    assert A.labeler_default(4, "max_drawdown", "sell") == 0.1


def _row(buy, live=None):
    return {"kept": {"buy": buy}, "liveness": live}


def test_floor_and_no_op_rules():
    assert A.status({"f00": _row(0.05), "f01": _row(0.02)}) == ("PASS", "")
    st, why = A.status({"f00": _row(0.05), "f01": _row(0.005)})
    assert st == "INCONCLUSIVE" and "f01" in why
    st, why = A.status({"f00": _row(0.05, {"buy.pt": {"status": "no_op"}})})
    assert st == "INCONCLUSIVE" and "buy.pt" in why
    assert A.status({"f00": _row(0.05, {"sell.sl": {"status": "equal_default"}})})[0] == "PASS"


def test_mfe_mae_ratio_sell_is_no_longer_the_buy_rule():
    """Before 034 mfe_mae_ratio had no sell variant and its raw SELL labels were its BUY labels
    (the audit recorded full overlap); the mirrored variant leaves none."""
    from Framework import TrainingSignals as T

    rng = np.random.default_rng(1)
    import pandas as pd

    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, 2000)))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0})
    p = {"horizon": 48, "min_gain": 0.02, "min_loss": 0.02}
    b = T.get_train_buy_signals(df, method=3, params=p).to_numpy() > 0.5
    s = T.get_train_sell_signals(df, method=3, params=p).to_numpy() > 0.5
    assert b.any() and s.any() and not (b & s).any()


def test_spearman_handles_constant_labels():
    assert A.spearman(np.zeros(10), np.arange(10.0)) is None
    assert A.spearman(np.r_[np.zeros(5), np.ones(5)], np.arange(10.0)) > 0.8
