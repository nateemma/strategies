"""031 T018: a trade with a position add (two entry fills) pools, pairs and risk-scores like any
other -- total stake, its own profit_abs, keyed by its ORIGINAL open date."""

from __future__ import annotations

import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[1]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from evaluation import pool as P  # noqa: E402
from evaluation import risk  # noqa: E402
from evaluation.compare_pooled import KEY, compare_data  # noqa: E402


def _order(entry, ts, amount=0.0, price=0.0):
    return {"ft_is_entry": entry, "order_filled_timestamp": ts, "amount": amount,
            "safe_price": price}


def test_entry_fields():
    one = {"orders": [_order(True, 1, 90, 10.0), _order(False, 5)]}
    two = {"orders": [_order(True, 1, 90, 10.0), _order(True, 3, 100, 9.0), _order(False, 5)]}
    assert P._entry_fields(one) == {"n_entries": 1, "entry_fills": None}
    assert P._entry_fields(two) == {"n_entries": 2, "entry_fills": [(1, 900.0), (3, 900.0)]}
    assert P._entry_fields({}) == {"n_entries": None, "entry_fills": None}


def _pooled(profits, stakes=None, n_entries=None):
    trades = []
    for i, p in enumerate(profits):
        trades.append({"fold_id": "f00", "pair": "SOL/USDT",
                       "open_date": f"2023-01-{i + 1:02d} 00:00:00+00:00",
                       "close_date": f"2023-01-{i + 1:02d} 06:00:00+00:00",
                       "profit_abs": p, "scale_pp_per_abs": 0.01,
                       "stake_amount": (stakes or {}).get(i, 900.0),
                       "n_entries": (n_entries or {}).get(i, 1)})
    return {"run_id": "x", "family": "NNNC", "folds_complete": 1, "status": "complete",
            "n_trades_pooled": len(trades), "trades": trades,
            "per_fold": [{"fold_id": "f00", "n_trades": len(trades),
                          "return_pct": sum(profits) * 0.01, "test_days": 90,
                          "status": "evaluated"}]}


def test_a_dca_trade_pairs_with_its_control_trade_on_the_original_open_date():
    assert KEY == ("fold_id", "pair", "open_date")
    ctl = _pooled([5.0] * 30)
    arm = _pooled([5.0] * 29 + [12.0], stakes={29: 1800.0}, n_entries={29: 2})
    out = compare_data(arm, ctl)
    # exactly one pair differs (the DCA trade); it is matched, not counted as arm-only
    assert out["bootstrap"]["n_diff"] == 1
    assert arm["trades"][29]["stake_amount"] == 1800.0


def test_risk_compare_runs_on_multifill_trades():
    ctl = _pooled([5.0, -3.0, 4.0] * 10)
    arm = _pooled([5.0, -3.0, 9.0] * 10, stakes={2: 1800.0}, n_entries={2: 2})
    rk = risk.compare(arm["trades"], ctl["trades"])
    assert rk["verdict_sharpe"] in ("IMPROVED", "UNCHANGED", "WORSENED", "UNRESOLVABLE")
