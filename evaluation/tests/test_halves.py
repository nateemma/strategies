"""T014 — the early/late halves split (023 research R13)."""

from __future__ import annotations

from evaluation import halves


def _pooled(rets, trade_profit):
    per_fold, trades = [], []
    for i, r in enumerate(rets):
        fid = f"f{i:02d}"
        per_fold.append(
            {
                "fold_id": fid,
                "n_trades": 30,
                "return_pct": r,
                "test_days": 90,
                "status": "evaluated",
            }
        )
        for j in range(30):
            trades.append(
                {
                    "fold_id": fid,
                    "pair": f"P{j}",
                    "open_date": f"2023-{i:02d}-{j:02d}",
                    "profit_abs": trade_profit(i, j),
                    "scale_pp_per_abs": 0.01,
                }
            )
    return {
        "family": "X",
        "per_fold": per_fold,
        "trades": trades,
        "n_trades_pooled": len(trades),
        "folds_complete": len(per_fold),
        "folds_planned": len(per_fold),
        "folds_missing": [],
        "status": "complete",
    }


def test_membership_and_sums():
    arm = _pooled([1.0] * 8 + [-1.0] * 7, lambda i, j: 2.0)
    ctl = _pooled([0.5] * 15, lambda i, j: 1.0)
    res = halves.split(arm, ctl)
    assert res["early"]["fold_ids"] == [f"f{i:02d}" for i in range(8)]
    assert res["late"]["fold_ids"] == [f"f{i:02d}" for i in range(8, 15)]
    assert res["early"]["arm"]["sum_pp"] == 8.0 and res["early"]["arm"]["positive_folds"] == 8
    assert res["late"]["arm"]["sum_pp"] == -7.0 and res["late"]["arm"]["positive_folds"] == 0
    assert res["late"]["ctl"]["sum_pp"] == 3.5
    assert "PROVISIONAL" in res["label"]


def test_each_half_gets_a_verdict():
    arm = _pooled([1.0] * 15, lambda i, j: 2.0 + (j % 3))
    ctl = _pooled([0.5] * 15, lambda i, j: 1.0)
    res = halves.split(arm, ctl)
    for h in ("early", "late"):
        assert res[h]["verdict"] in (
            "IMPROVED",
            "WORSENED",
            "UNCHANGED",
            "INCONCLUSIVE_UNDERPOWERED",
            "INCONCLUSIVE_ABSORBED",
            "INCONCLUSIVE_ARM_LOSES",
        )
    assert res["early"]["verdict"] == "IMPROVED"


def test_custom_split_point():
    arm = _pooled([1.0] * 15, lambda i, j: 1.0)
    res = halves.split(arm, arm, split_after="f04")
    assert len(res["early"]["fold_ids"]) == 5 and len(res["late"]["fold_ids"]) == 10
    assert res["early"]["verdict"] == "INCONCLUSIVE_ABSORBED"
