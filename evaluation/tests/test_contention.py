"""024 T023 (contract C6, data-model ContentionRecord): what combining two families cost each.

A "lost" entry is in a family's single-family arm but not in the combined arm. Its cause is read
from the combined arm's trades open at that instant.
"""

from __future__ import annotations

import json

import pytest
from evaluation import contention as C


def _t(fold, pair, open_, close, tag, profit=0.01):
    return {
        "fold_id": fold,
        "pair": pair,
        "open_date": open_,
        "close_date": close,
        "profit_ratio": profit,
        "enter_tag": tag,
    }


def _pooled(trades):
    return {"trades": trades}


def test_causes_are_classified():
    rev = _pooled(
        [
            _t(
                "f00",
                "A/USDT",
                "2023-01-01 10:00:00+00:00",
                "2023-01-01 12:00:00+00:00",
                "reversion:gbb",
            ),
            _t(
                "f00",
                "B/USDT",
                "2023-01-01 10:00:00+00:00",
                "2023-01-01 11:00:00+00:00",
                "reversion:gbb",
            ),
            _t(
                "f00",
                "C/USDT",
                "2023-01-02 10:00:00+00:00",
                "2023-01-02 11:00:00+00:00",
                "reversion:gbb",
            ),
            _t(
                "f09",
                "D/USDT",
                "2025-01-02 10:00:00+00:00",
                "2025-01-02 11:00:00+00:00",
                "reversion:gbb",
            ),
        ]
    )
    combo_trades = [
        # A: held by a momentum trade -> pair_held_by_other_family
        _t(
            "f00",
            "A/USDT",
            "2023-01-01 09:00:00+00:00",
            "2023-01-01 13:00:00+00:00",
            "momentum:breakout_gbb",
        ),
        # B: held by another reversion trade -> pair_held_same_family
        _t(
            "f00",
            "B/USDT",
            "2023-01-01 09:30:00+00:00",
            "2023-01-01 10:30:00+00:00",
            "reversion:gbb",
        ),
    ]
    # C: 2 open trades at the time on other pairs, max_open_trades=2 -> slots_full
    combo_trades += [
        _t(
            "f00",
            f"X{i}/USDT",
            "2023-01-02 09:00:00+00:00",
            "2023-01-02 12:00:00+00:00",
            "momentum:breakout_gbb",
        )
        for i in range(2)
    ]
    # D: nothing open -> divergence
    out = C.contention(_pooled(combo_trades), {"reversion": rev}, max_open_trades=2)
    early, late = out["reversion"]["early"], out["reversion"]["late"]
    assert early["lost"] == 3 and late["lost"] == 1
    assert early["lost_by_cause"] == {
        "pair_held_by_other_family": 1,
        "pair_held_same_family": 1,
        "slots_full": 1,
        "divergence": 0,
    }
    assert late["lost_by_cause"]["divergence"] == 1


def test_gained_overlap_and_counts():
    single = _pooled(
        [
            _t(
                "f00",
                "A/USDT",
                "2023-01-01 10:00:00+00:00",
                "2023-01-01 11:00:00+00:00",
                "reversion:gbb",
            )
        ]
    )
    combo = _pooled(
        [
            _t(
                "f00",
                "A/USDT",
                "2023-01-01 10:00:00+00:00",
                "2023-01-01 11:00:00+00:00",
                "reversion:gbb",
            ),
            _t(
                "f00",
                "E/USDT",
                "2023-01-01 10:00:00+00:00",
                "2023-01-01 11:00:00+00:00",
                "reversion:gbb+ov",
                0.02,
            ),
            _t(
                "f08",
                "F/USDT",
                "2024-12-25 10:00:00+00:00",
                "2024-12-25 11:00:00+00:00",
                "momentum:breakout_gbb",
            ),
        ]
    )
    out = C.contention(combo, {"reversion": single}, max_open_trades=10)
    e = out["reversion"]["early"]
    assert e["single_trades"] == 1 and e["combined_trades"] == 2
    assert e["lost"] == 0 and e["gained"] == 1
    assert e["gained_pnl_pct"] == pytest.approx(2.0)
    assert e["overlap_entries"] == 1
    assert out["reversion"]["late"]["combined_trades"] == 0  # the f08 trade is momentum


def test_missing_input_exits_2(tmp_path):
    rc = C.main(
        [
            "--combined",
            str(tmp_path / "nope.json"),
            "--single",
            f"reversion={tmp_path / 'nope2.json'}",
            "--config",
            str(tmp_path / "c.json"),
            "--out",
            str(tmp_path / "o.json"),
        ]
    )
    assert rc == 2


def test_cli_writes_json_and_md(tmp_path):
    combo = tmp_path / "combo.json"
    single = tmp_path / "rev.json"
    cfg = tmp_path / "cfg.json"
    combo.write_text(json.dumps(_pooled([])))
    single.write_text(json.dumps(_pooled([])))
    cfg.write_text(json.dumps({"max_open_trades": 10}))
    out = tmp_path / "c.json"
    assert (
        C.main(
            [
                "--combined",
                str(combo),
                "--single",
                f"reversion={single}",
                "--config",
                str(cfg),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["reversion"]["early"]["lost"] == 0
    assert out.with_suffix(".md").read_text().startswith("#")
