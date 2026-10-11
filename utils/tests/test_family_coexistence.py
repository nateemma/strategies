"""025 T002: FamilyCoexistenceMixin, rule A, a per-family pair cap (contract C1).

The cap only refuses NEW entries of a family that already holds `FAMILY_PAIR_CAP` open trades. The
parent's checks (the volume headroom) run first and unchanged. Refusals are logged to the
entry-condition sink so they reach the run record (feature 023 C4 path).
"""

from __future__ import annotations

import json
import types

import pytest
from utils import FamilyCoexistence as FC
from utils.FamilyCoexistence import FamilyCoexistenceError, FamilyCoexistenceMixin


class _Parent:
    PARENT_OK = True

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time,
                            entry_tag, side, **kwargs):
        return self.PARENT_OK


def _strategy(cap, parent_ok=True):
    cls = type("S", (FamilyCoexistenceMixin, _Parent), {"FAMILY_PAIR_CAP": cap})
    s = cls()
    s.PARENT_OK = parent_ok
    return s


def _open(*tags):
    return [types.SimpleNamespace(enter_tag=t, pair=f"P{i}/USDT") for i, t in enumerate(tags)]


def _call(s, tag):
    return s.confirm_trade_entry("X/USDT", "limit", 1.0, 1.0, "gtc", None, tag, "long")


@pytest.fixture
def opened(monkeypatch):
    state = {"trades": [], "calls": 0}

    def fake():
        state["calls"] += 1
        return state["trades"]

    monkeypatch.setattr(FC, "_open_trades", fake)
    return state


def test_no_cap_returns_parent_result_without_reading_trades(opened):
    assert _call(_strategy(None), "reversion:gbb") is True
    assert _call(_strategy(None, parent_ok=False), "reversion:gbb") is False
    assert opened["calls"] == 0


def test_parent_refusal_wins_and_trades_are_not_read(opened):
    assert _call(_strategy(2, parent_ok=False), "reversion:gbb") is False
    assert opened["calls"] == 0


def test_cap_is_per_family_and_logs_refusals(opened, monkeypatch, tmp_path):
    monkeypatch.setenv("FT_ENTRY_CONDITION_SINK", str(tmp_path))
    opened["trades"] = _open("reversion:gbb", "reversion:gbb+ov")
    s = _strategy(2)
    assert _call(s, "reversion:gbb") is False                  # reversion holds 2 = cap
    assert _call(s, "momentum:breakout_gbb") is True           # momentum holds 0
    lines = (tmp_path / "coexist_events.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in lines] == [{"rule": "pair_cap", "family": "reversion"}]


def test_overlap_suffix_counts_as_its_family(opened):
    opened["trades"] = _open("momentum:breakout_gbb+ov")
    assert _call(_strategy(1), "momentum:breakout_gbb") is False


def test_untagged_entry_is_not_capped(opened):
    opened["trades"] = _open("reversion:gbb", "reversion:gbb")
    assert _call(_strategy(1), "model_entry ") is True
    assert _call(_strategy(1), None) is True


def test_no_sink_env_writes_nothing(opened, monkeypatch, tmp_path):
    monkeypatch.delenv("FT_ENTRY_CONDITION_SINK", raising=False)
    monkeypatch.chdir(tmp_path)
    opened["trades"] = _open("reversion:gbb")
    assert _call(_strategy(1), "reversion:gbb") is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [0, 10, -1, 2.5, "3"])
def test_out_of_range_cap_is_refused_at_class_creation(bad):
    with pytest.raises(FamilyCoexistenceError, match="FAMILY_PAIR_CAP"):
        type("Bad", (FamilyCoexistenceMixin, _Parent), {"FAMILY_PAIR_CAP": bad})
