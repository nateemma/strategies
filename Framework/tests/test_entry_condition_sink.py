"""T007 — the entry-condition sink is read-only and counts correctly (023 contract C4).

`populate_entry_trend` ANDs a list of masks. The sink names each mask and writes its pass count
when FT_ENTRY_CONDITION_SINK is set. It must never change `enter_long`.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "utils"), str(ROOT / "NNNC")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Framework.BaseStrategy import BaseStrategy  # noqa: E402


N = 400


class _Stub(BaseStrategy):
    MASK = None

    def get_entry_conditions(self, dataframe):
        return pd.Series(self.MASK, index=dataframe.index)


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "volume": np.where(rng.random(N) < 0.3, 0.0, rng.random(N) * 100),
            "rvol": rng.random(N) * 3,
            "atr_pct_roll": rng.random(N) * 0.05,
            "guard_metric": rng.uniform(-1, 1, N),
            "close_norm": rng.uniform(-1, 1, N),
            "adx": rng.random(N) * 60,
            "bb_width": rng.random(N) * 0.2,
        }
    )


def _strategy(mask, guards: bool, thresholds: dict):
    s = object.__new__(_Stub)
    s.MASK = mask
    s.first_run = False
    s._gate_inertia_checked = True
    s.dp = types.SimpleNamespace(runmode=types.SimpleNamespace(value="live"))
    s.entry_enable_guards = types.SimpleNamespace(value=guards)
    for k, v in thresholds.items():
        setattr(s, k, types.SimpleNamespace(value=v))
    return s


THRESH = {
    "entry_rvol_threshold": 1.0,
    "entry_atr_pct": 0.01,
    "entry_guard_threshold": 0.3,
    "entry_close_norm_threshold": 0.5,
    "entry_adx_threshold": 20.0,
    "entry_bb_width_threshold": 0.05,
}


@pytest.mark.parametrize("guards", [True, False])
def test_enter_long_identical_with_and_without_sink(tmp_path, monkeypatch, guards):
    mask = np.random.default_rng(1).random(N) < 0.5
    monkeypatch.delenv("FT_ENTRY_CONDITION_SINK", raising=False)
    off = _strategy(mask, guards, THRESH).populate_entry_trend(_frame(), {"pair": "ABC/USDT"})
    monkeypatch.setenv("FT_ENTRY_CONDITION_SINK", str(tmp_path))
    on = _strategy(mask, guards, THRESH).populate_entry_trend(_frame(), {"pair": "ABC/USDT"})
    pd.testing.assert_series_equal(off["enter_long"].fillna(0), on["enter_long"].fillna(0))
    assert list(tmp_path.glob("*.json")), "sink wrote nothing"


def test_counts_are_correct(tmp_path, monkeypatch):
    mask = np.random.default_rng(2).random(N) < 0.4
    df = _frame(3)
    monkeypatch.setenv("FT_ENTRY_CONDITION_SINK", str(tmp_path))
    out = _strategy(mask, True, THRESH).populate_entry_trend(df.copy(), {"pair": "ABC/USDT"})
    rec = json.loads((tmp_path / "_Stub_ABC_USDT.json").read_text())["counts"]

    expected = {
        "signal": mask,
        "volume": (df["volume"] > 0).to_numpy(),
        "rvol": (df["rvol"] > 1.0).to_numpy(),
        "atr_pct": (df["atr_pct_roll"] > 0.01).to_numpy(),
        "guard": (df["guard_metric"] < 0.3).to_numpy(),
        "close_norm": (df["close_norm"] < 0.5).to_numpy(),
        "adx": (df["adx"] > 20.0).to_numpy(),
        "bb_width": (df["bb_width"] > 0.05).to_numpy(),
    }
    assert set(rec) == set(expected) | {"all"}
    for name, m in expected.items():
        assert rec[name] == {"pass": int(m.sum()), "total": N}, name
    combined = np.logical_and.reduce(list(expected.values()))
    assert rec["all"]["pass"] == int(combined.sum()) == int((out["enter_long"] == 1).sum())


def test_guards_off_records_only_mandatory(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_ENTRY_CONDITION_SINK", str(tmp_path))
    _strategy(np.ones(N, bool), False, THRESH).populate_entry_trend(_frame(), {"pair": "X/Y"})
    rec = json.loads((tmp_path / "_Stub_X_Y.json").read_text())["counts"]
    assert set(rec) == {"signal", "volume", "rvol", "all"}
