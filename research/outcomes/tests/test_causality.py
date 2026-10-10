"""T018 — the truncation causality checker (023 research R11, FR-003, SC-006).

A checker that cannot fail reports the same "PASS" as a real pass (constitution IV), so the
planted forward-reading triggers here MUST fail, on real data, through the real feature frame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "utils")):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.outcomes import causality as C  # noqa: E402
from research.outcomes import triggers as T  # noqa: E402


def _small_loader(pair, era):
    """One pair, one era, ~1,500 bars -- enough for the MINIMAL indicator warm-up."""
    from regime.regimedata import load_ohlcv

    o = load_ohlcv(pair)
    i0 = o.index.searchsorted(pd.Timestamp(era, tz="UTC"))
    return o.iloc[i0 - 600 : i0 + 900]


KW = {"pairs": ("SOL",), "eras": ("2023-10-01",), "cuts": 16, "slice_loader": _small_loader}


def _shift_trigger():
    return T.Trigger(
        name="planted_shift",
        evaluate=lambda df: (df["close"].shift(-3) > df["close"]).fillna(False).astype(bool),
        required_columns=("close",),
        source="CONTROL",
    )


def test_gbb_passes_on_real_data():
    T.ensure_builtins()
    res = C.check_triggers([T.get("gbb")], **KW)["gbb"]
    assert res["verdict"] == "PASS", res
    assert res["comparisons"] == 16


def test_planted_forward_reads_fail():
    res = C.check_triggers([_shift_trigger(), C.control_trigger()], **KW)
    for name in ("planted_shift", C.CONTROL_NAME):
        assert res[name]["verdict"] == "FAIL", res[name]
        fm = res[name]["first_mismatch"]
        assert fm and 0 <= fm["bars_before_cut"] <= 10, fm


def test_missing_columns_are_unscoreable_not_pass():
    t = T.Trigger(
        name="needs_ghost",
        evaluate=lambda df: df["ghost_col"] > 0,
        required_columns=("ghost_col",),
        source="CONTROL",
    )
    res = C.check_triggers([t], **KW)["needs_ghost"]
    assert res["verdict"] == "UNSCOREABLE" and res["missing_columns"] == ["ghost_col"]


def test_cli_exits_1_when_control_does_not_fail(monkeypatch):
    def fake_run(**kw):
        return {
            "checker_control": {
                "trigger": C.CONTROL_NAME,
                "verdict": "PASS",
                "first_mismatch": None,
            },
            "triggers": {},
            "slices_checked": [],
            "slices_skipped": [],
        }

    monkeypatch.setattr(C, "run", fake_run)
    assert C.main([]) == 1


def test_check_runnable_fails_closed(tmp_path):
    vf = tmp_path / "v.json"
    assert C.check_runnable("gbb", vf)[0] is False  # no file
    good = {
        "code_sha": C.code_sha(),
        "checker_control": {"verdict": "FAIL"},
        "triggers": {"gbb": {"verdict": "PASS"}, "bad": {"verdict": "FAIL"}},
    }
    vf.write_text(json.dumps(good))
    assert C.check_runnable("gbb", vf) == (True, "PASS")
    assert C.check_runnable("bad", vf)[0] is False
    assert C.check_runnable("absent", vf)[0] is False
    assert C.check_runnable(None, vf)[0] is False
    vf.write_text(json.dumps({**good, "code_sha": "stale"}))
    ok, why = C.check_runnable("gbb", vf)
    assert not ok and "STALE" in why
    vf.write_text(json.dumps({**good, "checker_control": {"verdict": "PASS"}}))
    assert C.check_runnable("gbb", vf)[0] is False


def test_non_contiguous_slice_is_rejected():
    # XRP was delisted from binance.us 2021-2023; a 2021 era would span the gap
    assert C._load_slice("XRP", "2021-10-01") is None
    assert C._load_slice("SOL", "2021-10-01") is not None
