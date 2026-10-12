"""029 T012/T027: NNNC derives from the shared entry framework.

Every loadable NNNC strategy carries EntryFrameworkMixin and StopInvariantMixin exactly once,
ahead of BaseStrategy; arm D enters in model mode, tagged reversion by its gbb_path label, with no
family exits; the variants that replace `get_entry_conditions` are known; Framework/ is untouched;
and an NNNC stop that violates an invariant halts the run (report mode records it instead).
"""

from __future__ import annotations

import re
import subprocess
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from evaluation.fingerprint import load_strategy_class  # noqa: E402


def _nnnc_strategies():
    from freqtrade.strategy import IStrategy

    out = []
    for py in sorted((STRAT / "NNNC").glob("*.py")):
        if not re.search(rf"^class {re.escape(py.stem)}\b", py.read_text(errors="ignore"), re.M):
            continue
        try:
            cls = load_strategy_class(py.stem)
        except Exception:  # noqa: BLE001 -- pre-existing load failures are scanned elsewhere
            continue
        # NNNCStrategy subclasses only: NNNC/ also holds scaler builders (CreateScalers*) and
        # NNProfitStrategy, which derive from Framework bases directly and are not NNNC arms.
        if (isinstance(cls, type) and issubclass(cls, IStrategy)
                and any(k.__name__ == "NNNCStrategy" for k in cls.__mro__)):
            out.append(cls)
    return out


NNNC_CLASSES = _nnnc_strategies()


def test_scan_found_the_nnnc_family():
    assert len(NNNC_CLASSES) > 250


def test_every_nnnc_class_carries_the_framework_once_before_basestrategy():
    from Framework.BaseStrategy import BaseStrategy
    from utils.EntryFramework import EntryFrameworkMixin
    from utils.StopInvariants import StopInvariantMixin

    bad = []
    for cls in NNNC_CLASSES:
        mro = cls.__mro__
        for mixin in (EntryFrameworkMixin, StopInvariantMixin):
            if mro.count(mixin) != 1 or mro.index(mixin) > mro.index(BaseStrategy):
                bad.append((cls.__name__, mixin.__name__))
    assert not bad, bad[:10]


def test_arm_d_is_model_mode_reversion_with_no_family_exits():
    cls = load_strategy_class("NNNC_A_Raw_Open")
    assert cls.ENTRY_MODE == "model"
    assert cls._model_family() == "reversion"  # gbb_path label (spec FR-007, A′)
    assert cls.FAMILY_EXITS == {}
    assert cls.REVERSION_REGIME_GATE is None and cls.MOMENTUM_RECENCY_BARS == 0


#: Variants that override get_entry_conditions (029 R2). Those that call super() reach the
#: framework's model source; the rest replace the mask outright and keep today's behaviour.
OVERRIDERS = {
    "NNNC_Breakout": True, "NNNC_BrkOra03": None, "NNNC_BrkOra02": None, "NNNC_BrkOra005": None,
    "NNNC_NoModel": None, "NNNC_Ora_Prec": None, "NNNC_Oracle": None,
    "NNNC_TriggerGate": None, "TriggerOnlyMixin": None,
}


def test_the_variants_that_replace_get_entry_conditions_are_known():
    found = {}
    for py in sorted((STRAT / "NNNC").glob("*.py")):
        src = py.read_text(errors="ignore")
        m = re.search(r"def get_entry_conditions\(self[^)]*\):(.*?)(?=\n    def |\nclass |\Z)",
                      src, re.S)
        if m:
            found[py.stem] = "super().get_entry_conditions" in m.group(1)
    assert set(found) == set(OVERRIDERS)
    assert found["NNNC_Breakout"] is True


def test_framework_directory_is_not_modified_by_this_feature():
    base = (STRAT / "specs/029-nnnc-simpletrigger-merge/baseline.md")
    if not base.is_file():
        pytest.skip("baseline not committed yet")
    m = re.search(r"Baseline commit: `([0-9a-f]{7,})`", base.read_text())
    if not m:
        pytest.skip("baseline commit not recorded yet")
    changed = subprocess.run(["git", "-C", str(STRAT), "diff", "--name-only", m.group(1), "--",
                              "Framework/"], capture_output=True, text=True, check=True).stdout
    assert changed.strip() == ""


# ------------------------------------------------------------- T027: invariants halt NNNC


T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _stub_nnnc(value):
    cls = load_strategy_class("NNNC_A_Raw_Open")
    s = object.__new__(cls)  # no __init__: no model loads
    s.stoploss_grace_hours = 6.5
    s.use_atr_adaptive_stoploss = True
    s.atr_stoploss_floor = -0.04
    s.stoploss = -0.05
    # replace the parent stop with a planted wrong-sign value
    import Framework.BaseStrategy as B

    orig = B.BaseStrategy.custom_stoploss
    B.BaseStrategy.custom_stoploss = lambda self, *a, **k: value
    return s, lambda: setattr(B.BaseStrategy, "custom_stoploss", orig)


def _call(s):
    trade = types.SimpleNamespace(id=7, open_rate=100.0, open_date_utc=T0)
    return s.custom_stoploss("SOL/USDT", trade, T0 + timedelta(hours=8), 95.0, -0.05, False)


def test_a_planted_violation_halts_an_nnnc_run(monkeypatch):
    from utils.StopInvariants import StopInvariantError

    monkeypatch.delenv("FT_STOP_INVARIANTS", raising=False)
    s, restore = _stub_nnnc(0.06)
    try:
        with pytest.raises(StopInvariantError, match="sign.*SOL/USDT trade 7"):
            _call(s)
    finally:
        restore()


def test_report_mode_records_it_instead(monkeypatch, tmp_path):
    monkeypatch.setenv("FT_STOP_INVARIANTS", "report")
    monkeypatch.setenv("FT_STOP_INVARIANTS_SINK", str(tmp_path))
    s, restore = _stub_nnnc(0.06)
    try:
        assert _call(s) == 0.06
    finally:
        restore()
    assert list(tmp_path.glob("*_stopinv.json"))
