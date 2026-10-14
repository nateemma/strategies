"""034 T006: the training wrapper's refusals and default folds (driver stubbed)."""

from __future__ import annotations

import pytest

from research.label_screen import train as T


OK = {"candidates": {"FMAE": {"status": "PASS"}, "MMR": {"status": "INCONCLUSIVE"}}}


def test_refuses_untrained_and_failed_audit():
    assert T.refusals("CONTROL", set(T.EARLY), OK, None)
    assert T.refusals("GFM", set(T.EARLY), OK, None)
    assert "S0a" in T.refusals("MMR", set(T.EARLY), OK, None)[0]
    assert T.refusals("TB", set(T.EARLY), OK, None)  # not audited at all
    assert T.refusals("FMAE", set(T.EARLY), OK, None) == []


def test_late_folds_only_for_the_s2_winner():
    late = {"f08", "f14"}
    assert T.refusals("FMAE", late, OK, None)
    assert T.refusals("FMAE", late, OK, {"s2_winner": {"tag": "TB"}})
    assert T.refusals("FMAE", late, OK, {"s2_winner": {"tag": "FMAE"}}) == []


class _Driver:
    PLANS: dict = {}
    BUSY = ("x",)

    def __init__(self):
        self.calls = []

    def run(self, key, folds, seed):
        self.calls.append((key, sorted(folds), seed))
        return []


def test_default_folds_and_registration(monkeypatch):
    monkeypatch.setattr(T, "_read", lambda name: OK if name == "audit.json" else None)
    d = _Driver()
    T.run("FMAE", None, driver=d)
    assert d.calls == [("LS_FMAE", list(T.EARLY), 1)]
    assert d.PLANS["LS_FMAE"] == "ls_fmae_es" and "research.label_screen.train " in d.BUSY
    with pytest.raises(SystemExit):
        T.run("MMR", None, driver=d)
