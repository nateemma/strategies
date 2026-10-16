"""037 T011a: discovery scope."""
from __future__ import annotations

from research.indicator_screen import discover as D


def test_scope_includes_indicator_strategies_and_excludes_trigger_arms():
    d = D.discover()
    names = {c["class"] for c in d["classes"]}
    assert {"CCI", "Combo", "Ehlers_BB", "BBBreakout", "Ichimoku", "HTDCP"} <= names
    ex = {c["class"]: c["reason"] for c in d["excluded"]}
    assert ex["GbbOpen"] == "trigger_arm" and ex["BreakoutVolOpen"] == "trigger_arm"
    assert "SimpleStrategy" not in names and not names & set(ex)
    assert len(names) >= 60
