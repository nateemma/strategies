"""036 T031: capacity refusals, INERT detection and the decision rule."""

from __future__ import annotations

import json
from pathlib import Path

from research.live_pnl import capacity as C


RULE = json.loads((Path(__file__).resolve().parents[2] / "nnnc_switches" /
                   "preregistration.json").read_text())["rule"]


def test_refusals():
    assert C.refusals(False, True) and C.refusals(True, False)
    assert C.refusals(True, True) == []


def test_inert_when_stakes_unchanged():
    ref = [100, 200, 300, 900] * 50
    assert C.inert(ref, ref)[0]
    assert not C.inert([x * 2 for x in ref], ref)[0]


def test_decide():
    assert C.decide(1, 1, "UNCHANGED", "UNCHANGED", "UNCHANGED", RULE, 500, 200)[0] == "PASS"
    v, failed = C.decide(1, -1, "UNCHANGED", "UNCHANGED", "WORSENED", RULE, 500, 200)
    assert v == "FAIL" and failed == ["late gain", "risk"]
    assert C.decide(1, 1, "UNRESOLVABLE", "UNCHANGED", "UNCHANGED", RULE, 500, 200)[0] == "INCONCLUSIVE"
    assert C.decide(1, 1, "UNCHANGED", "UNCHANGED", "UNCHANGED", RULE, 50, 200)[0] == "INCONCLUSIVE"
