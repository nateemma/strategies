"""024 T006: every registered trigger carries a family (spec FR-001, SC-006, contract C5).

A trigger's family says what its condition TESTS -- buying weakness (reversion), buying strength
(momentum), or neither (unassigned: broad filters that fire on most bars, the `empty` null, the
controls and the unscoreable labelers). It is decided from the condition, never the name:
`breakout_bb` is not a price breakout, but it buys high band position, so it is momentum.
"""

from __future__ import annotations

import pytest
from research.outcomes import triggers as T


FAMILIES = {"reversion", "momentum", "unassigned"}

#: spec FR-001, verbatim
EXPECTED = {
    **dict.fromkeys(
        ["gbb", "gbb_tight", "gbb_stale", "geometry", "votes_tech", "failed_breakout",
         "pair_cmo_macdhist", "pair_cmo_dcbreakout", "pair_kst_zscore", "pair_cci_copp"],
        "reversion",
    ),
    **dict.fromkeys(
        [
            "breakout_bb",
            "breakout_gbb",
            "breakout_gbb_nofresh",
            "breakout",
            "breakout_vol",
            "breakout_squeeze",
            "breakout_gm",
            "breakout_vol_gm",
            "breakout_squeeze_gm",
        ],
        "momentum",
    ),
    **dict.fromkeys(
        [
            "adx",
            "votes_ind2",
            "votes_ind3",
            "votes_ind4",
            "votes_consensus",
            "empty",
            "random_matched",
            "trends",
            "bands",
        ],
        "unassigned",
    ),
}


def _registered():
    T.ensure_builtins()
    return {t.name: t for t in T.all_triggers()}


def test_every_registered_trigger_has_a_family_and_reason():
    for name, trig in _registered().items():
        assert trig.family in FAMILIES, f"{name}: family {trig.family!r}"
        assert trig.family_reason.strip(), f"{name}: empty family_reason"


def test_classification_matches_spec_fr001():
    got = {name: trig.family for name, trig in _registered().items()}
    assert got == EXPECTED


def test_invalid_family_is_refused():
    with pytest.raises(T.TriggerError, match="family"):
        T.Trigger(
            name="planted",
            evaluate=lambda df: df,
            required_columns=(),
            family="sideways",
            family_reason="x",
        )


def test_family_without_reason_is_refused():
    with pytest.raises(T.TriggerError, match="family_reason"):
        T.Trigger(name="planted", evaluate=lambda df: df, required_columns=(), family="momentum")


def test_a_builtin_registered_without_a_family_fails_the_check(monkeypatch):
    # A newly registered trigger without a family must make the registry check fail.
    _registered()
    planted = T.Trigger(name="planted_no_family", evaluate=lambda df: df, required_columns=())
    monkeypatch.setitem(T._REGISTRY, planted.name, planted)
    with pytest.raises(AssertionError):
        test_every_registered_trigger_has_a_family_and_reason()
