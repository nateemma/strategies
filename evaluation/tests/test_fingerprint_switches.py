"""028 T016: promoting C2 into the shared exit leaves every recorded framework fingerprint
unchanged (FR-010), and a strategy that switches it on is distinguishable (FR-006)."""

from __future__ import annotations

import pytest
from evaluation import fingerprint as fp


CONFIG = fp.STRAT_DIR / "config" / "config_wf.json"

#: specs/028-recovery-aware-exits/baseline.md (T001), recorded before the switch existed.
RECORDED = {
    "NNNC_MLX_Open": "303456c1fa322b4c",
    "SimpleTrigger_BreakoutGbb_MomExit": "95dd915b90fc6217",
    "SimpleTrigger_RevOnlyVolreg2": "0e552be31732d222",
    "SimpleTrigger_RevOnlyVolreg2_XC4": "42b8856d4448fe86",
    "SimpleTrigger_RevOnlyVolreg2_XC2": "0e552be31732d222",
    "SimpleTrigger_RevOnlyVolreg2_XC1": "0e552be31732d222",
    "SimpleTrigger_RevOnlyVolreg2_XC42": "42b8856d4448fe86",
    # 027 baseline.md
    "SimpleTrigger_Gbb": "303456c1fa322b4c",
    "SimpleTrigger_FamCombined": "27617f430beca1cd",
    "SimpleTrigger_RegGateVolreg2": "f1492b64a9dd9257",
}


#: Moved on purpose since: NNNC's grace default 6.5 h -> 24 h (2026-10-13, research/nnnc_framework/).
#: These are checked with grace put back to 6.5 h -- everything else must still match the record.
GRACE_MOVED = {"NNNC_MLX_Open"}
#: ... and NNNC's in-profit held-pair add on by default (2026-10-13, research/held_pair/): checked
#: with the add off.
ADD_MOVED = {"NNNC_MLX_Open"}


def _sha(name):
    import copy

    cls = fp.load_strategy_class(name)
    if name in ADD_MOVED:
        cls = type(cls.__name__ + "NoAdd", (cls,), {"HELD_PAIR_ADD": None})
    pj = fp.load_params_json(name)
    if name in GRACE_MOVED:
        pj = copy.deepcopy(pj)
        pj["params"]["sell"]["stoploss_grace_hours"] = 6.5
    sha, _ = fp.resolve(cls, pj, fp.load_config(CONFIG))
    return sha


@pytest.mark.parametrize("name,sha", sorted(RECORDED.items()))
def test_recorded_fingerprints_unchanged_with_switch_off(name, sha):
    assert _sha(name) == sha


def test_c4_is_a_declared_parameter_not_a_switch():
    """A strategy declaring 12 h grace resolves XC4's framework: C4 needs no new code."""
    from SimpleTrigger.SimpleTrigger_RevOnlyVolreg2 import SimpleTrigger_RevOnlyVolreg2

    class Declared(SimpleTrigger_RevOnlyVolreg2):
        SELL_OVERRIDES = {**SimpleTrigger_RevOnlyVolreg2.SELL_OVERRIDES,
                          "stoploss_grace_hours": 12.0}

    sha, _ = fp.resolve(Declared, {}, fp.load_config(CONFIG))
    assert sha == RECORDED["SimpleTrigger_RevOnlyVolreg2_XC4"]


def test_promoted_c2_is_visible_to_the_fingerprint():
    """Unlike XC2's method override, the promoted switch is a field: its arm hashes differently
    from the control it was indistinguishable from."""
    assert _sha("SimpleTrigger_RevOnlyVolreg2_P2") != RECORDED["SimpleTrigger_RevOnlyVolreg2"]
