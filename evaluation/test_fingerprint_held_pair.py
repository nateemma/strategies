"""031 T013: the held-pair add switch enters framework_sha only when it is on, so every recorded
fingerprint is unchanged."""

from __future__ import annotations

import json
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[1]
for p in (str(STRAT), str(STRAT / "utils")):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.fingerprint import load_config, load_strategy_class, resolve  # noqa: E402


CONFIG = load_config(STRAT / "config" / "config_wf.json")
BASELINE = STRAT / "specs" / "031-held-pair-dca" / "baseline.json"


def test_off_classes_hash_as_recorded_and_on_differs():
    base = json.loads(BASELINE.read_text())["bases"]
    for name in ("NNNC_A_Raw_Open", "SimpleTrigger_RevOnlyVolreg2_P2_G24"):
        cls = load_strategy_class(name)
        sha, resolved = resolve(cls, None, CONFIG)
        assert "HELD_PAIR_ADD_RESOLVED" not in resolved["attributes"]
        on = type(name + "On", (cls,), {"HELD_PAIR_ADD": {}})
        sha_on, resolved_on = resolve(on, None, CONFIG)
        assert sha_on != sha
        assert resolved_on["attributes"]["HELD_PAIR_ADD_RESOLVED"]["max_adds"] == 1
        assert resolved_on["attributes"]["position_adjustment_enable"] is True
    # the recorded baseline (params JSON included) is reproduced by name
    from evaluation.fingerprint import resolve_by_name

    for name, sha in base.items():
        assert resolve_by_name(name, STRAT / "config" / "config_wf.json")[0] == sha
