# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402
"""GbbGuardGridOpen — the `gbb` trigger with PER-FOLD tuned guards and NO model (feature 022).

Arm C. The guard threshold values are NOT set here: they arrive from the fold's GENERATED
params file, chosen by `evaluation/pick_guards.py` on that fold's OWN training window and
routed through `run_folds.py`'s `guards` plan key. freqtrade merges a params file over the
class with the file winning, so the generated values override `TriggerOnlyStrategy.buy_params`
without this class naming a single threshold.

That is deliberate: a class that pinned its own guard values would make the per-fold choice
unreachable, and 15 near-identical classes is the duplication this repository keeps paying for.

NO EXIT FIELD IS DECLARED HERE. Everything exit-side is inherited unchanged from
`TriggerOnlyStrategy`, which is what makes arm C comparable to arms A, B and D exit-for-exit —
proven by `exit_config_hash` equality (FR-009, SC-004), not asserted.

NO MODEL. `TriggerOnlyStrategy` is model-free, so nothing is fitted anywhere in this arm
(FR-020).

`entry_enable_guards` stays True and `run_folds.generate_fold` forces it True on write as well,
because that flag also gates the `volume > 0` data-sanity check and 34.26% of bars in this
dataset have `volume <= 0` (FR-008).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from SimpleStrategies.TriggerOnly import TriggerOnlyStrategy


class GbbGuardGridOpen(TriggerOnlyStrategy):
    """`gbb` + per-fold tuned guards, no model. Guard values come from the generated params."""

    TRIGGER_NAME = "gbb"
