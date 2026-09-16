"""Signal-context timing study (feature 017).

Does NNNC_MLX entry-signal QUALITY depend on signal CONTEXT, and can that be acted on?

THE STRUCTURAL FACT THIS PACKAGE IS BUILT AROUND. freqtrade admits an entry only when the pair
has no open position (`optimize/backtesting.py:1552`), so a repeat signal has ALREADY been
discarded by the time it appears. Filtering it is a no-op. Read correctly the repeat signal is
not about the signal at all -- it is information about the POSITION already open, arriving
before that trade's outcome is known. Its only actionable home is the exit side.

TWO CURRENCIES, NEVER MIXED. The confound test measures a WIN-RATE difference; the study's
resolution limit of 7.633 pp is an ANNUALISED RETURN quantity. The motivating effect is 14.8
win-rate points, which clears 7.633 numerically while carrying no return information whatever.
Applying the floor to the wrong currency turns Principle III's gate from a defence into a rubber
stamp, so `confound.effect_sizes` returns both and `confound.verdict` gates on only one.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pinned thresholds. Every kill gate in this feature fires on a number from here, so a reader
# can see all of them in one place and no module can quietly hold a different one.
# ---------------------------------------------------------------------------
NOISE_FLOOR_PP = 7.633        # SC-015  pooled annualised RETURN resolution limit
RETURN_FLOOR_PP_YR = 3.350    # SC-005  acceptance floor
FIDELITY_MIN = 0.95           # T011a   reconstruction vs recorded trades, per fold
OVERLAP_INERT = 0.70          # FR-022  trigger overlap with the existing stop stack
OVERLAP_WITHIN_BARS = 8       # FR-022  "the existing stack fired anyway" window
RECONCILE_TOL = 0.002         # T008    profit_ratio units, ~2x a round-trip fee
TIGHTEN_FACTOR = 0.5          # FR-019a frozen; an outright exit is a sensitivity only
STAKE_PP_PER_RATIO = 9.0      # 900 stake on a 10,000 wallet (established in brief 013)

DRAWDOWN_LOOKBACK_BARS = 96   # FR: 1 day, this repository's window convention
BURST_GAP_BARS = 24           # FR: 6h, chosen to exceed the 3-10h median hold

ROOT = Path(__file__).resolve().parents[2]
# Preserved by T001. NOT /tmp: a clear there costs ~44 minutes of backtests to rebuild.
SIDECAR_ROOT = ROOT.parent / "sidecars" / "es_sink"
OUT = Path(__file__).resolve().parent / "out"


def assert_interpreter() -> None:
    """The pyenv shim is a DIFFERENT 3.12.13 carrying no pandas and no scipy.

    `sys.prefix`, never `Path(sys.executable).resolve()`: the venv's `bin/python` is a SYMLINK
    to the pyenv binary, so resolving it lands outside the venv and rejects the CORRECT
    interpreter. Recorded 2026-09-15, when exactly that guard fired a false positive.
    """
    if sys.prefix == sys.base_prefix or "freqtrade/.venv" not in sys.prefix:
        raise SystemExit(
            f"wrong interpreter (prefix {sys.prefix})\n"
            f"use {ROOT.parent.parent}/.venv/bin/python")


assert_interpreter()
