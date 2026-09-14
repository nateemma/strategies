"""Brief 05 — gradient-free policy search over the exit and guard levers (feature 013).

WHAT THIS IS. Briefs 01, 02 and 04 failed the SAME objective through three different
searches. This is a fourth search over that objective, but over a DIFFERENT space: the
discrete exit and guard levers none of the three could touch, searched JOINTLY rather than
one at a time.

THE TEN DIMENSIONS (space.py). Seven exit levers, two entry guards, one holding-time lever.
`cexit_take_profit` and `hold_window_hours` were added during planning and are not in the
brief's own list; the reasons are recorded in space.py and in specs/013-*/research.md R2.

THE MEASUREMENT THAT GOVERNS THE FEATURE. An entire rvol parameter sweep spans 4.93 pp
inside a 7.633 pp noise floor: the search space's total range of effect is smaller than the
uncertainty on measuring it. A quantified negative is the expected deliverable, and briefs
01-04 delivered their value exactly that way.

THE BLOCKING GATE (calibrate.py). The surrogate in exits.py re-implements freqtrade's exit
stack. Duplication drifts, and this study has paid for that repeatedly -- a loss term that
silently never ran, a purge landing in an already-discarded tail, four inert parameters. So
`calibrate` scores candidates BOTH ways against real backtests, and every downstream entry
point refuses to run without a passing calibration.json. An uncalibrated surrogate does not
merely risk a wrong answer: it means the entire search budget optimises a fiction.
"""
from __future__ import annotations

# T002: the interpreter guard. `~/.pyenv/shims/python3` is a DIFFERENT 3.12.13 that carries
# `cma` but neither pandas nor scipy, so an import failure here is far more likely to be the
# wrong interpreter than a missing install. Say so, rather than emitting a bare ImportError.
_VENV = "freqtrade/.venv/bin/python"
_missing = []
for _m in ("numpy", "pandas", "scipy", "cma"):
    try:
        __import__(_m)
    except ImportError:                                        # pragma: no cover
        _missing.append(_m)
if _missing:                                                   # pragma: no cover
    import sys as _sys
    raise ImportError(
        f"research.labelfree.es needs {', '.join(_missing)}, missing from {_sys.executable}.\n"
        f"Run this feature under {_VENV} — note that ~/.pyenv/shims/python3 is a different\n"
        f"3.12.13 carrying cma but neither pandas nor scipy."
    )
del _m, _missing
