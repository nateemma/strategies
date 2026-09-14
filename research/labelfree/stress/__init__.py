"""Brief 06 — a stress harness: how does a policy behave on paths it has never seen?

INFRASTRUCTURE, NOT AN ALPHA SOURCE. No policy is trained on synthetic data here. The
literature is unambiguous that a generator cannot add information the ~1.5M real bars lack:
agents trained inside learned simulators learn the simulator's artefacts (Coletta 2023), the
most faithful crypto generator had negative CAGR (CTBench 2025), recursive self-training
collapses tails (Shumailov 2024).

ONE GENERATOR, and the reason is measured rather than assumed. The repo's three rebuilt
generators cannot emit a coherent sequence -- TabDDPM and WGAN return (n, 1, F), CTAB-GAN+
returns a DataFrame. They are class-conditional ROW samplers built for class balancing, which
is what they were needed for and is not a defect. But a row sampler cannot produce a price
PATH, and the policy under test is path-dependent: stops, grace windows, holding limits and
take-profit all depend on the order bars arrive in. CGAN is the only sequence-capable type and
is unmaintained (clarification Q3). So the generator is a stationary block bootstrap.

  gencensus.py   what each generator can actually emit, cited by file:line
  bootstrap.py   stationary block bootstrap over real panels (Politis & Romano 1994)
  bank.py        persisted path banks, with the provenance to detect staleness
  calibrate.py   the Tail-GAN criterion at PRE-REGISTERED tolerances -- the blocking gate
  run.py         stress a policy across a bank -> a distribution, never a point estimate
  report.py      the protocol §6 result document

A VETO, NEVER A GATE. "NO VETO" means the harness found no reason to object, not that the
policy is good. No policy may be promoted on stress results alone (SC-009, FR-014).
"""
from __future__ import annotations

# T002: the interpreter guard. `~/.pyenv/shims/python3` is a DIFFERENT 3.12.13 carrying neither
# pandas nor scipy, so an import failure here is far more likely to be the wrong interpreter
# than a missing install. Say so rather than emitting a bare ImportError.
_VENV = "freqtrade/.venv/bin/python"
_missing = []
for _m in ("numpy", "pandas", "scipy"):
    try:
        __import__(_m)
    except ImportError:                                        # pragma: no cover
        _missing.append(_m)
if _missing:                                                   # pragma: no cover
    import sys as _sys
    raise ImportError(
        f"research.labelfree.stress needs {', '.join(_missing)}, missing from {_sys.executable}.\n"
        f"Run this feature under {_VENV} — ~/.pyenv/shims/python3 is a different 3.12.13 that "
        f"carries neither pandas nor scipy."
    )
del _m, _missing
