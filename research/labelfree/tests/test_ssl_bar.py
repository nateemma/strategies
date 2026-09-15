"""The comparison bar — T009, the test that should have existed from the start.

THE BAR IS THE GATE. Every screen result is a comparison against it, so a drifted bar changes
every verdict with nothing failing. That is precisely what happened during implementation: the
bar was measured wrongly twice and each wrong version produced confident, plausible, wrong
conclusions.

  generation 1   split by PAIR while claiming to split by time
  generation 2   target started at open[t], so it included bar t's own move
  generation 3   target open[t+1] -> open[t+1+H]                  <- current

Generation 2 inverted a conclusion: raw OHLCV appeared to beat the engineered feature set by
~1.95x at every horizon, 25/25 fold-horizon comparisons, and it was proposed as its own brief.
The tell was the stability -- a ratio that flat across horizons is mechanical, not robust.

So these tests guard the TARGET'S ALIGNMENT, not just the numbers. A number can be re-measured;
an alignment error is invisible and self-consistent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.ssl import inputs as IN                 # noqa: E402
from research.labelfree.ssl import screen as SC                 # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "ssl" / "baseline.json"


# ------------------------------------------------------------------ the alignment itself

def test_forward_return_enters_at_t_plus_one():
    """A signal at bar t is known at close[t]; the strategy enters at the NEXT bar's open.

    Element i must therefore be open[i+1] -> open[i+1+H]. Starting at open[i] includes bar i's
    own move, which the inputs reveal directly — the leak that inverted a conclusion.
    """
    px = np.arange(1, 21, dtype=float)
    H = 3
    r = IN.forward_return(px, H)
    assert r[0] == pytest.approx(px[1 + H] / px[1] - 1.0), "does not enter at t+1"
    assert r[0] != pytest.approx(px[H] / px[0] - 1.0), "entered at t — the leaked alignment"
    assert len(r) == len(px) - H - 1


def test_forward_return_cannot_see_bar_t():
    """Perturbing bar t's own price must not change the target for a signal at bar t."""
    px = np.linspace(100, 120, 40)
    a = IN.forward_return(px, 6)
    px2 = px.copy()
    px2[5] *= 1.5                      # bar 5's own open moves wildly
    b = IN.forward_return(px2, 6)
    assert a[5] == pytest.approx(b[5]), (
        "the target for a signal at bar 5 changed when bar 5's own price moved — it starts too "
        "early")


def test_forward_return_does_see_the_entry_bar():
    """The converse: it MUST depend on t+1, or it is measuring the wrong window."""
    px = np.linspace(100, 120, 40)
    a = IN.forward_return(px, 6)
    px2 = px.copy()
    px2[6] *= 1.5                      # the ENTRY bar for a signal at bar 5
    b = IN.forward_return(px2, 6)
    assert a[5] != pytest.approx(b[5])


# ------------------------------------------------------------------ the measured bar

@pytest.mark.skipif(not BASELINE.exists(), reason="run: ssl.screen --baseline-only")
def test_measured_ceiling_matches_the_recorded_baseline():
    """The constant and the measured file must not drift apart.

    If they do, CHECK THE TARGET'S ALIGNMENT before adjusting either — both earlier generations
    of this constant looked plausible and both were leaking.
    """
    import json
    raw = json.loads(BASELINE.read_text())
    tab = raw.get("by_horizon", raw)
    folds = raw.get("folds")
    if folds is not None and len(folds) != 10:
        pytest.skip(f"baseline.json came from {len(folds)} folds; the constant is the 10-fold "
                    f"f05..f14 measurement, so a subset run is not comparable")
    base = {int(k): v["median"] for k, v in tab.items()}
    for H, v in SC.MEASURED_CEILING.items():
        if H in base:
            assert abs(base[H] - v) < 0.005, (
                f"H={H}: constant {v} vs measured {base[H]}. Do NOT edit the constant to match "
                f"without checking forward_return's alignment first.")


@pytest.mark.skipif(not BASELINE.exists(), reason="run: ssl.screen --baseline-only")
def test_the_bar_is_flat_across_horizons_not_decaying():
    """Under causal alignment predictability does NOT decay with horizon.

    The familiar decay — ~0.15 at H=3 falling to ~0.03 at H=96 — was the leak's own signature:
    bar t's move is proportionally the largest part of a short-horizon target. This test pins the
    corrected shape, because 'predictability decays with horizon' was stated as a finding twice
    in this feature's planning documents and does not survive.
    """
    import json
    raw = json.loads(BASELINE.read_text())
    base = {int(k): v["median"] for k, v in raw.get("by_horizon", raw).items()}
    vals = np.array([base[h] for h in sorted(base)])
    assert vals.max() < 0.08, f"bar is far higher than the causal measurement: {vals}"
    assert vals.max() / max(vals.min(), 1e-9) < 3.0, (
        f"a >3x spread across horizons suggests the decay is back, i.e. the target has "
        f"regressed to the leaked alignment: {dict(sorted(base.items()))}")
