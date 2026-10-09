"""The three required triggers (T023, FR-013, FR-014a).

The load-bearing test here is `test_geometry_is_CAUSAL_...`: it mutates future bars and
asserts the mask over earlier bars does not move. That is the property `labels_geometry`
lacks, it is why `geometry` is a substitute rather than a port, and it is the one claim
that cannot be established by reading the code -- `find_peaks` over a whole array looks
backward-compatible until you perturb the future and watch the past change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "utils"), str(ROOT / "NNNC")):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.outcomes import builtin_triggers as B  # noqa: E402
from research.outcomes import triggers as T  # noqa: E402


@pytest.fixture(autouse=True)
def _registry():
    T.clear_registry()
    B.register_builtins()
    yield
    T.clear_registry()


def _frame(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "close": 100.0 + rng.normal(0, 1, n).cumsum(),
        "guard_metric": rng.uniform(0.0, 1.0, n),
        "bb_width": rng.uniform(0.0, 0.08, n),
        "rvol": rng.uniform(0.0, 3.0, n),
    })


def test_all_three_required_members_are_registered():
    assert T.names() == ["empty", "gbb", "geometry"]


def test_register_builtins_is_idempotent_via_replace():
    B.register_builtins(replace=True)
    assert T.names() == ["empty", "gbb", "geometry"]


# =============================================================================================
# empty — the verdict null
# =============================================================================================

def test_empty_fires_on_every_bar_and_needs_no_columns():
    df = _frame()
    assert T.evaluate("empty", df).all()
    assert T.get("empty").required_columns == ()


def test_empty_names_forward_mae_as_its_source():
    """Not cosmetic: the arm that lost to cash and the verdict rule's null are the same
    object, because forward_mae applies no entry condition at all."""
    assert T.get("empty").source == "labels_forward_return_mae_cap"


# =============================================================================================
# gbb — faithful to labels_gbb's backward-looking half
# =============================================================================================

def _live_gbb_trigger(df: pd.DataFrame) -> np.ndarray:
    """`labels_gbb`'s mask with its `future_gain` clause removed, rebuilt here from the
    live source INDEPENDENTLY of the implementation under test."""
    n = len(df)
    guard = np.nan_to_num(np.asarray(df["guard_metric"], dtype=float), nan=0.0)
    bbw = np.nan_to_num(np.asarray(df["bb_width"], dtype=float), nan=0.0)
    rvol = np.nan_to_num(
        np.asarray(df.get("rvol", pd.Series(np.zeros(n))), dtype=float), nan=0.0
    )
    return (guard < 0.3) & (bbw > 0.035) & (rvol > -1.0)


def test_gbb_matches_the_live_backward_looking_mask():
    df = _frame()
    np.testing.assert_array_equal(T.evaluate("gbb", df).to_numpy(), _live_gbb_trigger(df))


def test_gbb_reproduces_labels_gbbs_ASYMMETRIC_NaN_convention():
    """`nan_to_num(nan=0.0)` means a NaN `guard_metric` FIRES (0.0 < 0.3) while a NaN
    `bb_width` does NOT (0.0 > 0.035 is False). Writing the obvious
    `df["guard_metric"] < 0.3` would make a NaN compare False and silently diverge from
    the live label on exactly those bars.

    Latent on the production data (SOL/USDT carries zero NaNs in either column), so this
    is fidelity insurance rather than a live difference -- which is why it needs a test
    that constructs the case rather than waiting to meet it."""
    df = pd.DataFrame({
        "guard_metric": [np.nan, 0.1, np.nan, 0.9],
        "bb_width": [0.05, np.nan, np.nan, 0.05],
        "rvol": [1.0, 1.0, 1.0, 1.0],
    })
    out = T.evaluate("gbb", df).tolist()
    assert out == [True, False, False, False]
    np.testing.assert_array_equal(out, _live_gbb_trigger(df))


def test_gbb_carries_the_inert_rvol_clause_rather_than_dropping_it():
    """The clause is inert at `rvol > -1.0`, not absent. Dropping it would silently change
    what this trigger means the moment a family sets LABEL_RVOL_THRESHOLD."""
    assert B.GBB_RVOL_THRESHOLD == -1.0
    df = _frame()
    df["rvol"] = 0.0                      # every bar fails any threshold >= 0
    assert T.evaluate("gbb", df).any(), "a zero rvol must still pass the inert clause"


def test_gbb_requires_its_columns_and_refuses_to_guess():
    df = _frame().drop(columns=["bb_width"])
    with pytest.raises(T.TriggerError, match="bb_width"):
        T.evaluate("gbb", df)


# =============================================================================================
# geometry — the causality that labels_geometry lacks
# =============================================================================================

def test_geometry_is_CAUSAL_future_bars_cannot_change_the_past():
    """THE test. Perturb every bar after a cut point and assert the mask at or before it is
    unchanged. `labels_geometry` fails this by construction: `find_peaks` scans the whole
    array and the loop additionally labels the three bars BEFORE each valley, so a bar's
    label depends on what happens after it."""
    df = _frame(n=300)
    cut = 200
    before = T.evaluate("geometry", df).iloc[: cut + 1]

    perturbed = df.copy()
    rng = np.random.default_rng(99)
    perturbed.loc[cut + 1:, "close"] = 100.0 + rng.normal(0, 20, len(df) - cut - 1).cumsum()
    after = T.evaluate("geometry", perturbed).iloc[: cut + 1]

    pd.testing.assert_series_equal(before, after)


def test_the_live_labels_geometry_FAILS_that_same_test():
    """The control, and the justification for not porting it. Without this, "it looks
    forward" is an assertion about code I read; with it, it is a measurement."""
    from Framework.TrainingSignals import labels_geometry

    df = _frame(n=300)
    cut = 200
    before = labels_geometry(df, min_gain=0.01).iloc[: cut + 1]

    perturbed = df.copy()
    rng = np.random.default_rng(99)
    perturbed.loc[cut + 1:, "close"] = 100.0 + rng.normal(0, 20, len(df) - cut - 1).cumsum()
    after = labels_geometry(perturbed, min_gain=0.01).iloc[: cut + 1]

    assert not before.equals(after), (
        "labels_geometry unexpectedly looks causal here; if that is now true, re-examine "
        "GEOMETRY_FINDINGS and whether a faithful port has become possible"
    )


def test_labels_geometry_still_raises_with_its_own_defaults():
    """Pins finding 1 of GEOMETRY_FINDINGS, so "never tested" stays literal and
    discoverable. If someone gives `min_gain` a default, this fails and the finding needs
    revisiting."""
    from Framework.TrainingSignals import labels_geometry

    with pytest.raises(TypeError):
        labels_geometry(_frame())


def test_geometry_fires_after_the_low_not_at_it():
    """A V-shaped move: the low sits at a known index, and the trigger must fire
    GEOMETRY_CONFIRM_BARS later -- never at the low itself, which no strategy could have
    identified at the time."""
    k = B.GEOMETRY_CONFIRM_BARS
    down = list(np.linspace(100.0, 90.0, 20))
    up = list(np.linspace(90.0, 100.0, 20))[1:]
    close = down + up
    low_idx = len(down) - 1
    df = pd.DataFrame({"close": close})

    out = T.evaluate("geometry", df)
    assert not out.iloc[low_idx], "must not fire at the low; that is the lookahead"
    assert out.iloc[low_idx + k], f"must fire {k} bars after the low"


def test_geometry_ignores_a_low_that_is_not_deep_enough():
    """The prominence analogue. A shallow dip must not register as a valley."""
    rng = np.random.default_rng(1)
    flat = 100.0 + rng.normal(0, 0.01, 200)      # moves far less than GEOMETRY_MIN_DEPTH
    assert not T.evaluate("geometry", pd.DataFrame({"close": flat})).any()


def test_geometry_warmup_is_False_not_missing():
    """`triggers.evaluate` rejects a mask carrying missingness, so the warm-up must be
    filled by the trigger -- explicitly, by its author, not coerced by the framework."""
    out = T.evaluate("geometry", _frame(n=60))
    assert out.dtype == bool
    assert not out.iloc[: 2 * B.GEOMETRY_CONFIRM_BARS + 1].any()


def test_geometry_is_selective_unlike_the_labeler_it_replaces():
    """labels_geometry marks four bars per valley at distance 4, firing on ~54% of bars
    (measured). A trigger that fires on half of all bars cannot select anything."""
    rate = T.evaluate("geometry", _frame(n=2000)).mean()
    assert 0.001 < rate < 0.20, f"base rate {rate:.3f} is not selective"


def test_geometry_declares_itself_not_a_port():
    """The divergence is a reporting obligation (T025, the leaderboard), so it has to be
    legible from the registry and not only from a comment."""
    assert "NOT a port" in T.get("geometry").description
    assert T.get("geometry").source == T.SOURCE_NATIVE
    assert "forward-looking" in B.GEOMETRY_FINDINGS


# =============================================================================================
# The registry's own constraints still hold for the builtins
# =============================================================================================

def test_no_builtin_declares_an_outcome_or_pair_column():
    for t in T.all_triggers():
        assert "pair" not in t.required_columns
        assert not any(c.startswith("__outcome_") for c in t.required_columns)


def test_every_builtin_is_pair_label_invariant():
    df = _frame()
    for name in T.names():
        a = T.evaluate(name, df.assign(pair="ZEC/USDT"))
        b = T.evaluate(name, df.assign(pair="SOL/USDT"))
        pd.testing.assert_series_equal(a, b, obj=name)


def test_the_module_entrypoint_lists_the_builtins_as_a_SUBPROCESS():
    """`python -m research.outcomes.triggers --list` must work for real.

    An in-process `main([...])` call CANNOT catch the bug this guards: under `python -m`
    the module is imported twice -- once as `__main__`, once as
    `research.outcomes.triggers` when `builtin_triggers` imports it to reach `register` --
    and each copy holds its own `_REGISTRY`. The builtins landed in one while `main` read
    the other, so `--list` printed "no triggers registered" immediately after registering
    three. Only a subprocess exercises the `__main__` path, so only a subprocess can pin
    the fix.
    """
    import os
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "utils"), str(ROOT / "NNNC"), env.get("PYTHONPATH", "")]
    )
    r = subprocess.run(
        [sys.executable, "-m", "research.outcomes.triggers", "--list"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "no triggers registered" not in r.stdout
    for name in ("empty", "gbb", "geometry"):
        assert name in r.stdout, f"{name} missing from --list output:\n{r.stdout}"


# =============================================================================================
# T025's audit triggers (FR-012)
# =============================================================================================

def test_audit_triggers_register():
    B.register_audit_triggers()
    assert {"adx", "breakout", "breakout_gbb"} <= set(T.names())


def test_adx_matches_the_live_filter_in_the_three_future_labelers():
    """Reproduces `np.where(adx_scaled > -0.2, labels, 0)` exactly. The clause is BINDING --
    38.35% of bars pooled -- so it is a real trigger, not an inert one like gbb's rvol."""
    B.register_audit_triggers()
    df = pd.DataFrame({"adx_scaled": [-1.0, -0.2, -0.19, 0.0, 1.0, np.nan]})
    assert T.evaluate("adx", df).tolist() == [False, False, True, True, True, True]


def test_adx_treats_a_missing_value_the_way_the_live_code_does():
    """`df.get("adx_scaled", zeros)` plus `0.0 > -0.2` means a missing/NaN value PASSES the
    live filter. Reproduced rather than tidied: tightening it here would make the trigger
    stricter than the labelers it was extracted from."""
    B.register_audit_triggers()
    assert T.evaluate("adx", pd.DataFrame({"adx_scaled": [np.nan]})).iloc[0]


def test_breakout_is_CAUSAL():
    """`rolling(lookback).max().shift(1)` reads only closed bars, so unlike `geometry` this
    one ports exactly -- and that is worth proving, not assuming."""
    B.register_audit_triggers()
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"close": 100 + rng.normal(0, 1, 300).cumsum()})
    df["high"] = df["close"] + 0.5
    cut = 200
    before = T.evaluate("breakout", df).iloc[: cut + 1]
    pert = df.copy()
    pert.loc[cut + 1:, ["close", "high"]] = 500.0
    pd.testing.assert_series_equal(before, T.evaluate("breakout", pert).iloc[: cut + 1])


def test_breakout_fires_only_above_the_prior_window_high():
    B.register_audit_triggers()
    n = B.BREAKOUT_LOOKBACK + 5
    df = pd.DataFrame({"high": [10.0] * n, "close": [9.0] * n})
    df.loc[n - 1, "close"] = 10.5                       # exceeds the prior 20-bar high
    out = T.evaluate("breakout", df)
    assert out.iloc[-1]
    assert not out.iloc[:-1].any()


def test_breakout_warmup_is_False_not_missing():
    B.register_audit_triggers()
    n = B.BREAKOUT_LOOKBACK - 1
    df = pd.DataFrame({"high": np.arange(n, dtype=float), "close": np.arange(n, dtype=float)})
    out = T.evaluate("breakout", df)
    assert out.dtype == bool and not out.any()


def test_breakout_gbbs_guard_direction_is_OPPOSITE_to_gbbs():
    """Not a variant of one condition: `breakout_gbb` wants guard_metric ABOVE 0.2 while
    `gbb` wants it BELOW 0.3. Worth a test because the shared name invites the assumption
    that they select the same regime, and any leaderboard row mentioning gbb is misread if
    so."""
    B.register_audit_triggers()
    df = pd.DataFrame({"guard_metric": [0.1, 0.25, 0.5], "bb_width": [0.05] * 3,
                       "rvol": [1.0] * 3})
    assert T.evaluate("gbb", df).tolist() == [True, True, False]
    assert T.evaluate("breakout_gbb", df).tolist() == [False, True, True]


@pytest.mark.parametrize("labeler,trigger,kwargs", [
    ("labels_gbb", "gbb", {"min_gain": 0.01}),
    ("labels_gbb_path", "gbb", {"min_gain": 0.01}),
    ("labels_future_sharpe", "adx", {}),
    ("labels_future_sortino", "adx", {}),
    ("labels_future_expectancy", "adx", {}),
    ("labels_breakout", "breakout", {"min_gain": 0.01}),
    ("labels_breakout_tb", "breakout", {"min_gain": 0.01}),
    ("labels_breakout_vol", "breakout", {"min_gain": 0.01}),
    ("labels_breakout_squeeze", "breakout", {"min_gain": 0.01}),
    ("labels_breakout_gbb", "breakout_gbb", {"min_gain": 0.01}),
])
def test_every_claimed_reduction_CONTAINS_its_labelers_output(labeler, trigger, kwargs):
    """THE proof of the audit. If a label ever fires where its candidate trigger does not,
    the claimed factorisation `mask = trigger AND forward_test` is simply wrong. Verified on
    real data in the audit (SOL/USDT, 164,918 bars, zero violations); pinned here on
    synthetic data so it runs without the feather cache.
    """
    import Framework.TrainingSignals as TS

    B.register_audit_triggers()
    rng = np.random.default_rng(17)
    n = 1200
    close = 100 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame({
        "close": close,
        "open": close,
        "high": close + np.abs(rng.normal(0, 0.6, n)),
        "low": close - np.abs(rng.normal(0, 0.6, n)),
        "volume": rng.uniform(1e4, 1e6, n),
        "guard_metric": rng.uniform(-1.0, 1.0, n),
        "bb_width": rng.uniform(0.0, 0.08, n),
        "rvol": rng.uniform(0.0, 3.0, n),
        "adx_scaled": rng.uniform(-1.0, 1.0, n),
    })
    fn = getattr(TS, labeler)
    label = fn(df, **kwargs).to_numpy() > 0.5
    fired = T.evaluate(trigger, df).to_numpy()
    outside = int((label & ~fired).sum())
    assert outside == 0, (
        f"{labeler} fired on {outside} bars outside its claimed trigger {trigger!r}; "
        f"the factorisation in label-method-audit.md is wrong"
    )
    assert label.sum() > 0, "fixture produced no labels; containment is vacuous"


def test_local_min_followthroughs_condition_is_NOT_causal():
    """The second non-reducible member, and the same failure as geometry: `is_local_min`
    uses a CENTRED window `close[t-k : t+k+1]`, so a bar's eligibility depends on the k bars
    after it. Measured the same way -- perturb the future, watch the past move."""
    from Framework.TrainingSignals import labels_local_min_followthrough

    rng = np.random.default_rng(23)
    n = 300
    close = 100 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame({"close": close, "high": close + 0.5, "low": close - 0.5})
    cut = 200
    before = labels_local_min_followthrough(df, min_gain=0.01).iloc[: cut + 1]
    pert = df.copy()
    pert.loc[cut + 1:, ["close", "high", "low"]] = 50.0
    after = labels_local_min_followthrough(pert, min_gain=0.01).iloc[: cut + 1]
    assert not before.equals(after), (
        "local_min_followthrough now looks causal; if so, re-examine the audit -- a "
        "faithful port may have become possible"
    )
