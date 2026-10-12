"""027 T004: early-only choice, both-halves rules, N/A, ambiguity, exit codes, fidelity gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from research.exit_coupling import feasibility as F


def _df(early, late, coupling="C3", values=("close", "breakeven")):
    """Rows with per-value deltas; `early`/`late` = {value: delta per row}, 4 rows per half."""
    rows = []
    for half, d in (("early", early), ("late", late)):
        for _ in range(4):
            rows.append({"half": half, **{f"d_{coupling}_{v}": d[v] for v in values}})
    return pd.DataFrame(rows)


def test_choice_reads_early_rows_only_planted_late_effect_ignored():
    df = _df({"close": 1.0, "breakeven": 0.5}, {"close": -100.0, "breakeven": 100.0})
    assert F.choose(df, "C3", ["close", "breakeven"]) == "close"


def test_c4_depth_choice_is_early_only_too():
    df = _df({"base": 0.1, "p95": 0.2}, {"base": 50.0, "p95": -50.0}, "C4", ("base", "p95"))
    assert F.choose(df, "C4", ["base", "p95"]) == "p95"


def test_rule_holding_in_one_half_only_is_not_supported():
    assert F.supported({"early": [True, True], "late": [True, True]})
    assert not F.supported({"early": [True, True], "late": [True, False]})
    assert not F.supported({"early": [False], "late": [True]})


def _summ(killed=0.3, kag=0.6, kag_lp=0.2):
    return {h: {"b_killed_share": killed, "b_killed_after_grace_share_of_killed": kag,
                "b_killed_after_grace_share_of_lp": kag_lp} for h in ("early", "late")}


def _vdf(gains: dict):
    """A replay frame with the given per-half delta for every coupling column."""
    cols = {f"d_C1_{p}": None for p in F.GRID["C1"]} | {f"d_C3_{a}": None for a in F.GRID["C3"]}
    cols |= {f"d_C4_{d}": None for d in F.GRID["C4"]} | {f"d_C5_{x}": None for x in F.GRID["C5"]}
    cols |= {"d_C2_k": None}
    rows = []
    for half, g in gains.items():
        rows.append({"half": half, "atr_signal": 0.01, "touch_h": 1.0, **{c: g for c in cols}})
    return pd.DataFrame(rows)


PROF = {"C1": {80: -0.05, 90: -0.07}, "C4": {"base": None, "p95": -0.1}, "C2": None,
        "atr_tercile_cuts": (0.0, 1.0)}


def test_c2_is_na_without_a_take_profit():
    v = F.verdicts(_vdf({"early": 1.0, "late": 1.0}), PROF, _summ())
    assert v["C2"]["verdict"] == "N/A"


def test_c2_and_c4_need_a_positive_replay_gain_as_well():
    prof = {**PROF, "C2": 5.0}
    df = _vdf({"early": 1.0, "late": -1.0})
    df.loc[df["half"] == "early", "atr_signal"] = 0.0
    v = F.verdicts(df, prof, _summ())
    assert v["C4"]["verdict"] == "NOT SUPPORTED"  # descriptive holds, late gain < 0
    assert v["C2"]["verdict"] == "NOT SUPPORTED"
    v = F.verdicts(_vdf({"early": 1.0, "late": 1.0}), PROF, _summ(kag=0.2))
    assert v["C4"]["verdict"] == "NOT SUPPORTED"  # gain > 0, descriptive fails
    v = F.verdicts(_vdf({"early": 1.0, "late": 1.0}), PROF, _summ())
    assert v["C4"]["verdict"] == "SUPPORTED" and v["C1"]["verdict"] == "SUPPORTED"


def test_ambiguous_rows_are_excluded_from_a_and_b_and_counted():
    base = {"half": "early", "e": 0, "rec_idx": 10, "rec_pp": 0.0, "rec_profit": 0.0,
            "target_net": 0.018, "touch_m": np.nan, "touch_h_after": True,
            "touch_m_after": True, "stop_at_touch": -0.04, "breach_after_grace": True}
    df = pd.DataFrame([
        {**base, "ambiguous": True, "touch_h": 5.0, "mae": -0.05, "breach": 3.0,
         "rec_reason": "stop"},
        {**base, "ambiguous": False, "touch_h": 5.0, "mae": -0.01, "breach": np.nan,
         "rec_reason": "take_profit"},
    ])
    s = F.summarise(df)
    assert s["ambiguous"] == 1 and s["label_positive"] == 1
    assert s["a_stop_exits"] == 0 and s["b_killed_share"] == 0.0


def test_exit_codes():
    sup = {"x": {"verdict": "SUPPORTED"}}
    nos = {"x": {"verdict": "NOT SUPPORTED"}}
    assert F.exit_code(True, True, sup) == 0
    assert F.exit_code(True, True, nos) == 3
    assert F.exit_code(False, True, sup) == 2


def test_failed_fidelity_gate_gives_no_verdict():
    assert F.exit_code(True, False, {"x": {"verdict": "SUPPORTED"}}) == 2


def test_fidelity_counts_reason_and_bar():
    df = pd.DataFrame({
        "rec_reason": ["stop", "stop", "take_profit"], "rec_idx": [5, 6, 7],
        "null_reason": ["stop", "stop", "unclog_12h"], "null_idx": [5, 9, 7],
        "model_stop_at_exit": [96.0, 96.0, 0.0], "rec_stop_abs": [96.0, 95.0, 0.0],
        "open_rate": [100.0, 100.0, 100.0],
    })
    f = F.fidelity(df)
    assert f["null_replay_agreement"] == 1 / 3 and f["stop_model_agreement"] == 0.5
