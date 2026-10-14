"""034 T013: the staged judge -- stage order, two roles, winner choice, seed-2, refusal."""

from __future__ import annotations

import os

from research.label_screen import judge as J


F = [f"f{i:02d}" for i in range(15)]
PR = {"label": "x", "early": F[:8], "late": F[8:], "small_margin_pp": 1.0,
      "confirmation_route": "post-2026-09-07 folds or dry-run", "exempt_records": [],
      "comparators": {"ARO": {"width": "w_aro", "control": "wf_aro_g24"},
                      "GBR": {"width": "w_gbr", "control": "fs_gbr_gr"}}}


def _pool(pp: dict, read: list | None = None):
    def f(tag):
        if read is not None:
            read.append(tag)
        return {"folds_missing": [], "per_fold": [{"fold_id": k, "return_pct": pp[tag]} for k in F]}
    return f


def _raise(*a, **k):
    raise AssertionError("a later stage's records were read")


OKI = {"S0_pass": True, "S0_inconclusive": False, "zero_trade_folds": []}


def test_candidate_stage_order():
    audit = {"candidates": {"A": {"status": "INCONCLUSIVE", "reason": "floor"},
                            "B": {"status": "PASS"}, "C": {"status": "PASS"}}}
    inst = {"candidates": {"B": {"S0_pass": True}, "C": {"S0_pass": True}}}
    sep = {"references": {}, "candidates": {"B": {"S1_pass": False, "gap": 0.01},
                                            "C": {"S1_pass": True, "gap": 0.2}}}
    surv = {"candidates": {"C": {"survives": True, "reason": "S1 pass"}}}
    a = J.judge_candidate("A", audit, inst, sep, surv)
    assert a["stopped_at"] == "S0a" and a["S0"] == J.NR
    b = J.judge_candidate("B", audit, inst, sep, surv)
    assert b["decision"] == "FAIL" and b["stopped_at"] == "S1" and b["survivor"] == J.NR
    assert J.judge_candidate("C", audit, inst, sep, surv)["decision"] == "SURVIVOR"


def test_s2_needs_both_comparators_and_reads_nothing_after_an_instrument_failure():
    comps = PR["comparators"]["ARO"]
    pp = {"a": 3.0, "w_aro": 1.0, "wf_aro_g24": 2.0}
    assert J.judge_s2(PR, "a", comps, OKI, _pool(pp))["decision"] == "S2_PASS"
    assert J.judge_s2(PR, "a", comps, OKI, _pool(pp | {"a": 1.5}))["decision"] == "FAIL"
    r = J.judge_s2(PR, "a", comps, OKI | {"S0_pass": False}, _raise)
    assert r["decision"] == "INCONCLUSIVE" and r["S2"] == J.NR


def test_small_margin_seed2_both_ways():
    comps = PR["comparators"]["ARO"]
    pp = {"a": 2.1, "w_aro": 1.0, "wf_aro_g24": 2.0}  # 8 x 0.1 = 0.8 pp over control
    assert J.judge_s2(PR, "a", comps, OKI, _pool(pp))["decision"] == "INCONCLUSIVE"
    good = lambda arm: {"margins": {"vs_width": 3.0, "vs_control": 0.2}}
    bad = lambda arm: {"margins": {"vs_width": 3.0, "vs_control": -0.2}}
    assert J.judge_s2(PR, "a", comps, OKI, _pool(pp), good)["decision"] == "S2_PASS"
    assert J.judge_s2(PR, "a", comps, OKI, _pool(pp), bad)["decision"] == "FAIL"


def test_winner_is_the_largest_minimum_margin():
    s2 = {"x": {"decision": "S2_PASS", "min_margin": 2.0}, "y": {"decision": "S2_PASS", "min_margin": 5.0},
          "z": {"decision": "FAIL"}}
    assert J.pick_winner(s2) == "y" and J.pick_winner({"z": {"decision": "FAIL"}}) is None


def test_judge_all_two_roles_bog_sizer_only_and_late_read_only_for_the_winner():
    audit = {"candidates": {t: {"status": "PASS"} for t in
                            ("GFM", "FMAE", "TB", "MMR", "FEXP", "QF", "LMF", "BOG")}}
    inst = {"candidates": {t: {"S0_pass": True} for t in audit["candidates"]},
            "arms": {a: OKI for a in ("ls_fmae_sz", "ls_fmae_es", "ls_bog_sz")}}
    sep = {"references": {}, "candidates": {t: {"S1_pass": False, "gap": 0.0}
                                            for t in audit["candidates"]}}
    sep["candidates"]["FMAE"] = sep["candidates"]["BOG"] = {"S1_pass": True, "gap": 0.1}
    surv = {"survivors": ["FMAE", "BOG"],
            "candidates": {"FMAE": {"survives": True, "reason": ""}, "BOG": {"survives": True, "reason": ""}}}
    pp = {"ls_fmae_sz": 4.0, "ls_fmae_es": 9.0, "ls_bog_sz": 3.0, "w_aro": 1.0, "wf_aro_g24": 2.0}
    read = []
    calls = []
    jo = lambda pr, a, c: calls.append((a, c)) or {"verdict": "FAIL", "late_gain_pp": -1.0}
    out = J.judge_all(PR, audit, inst, sep, surv, _pool(pp, read), jo)
    assert set(out["arms"]) == {"ls_fmae_sz", "ls_fmae_es", "ls_bog_sz"}
    assert out["s2_winner"]["arm"] == "ls_fmae_es"
    assert calls == [("ls_fmae_es", "wf_aro_g24")]  # judge_one only on the winner
    assert out["arms"]["ls_fmae_es"]["decision"] == "FAIL" and "gbr_repeat" not in out


def test_refuses_a_record_older_than_the_commit(tmp_path):
    p = tmp_path / "ls_fmae_sz_f02_F02.json"
    p.write_text("{}")
    os.utime(p, (1000, 1000))
    assert J.refusals(PR, commit_ts=2000.0, arms_dir=tmp_path) == [
        "record older than the pre-registration commit: ls_fmae_sz_f02_F02.json"]
    (tmp_path / "ls_fmae_sz_f00_F00.smoke.json").write_text("{}")
    os.utime(p, (3000, 3000))
    assert J.refusals(PR, commit_ts=2000.0, arms_dir=tmp_path) == []
