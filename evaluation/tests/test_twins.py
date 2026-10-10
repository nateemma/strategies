"""T037 — twins check / compare gates (023 contract C5, FR-008/FR-013/FR-014)."""

from __future__ import annotations

import json

import pytest
from evaluation import twins


PLAN = {
    "config": "user_data/strategies/config/config_wf.json",
    "folds": [{"fold_id": "f00"}, {"fold_id": "f01"}],
}


def _write(d, tag, fid, **kw):
    rec = {
        "returncode": 0,
        "strategy": "X",
        "framework_sha": "aaaa",
        "trades_hash": "h",
        "n_trades": 10,
        **kw,
    }
    (d / f"{tag}_{fid}_{fid.upper()}.json").write_text(json.dumps(rec))


@pytest.fixture
def arms(tmp_path, monkeypatch):
    monkeypatch.setattr(twins, "ARMS", tmp_path)
    for tag in ("nn", "st", "eq"):
        for fid in ("f00", "f01"):
            _write(tmp_path, tag, fid, trades_hash=("nnh" if tag == "nn" else f"h{fid}"))
    return tmp_path


def test_pass(arms):
    code, res = twins.check(PLAN, "nn", "st", "eq")
    assert code == 0 and res["passed"] and res["verdict"] == "PASS"


def test_fingerprint_mismatch_exits_2(arms):
    _write(arms, "st", "f01", framework_sha="bbbb", trades_hash="hf01")
    code, res = twins.check(PLAN, "nn", "st", "eq")
    assert code == 2 and res["fingerprint_mismatch_folds"] == ["f01"]


def test_equivalence_mismatch_exits_3(arms):
    _write(arms, "eq", "f00", trades_hash="different")
    code, res = twins.check(PLAN, "nn", "st", "eq")
    assert code == 3 and res["equivalence_mismatch_folds"] == ["f00"]


def test_missing_or_failed_fold_exits_4(arms):
    _write(arms, "eq", "f01", error="backtest FAILED")
    code, res = twins.check(PLAN, "nn", "st", "eq")
    assert code == 4 and res["missing"][0]["fold"] == "f01"


def test_compare_refuses_without_a_passing_check():
    with pytest.raises(PermissionError):
        twins.compare({}, {}, None)
    with pytest.raises(PermissionError):
        twins.compare({}, {}, {"passed": False})


def _pooled(profits_by_fold, key_prefix):
    trades, per_fold = [], []
    for fid, profits in profits_by_fold.items():
        per_fold.append(
            {
                "fold_id": fid,
                "n_trades": len(profits),
                "test_days": 90,
                "return_pct": sum(profits) * 0.01,
                "status": "evaluated",
            }
        )
        trades += [
            {
                "fold_id": fid,
                "pair": f"P{i}",
                "open_date": f"{key_prefix}{fid}{i}",
                "profit_abs": p,
                "scale_pp_per_abs": 0.01,
            }
            for i, p in enumerate(profits)
        ]
    return {
        "family": "X",
        "per_fold": per_fold,
        "trades": trades,
        "n_trades_pooled": len(trades),
        "folds_complete": len(per_fold),
        "folds_planned": len(per_fold),
        "folds_missing": [],
        "status": "complete",
    }


def test_compare_reports_all_sections_and_no_directional_verdict_when_inconclusive(monkeypatch):
    nn = _pooled({"f00": [5.0] * 30, "f08": [5.0] * 30}, "a")
    st = _pooled({"f00": [1.0, -1.0] * 60, "f08": [1.0, -1.0] * 60}, "b")
    monkeypatch.setattr(twins, "pool", lambda plan: nn if plan == "NN" else st)
    res = twins.compare("NN", "ST", {"passed": True})
    for k in (
        "verdict",
        "delta_pp",
        "ci_pp_yr",
        "halves",
        "shared_fraction_of_nnnc",
        "matched_count",
        "label",
    ):
        assert k in res, k
    assert "PROVISIONAL" in res["label"]
    assert res["matched_count"]["pooled"]["z"] > 3
    # an INCONCLUSIVE paired verdict never becomes a directional one (FR-014)
    monkeypatch.setattr(
        twins,
        "compare_data",
        lambda a, c: {"verdict": "INCONCLUSIVE_UNDERPOWERED", "bootstrap": {"shared": 0}},
    )
    res2 = twins.compare("NN", "ST", {"passed": True})
    assert res2["directional_verdict"] is None
