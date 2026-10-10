"""T045 — attribution statuses and row contents (023 US2, FR-011/FR-012)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation import attribution as A


def _comp(name, metric="trades_hash", neutral=()):
    return SimpleNamespace(name=name, overrides={"sell": {"x": 1}}, liveness_metric=metric,
                           neutral_is_identity_under=tuple(neutral))


def _plan(tag):
    return {"folds": [{"fold_id": f, "arm_tag": f"{tag}_{f}"} for f in ("f00", "f08")]}


def _pooled(profits, prefix):
    per_fold, trades = [], []
    for fid in ("f00", "f08"):
        per_fold.append({"fold_id": fid, "n_trades": len(profits), "test_days": 90,
                         "return_pct": sum(profits) * 0.01, "status": "evaluated"})
        trades += [{"fold_id": fid, "pair": f"P{i}", "open_date": f"{prefix}{fid}{i}",
                    "profit_abs": p, "scale_pp_per_abs": 0.01} for i, p in enumerate(profits)]
    return {"family": "X", "per_fold": per_fold, "trades": trades, "n_trades_pooled": len(trades),
            "folds_complete": 2, "folds_planned": 2, "folds_missing": [], "status": "complete"}


@pytest.fixture
def world(monkeypatch):
    pooled = {"full": _pooled([2.0, -1.0] * 20, "a"),
              "live": _pooled([1.0, -2.0] * 25, "b"),
              "inert": _pooled([2.0, -1.0] * 20, "a"),
              "zero": _pooled([], "c")}
    recs = {"full": [{"trades_hash": "h0", "exit_reason_counts": {"take_profit": 10}},
                     {"trades_hash": "h1", "exit_reason_counts": {"take_profit": 12}}],
            "live": [{"trades_hash": "x0", "exit_reason_counts": {"take_profit": 3}},
                     {"trades_hash": "x1", "exit_reason_counts": {"take_profit": 4}}],
            "inert": [{"trades_hash": "h0"}, {"trades_hash": "h1"}],
            "zero": [{"trades_hash": None}, {"trades_hash": None}]}
    tag = lambda plan: plan["folds"][0]["arm_tag"].split("_")[0]
    monkeypatch.setattr(A, "pool", lambda plan: pooled[tag(plan)])
    monkeypatch.setattr(A, "_records", lambda plan: recs[tag(plan)])
    return pooled


def test_statuses(world):
    comps = [_comp("live", "exit_reason_counts.take_profit"), _comp("inert"), _comp("zero"),
             _comp("guard", neutral=("ARM_D_OPEN",))]
    plans = {"live": _plan("live"), "inert": _plan("inert"), "zero": _plan("zero")}
    rep = A.build_report(_plan("full"), plans, comps, "ARM_D_OPEN")
    st = {r["component"]: r for r in rep["rows"]}
    assert st["inert"]["status"] == "INERT"
    assert st["zero"]["status"] == "ZERO_TRADES"
    assert st["guard"]["status"] == "ALREADY_NEUTRAL" and "arm_tag" not in st["guard"]
    live = st["live"]
    assert live["status"] == "LIVE"
    assert live["liveness_evidence"] == {"metric": "exit_reason_counts.take_profit",
                                         "before": 22, "after": 7, "moved": True}
    for k in ("halves", "fold_detail", "power", "verdict", "component_contribution_pp"):
        assert k in live, k
    assert live["power"]["n_trades"] == 100 and live["power"]["half_width_pp_yr"] > 0
    assert "PROVISIONAL" in rep["label"]


def test_generate_merges_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "STRAT", tmp_path)
    (tmp_path / "SimpleTrigger").mkdir()
    base = type("SimpleTrigger_X", (), {"BUY_OVERRIDES": {"a": 1}, "SELL_OVERRIDES": {},
                                        "ATTR_OVERRIDES": {}})
    comp = SimpleNamespace(name="guard_adx", overrides={"buy": {"entry_adx_threshold": -1.0}})
    src = A.generate(base, comp).read_text()
    assert "class SimpleTrigger_X__C_guard_adx(SimpleTrigger_X)" in src
    assert "BUY_OVERRIDES = {'a': 1, 'entry_adx_threshold': -1.0}" in src
