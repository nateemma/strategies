"""T031 / SC-001 — the SimpleTrigger family reproduces every recorded 021/022 no-model arm exactly.

A DATA test over `regime/arms/`: for each legacy arm, the per-fold `trades_hash` of the
SimpleTrigger run (`stl_*`, LEGACY_TRIGGERONLY preset) must equal the recorded one (`wf_*`) in
every fold. Skips, never passes, when a run's records are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ARMS = Path(__file__).resolve().parents[2] / "regime/arms"
FOLDS = [f"f{i:02d}" for i in range(15)]
PAIRS = [
    ("stl_bo", "wf_bo"),
    ("stl_bvo", "wf_bvo"),
    ("stl_bbo", "wf_bbo"),
    ("stl_bgo", "wf_bgo"),
    ("stl_geo", "wf_geo"),
    ("stl_gbbo", "wf_gbbo"),
    ("stl_gbbr", "wf_gbbr"),
]


def _rec(tag, fid):
    p = ARMS / f"{tag}_{fid}_{fid.upper()}.json"
    return json.loads(p.read_text()) if p.is_file() else None


@pytest.mark.parametrize("new,old", PAIRS)
def test_trade_sets_identical_in_every_fold(new, old):
    recs = [(_rec(new, f), _rec(old, f)) for f in FOLDS]
    if any(n is None for n, _ in recs):
        pytest.skip(f"{new} has not been run on every fold")
    diffs = [
        (f, n.get("trades_hash"), o.get("trades_hash"), n.get("n_trades"), o.get("n_trades"))
        for f, (n, o) in zip(FOLDS, recs, strict=True)
        if n.get("trades_hash") != o.get("trades_hash")
    ]
    assert not diffs, f"{new} differs from {old}: {diffs}"
    assert all(n.get("returncode") == 0 and not n.get("error") for n, _ in recs)
    assert all(
        n.get("no_model") is True and n.get("preset") == "LEGACY_TRIGGERONLY" for n, _ in recs
    )


def test_wf_gbbr_totals_458_trades():
    recs = [_rec("stl_gbbr", f) for f in FOLDS]
    if any(r is None for r in recs):
        pytest.skip("stl_gbbr has not been run on every fold")
    assert sum(r.get("n_trades") or 0 for r in recs) == 458


def test_no_model_artefacts_were_created():
    saved = Path(__file__).resolve().parents[2] / "saved_data"
    assert not list(saved.glob("SimpleTrigger*")), "a SimpleTrigger run created model artefacts"
