"""Path bank provenance and the identity-resample equivalence (T014, T015, T019)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.stress import bank as BK               # noqa: E402
from research.labelfree.stress import bootstrap as BS          # noqa: E402

SINK = Path("/tmp/lf/es_sink/f07_test")
needs_panel = pytest.mark.skipif(
    not SINK.exists(),
    reason="needs the f07 side-car: es.signals --folds f07 --which test")


# ------------------------------------------------------------------ T014: THE DECISIVE TEST

@needs_panel
def test_bank_and_fold_panel_score_identically():
    """A DEGENERATE resample — the identity index — must reproduce the real-panel result exactly.

    This is the test the whole bank rests on. If the resampled panel's shape or column set has
    drifted from what the surrogate expects, every downstream number is meaningless while
    everything still runs — the same class of failure as feature 013's four hand-found fidelity
    bugs, none of which failed a self-consistent unit test.
    """
    from research.labelfree.es import evaluate as EV
    from research.labelfree.es import panel as PA
    from research.labelfree.es import space as SP

    real = PA.load("f07", train=False)
    n = max(len(d["open"]) for d in real["pairs"].values())
    identity = np.arange(n, dtype=np.int64)
    synth = BS.resample_panel(real, identity)

    params = dict(SP.INCUMBENT)
    a = EV.run(real, params)
    b = EV.run(synth, params)

    assert len(a) == len(b), f"trade count differs: real {len(a)} vs identity-resample {len(b)}"
    assert np.allclose(a.pnl, b.pnl, atol=1e-12), "per-trade P&L differs under the identity"
    assert a.reason == b.reason, "exit reasons differ under the identity"
    assert abs(a.annualised_pp - b.annualised_pp) < 1e-9


@needs_panel
def test_resample_preserves_every_column_the_surrogate_reads():
    from research.labelfree.es import panel as PA
    real = PA.load("f07", train=False)
    n = max(len(d["open"]) for d in real["pairs"].values())
    synth = BS.resample_panel(real, np.arange(n, dtype=np.int64))
    for pair, d in synth["pairs"].items():
        for col in ("open", "high", "low", "cost", "rvol_stop", "atr_pct_roll",
                    "rvol", "guard_metric", "close_norm", "adx", "bb_width",
                    "p_buy", "p_sell", "p_hold"):
            assert col in d and d[col] is not None, f"{pair} lost {col}"


# --------------------------------------------------------------------- provenance / staleness

def _fake_panels():
    n = 400
    px = np.linspace(100, 110, n)
    d = {c: np.full(n, 1.0) for c in BS.PANEL_COLUMNS}
    d.update(open=px, high=px, low=px, close=px)
    return {"f00": {"pairs": {"A/USDT": dict(d)}, "span_years": 0.1}}


def test_provenance_records_what_a_stale_bank_needs():
    p = _fake_panels()
    prov = BK.provenance(["f00"], 192, 7, 10, p)
    for k in ("generator", "block_length", "seed", "fold_ids", "n_paths",
              "feature_columns", "source_sha256", "shared_block_index", "created"):
        assert k in prov, f"provenance lacks {k}"
    assert prov["shared_block_index"] is True
    assert prov["generator"] == "stationary_block_bootstrap"


def test_source_hash_changes_when_the_panel_moves():
    """The staleness detector must actually detect. A hash that ignored the data would pass
    every check and let a bank silently stress the wrong thing."""
    a = _fake_panels()
    h1 = BK.source_hash(a)
    a["f00"]["pairs"]["A/USDT"]["open"] = a["f00"]["pairs"]["A/USDT"]["open"] + 1.0
    assert BK.source_hash(a) != h1


def test_source_hash_is_order_independent():
    a, b = _fake_panels(), _fake_panels()
    b["f00"]["pairs"] = dict(reversed(list(b["f00"]["pairs"].items())))
    assert BK.source_hash(a) == BK.source_hash(b)


@needs_panel
def test_moved_source_is_refused_not_warned(tmp_path):
    """FR-002: a bank whose source has moved is REFUSED."""
    out = BK.build(["f07"], block_len=192, n_paths=2, seed=7, out=tmp_path / "b.npz")
    prov, idx, panels = BK.load(out)               # clean load
    assert prov["n_paths"] == 2 and idx["f07"].shape[0] == 2

    import json
    z = dict(np.load(out, allow_pickle=False))
    bad = json.loads(str(z["provenance"]))
    bad["source_sha256"] = "0" * 64
    z["provenance"] = json.dumps(bad)
    np.savez_compressed(tmp_path / "stale.npz", **z)
    with pytest.raises(SystemExit, match="STALE BANK"):
        BK.load(tmp_path / "stale.npz")
