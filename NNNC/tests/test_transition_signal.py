import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _transition_signal as TS  # noqa: E402

DATES = pd.date_range("2023-01-01", periods=6, freq="15min", tz="UTC")


@pytest.fixture
def tables(tmp_path, monkeypatch):
    tail = pd.DataFrame({"date": pd.date_range("2022-10-01", periods=100, freq="h", tz="UTC"), "pair": "A/USDT",
                         "part": "tail", "p": np.linspace(0.0, 0.99, 100)})
    test = pd.DataFrame({"date": DATES[:5], "pair": "A/USDT", "part": "test", "p": [0.1, 0.95, 0.5, 0.85, 0.99]})
    t = pd.concat([tail.assign(cell=c) for c in ("up_x", "down_x")] + [test.assign(cell=c) for c in ("up_x", "down_x")])
    t.to_parquet(tmp_path / "pred_f03.parquet")
    monkeypatch.setattr(TS.TransitionSignalMixin, "TR_TABLES", tmp_path)
    return tmp_path


class _Book:
    curr_pair = "A/USDT"

    def get_entry_conditions(self, df):
        self._st_tags = np.array(["gbb:model:agree"] * len(df), dtype=object)
        return pd.Series(True, index=df.index)

    def stake_weight(self, tag):
        return 2.0 if str(tag).endswith(":agree") else 1.0


def _cls(mixin, **kw):
    return type("X_f03", (mixin, _Book), {"TR_UP_CELL": "up_x", "TR_DOWN_CELL": "down_x", **kw})()


def test_threshold_is_tail_percentile_and_missing_bar_never_fires(tables):
    s = _cls(TS.TrGateMixin)
    assert s._tr_threshold("down_x", 80) == pytest.approx(np.percentile(np.linspace(0, 0.99, 100), 80))
    fires = s._tr_fires("down_x", 80, "A/USDT", DATES)
    assert fires.tolist() == [False, True, False, True, True, False]  # 6th bar has no row


def test_gate_skips_high_down_bars(tables):
    df = pd.DataFrame({"date": DATES})
    assert _cls(TS.TrGateMixin).get_entry_conditions(df).tolist() == [True, False, True, False, False, True]


def test_scale_tags_and_weights(tables):
    s = _cls(TS.TrScaleMixin, TR_UP_CELL="up_x", TR_DOWN_CELL=None)
    s.get_entry_conditions(pd.DataFrame({"date": DATES}))
    assert s._st_tags[1] == "gbb:model:agree|tr_up" and s._st_tags[0] == "gbb:model:agree"
    assert s.stake_weight(s._st_tags[1]) == pytest.approx(2.0 * 4 / 3)
    assert s.stake_weight("gbb:model:agree|tr_down") == pytest.approx(2.0 * 2 / 3)
    assert s.stake_weight("gbb:model:disagree") == 1.0


def test_scale_both_sides_firing_is_unchanged(tables):
    s = _cls(TS.TrScaleMixin)  # same table for up and down: both fire together
    s.get_entry_conditions(pd.DataFrame({"date": DATES}))
    assert all(not t.endswith(("|tr_up", "|tr_down")) for t in s._st_tags)


def test_long_entries_at_p99(tables):
    s = _cls(TS.TrLongMixin)
    m = s.get_entry_conditions(pd.DataFrame({"date": DATES}))
    assert m.tolist() == [False, False, False, False, True, False]
    assert s._st_tags[4] == "reversion:tr_up"


def test_fold_id_from_class_name(tables):
    with pytest.raises(ValueError):
        type("NoFold", (TS.TrGateMixin, _Book), {})()._tr_fold()


def test_tr_tables_redirects_to_the_seed_rerun_tables(tables, tmp_path_factory):
    rs1 = tmp_path_factory.mktemp("rs1")
    t = pd.read_parquet(tables / "pred_f03.parquet")
    t.assign(p=1.0 - t["p"]).to_parquet(rs1 / "pred_f03.parquet")
    a, b = _cls(TS.TrGateMixin), _cls(TS.TrGateMixin, TR_TABLES=rs1)
    pa, pb = a._tr_p("down_x", "A/USDT", DATES[:5]), b._tr_p("down_x", "A/USDT", DATES[:5])
    np.testing.assert_allclose(pb, 1.0 - pa)
