"""035 T011: NewInputMixin -- off is a no-op, existing keys unchanged, survivor key distinct, the
control's scaler untouched, merge keeps rows, stale cache refused, end-to-end causality."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


STRAT = Path(__file__).resolve().parents[3]
for p in (STRAT, STRAT / "NNNC"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import _new_inputs as NI  # noqa: E402
from research.new_inputs import build as B  # noqa: E402
from research.new_inputs import inputs as I  # noqa: E402


KEYS = STRAT / "research" / "label_screen" / "tests" / "fixtures" / "model_keys_pre034.json"


class _Parent:
    curr_pair = "P0/USDT"

    def add_additional_indicators(self, df):
        df["regime"] = 1.0
        return df


def _cls(tags, sha=I.TABLE_SHA):
    cols = tuple(c for t in tags for c in I.columns(t))
    return type("T", (NI.NewInputMixin, _Parent), {"NEW_INPUTS": tuple(tags), "NEW_INPUTS_SHA": sha,
                                                   "NEW_INPUT_COLUMNS": cols,
                                                   "_new_input_cache": None})


def _write_cache(tmp: Path, built: dict, sha=I.TABLE_SHA):
    for p, df in built.items():
        df.to_parquet(tmp / f"{I.file_stem(p)}.parquet", index=False)
    (tmp / "build_meta.json").write_text(json.dumps({"table_sha": sha}))


def test_off_returns_the_parents_frame():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")})
    out = _cls(())().add_additional_indicators(df.copy())
    assert list(out.columns) == ["date", "regime"]


def test_existing_model_keys_unchanged():
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    cfg = load_config(STRAT / "config" / "config_wf.json")
    keys = json.loads(KEYS.read_text())
    for name in sorted(keys)[:6]:
        assert MK.key_hash(MK.resolve(load_strategy_class(name), cfg)) == keys[name], name


def test_survivor_class_extends_inputs_and_keeps_the_control_scaler():
    import pickle

    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    ctl = load_strategy_class("NNNC_A_Raw_Open")
    cols = I.columns("PO_OI")
    surv = type("NNNC_NI_TEST", (NI.NewInputMixin, ctl), {
        "NEW_INPUTS": ("PO_OI",), "NEW_INPUTS_SHA": I.TABLE_SHA, "NEW_INPUT_COLUMNS": tuple(cols),
        "include_list": list(ctl.include_list) + cols,
        "pre_normalized_columns": list(ctl.pre_normalized_columns) + cols,
        "__module__": ctl.__module__})
    assert surv.include_list[:len(ctl.include_list)] == ctl.include_list
    assert set(surv.include_list) - set(ctl.include_list) == set(cols)
    off = type("NNNC_NI_OFF", (NI.NewInputMixin, ctl), {"__module__": ctl.__module__})
    assert off.include_list == ctl.include_list
    assert off.pre_normalized_columns == ctl.pre_normalized_columns
    scaled = {c for c in surv.include_list if c not in surv.pre_normalized_columns}
    with (STRAT / "saved_data" / "main_scaler.pkl").open("rb") as fh:
        sc = pickle.load(fh)  # noqa: S301 -- the repo's own scaler, not downloaded data
    assert set(sc.feature_names_in_) == scaled  # the new columns never reach the scaler
    cfg = load_config(STRAT / "config" / "config_wf.json")
    assert MK.key_hash(MK.resolve(surv, cfg)) != MK.key_hash(MK.resolve(ctl, cfg))


def test_merge_keeps_rows_on_a_frame_with_gaps(tmp_path, monkeypatch, synth):
    built = B.build_all(synth)
    _write_cache(tmp_path, built)
    monkeypatch.setattr(NI, "CACHE_DIR", tmp_path)
    o = synth["ohlcv"]["P0/USDT"].iloc[::3].reset_index(drop=True)  # a frame with holes
    out = _cls(("OF_TBG", "PO_OI"))().add_additional_indicators(o[["date"]].copy())
    assert len(out) == len(o)
    ref = built["P0/USDT"].set_index("date").loc[o["date"], "ni_po_oi_missing"].to_numpy()
    assert np.array_equal(out["ni_po_oi_missing"].to_numpy(), ref)


def test_stale_or_missing_cache_is_refused(tmp_path, monkeypatch, synth):
    monkeypatch.setattr(NI, "CACHE_DIR", tmp_path)
    df = pd.DataFrame({"date": synth["ohlcv"]["P0/USDT"]["date"][:5]})
    with pytest.raises(ValueError, match="no new-input cache"):
        _cls(("OF_TBG",))().add_additional_indicators(df.copy())
    _write_cache(tmp_path, {"P0/USDT": B.build_all(synth)["P0/USDT"]}, sha="other")
    with pytest.raises(ValueError, match="input table"):
        _cls(("OF_TBG",))().add_additional_indicators(df.copy())


def test_end_to_end_causality_through_the_merge(tmp_path, monkeypatch, synth):
    full = B.build_all(synth)
    cut = synth["ohlcv"]["P0/USDT"]["date"].iloc[3500]
    _write_cache(tmp_path, B.build_all(synth, cut))
    monkeypatch.setattr(NI, "CACHE_DIR", tmp_path)
    tags = tuple(i.tag for i in I.TABLE)
    o = synth["ohlcv"]["P0/USDT"]
    out = _cls(tags)().add_additional_indicators(o.loc[o["date"] <= cut, ["date"]].copy())
    got = out.iloc[-1][list(I.all_columns())].to_numpy(float)
    ref = full["P0/USDT"].set_index("date").loc[cut, list(I.all_columns())].to_numpy(float)
    assert np.allclose(got, ref, equal_nan=True, atol=1e-12)
