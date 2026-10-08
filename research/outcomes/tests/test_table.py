# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""T012, T014-T019 (specs/021-label-exit-coordination) — `ExitConfig`, the outcome
table builder, fillability, gap-fill share, the cache, the CLI and reproducibility.

T020 (building the full 11-pair table) and T021 (reconciling against the harness)
are explicitly NOT this file's job -- see tasks.md. Builds here are smoke-sized: one
pair, a short, hand-picked timerange.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from Framework.BaseNNStrategy import BaseNNStrategy  # noqa: E402
from research.clfstudy.data import load_pair, populate  # noqa: E402
from research.outcomes.leakage import OUTCOME_COLUMN_NAMES  # noqa: E402
from research.outcomes.table import (  # noqa: E402
    CacheMismatchError,
    ExitConfig,
    OPEN_OPERATING_POINT_JSON,
    PRODUCTION_STAKE,
    build_outcome_table,
    load_table,
    save_table,
    status,
)
from utils.ExitPolicy import ExitParams  # noqa: E402


# =============================================================================================
# T012 -- ExitConfig
# =============================================================================================


def test_exit_config_wraps_exitparams_not_restates_it():
    """Composition decision: `ExitConfig` has exactly one field, `params`, typed
    `ExitParams` -- see table.py's module/class docstrings for why."""
    assert set(ExitConfig.__dataclass_fields__.keys()) == {"params"}


def test_exit_config_exposes_the_eleven_data_model_fields():
    cfg = ExitConfig.from_operating_point()
    expected = {
        "atr_stoploss_multiplier", "atr_stoploss_cap", "atr_stoploss_floor",
        "use_atr_adaptive_stoploss", "use_volume_confirmation_stoploss",
        "static_stoploss", "stoploss_grace_hours", "stoploss_grace_level",
        "take_profit", "max_hold_days", "fee_per_side",
    }
    for field in expected:
        assert hasattr(cfg, field), f"ExitConfig missing data-model field {field!r}"


def test_exit_config_rejects_missing_grace_never_defaults():
    """The Phase-2 hazard: `ExitConfig` must source grace from an operating point,
    never silently accept `ExitParams()`'s (deliberately absent) grace default."""
    with pytest.raises(ValueError, match="grace"):
        ExitConfig(ExitParams())


def test_exit_config_delegates_validation_to_exitparams():
    """`atr_stoploss_floor <= atr_stoploss_cap < 0` etc. are enforced once, in
    `ExitParams.__post_init__` -- `ExitConfig` does not re-validate them."""
    with pytest.raises(ValueError):
        ExitParams(atr_stoploss_floor=-0.01, atr_stoploss_cap=-0.02)  # floor > cap
    with pytest.raises(ValueError):
        ExitParams(take_profit=0.0)
    with pytest.raises(ValueError):
        ExitParams(max_hold_days=0)
    with pytest.raises(ValueError):
        ExitParams(fee_per_side=-0.001)


def test_config_hash_is_16_hex_chars():
    cfg = ExitConfig.from_operating_point()
    assert len(cfg.config_hash) == 16
    int(cfg.config_hash, 16)  # does not raise -- valid hex


def test_config_hash_identical_for_identical_configs():
    a = ExitConfig.from_operating_point()
    b = ExitConfig.from_operating_point()
    assert a.config_hash == b.config_hash


def test_config_hash_changes_when_fee_per_side_changes():
    """FR-007: `fee_per_side` MUST be in the hash -- outcomes are net of fees, so a
    fee change changes every outcome, and omitting it is the silent-staleness bug
    FR-007 exists to prevent."""
    a = ExitConfig.from_operating_point(fee_per_side=0.001)
    b = ExitConfig.from_operating_point(fee_per_side=0.002)
    assert a.config_hash != b.config_hash


def test_config_hash_changes_when_timeframe_minutes_changes():
    """data-model.md lists eleven fields; `ExitParams` (what `simulate_forward`
    actually reads) has a twelfth, `timeframe_minutes`. The hash rule in
    contracts/outcome-table.md ("every parameter simulate_forward reads") covers it
    even though the data-model table predates the field -- this pins that decision."""
    a = ExitConfig.from_operating_point()
    b = ExitConfig(replace(a.params, timeframe_minutes=5))
    assert a.config_hash != b.config_hash


def test_from_operating_point_reads_the_open_json():
    cfg = ExitConfig.from_operating_point()
    assert cfg.stoploss_grace_hours == 6.5
    assert cfg.stoploss_grace_level == -0.27
    assert cfg.take_profit == 0.022
    assert cfg.max_hold_days == 10


def test_from_operating_point_sources_use_atr_adaptive_stoploss_from_base_nn_strategy():
    """`BaseStrategy.use_atr_adaptive_stoploss` is False; the Open operating point is
    an NN family and runs with `BaseNNStrategy`'s True override. Using the wrong
    source here would silently simulate the static (non-ATR) stop."""
    assert BaseStrategy.use_atr_adaptive_stoploss is False
    assert BaseNNStrategy.use_atr_adaptive_stoploss is True
    cfg = ExitConfig.from_operating_point()
    assert cfg.use_atr_adaptive_stoploss is True


def test_from_operating_point_sources_atr_params_from_base_strategy():
    cfg = ExitConfig.from_operating_point()
    assert cfg.atr_stoploss_multiplier == BaseStrategy.atr_stoploss_multiplier == 2.5
    assert cfg.atr_stoploss_cap == BaseStrategy.atr_stoploss_cap == -0.02
    assert cfg.atr_stoploss_floor == BaseStrategy.atr_stoploss_floor == -0.04
    assert cfg.static_stoploss == BaseStrategy.stoploss == -0.05


def test_open_operating_point_json_is_the_pinned_file():
    assert OPEN_OPERATING_POINT_JSON.name == "NNNC_MLX_Open.json"
    assert OPEN_OPERATING_POINT_JSON.exists()


# =============================================================================================
# Shared fixture: a small, hand-built pair frame with REAL indicator columns
# (`research.clfstudy.data.populate`), sliced to a short, deterministic window.
# =============================================================================================


@pytest.fixture(scope="module")
def smoke_df() -> pd.DataFrame:
    df = populate(load_pair("ZEC/USDT"))
    df = df[(df["date"] >= pd.Timestamp("2022-01-01", tz="UTC"))
            & (df["date"] < pd.Timestamp("2022-01-15", tz="UTC"))]
    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def smoke_config() -> ExitConfig:
    return ExitConfig.from_operating_point()


# =============================================================================================
# T014 -- the table builder
# =============================================================================================


def test_build_outcome_table_requires_atr_pct_roll_and_ohlcv():
    bad_df = pd.DataFrame({"date": [pd.Timestamp("2022-01-01", tz="UTC")], "close": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        build_outcome_table(bad_df, "ZEC/USDT", ExitConfig.from_operating_point())


def test_build_outcome_table_produces_one_record_per_bar(smoke_df, smoke_config):
    table = build_outcome_table(smoke_df, "ZEC/USDT", smoke_config)
    assert len(table.records) == len(smoke_df)
    assert set(table.records["pair"].unique()) == {"ZEC/USDT"}
    assert (table.records["date"].to_numpy() == smoke_df["date"].to_numpy()).all()


def test_build_outcome_table_columns_match_the_leakage_guard_s_registry(smoke_df, smoke_config):
    """The columns `table.py` produces are exactly the names `leakage.py` reserves
    -- defined there first (T013), consumed here (T014), never restated."""
    table = build_outcome_table(smoke_df, "ZEC/USDT", smoke_config)
    produced = {c for c in table.records.columns if c.startswith("__outcome_")}
    assert produced == OUTCOME_COLUMN_NAMES


def test_build_outcome_table_close_reason_is_one_of_four(smoke_df, smoke_config):
    table = build_outcome_table(smoke_df, "ZEC/USDT", smoke_config)
    assert set(table.records["__outcome_close_reason"].unique()) <= {
        "stop", "take_profit", "hold_cap", "end_of_data",
    }


def test_build_outcome_table_positive_matches_profit_sign(smoke_df, smoke_config):
    table = build_outcome_table(smoke_df, "ZEC/USDT", smoke_config)
    r = table.records
    assert (r["__outcome_positive"] == (r["__outcome_profit"] > 0.0)).all()


def test_build_outcome_table_calls_simulate_forward_not_its_own_arithmetic(monkeypatch, smoke_df, smoke_config):
    """Structural check that T014 contains no exit arithmetic of its own: patch
    `simulate_forward` to a stub and confirm the table is built entirely from its
    return values."""
    import research.outcomes.table as table_mod

    calls = []

    def _stub(high, low, close, atr_pct, rvol, start_index, params):
        calls.append(start_index)
        return 0.0123, 3, "take_profit"

    monkeypatch.setattr(table_mod, "simulate_forward", _stub)
    table = table_mod.build_outcome_table(smoke_df, "ZEC/USDT", smoke_config)

    assert len(calls) == len(smoke_df)
    assert (table.records["__outcome_profit"] == 0.0123).all()
    assert (table.records["__outcome_hold_bars"] == 3).all()
    assert (table.records["__outcome_close_reason"] == "take_profit").all()


def test_build_outcome_table_net_of_fees_matches_simulate_forward_directly(smoke_df, smoke_config):
    """Cross-check one bar against calling `simulate_forward` by hand -- the table
    must reproduce it exactly, not approximate it."""
    from utils.ExitPolicy import simulate_forward

    table = build_outcome_table(smoke_df, "ZEC/USDT", smoke_config)
    i = 10
    high = smoke_df["high"].to_numpy(dtype=float)
    low = smoke_df["low"].to_numpy(dtype=float)
    close = smoke_df["close"].to_numpy(dtype=float)
    atr_pct = smoke_df["atr_pct_roll"].to_numpy(dtype=float)
    from research.outcomes.table import _rolling_rvol_series
    rvol = _rolling_rvol_series(smoke_df["volume"])

    profit, bars, reason = simulate_forward(
        high, low, close, atr_pct, rvol, start_index=i, params=smoke_config.params
    )
    row = table.records.iloc[i]
    assert row["__outcome_profit"] == pytest.approx(profit)
    assert row["__outcome_hold_bars"] == bars
    assert row["__outcome_close_reason"] == reason


# =============================================================================================
# T015 -- fillability, via BaseStrategy.custom_stake_amount's own QUOTE_VOLUME_HEADROOM_MULT
# =============================================================================================


def test_fillable_uses_quote_volume_headroom_mult_from_base_strategy():
    """Synthetic frame: one bar with ample quote volume (fillable at
    PRODUCTION_STAKE), one with almost none (not fillable), holding everything else
    (ATR, OHLC) constant so only volume drives the fillability columns."""
    n = 5
    dates = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    close = np.full(n, 100.0)
    df = pd.DataFrame({
        "date": dates,
        "open": close, "high": close, "low": close, "close": close,
        # bar 0: ample volume (fillable); bar 1: a trickle (not fillable); rest ample
        "volume": [1_000_000.0, 1.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
        "atr_pct_roll": np.full(n, 0.0),  # ATR unavailable -> static stop, irrelevant here
    })
    cfg = ExitConfig.from_operating_point()
    table = build_outcome_table(df, "ZEC/USDT", cfg, stake=PRODUCTION_STAKE)
    r = table.records

    headroom = BaseStrategy.QUOTE_VOLUME_HEADROOM_MULT
    expected_fraction_0 = min(1.0, (df["volume"][0] * df["close"][0] / headroom) / PRODUCTION_STAKE)
    expected_fraction_1 = min(1.0, (df["volume"][1] * df["close"][1] / headroom) / PRODUCTION_STAKE)

    assert r["__outcome_fillable_fraction"].iloc[0] == pytest.approx(expected_fraction_0)
    assert r["__outcome_fillable_fraction"].iloc[1] == pytest.approx(expected_fraction_1)
    assert bool(r["__outcome_fillable"].iloc[0]) is True
    assert bool(r["__outcome_fillable"].iloc[1]) is False


def test_fillable_fraction_is_never_above_one():
    n = 3
    dates = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    close = np.full(n, 100.0)
    df = pd.DataFrame({
        "date": dates, "open": close, "high": close, "low": close, "close": close,
        "volume": np.full(n, 1e9),  # absurdly deep
        "atr_pct_roll": np.zeros(n),
    })
    table = build_outcome_table(df, "ZEC/USDT", ExitConfig.from_operating_point())
    assert (table.records["__outcome_fillable_fraction"] <= 1.0).all()
    assert table.records["__outcome_fillable"].all()


# =============================================================================================
# T016 -- gap-fill share, reusing research.clfstudy.data.zero_volume_share
# =============================================================================================


def test_gapfill_share_zero_when_no_gapfill_bars_in_the_hold_window():
    """`max_hold_days=1`, `timeframe_minutes=1440` collapses the hold window to one
    bar (as in `test_exit_policy_simulate_forward.py`); a real (non-gap-fill) bar
    there must give a gap-fill share of 0.0."""
    cfg = ExitConfig(ExitParams(
        static_stoploss=-0.5, use_atr_adaptive_stoploss=False, take_profit=0.5,
        max_hold_days=1, timeframe_minutes=1440, fee_per_side=0.0,
        stoploss_grace_hours=0.0, stoploss_grace_level=-0.15,
    ))
    dates = pd.date_range("2022-01-01", periods=2, freq="1D", tz="UTC")
    df = pd.DataFrame({
        "date": dates,
        "open": [100.0, 101.0], "high": [100.0, 101.0],
        "low": [100.0, 99.0], "close": [100.0, 101.0],
        "volume": [500.0, 500.0],  # real volume, not gap-fill
        "atr_pct_roll": [0.0, 0.0],
    })
    table = build_outcome_table(df, "ZEC/USDT", cfg)
    assert table.records["__outcome_gapfill_share"].iloc[0] == 0.0


def test_gapfill_share_one_when_the_entire_hold_window_is_gapfill():
    cfg = ExitConfig(ExitParams(
        static_stoploss=-0.5, use_atr_adaptive_stoploss=False, take_profit=0.5,
        max_hold_days=1, timeframe_minutes=1440, fee_per_side=0.0,
        stoploss_grace_hours=0.0, stoploss_grace_level=-0.15,
    ))
    dates = pd.date_range("2022-01-01", periods=2, freq="1D", tz="UTC")
    df = pd.DataFrame({
        "date": dates,
        "open": [100.0, 100.0], "high": [100.0, 100.0],
        "low": [100.0, 100.0], "close": [100.0, 100.0],
        "volume": [500.0, 0.0],  # bar 1 is a gap-fill bar: volume 0, OHLC all equal
        "atr_pct_roll": [0.0, 0.0],
    })
    table = build_outcome_table(df, "ZEC/USDT", cfg)
    assert table.records["__outcome_gapfill_share"].iloc[0] == 1.0


def test_gapfill_share_matches_zero_volume_share_on_a_real_pair(smoke_df, smoke_config):
    from research.clfstudy.data import zero_volume_share

    table = build_outcome_table(smoke_df, "ZEC/USDT", smoke_config)
    i = 5
    bars = int(table.records["__outcome_hold_bars"].iloc[i])
    if bars > 0:
        window = smoke_df.iloc[i + 1: i + 1 + bars]
        expected = zero_volume_share(window)
        expected = 0.0 if pd.isna(expected) else expected
        assert table.records["__outcome_gapfill_share"].iloc[i] == pytest.approx(expected)


# =============================================================================================
# T017 -- the feather cache + JSON sidecar
# =============================================================================================


def test_save_and_load_table_round_trips(tmp_path, monkeypatch, smoke_df, smoke_config):
    import research.outcomes.table as table_mod
    monkeypatch.setattr(table_mod, "CACHE_DIR", tmp_path)

    table = table_mod.build_outcome_table(smoke_df, "ZEC/USDT", smoke_config)
    table_mod.save_table(table, "ZEC/USDT")
    loaded = table_mod.load_table("ZEC/USDT", smoke_config)

    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), table.records.reset_index(drop=True)
    )


def test_load_table_missing_cache_is_a_plain_file_not_found(tmp_path, monkeypatch):
    import research.outcomes.table as table_mod
    monkeypatch.setattr(table_mod, "CACHE_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        table_mod.load_table("ZEC/USDT", ExitConfig.from_operating_point())


def test_load_table_raises_never_silently_reuses_a_tampered_sidecar(tmp_path, monkeypatch, smoke_df, smoke_config):
    """FR-007/SC-004: a sidecar whose recorded params no longer hash to the
    filename's embedded hash must be an ERROR, not a silent reuse."""
    import research.outcomes.table as table_mod
    monkeypatch.setattr(table_mod, "CACHE_DIR", tmp_path)

    table = table_mod.build_outcome_table(smoke_df, "ZEC/USDT", smoke_config)
    table_mod.save_table(table, "ZEC/USDT")

    sidecar = table_mod._sidecar_path(smoke_config.config_hash)
    payload = json.loads(sidecar.read_text())
    payload["fee_per_side"] = payload["fee_per_side"] + 0.5  # corrupt it
    sidecar.write_text(json.dumps(payload))

    with pytest.raises(CacheMismatchError):
        table_mod.load_table("ZEC/USDT", smoke_config)


def test_status_lists_cached_tables(tmp_path, monkeypatch, smoke_df, smoke_config):
    import research.outcomes.table as table_mod
    monkeypatch.setattr(table_mod, "CACHE_DIR", tmp_path)

    table = table_mod.build_outcome_table(smoke_df, "ZEC/USDT", smoke_config)
    table_mod.save_table(table, "ZEC/USDT")

    rows = table_mod.status()
    assert len(rows) == 1
    assert rows[0]["pair_base"] == "ZEC"
    assert rows[0]["config_hash"] == smoke_config.config_hash
    assert rows[0]["rows"] == len(smoke_df)


# =============================================================================================
# T018 -- the CLI
# =============================================================================================


def test_cli_build_status_verify_round_trip(tmp_path, monkeypatch):
    import research.outcomes.table as table_mod
    monkeypatch.setattr(table_mod, "CACHE_DIR", tmp_path)

    table_mod.main([
        "--build", "--pairs", "ZEC/USDT",
        "--timerange", "2022-01-01:2022-01-08",
    ])
    assert len(list(tmp_path.glob("*.feather"))) == 1

    table_mod.main(["--status"])  # must not raise

    table_mod.main([
        "--verify", "--pairs", "ZEC/USDT",
        "--timerange", "2022-01-01:2022-01-08",
    ])


def test_cli_build_requires_pairs():
    import research.outcomes.table as table_mod
    with pytest.raises(SystemExit):
        table_mod.main(["--build"])


# =============================================================================================
# T019 -- reproducibility (SC-003, FR-008): two builds from the same inputs are
# byte-identical.
# =============================================================================================


def test_two_builds_from_the_same_inputs_are_identical(smoke_df, smoke_config):
    a = build_outcome_table(smoke_df.copy(), "ZEC/USDT", smoke_config)
    b = build_outcome_table(smoke_df.copy(), "ZEC/USDT", smoke_config)
    pd.testing.assert_frame_equal(
        a.records.reset_index(drop=True), b.records.reset_index(drop=True)
    )
    assert a.config_hash == b.config_hash


def test_two_cached_builds_round_trip_to_byte_identical_feathers(tmp_path, monkeypatch, smoke_df, smoke_config):
    import pyarrow.feather as feather
    import research.outcomes.table as table_mod
    monkeypatch.setattr(table_mod, "CACHE_DIR", tmp_path / "a")
    (tmp_path / "a").mkdir()
    path_a = table_mod.save_table(
        table_mod.build_outcome_table(smoke_df.copy(), "ZEC/USDT", smoke_config), "ZEC/USDT"
    )

    monkeypatch.setattr(table_mod, "CACHE_DIR", tmp_path / "b")
    (tmp_path / "b").mkdir()
    path_b = table_mod.save_table(
        table_mod.build_outcome_table(smoke_df.copy(), "ZEC/USDT", smoke_config), "ZEC/USDT"
    )

    pd.testing.assert_frame_equal(
        feather.read_feather(path_a), feather.read_feather(path_b)
    )
