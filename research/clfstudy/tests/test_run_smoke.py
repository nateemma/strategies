# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""Smoke tests for `research/clfstudy/run.py` and `phase.py` (T017/T018).

Builds a tiny 2-pair synthetic dataset (all IND24 columns as random normals, a random
3-class `label` column) and a single toy `Window` sized to fit inside it, bypassing
`data.load_all()` and split-set loading entirely so the test never touches real feather
data or the real split-set files. Runs `run_one` once for the training-free `MAJORITY`
baseline and once for a 1-epoch `MLP`, into a tmp `results.jsonl`, and checks every
`RunResult` field (data-model.md §RunResult) is present with an allowed `status`. Also
exercises `phase.py`'s existing-key skip logic directly, its MLX memory-release hook
(`run_one`'s `finally` block, T029), and its `--isolate` subprocess command/config_id
round-trip (T030).
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from research.clfstudy import features as F
from research.clfstudy import phase as phase_mod
from research.clfstudy import registry
from research.clfstudy import run as run_mod
from research.clfstudy import splits as S
from research.clfstudy.registry import ExperimentConfig

N_ROWS = 4000
PAIRS = ["FOO/USDT", "BAR/USDT"]

ALLOWED_STATUSES = {"ok", "failed", "collapsed", "class_absent"}

_ROW_FIELDS = [
    "config_id", "split_set", "window_id", "seed",
    "phase", "arch", "hp", "input_set", "seq_len", "train", "tags",
    "scaler_mode",
    "status", "error",
    "pairs_included", "pairs_excluded", "n_train", "n_val", "n_test",
    "class_counts_train", "class_counts_test", "pred_counts",
    "degenerate_iqr_cols", "constant_cols",
    "zero_vol_share_train", "zero_vol_share_test",
    "bal_acc", "macro_f1", "mcc",
    "prec_sell", "prec_hold", "prec_buy", "rec_sell", "rec_hold", "rec_buy",
    "logloss", "auroc_sell", "auroc_hold", "auroc_buy", "auroc_macro", "ece",
    "reliability_bins", "confusion",
    "buy_prec_at_08", "buy_cov_at_08",
    "trig_buy_n", "trig_buy_base_rate", "trig_buy_prec_argmax", "trig_buy_auroc",
    "trig_buy_mcc",
    "trig_sell_n", "trig_sell_base_rate", "trig_sell_prec_argmax", "trig_sell_auroc",
    "trig_sell_mcc",
    "trig_rule_mcc",
    "n_params", "train_seconds", "epochs_run", "best_epoch",
    "infer_ms_per_window_b256", "infer_ms_per_window_b1", "mlx_peak_mb",
    "harness_version", "git_sha", "started_at", "finished_at", "weights_path",
    "probs_path", "ckpt_name",
]

_PROBS_COLUMNS = [
    "date", "pair", "label", "p_sell", "p_hold", "p_buy",
    "guard_metric", "bb_width", "buy_trigger", "sell_trigger",
]


def _make_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=N_ROWS, freq="15min", tz="UTC")
    data = {c: rng.normal(size=N_ROWS) for c in F.IND24}
    data["date"] = dates
    data["label"] = rng.integers(0, 3, N_ROWS).astype(np.int8)
    # RAW guard_metric/bb_width (`_build_tensors` reads these directly off the frame
    # `data.load_all()` would otherwise provide, to feed `metrics.trigger_masks`): natural
    # ranges ([-1, 1] and a small positive ratio) straddling the default thresholds (0.3,
    # 0.035) so BUY-trigger, SELL-trigger and non-trigger rows all occur -> trig_buy_n > 0.
    # `bb_width` here overwrites the unrelated scaled-feature-space value the IND24 loop
    # above already produced (bb_width is itself part of IND24) since only this raw one is
    # ever read for the trigger rule.
    data["guard_metric"] = rng.uniform(-1.0, 1.0, size=N_ROWS)
    data["bb_width"] = rng.uniform(0.0, 0.08, size=N_ROWS)
    # A known, deterministic zero-volume pattern (freqtrade's gap-fill contamination,
    # research FIX 2): exactly every 4th bar is zero-volume, spread evenly across the
    # whole span so both the train and test role slices contain a mix of zero/nonzero
    # rows -- `test_run_one_records_zero_volume_shares` below checks the row's
    # `zero_vol_share_{train,test}` against this pattern directly.
    volume = rng.uniform(1.0, 100.0, size=N_ROWS)
    volume[np.arange(N_ROWS) % 4 == 0] = 0.0
    data["volume"] = volume
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_frames():
    return {pair: _make_frame(seed) for seed, pair in enumerate(PAIRS)}


@pytest.fixture
def toy_splitset(synthetic_frames):
    dates = synthetic_frames[PAIRS[0]]["date"]
    # ~80% train+val, ~6.7% test; leaves ample train rows for both classifiers and
    # enough test rows (266) for every class to plausibly appear.
    w = S._make_window("A", dates.iloc[0], dates.iloc[3200], dates.iloc[3466], distant=True)
    return S.SplitSet(
        split_set="v1",
        embargo_bars=S.EMBARGO_BARS,
        val_fraction=S.VAL_FRACTION,
        min_history_days=1,
        timeframe="15m",
        pairs=list(PAIRS),
        windows=[w],
    )


@pytest.fixture(autouse=True)
def _patch_run_module(monkeypatch, synthetic_frames, toy_splitset, tmp_path):
    """Bypass real feather data / real split-set files; write results to a tmp file."""
    monkeypatch.setattr(run_mod, "_FRAMES_CACHE", None)
    monkeypatch.setattr(run_mod, "_SPLITSET_CACHE", {})
    monkeypatch.setattr(run_mod.data, "load_all", lambda *a, **k: dict(synthetic_frames))
    monkeypatch.setattr(run_mod, "load_or_build_splitset", lambda split_set: toy_splitset)
    # a pair needs >=1 day of history before val_start to be included (not 365).
    monkeypatch.setattr(S, "MIN_HISTORY_DAYS", 1)

    results_path = tmp_path / "results.jsonl"
    monkeypatch.setattr(run_mod, "RESULTS_PATH", results_path)
    return results_path


def _last_row(results_path) -> dict:
    lines = results_path.read_text().strip().splitlines()
    return json.loads(lines[-1])


def test_run_one_majority_baseline(_patch_run_module):
    cfg = ExperimentConfig(
        arch="MAJORITY", hp={}, input_set="ind24", seq_len=16, split_set="v1",
        train={}, phase=0, seeds=[0], tags=["smoke"],
    )
    cfg.validate()

    row = run_mod.run_one(cfg, "A", seed=0)

    assert set(_ROW_FIELDS) <= set(row.keys())
    assert row["status"] in ALLOWED_STATUSES
    assert row["status"] != "failed", row["error"]
    assert row["n_params"] == 0
    assert row["train_seconds"] == 0.0
    assert row["epochs_run"] == 0
    assert row["pairs_included"] == PAIRS
    # the synthetic guard_metric/bb_width columns straddle both trigger thresholds, so the
    # BUY-trigger conditional fields are actually populated (not just present-but-empty).
    assert row["trig_buy_n"] > 0
    assert row["trig_sell_n"] > 0
    # default (no --save-probs): nothing written, field stays None.
    assert row["probs_path"] is None

    on_disk = _last_row(_patch_run_module)
    assert set(_ROW_FIELDS) <= set(on_disk.keys())
    assert on_disk["config_id"] == row["config_id"]
    assert on_disk["status"] == row["status"]
    assert on_disk["mcc"] == pytest.approx(row["mcc"])


def test_run_one_mlp_one_epoch(_patch_run_module):
    cfg = ExperimentConfig(
        arch="MLP", hp={}, input_set="ind24", seq_len=16, split_set="v1",
        train={"max_epochs": 1, "patience": 1, "batch_size": 64},
        phase=0, seeds=[0], tags=["smoke"],
    )
    cfg.validate()

    row = run_mod.run_one(cfg, "A", seed=0)

    assert set(_ROW_FIELDS) <= set(row.keys())
    assert row["status"] in ALLOWED_STATUSES
    assert row["status"] != "failed", row["error"]
    assert row["n_params"] > 0
    assert row["epochs_run"] == 1
    assert row["train_seconds"] > 0.0
    # default (no --save-probs): nothing written, field stays None.
    assert row["probs_path"] is None

    on_disk = _last_row(_patch_run_module)
    assert set(_ROW_FIELDS) <= set(on_disk.keys())
    assert on_disk["config_id"] == row["config_id"]
    assert on_disk["status"] == row["status"]
    assert on_disk["epochs_run"] == row["epochs_run"]


def test_run_one_records_zero_volume_shares(_patch_run_module, synthetic_frames, toy_splitset):
    """`zero_vol_share_{train,test}` (research FIX 2) must equal the true zero-volume
    fraction of exactly the train-role / test-role rows of the *included* pairs -- computed
    here independently via `splits.assign_roles` against the same synthetic `volume`
    column, never by re-deriving it from `run_one` internals."""
    cfg = ExperimentConfig(
        arch="MAJORITY", hp={}, input_set="ind24", seq_len=16, split_set="v1",
        train={}, phase=0, seeds=[0], tags=["smoke"],
    )
    cfg.validate()

    row = run_mod.run_one(cfg, "A", seed=0)
    assert row["status"] != "failed", row.get("error")

    w = toy_splitset.windows[0]
    train_vol_parts, test_vol_parts = [], []
    for frame in synthetic_frames.values():
        role = S.assign_roles(frame["date"], w)
        train_vol_parts.append(frame.loc[role == "train", "volume"].to_numpy())
        test_vol_parts.append(frame.loc[role == "test", "volume"].to_numpy())
    train_vol = np.concatenate(train_vol_parts)
    test_vol = np.concatenate(test_vol_parts)

    # sanity: the 1-in-4 zero-volume pattern lands inside both roles for this toy window.
    assert 0.0 < float((train_vol == 0).mean()) < 1.0
    assert 0.0 < float((test_vol == 0).mean()) < 1.0

    assert row["zero_vol_share_train"] == pytest.approx(float((train_vol == 0).mean()))
    assert row["zero_vol_share_test"] == pytest.approx(float((test_vol == 0).mean()))


def test_run_one_reports_degenerate_scaler_columns(_patch_run_module, synthetic_frames):
    """`fit_scaler`'s per-window diagnostic (FIX 1) must thread through `run_one` into the
    row: a column made zero-IQR-but-nonzero-std across the whole (train-role) span is
    named in `degenerate_iqr_cols`, and `scaler_mode` on the row matches the config."""
    degenerate_col = "di_diff_scaled"  # IND24 member that needs_norm() actually scales
    for frame in synthetic_frames.values():
        vals = np.zeros(len(frame))
        vals[:20] = np.random.default_rng(1).normal(loc=10.0, scale=1.0, size=20)
        frame[degenerate_col] = vals

    cfg = ExperimentConfig(
        arch="MAJORITY", hp={}, input_set="ind24", seq_len=16, split_set="v1",
        train={}, phase=0, seeds=[0], tags=["smoke"], scaler_mode="robust_std",
    )
    cfg.validate()

    row = run_mod.run_one(cfg, "A", seed=0)

    assert row["status"] != "failed", row.get("error")
    assert row["scaler_mode"] == "robust_std"
    assert degenerate_col in row["degenerate_iqr_cols"]
    assert row["constant_cols"] == []


def test_run_one_mlp_save_probs_writes_parquet(_patch_run_module):
    """`--save-probs`/`save_probs=True` writes one row per test example to
    `PROBS_DIR/<config_id>_<window_id>_<seed>.parquet` with the documented columns, and
    the on-disk `RunResult` row records the (relative) path."""
    cfg = ExperimentConfig(
        arch="MLP", hp={}, input_set="ind24", seq_len=16, split_set="v1",
        train={"max_epochs": 1, "patience": 1, "batch_size": 64},
        phase=0, seeds=[0], tags=["smoke"],
    )
    cfg.validate()

    from research.clfstudy import PKG
    from research.clfstudy.run import PROBS_DIR

    cid = registry.config_id(cfg)
    dest = PROBS_DIR / f"{cid}_A_0.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)

    try:
        row = run_mod.run_one(cfg, "A", seed=0, save_probs=True)

        assert row["status"] != "failed", row.get("error")
        assert row["probs_path"] == str(dest.relative_to(PKG))
        assert dest.exists()

        df = pd.read_parquet(dest)
        assert list(df.columns) == _PROBS_COLUMNS
        assert len(df) == row["n_test"]
        assert df["label"].between(0, 2).all()
        assert df["p_sell"].dtype == np.float32
        assert df["p_hold"].dtype == np.float32
        assert df["p_buy"].dtype == np.float32
        np.testing.assert_allclose(
            df[["p_sell", "p_hold", "p_buy"]].sum(axis=1).to_numpy(), 1.0, atol=1e-3
        )
        # pyarrow round-trips tz-aware UTC timestamps at either ns or us precision
        # depending on build; only the "tz-aware UTC" part is contractual here.
        assert str(df["date"].dtype).startswith("datetime64[") and str(df["date"].dtype).endswith(", UTC]")
        assert set(df["pair"].unique()) <= set(PAIRS)
        assert df["buy_trigger"].dtype == bool
        assert df["sell_trigger"].dtype == bool
        # mutually exclusive by construction (metrics.trigger_masks).
        assert not (df["buy_trigger"] & df["sell_trigger"]).any()

        on_disk = _last_row(_patch_run_module)
        assert on_disk["probs_path"] == row["probs_path"]
    finally:
        dest.unlink(missing_ok=True)


def test_run_one_releases_mlx_memory(_patch_run_module, monkeypatch):
    """`run_one`'s `finally` block (T029) must release MLX's Metal buffer cache and stamp
    `mlx_peak_mb` on the row, on every path (including this training-free baseline)."""
    calls = []
    monkeypatch.setattr(run_mod.mx, "clear_cache", lambda: calls.append("clear_cache"))

    cfg = ExperimentConfig(
        arch="MAJORITY", hp={}, input_set="ind24", seq_len=16, split_set="v1",
        train={}, phase=0, seeds=[0], tags=["smoke"],
    )
    cfg.validate()

    row = run_mod.run_one(cfg, "A", seed=0)

    assert row["status"] != "failed", row.get("error")
    assert "mlx_peak_mb" in row
    assert calls == ["clear_cache"]


def test_run_one_dry_run_returns_empty_and_appends_nothing(_patch_run_module):
    cfg = ExperimentConfig(
        arch="MAJORITY", hp={}, input_set="ind24", seq_len=16, split_set="v1",
        train={}, phase=0, seeds=[0], tags=["smoke"],
    )
    cfg.validate()

    row = run_mod.run_one(cfg, "A", seed=0, dry_run=True)

    assert row == {}
    assert not _patch_run_module.exists()


def test_run_one_removes_stale_model_file(_patch_run_module):
    """A leftover weight file at the exact path `registry.build()` would route this
    (config, window, seed) to (e.g. from a killed-mid-run resume, or a retry) must be deleted
    before `train()` runs -- otherwise `MLXClassifierNary.train()`'s unconditional
    `self.model = self.load()` silently warm-starts from it. Regression test for the
    bug where `force_train=True` did not prevent this."""
    cfg = ExperimentConfig(
        arch="MLP", hp={}, input_set="ind24", seq_len=16, split_set="v1",
        train={"max_epochs": 1, "patience": 1, "batch_size": 64},
        phase=0, seeds=[0], tags=["smoke"],
    )
    cfg.validate()

    from research.clfstudy import registry as registry_mod
    from research.clfstudy import CACHE_DIR

    cid = registry_mod.config_id(cfg)
    # `CACHE_DIR` is the real, shared checkpoint cache (also used by concurrent study
    # runs), not a tmp_path -- clean up our probe file in `finally` regardless of outcome.
    ckpt_path = CACHE_DIR / "ckpt" / f"{cid}_A_0.safetensors"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_bytes(b"not a real checkpoint")
    dummy_mtime = ckpt_path.stat().st_mtime

    try:
        row = run_mod.run_one(cfg, "A", seed=0)

        assert row["status"] != "failed", row.get("error")
        assert row["stale_files_removed"] >= 1
        assert ckpt_path.exists()
        assert ckpt_path.stat().st_mtime >= dummy_mtime
        assert ckpt_path.read_bytes() != b"not a real checkpoint"
    finally:
        ckpt_path.unlink(missing_ok=True)


def test_phase_skip_logic_skips_existing_key(monkeypatch, tmp_path):
    results_path = tmp_path / "phase_results.jsonl"
    done_row = {
        "config_id": "abc123", "split_set": "v1", "window_id": "A", "seed": 0,
        "status": "ok",
    }
    failed_row = {
        "config_id": "def456", "split_set": "v1", "window_id": "A", "seed": 0,
        "status": "failed",
    }
    results_path.write_text(json.dumps(done_row) + "\n" + json.dumps(failed_row) + "\n")
    monkeypatch.setattr(phase_mod, "RESULTS_PATH", results_path)

    existing = phase_mod._load_existing_statuses()
    done_key = ("abc123", "v1", "A", 0)
    failed_key = ("def456", "v1", "A", 0)
    missing_key = ("nope", "v1", "A", 0)

    assert existing[done_key] == "ok"
    assert existing[failed_key] == "failed"

    assert phase_mod._should_skip(existing.get(done_key), retry_failed=False) is True
    assert phase_mod._should_skip(existing.get(failed_key), retry_failed=False) is True
    assert phase_mod._should_skip(existing.get(failed_key), retry_failed=True) is False
    assert phase_mod._should_skip(existing.get(missing_key), retry_failed=False) is False


def test_run_isolated_builds_matching_config_id(monkeypatch, tmp_path):
    """`phase._run_isolated` (T030) must build a `python -m research.clfstudy.run`
    command whose inline `--config` JSON round-trips through `run._load_config` to the
    *same* `config_id` as the parent's `cfg` -- verified here without spawning a real
    subprocess (the monkeypatched `subprocess.run` simulates only the id-matching part of
    what the real child would do, then appends a matching row itself)."""
    results_path = tmp_path / "iso_results.jsonl"
    monkeypatch.setattr(phase_mod, "RESULTS_PATH", results_path)

    cfg = ExperimentConfig(
        arch="MAJORITY", hp={"foo": 1}, input_set="ind24", seq_len=16, split_set="v1",
        train={"max_epochs": 5}, phase=0, seeds=[0], tags=["smoke"],
    )
    cfg.validate()
    expected_cid = registry.config_id(cfg)

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        config_json = cmd[cmd.index("--config") + 1]
        child_cfg = run_mod._load_config(config_json)
        child_cid = registry.config_id(child_cfg)
        assert child_cid == expected_cid
        row = {
            "config_id": child_cid,
            "split_set": child_cfg.split_set,
            "window_id": cmd[cmd.index("--window") + 1],
            "seed": int(cmd[cmd.index("--seed") + 1]),
            "status": "ok",
        }
        results_path.write_text(json.dumps(row) + "\n")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(phase_mod.subprocess, "run", fake_run)

    returncode, status = phase_mod._run_isolated(cfg, "A", 0)

    assert captured["cmd"][:3] == [sys.executable, "-m", "research.clfstudy.run"]
    assert returncode == 0
    assert status == "ok"


def test_isolated_cmd_carries_the_label():
    """Regression: phase.py's child argv must include cfg.label (a triple_barrier config ran
    on the gbb label when it did not, 2026-09-19)."""
    from research.clfstudy import phase as phase_mod, registry
    from research.clfstudy.run import _load_config
    import json as _json
    cfg = registry.ExperimentConfig(arch="MLP", hp={}, input_set="ind24", seq_len=16, split_set="v1",
                                    train={}, phase=19, seeds=[0],
                                    label={"name": "triple_barrier", "horizon": 48, "up": 0.02, "down": 0.02})
    cfg.validate()
    cmd = phase_mod._isolated_cmd(cfg, "A", 0)
    payload = _json.loads(cmd[cmd.index("--config") + 1])
    assert payload["label"]["name"] == "triple_barrier"
    child = _load_config(cmd[cmd.index("--config") + 1])
    assert child.label == cfg.label
    assert registry.config_id(child) == registry.config_id(cfg)
