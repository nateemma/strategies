# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
End-to-end harness test (regression for the window/checkpoint-collision defect).

A defect went undetected for a day because no test ran several windows of the same
(config, seed) through `phase.py` with a shared checkpoint directory: the checkpoint path
lacked the window id, and `MLXClassifierNary.train()` loads any existing file at
`model_path` unconditionally -- so window B silently warm-started from window A's saved
weights. Both are fixed now (the checkpoint path is `<config_id>_<window_id>_<seed>
.safetensors`, and `run.py` deletes stale files before every `train()` call), but nothing
proved it end to end, across real windows, through the real resumability path.

This builds a tiny, fully synthetic 2-pair / 3-window dataset (bypassing `data.load_all`
and the real split-set files entirely) with a LEARNABLE label -- SELL/HOLD/BUY from
thresholding one IND24 feature at the current bar -- and drives the harness through
`phase.py`'s `--no-isolate` in-process path: a fresh phase run, a resumed (all-skipped)
rerun, a rerun after deleting one results row (checks the stale-checkpoint cleanup that
the defect above needed), and `leaderboard.write_outputs` / `gate.decide` on the resulting
`results.jsonl`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
import numpy as np
import pandas as pd
import pytest
import yaml

# CPU only: a GPU job may be running concurrently in this environment.
mx.set_default_device(mx.cpu)

from research.clfstudy import features as F
from research.clfstudy import gate, leaderboard
from research.clfstudy import phase as phase_mod
from research.clfstudy import registry
from research.clfstudy import run as run_mod
from research.clfstudy import splits as S
from research.clfstudy.registry import ExperimentConfig

N_ROWS = 6000
PAIRS = ["FOO/USDT", "BAR/USDT"]
SIGNAL_COL = F.IND24[0]
THRESH = 0.25
#: subsample training rows 4x (test rows are unaffected) -- keeps the whole test <90s.
STRIDE = 4

ALLOWED_STATUSES = {"ok", "collapsed", "class_absent"}

_RUN_MARKER_RE = re.compile(r"\[\d+/\d+\] config_id=")


def _make_frame(seed: int) -> pd.DataFrame:
    """24 IND24 columns of random normals; `label` is a deterministic, learnable function
    of `SIGNAL_COL` at the current (last) bar -- SELL below -THRESH, BUY above +THRESH,
    HOLD in between."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=N_ROWS, freq="15min", tz="UTC")
    data = {c: rng.normal(size=N_ROWS) for c in F.IND24}
    signal = data[SIGNAL_COL]
    label = np.where(signal > THRESH, 2, np.where(signal < -THRESH, 0, 1)).astype(np.int8)
    data["date"] = dates
    data["label"] = label
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_frames() -> Dict[str, pd.DataFrame]:
    return {pair: _make_frame(seed) for seed, pair in enumerate(PAIRS)}


@pytest.fixture
def toy_splitset(synthetic_frames) -> S.SplitSet:
    """Three anchored windows (A distant, B/C not) sized to fit inside the 6000-bar
    synthetic span, embargo respected -- same construction as `splits.build`, just
    compressed to fit a tiny dataset (mirrors `test_run_smoke.py`'s single-window fixture,
    extended to three)."""
    dates = synthetic_frames[PAIRS[0]]["date"]
    windows = [
        S._make_window("A", dates.iloc[0], dates.iloc[3000], dates.iloc[3299], distant=True),
        S._make_window("B", dates.iloc[0], dates.iloc[4000], dates.iloc[4299], distant=False),
        S._make_window("C", dates.iloc[0], dates.iloc[5000], dates.iloc[5299], distant=False),
    ]
    ss = S.SplitSet(
        split_set="v1",
        embargo_bars=S.EMBARGO_BARS,
        val_fraction=S.VAL_FRACTION,
        min_history_days=1,
        timeframe="15m",
        pairs=list(PAIRS),
        windows=windows,
    )
    S.validate(ss)
    return ss


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch, synthetic_frames, toy_splitset, tmp_path):
    """Bypass real feather data / real split-set files, and route results/checkpoints/
    phase-configs to `tmp_path` -- so this test never touches the real study's
    `results.jsonl`, `.cache/ckpt/`, or `configs/`."""
    monkeypatch.setattr(run_mod, "_FRAMES_CACHE", None)
    monkeypatch.setattr(run_mod, "_SPLITSET_CACHE", {})
    monkeypatch.setattr(run_mod.data, "load_all", lambda *a, **k: dict(synthetic_frames))
    monkeypatch.setattr(run_mod, "load_or_build_splitset", lambda split_set: toy_splitset)
    monkeypatch.setattr(phase_mod, "load_or_build_splitset", lambda split_set: toy_splitset)
    monkeypatch.setattr(S, "MIN_HISTORY_DAYS", 1)

    results_path = tmp_path / "results.jsonl"
    cache_dir = tmp_path / "cache"
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    monkeypatch.setattr(run_mod, "RESULTS_PATH", results_path)
    monkeypatch.setattr(phase_mod, "RESULTS_PATH", results_path)
    monkeypatch.setattr(run_mod, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(phase_mod, "CONFIGS_DIR", configs_dir)

    return {
        "results_path": results_path,
        "ckpt_dir": cache_dir / "ckpt",
        "configs_dir": configs_dir,
    }


def _write_phase_yaml(path: Path) -> None:
    train = {"max_epochs": 2, "patience": 5, "batch_size": 64, "stride": STRIDE}
    doc = {
        "phase": 0,
        "configs": [
            {
                "arch": "MLP", "hp": {}, "input_set": "ind24", "seq_len": 16,
                "split_set": "v1", "train": train, "phase": 0,
                "seeds": [0, 1], "tags": ["e2e"],
            },
            {
                "arch": "LSTM_FAM", "hp": {"hidden": 8}, "input_set": "ind24", "seq_len": 16,
                "split_set": "v1", "train": train, "phase": 0,
                "seeds": [0, 1], "tags": ["e2e"],
            },
        ],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _run_segments(stdout: str) -> List[str]:
    """Split `stdout` on phase.py's `"[i/n] config_id=..."` per-run marker; the piece
    before the first marker (if any) is discarded."""
    pieces = _RUN_MARKER_RE.split(stdout)[1:]
    return pieces


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


def test_pipeline_e2e(_patch_pipeline, capsys):
    paths = _patch_pipeline
    phase_path = paths["configs_dir"] / "phase_0.yaml"
    _write_phase_yaml(phase_path)

    # ---- (a) fresh run, in-process (--no-isolate) --------------------------------
    rc = phase_mod.main(["--phase", "0", "--no-isolate"])
    out = capsys.readouterr().out
    assert rc == 0

    rows = _read_rows(paths["results_path"])

    # ---- (b) every row well-formed; no cross-window warm start -------------------
    assert len(rows) == 12

    ckpt_names = set()
    for row in rows:
        assert row["status"] in ALLOWED_STATUSES, row.get("error")
        assert row["epochs_run"] >= 1
        assert row["ckpt_name"] not in ckpt_names, "ckpt_name must be unique per run"
        ckpt_names.add(row["ckpt_name"])
        assert row["window_id"] in row["ckpt_name"]

    assert "MLX model loaded" not in out, (
        "a warm-started model means a window trained from another window's weights"
    )
    segments = _run_segments(out)
    assert len(segments) == 12
    for seg in segments:
        assert "MLX model loaded" not in seg
        assert ("MLX model not found" in seg) or ("fresh-start: removed stale" in seg)

    # ---- weights actually differ between window A and window B (same config/seed) ----
    by_key = {(r["config_id"], r["window_id"], r["seed"]): r for r in rows}
    mlp_cfg_id = next(r["config_id"] for r in rows if r["arch"] == "MLP")
    row_a = by_key[(mlp_cfg_id, "A", 0)]
    row_b = by_key[(mlp_cfg_id, "B", 0)]
    weights_a = mx.load(str(paths["ckpt_dir"] / row_a["ckpt_name"]))
    weights_b = mx.load(str(paths["ckpt_dir"] / row_b["ckpt_name"]))
    assert set(weights_a.keys()) == set(weights_b.keys())
    assert any(
        not np.array_equal(np.asarray(weights_a[k]), np.asarray(weights_b[k]))
        for k in weights_a
    ), "window B's saved weights are byte-identical to window A's -- warm-start regression"

    # ---- (c) resumability: rerunning the same phase skips every key --------------
    rc2 = phase_mod.main(["--phase", "0", "--no-isolate"])
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert "phase 0: skipped=12 ran=0 failed=0" in out2
    assert len(_read_rows(paths["results_path"])) == 12

    # ---- (d) delete one row -> exactly one run, stale checkpoint cleaned ----------
    lines = paths["results_path"].read_text().strip().splitlines()
    assert len(lines) == 12
    removed_row = json.loads(lines.pop(6))
    paths["results_path"].write_text("\n".join(lines) + "\n")

    rc3 = phase_mod.main(["--phase", "0", "--no-isolate"])
    out3 = capsys.readouterr().out
    assert rc3 == 0
    assert "phase 0: skipped=11 ran=1 failed=0" in out3

    segments3 = _run_segments(out3)
    assert len(segments3) == 1

    rows_after = _read_rows(paths["results_path"])
    assert len(rows_after) == 12
    rerun_row = next(
        r for r in rows_after
        if (r["config_id"], r["window_id"], r["seed"])
        == (removed_row["config_id"], removed_row["window_id"], removed_row["seed"])
    )
    assert rerun_row["stale_files_removed"] == 1

    # ---- (e) leaderboard + gate on the resulting results.jsonl --------------------
    # gate.decide requires a RIDGE reference; add it directly (bypassing phase.py, so
    # the "12 rows" invariant above stays exact) rather than folding it into the phase.
    ridge_cfg = ExperimentConfig(
        arch="RIDGE", hp={}, input_set="ind24", seq_len=16, split_set="v1",
        train={"max_epochs": 2, "patience": 5, "batch_size": 64, "stride": STRIDE},
        phase=2, seeds=[0], tags=["e2e-ridge"],
    )
    ridge_cfg.validate()
    for window_id in ("A", "B", "C"):
        ridge_row = run_mod.run_one(ridge_cfg, window_id, seed=0)
        assert ridge_row["status"] != "failed", ridge_row.get("error")

    out_dir = paths["configs_dir"].parent / "report"
    written = leaderboard.write_outputs(
        phase=0, out_dir=out_dir, results_path=paths["results_path"],
    )
    assert Path(written["results_csv"]).exists()
    assert Path(written["leaderboard_csv"]).exists()
    assert Path(written["leaderboard_md"]).exists()

    decision = gate.decide(phase=0, results_path=paths["results_path"])
    mlp_cfg_ids = {r["config_id"] for r in rows_after if r["arch"] == "MLP"}
    lstm_cfg_ids = {r["config_id"] for r in rows_after if r["arch"] == "LSTM_FAM"}
    assert mlp_cfg_ids and mlp_cfg_ids <= set(decision["per_config"].keys())
    assert lstm_cfg_ids and lstm_cfg_ids <= set(decision["per_config"].keys())
