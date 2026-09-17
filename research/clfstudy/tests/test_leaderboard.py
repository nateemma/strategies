"""Tests for research/clfstudy/leaderboard.py (T019).

Builds a small synthetic `results.jsonl` (`make_row`/`write_jsonl` below, same shape as
`test_gate.py`'s helper but kept self-contained here) with known per-seed MCC values, checks
`aggregate()`'s per-window mean/std (ddof=1) and pooled means by hand, that a duplicate
(config_id, split_set, window_id, seed) key keeps the LAST line, that `failed` rows are
excluded, and that `write_outputs()` produces its csv/md/png files without raising.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from research.clfstudy import leaderboard as LB

CFG = "cfg1aaaaaaaa"


def make_row(config_id, window_id, seed, *, arch="LSTM", phase=4, status="ok",
             mcc=0.3, macro_f1=0.3, bal_acc=0.3, input_set="ind24", seq_len=16,
             hp=None, tags=None, n_train=1000, n_val=100, n_test=200,
             n_params=1234, train_seconds=5.0, epochs_run=5, best_epoch=3,
             infer_ms_per_window_b256=0.5, infer_ms_per_window_b1=1.0,
             split_set="v1", **overrides) -> dict:
    """One synthetic RunResult row (data-model.md §RunResult), overridable via kwargs."""
    row = {
        "config_id": config_id,
        "split_set": split_set,
        "window_id": window_id,
        "seed": seed,
        "phase": phase,
        "arch": arch,
        "hp": hp or {},
        "input_set": input_set,
        "seq_len": seq_len,
        "train": {
            "max_epochs": 5,
            "patience": 5,
            "batch_size": 64,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "stride": 1,
        },
        "tags": tags or [],
        "status": status,
        "error": None,
        "pairs_included": ["ZEC/USDT"],
        "pairs_excluded": [],
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "class_counts_train": [n_train // 3] * 3,
        "class_counts_test": [n_test // 3] * 3,
        "pred_counts": [n_test // 3] * 3,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "prec_sell": 0.3,
        "prec_hold": 0.3,
        "prec_buy": 0.3,
        "rec_sell": 0.3,
        "rec_hold": 0.3,
        "rec_buy": 0.3,
        "logloss": 1.0,
        "auroc_sell": 0.5,
        "auroc_hold": 0.5,
        "auroc_buy": 0.5,
        "auroc_macro": 0.5,
        "ece": 0.1,
        "reliability_bins": [
            {"lo": lo, "hi": lo + 1 / 15, "conf": 0.5, "acc": 0.5, "n": 1}
            for lo in np.linspace(0.0, 14 / 15, 15)
        ],
        "confusion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "buy_prec_at_08": 0.5,
        "buy_cov_at_08": 0.1,
        "n_params": n_params,
        "train_seconds": train_seconds,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "infer_ms_per_window_b256": infer_ms_per_window_b256,
        "infer_ms_per_window_b1": infer_ms_per_window_b1,
        "harness_version": "test",
        "git_sha": "deadbeef",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "weights_path": None,
    }
    row.update(overrides)
    return row


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# Window A: seed0=.4, seed1=.6 -> mean=.5, std=|.6-.4|/sqrt(2)
# Window B: seed0=.2, seed1=.4 -> mean=.3, std=|.4-.2|/sqrt(2)
# Window C: seed0=.5, seed1=.5 -> mean=.5, std=0
WINDOW_MCC = {"A": (0.4, 0.6), "B": (0.2, 0.4), "C": (0.5, 0.5)}
EXPECTED_MEAN = {w: sum(v) / 2 for w, v in WINDOW_MCC.items()}
EXPECTED_STD = {
    w: abs(v[1] - v[0]) / math.sqrt(2) for w, v in WINDOW_MCC.items()
}
EXPECTED_POOLED_MEAN = sum(EXPECTED_MEAN.values()) / 3
EXPECTED_POOLED_STD = sum(EXPECTED_STD.values()) / 3


def _base_rows():
    rows = []
    for w, (m0, m1) in WINDOW_MCC.items():
        rows.append(make_row(CFG, w, 0, mcc=m0))
        rows.append(make_row(CFG, w, 1, mcc=m1))
    return rows


def test_aggregate_per_window_mean_std_ddof1_and_pooled_mean(tmp_path):
    df = pd.DataFrame(_base_rows())
    agg = LB.aggregate(df, phase=4)

    assert len(agg) == 1
    row = agg.iloc[0]
    for w in ("A", "B", "C"):
        assert row[f"mcc_mean_{w}"] == pytest.approx(EXPECTED_MEAN[w])
        assert row[f"mcc_std_{w}"] == pytest.approx(EXPECTED_STD[w])
        assert row[f"n_seeds_{w}"] == 2

    assert row["pooled_mcc_mean"] == pytest.approx(EXPECTED_POOLED_MEAN)
    assert row["pooled_mcc_std"] == pytest.approx(EXPECTED_POOLED_STD)
    assert row["n_windows"] == 3


def test_load_results_duplicate_key_keeps_last_line(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = _base_rows()
    # A duplicate of the (config_id, split_set, window_id, seed=0) key for window A, with a
    # different mcc -- this must win over the original since it comes later in the file.
    rows.append(make_row(CFG, "A", 0, mcc=0.99))
    write_jsonl(path, rows)

    df = LB.load_results(path)
    a_seed0 = df[(df["config_id"] == CFG) & (df["window_id"] == "A") & (df["seed"] == 0)]
    assert len(a_seed0) == 1
    assert a_seed0["mcc"].iloc[0] == pytest.approx(0.99)


def test_aggregate_excludes_failed_rows(tmp_path):
    rows = _base_rows()
    # A failed row for window A with an extreme mcc that would badly skew the mean/std if it
    # were included.
    rows.append(make_row(CFG, "A", 2, mcc=-1.0, status="failed"))
    df = pd.DataFrame(rows)

    agg = LB.aggregate(df, phase=4)
    row = agg.iloc[0]
    assert row["mcc_mean_A"] == pytest.approx(EXPECTED_MEAN["A"])
    assert row["mcc_std_A"] == pytest.approx(EXPECTED_STD["A"])
    assert row["n_seeds_A"] == 2  # the failed seed does not count


def test_aggregate_keeps_collapsed_and_class_absent_but_flags_them(tmp_path):
    rows = _base_rows()
    rows.append(make_row(CFG, "A", 2, mcc=0.5, status="collapsed"))
    df = pd.DataFrame(rows)

    agg = LB.aggregate(df, phase=4)
    row = agg.iloc[0]
    assert row["n_seeds_A"] == 3  # collapsed row IS included in the mean/std
    assert bool(row["any_collapsed_A"]) is True
    assert bool(row["any_collapsed"]) is True


def test_baselines_always_included_when_filtering_by_phase(tmp_path):
    rows = _base_rows()  # phase=4 by default
    rows += [
        make_row("ridgeconfigid", w, s, arch="RIDGE", phase=2, mcc=0.1)
        for w in ("A", "B", "C")
        for s in (0, 1)
    ]
    df = pd.DataFrame(rows)

    agg = LB.aggregate(df, phase=4)
    assert "ridgeconfigid" in set(agg["config_id"])
    assert CFG in set(agg["config_id"])


def test_aggregate_sorted_by_pooled_mcc_desc(tmp_path):
    rows = _base_rows()
    rows += [make_row("lowcfgid0001", w, s, mcc=-0.5) for w in ("A", "B", "C") for s in (0, 1)]
    df = pd.DataFrame(rows)

    agg = LB.aggregate(df, phase=4)
    assert list(agg["pooled_mcc_mean"]) == sorted(agg["pooled_mcc_mean"], reverse=True)
    assert agg.iloc[0]["config_id"] == CFG


def test_write_outputs_produces_files_without_error(tmp_path):
    results_path = tmp_path / "results" / "results.jsonl"
    results_path.parent.mkdir(parents=True)
    rows = _base_rows()
    rows += [
        make_row("ridgeconfigid", w, s, arch="RIDGE", phase=2, mcc=0.1)
        for w in ("A", "B", "C")
        for s in (0, 1)
    ]
    write_jsonl(results_path, rows)

    out_dir = tmp_path / "report"
    paths = LB.write_outputs(4, out_dir=out_dir, results_path=results_path)

    assert (out_dir / "leaderboard_4.csv").exists()
    assert (out_dir / "leaderboard_4.md").exists()
    md_text = (out_dir / "leaderboard_4.md").read_text()
    assert "Constitution III" in md_text
    assert (tmp_path / "results" / "results.csv").exists()
    assert (tmp_path / "results" / "results.parquet").exists()

    for name, path in paths["plots"].items():
        assert path is not None, f"plot {name} was skipped unexpectedly"
        assert (out_dir / "plots").exists()
