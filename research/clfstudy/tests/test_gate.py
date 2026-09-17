"""Tests for research/clfstudy/gate.py (T021, spec FR-028).

Builds a small synthetic `results.jsonl` (via `make_row`/`write_jsonl` below) with a known
RIDGE baseline and several candidate configs engineered to hit exactly one FR-028 clause
each, then checks `gate.decide()`'s arithmetic and viability calls against hand-computed
values.
"""
from __future__ import annotations

import json
import math

import pytest

from research.clfstudy import gate

RIDGE_ID = "ridge000000"
CAND_2OF3 = "cand2of3aaa"
CAND_1OF3 = "cand1of3bbb"
CAND_COLLAPSED = "candcollaps"
CAND_STDFAIL = "candstdfail"
CAND_N1 = "candn1seedx"


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
        "reliability_bins": [],
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


def _ridge_rows():
    """RIDGE, phase 2, 2 seeds/window: MCC std known exactly (A=.02828, B=0, C=.01414)."""
    rows = []
    for w, (m0, m1) in {"A": (0.10, 0.14), "B": (0.10, 0.10), "C": (0.10, 0.12)}.items():
        for seed, mcc in ((0, m0), (1, m1)):
            rows.append(
                make_row(RIDGE_ID, w, seed, arch="RIDGE", phase=2, mcc=mcc, macro_f1=0.30)
            )
    return rows


RIDGE_STD_A = abs(0.14 - 0.10) / math.sqrt(2)
RIDGE_STD_B = 0.0
RIDGE_STD_C = abs(0.12 - 0.10) / math.sqrt(2)
RIDGE_MEAN_STD = (RIDGE_STD_A + RIDGE_STD_B + RIDGE_STD_C) / 3.0
EXPECTED_MARGIN = max(0.02, 2.0 * RIDGE_MEAN_STD)


def _build_results_file(tmp_path, extra_rows):
    path = tmp_path / "results.jsonl"
    write_jsonl(path, _ridge_rows() + extra_rows)
    return path


def test_margin_used_equals_max_floor_or_2x_ridge_pooled_std(tmp_path):
    path = _build_results_file(tmp_path, [])
    decision = gate.decide(phase=4, results_path=path)
    assert decision["rule"]["margin_used"] == pytest.approx(EXPECTED_MARGIN)
    assert decision["ridge_reference"]["pooled_seed_std_mcc"] == pytest.approx(RIDGE_MEAN_STD)
    assert decision["rule"]["margin_floor"] == 0.02
    assert decision["rule"]["majority_of"] == 3


def test_config_passing_2_of_3_windows_is_viable(tmp_path):
    rows = []
    # Window A: passes (mcc, f1 well above ridge+margin)
    for seed in (0, 1):
        rows.append(make_row(CAND_2OF3, "A", seed, phase=4, mcc=0.20, macro_f1=0.40))
    # Window B: passes
    for seed in (0, 1):
        rows.append(make_row(CAND_2OF3, "B", seed, phase=4, mcc=0.20, macro_f1=0.40))
    # Window C: fails (well below ridge+margin)
    for seed in (0, 1):
        rows.append(make_row(CAND_2OF3, "C", seed, phase=4, mcc=0.05, macro_f1=0.10))

    path = _build_results_file(tmp_path, rows)
    decision = gate.decide(phase=4, results_path=path)

    entry = decision["per_config"][CAND_2OF3]
    assert entry["A"]["window_pass"] is True
    assert entry["B"]["window_pass"] is True
    assert entry["C"]["window_pass"] is False
    assert entry["viable"] is True
    assert CAND_2OF3 in decision["viable_set"]


def test_config_passing_1_of_3_windows_is_not_viable(tmp_path):
    rows = []
    for seed in (0, 1):
        rows.append(make_row(CAND_1OF3, "A", seed, phase=4, mcc=0.20, macro_f1=0.40))
    for seed in (0, 1):
        rows.append(make_row(CAND_1OF3, "B", seed, phase=4, mcc=0.05, macro_f1=0.10))
    for seed in (0, 1):
        rows.append(make_row(CAND_1OF3, "C", seed, phase=4, mcc=0.05, macro_f1=0.10))

    path = _build_results_file(tmp_path, rows)
    decision = gate.decide(phase=4, results_path=path)

    entry = decision["per_config"][CAND_1OF3]
    assert sum(entry[w]["window_pass"] for w in ("A", "B", "C")) == 1
    assert entry["viable"] is False
    assert CAND_1OF3 not in decision["viable_set"]


def test_config_with_one_collapsed_window_never_viable_even_if_metrics_pass(tmp_path):
    rows = []
    for seed in (0, 1):
        rows.append(make_row(CAND_COLLAPSED, "A", seed, phase=4, mcc=0.20, macro_f1=0.40))
    # Window B: metrics would pass, but status is collapsed on both seeds.
    for seed in (0, 1):
        rows.append(
            make_row(
                CAND_COLLAPSED, "B", seed, phase=4, mcc=0.20, macro_f1=0.40, status="collapsed"
            )
        )
    for seed in (0, 1):
        rows.append(make_row(CAND_COLLAPSED, "C", seed, phase=4, mcc=0.20, macro_f1=0.40))

    path = _build_results_file(tmp_path, rows)
    decision = gate.decide(phase=4, results_path=path)

    entry = decision["per_config"][CAND_COLLAPSED]
    # Metrics-wise this would be a 2-of-3 majority (A and C pass; only B fails on status).
    assert entry["A"]["window_pass"] is True
    assert entry["C"]["window_pass"] is True
    assert entry["B"]["status_ok"] is False
    assert entry["B"]["window_pass"] is False
    assert entry["viable"] is False
    assert CAND_COLLAPSED not in decision["viable_set"]


def test_std_0_031_fails_the_stability_clause_on_that_window(tmp_path):
    # mean=.12 for ridge A; margin ~= .0283 -> threshold ~= .1483. Pick seeds whose MEAN
    # clears that threshold but whose ddof=1 std is exactly 0.031. Windows B and C pass
    # every clause cleanly, so this config clears the margin on a strict majority (2 of 3)
    # and has no bad status -- it fails ONLY the stability clause, on window A.
    diff = 0.031 * math.sqrt(2)
    seed0_mcc = 0.20
    seed1_mcc = 0.20 + diff

    rows = [
        make_row(CAND_STDFAIL, "A", 0, phase=4, mcc=seed0_mcc, macro_f1=0.40),
        make_row(CAND_STDFAIL, "A", 1, phase=4, mcc=seed1_mcc, macro_f1=0.40),
    ]
    for seed in (0, 1):
        rows.append(make_row(CAND_STDFAIL, "B", seed, phase=4, mcc=0.20, macro_f1=0.40))
        rows.append(make_row(CAND_STDFAIL, "C", seed, phase=4, mcc=0.20, macro_f1=0.40))

    path = _build_results_file(tmp_path, rows)

    # Default std_mode="every" (spec FR-028(ii)): std_ok must hold on EVERY window, so one
    # unstable window fails the config even though it clears the margin on a majority and
    # is reported as a stability-only ("third seed") candidate instead (FR-004).
    decision = gate.decide(phase=4, results_path=path)

    clause = decision["per_config"][CAND_STDFAIL]["A"]
    assert clause["std_mcc"] == pytest.approx(0.031, abs=1e-9)
    assert clause["mcc_ok"] is True
    assert clause["f1_ok"] is True
    assert clause["status_ok"] is True
    assert clause["std_ok"] is False
    assert clause["window_pass"] is False

    entry = decision["per_config"][CAND_STDFAIL]
    assert entry["viable"] is False
    assert CAND_STDFAIL not in decision["viable_set"]
    stability_only = {item["config_id"]: item for item in decision["stability_only_failures"]}
    assert CAND_STDFAIL in stability_only
    assert stability_only[CAND_STDFAIL]["windows"] == ["A"]

    # std_mode="majority" reproduces the pre-fix behaviour: std_ok folds into window_pass,
    # so B and C (2 of 3, a strict majority) outvote the unstable window A.
    decision_majority = gate.decide(phase=4, results_path=path, std_mode="majority")
    entry_majority = decision_majority["per_config"][CAND_STDFAIL]
    assert sum(entry_majority[w]["window_pass"] for w in ("A", "B", "C")) == 2
    assert entry_majority["viable"] is True
    assert CAND_STDFAIL in decision_majority["viable_set"]


def test_single_seed_window_fails_std_clause_as_insufficient_seeds(tmp_path):
    rows = [make_row(CAND_N1, "A", 0, phase=4, mcc=0.50, macro_f1=0.40)]
    for seed in (0, 1):
        rows.append(make_row(CAND_N1, "B", seed, phase=4, mcc=0.20, macro_f1=0.40))
        rows.append(make_row(CAND_N1, "C", seed, phase=4, mcc=0.20, macro_f1=0.40))

    path = _build_results_file(tmp_path, rows)
    decision = gate.decide(phase=4, results_path=path)

    clause = decision["per_config"][CAND_N1]["A"]
    assert clause["n_seeds"] == 1
    assert clause["std_ok"] is False
    assert "insufficient_seeds" in clause["notes"]


def test_ridge_itself_is_never_viable(tmp_path):
    path = _build_results_file(tmp_path, [])
    decision = gate.decide(phase=4, results_path=path)

    assert decision["per_config"][RIDGE_ID]["viable"] is False
    assert RIDGE_ID not in decision["viable_set"]


def test_write_decision_produces_json_and_md(tmp_path):
    rows = []
    for seed in (0, 1):
        rows.append(make_row(CAND_2OF3, "A", seed, phase=4, mcc=0.20, macro_f1=0.40))
        rows.append(make_row(CAND_2OF3, "B", seed, phase=4, mcc=0.20, macro_f1=0.40))
        rows.append(make_row(CAND_2OF3, "C", seed, phase=4, mcc=0.05, macro_f1=0.10))
    path = _build_results_file(tmp_path, rows)

    out_dir = tmp_path / "gates"
    decision = gate.write_decision(phase=4, results_path=path, out_dir=out_dir)

    json_path = out_dir / "phase_4.json"
    md_path = out_dir / "phase_4.md"
    assert json_path.exists()
    assert md_path.exists()
    reloaded = json.loads(json_path.read_text())
    assert reloaded["viable_set"] == decision["viable_set"]
