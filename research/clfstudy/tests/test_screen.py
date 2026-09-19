"""Tests for research/clfstudy/screen.py.

Builds a small synthetic `results.jsonl` (`make_row`/`write_jsonl`, same shape as
`test_leaderboard.py`'s helper) with a baseline-phase (19) RIDGE reference and four phase-20
variants engineered to hit each branch of the author's promotion rule (`gates/review_19.md`
s.6): promote-by-delta, no-promotion, promote-by-ridge-shift ("label harder"), and a variant
missing MLP_FAM rows entirely (INCOMPLETE).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.clfstudy import screen as SC

BASELINE_PHASE = 19
PHASE = 20

LABEL_1 = {"name": "triple_barrier", "horizon": 48, "up": 0.02, "down": 0.02}
LABEL_2 = {"name": "triple_barrier", "horizon": 48, "up": 0.03, "down": 0.02}
LABEL_3 = {"name": "triple_barrier", "horizon": 48, "up": 0.02, "down": 0.03}
LABEL_4 = {"name": "triple_barrier", "horizon": 24, "up": 0.02, "down": 0.02}

WINDOWS = ("A", "B", "C")


def make_row(config_id, window_id, seed, *, arch="RIDGE", phase=PHASE, status="ok",
             mcc=0.2, macro_f1=0.3, bal_acc=0.3, input_set="ind24", seq_len=16,
             hp=None, tags=None, n_train=1000, n_val=100, n_test=200,
             n_params=1234, train_seconds=5.0, epochs_run=5, best_epoch=3,
             infer_ms_per_window_b256=0.5, infer_ms_per_window_b1=1.0,
             split_set="v1", label=None, class_counts_test=None, **overrides) -> dict:
    """One synthetic RunResult row (data-model.md §RunResult), overridable via kwargs."""
    row = {
        "config_id": config_id,
        "split_set": split_set,
        "window_id": window_id,
        "seed": seed,
        "phase": phase,
        "arch": arch,
        "hp": hp or {},
        "label": label,
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
        "class_counts_test": class_counts_test or [n_test // 3] * 3,
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


def _ridge_rows(config_id, label, mcc_by_window, seeds=(0, 1, 2)):
    return [
        make_row(config_id, w, s, arch="RIDGE", phase=PHASE, label=label, mcc=mcc_by_window[w])
        for w in WINDOWS
        for s in seeds
    ]


def _mlp_rows(config_id, label, mcc_by_window, seeds=(0, 1)):
    return [
        make_row(config_id, w, s, arch="MLP_FAM", phase=PHASE, label=label,
                  hp={"width": 64}, mcc=mcc_by_window[w])
        for w in WINDOWS
        for s in seeds
    ]


def _baseline_rows(mcc=0.20):
    """Baseline-phase (19) RIDGE rows: pooled MCC == `mcc` (identical on every window/seed)."""
    return [
        make_row("baselineridge", w, s, arch="RIDGE", phase=BASELINE_PHASE, label={"name": "gbb"}, mcc=mcc)
        for w in WINDOWS
        for s in (0, 1, 2)
    ]


def _all_rows():
    rows = list(_baseline_rows(mcc=0.20))

    # Variant 1: MLP beats RIDGE by 0.03 on A and B, 0.0 on C -> PROMOTE (delta, 2 of 3 windows).
    rows += _ridge_rows("variant1cfg", LABEL_1, {"A": 0.20, "B": 0.20, "C": 0.20})
    rows += _mlp_rows("variant1cfg", LABEL_1, {"A": 0.23, "B": 0.23, "C": 0.20})

    # Variant 2: deltas 0.005 everywhere, ridge_shift 0.01 -> NO.
    rows += _ridge_rows("variant2cfg", LABEL_2, {"A": 0.21, "B": 0.21, "C": 0.21})
    rows += _mlp_rows("variant2cfg", LABEL_2, {"A": 0.215, "B": 0.215, "C": 0.215})

    # Variant 3: RIDGE itself moves -0.06 vs baseline -> NO (a drop is noted as "label harder", not promoted).
    rows += _ridge_rows("variant3cfg", LABEL_3, {"A": 0.14, "B": 0.14, "C": 0.14})
    rows += _mlp_rows("variant3cfg", LABEL_3, {"A": 0.14, "B": 0.14, "C": 0.14})

    # Variant 4: RIDGE only, no MLP_FAM rows at all -> INCOMPLETE.
    rows += _ridge_rows("variant4cfg", LABEL_4, {"A": 0.20, "B": 0.20, "C": 0.20})

    return rows


def _by_display(variants, key_fn):
    return {key_fn(v): v for v in variants}


def test_promote_by_delta_two_of_three_windows():
    df = pd.DataFrame(_all_rows())
    result = SC.screen(df, PHASE, baseline_phase=BASELINE_PHASE)

    assert result["baseline_ridge_pooled"] == pytest.approx(0.20)

    by_label = {v["label"]["up"]: v for v in result["variants"] if v["label"].get("down") == 0.02 and v["label"]["horizon"] == 48}
    v1 = by_label[0.02]
    assert v1["ridge"]["pooled"] == pytest.approx(0.20)
    assert v1["mlp"]["pooled"] == pytest.approx(0.22)
    assert v1["delta"]["A"] == pytest.approx(0.03)
    assert v1["delta"]["B"] == pytest.approx(0.03)
    assert v1["delta"]["C"] == pytest.approx(0.0)
    assert v1["windows_passing_delta"] == 2
    assert v1["ridge_shift"] == pytest.approx(0.0)
    assert v1["verdict"] == "PROMOTE"
    assert v1["balance"] is not None
    assert sum(v1["balance"]) == pytest.approx(1.0)


def test_no_promotion_small_delta_and_small_ridge_shift():
    df = pd.DataFrame(_all_rows())
    result = SC.screen(df, PHASE, baseline_phase=BASELINE_PHASE)

    v2 = next(v for v in result["variants"] if v["label"]["up"] == 0.03)
    assert v2["ridge"]["pooled"] == pytest.approx(0.21)
    for w in WINDOWS:
        assert v2["delta"][w] == pytest.approx(0.005)
    assert v2["windows_passing_delta"] == 0
    assert v2["ridge_shift"] == pytest.approx(0.01)
    assert v2["verdict"] == "NO"


def test_promote_by_ridge_shift_label_harder():
    df = pd.DataFrame(_all_rows())
    result = SC.screen(df, PHASE, baseline_phase=BASELINE_PHASE)

    v3 = next(v for v in result["variants"] if v["label"]["down"] == 0.03)
    assert v3["ridge"]["pooled"] == pytest.approx(0.14)
    assert v3["ridge_shift"] == pytest.approx(-0.06)
    assert v3["windows_passing_delta"] == 0
    assert v3["verdict"] == "NO"
    assert any("label harder" in n for n in v3["notes"])


def test_missing_mlp_rows_is_incomplete():
    df = pd.DataFrame(_all_rows())
    result = SC.screen(df, PHASE, baseline_phase=BASELINE_PHASE)

    v4 = next(v for v in result["variants"] if v["label"]["horizon"] == 24)
    assert v4["verdict"] == "INCOMPLETE"
    assert any("MLP_FAM" in n for n in v4["notes"])
    assert v4["mlp"]["by_window"] == {}


def test_render_markdown_contains_every_variant_and_constitution_note():
    df = pd.DataFrame(_all_rows())
    result = SC.screen(df, PHASE, baseline_phase=BASELINE_PHASE)
    md = SC.render_markdown(result)

    assert "Constitution III" in md
    assert "PROMOTE" in md
    assert "INCOMPLETE" in md
    for v in result["variants"]:
        assert v["variant_display"] in md


def test_write_screen_produces_md_and_json(tmp_path):
    results_path = tmp_path / "results" / "results.jsonl"
    results_path.parent.mkdir(parents=True)
    write_jsonl(results_path, _all_rows())

    out_dir = tmp_path / "report"
    result = SC.write_screen(PHASE, baseline_phase=BASELINE_PHASE, results_path=results_path, out_dir=out_dir)

    md_path = out_dir / f"screen_{PHASE}.md"
    json_path = out_dir / f"screen_{PHASE}.json"
    assert md_path.exists()
    assert json_path.exists()

    loaded = json.loads(json_path.read_text())
    assert loaded["phase"] == PHASE
    assert len(loaded["variants"]) == len(result["variants"])


def test_no_rows_for_phase_returns_empty_variants():
    df = pd.DataFrame(_baseline_rows())
    result = SC.screen(df, PHASE, baseline_phase=BASELINE_PHASE)
    assert result["variants"] == []
    assert result["baseline_ridge_pooled"] == pytest.approx(0.20)


def test_empty_dataframe():
    result = SC.screen(pd.DataFrame(), PHASE, baseline_phase=BASELINE_PHASE)
    assert result["variants"] == []
    assert np.isnan(result["baseline_ridge_pooled"])
