"""Tests for research/clfstudy/report.py (T061, data-model.md §Report, spec FR-035, SC-009).

Builds a tiny synthetic artefact tree (`methods.md`, `leaderboard_2.md` with Table 1/2/3
headings, `gates/phase_2.md`, `gates/review_2.md`, a two-row `results.jsonl` carrying
`trig_*` fields) under `tmp_path`, runs `report.assemble()`/`write_report()` against it
(paths parametrised, never the real study artefacts), and checks the section skeleton and
the parsed leaderboard tables are present. Also checks nothing raises when files (gates,
review memos, calibration.md, quickstart.md) are missing, and exercises
`scan_for_forbidden_terms()`.
"""
from __future__ import annotations

import json

import pytest

from research.clfstudy import report as R


def make_row(config_id, window_id, seed, *, arch="LSTM", phase=4, status="ok",
             mcc=0.3, macro_f1=0.3, bal_acc=0.3, input_set="ind24", seq_len=16,
             split_set="v1", trig=None, **overrides) -> dict:
    """One minimal RunResult-shaped row, optionally carrying `trig_*` fields."""
    row = {
        "config_id": config_id,
        "split_set": split_set,
        "window_id": window_id,
        "seed": seed,
        "phase": phase,
        "arch": arch,
        "hp": {},
        "input_set": input_set,
        "seq_len": seq_len,
        "train": {"max_epochs": 5, "patience": 5, "batch_size": 64, "lr": 1e-4,
                   "weight_decay": 0.0, "stride": 1},
        "tags": [],
        "status": status,
        "error": None,
        "pairs_included": ["ZEC/USDT"],
        "pairs_excluded": [],
        "n_train": 100,
        "n_val": 10,
        "n_test": 20,
        "class_counts_train": [33, 34, 33],
        "class_counts_test": [7, 7, 6],
        "pred_counts": [7, 7, 6],
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "prec_sell": 0.3, "prec_hold": 0.3, "prec_buy": 0.3,
        "rec_sell": 0.3, "rec_hold": 0.3, "rec_buy": 0.3,
        "logloss": 1.0,
        "auroc_sell": 0.5, "auroc_hold": 0.5, "auroc_buy": 0.5, "auroc_macro": 0.5,
        "ece": 0.1,
        "reliability_bins": [],
        "confusion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "buy_prec_at_08": 0.5,
        "buy_cov_at_08": 0.1,
        "n_params": 100,
        "train_seconds": 1.0,
        "epochs_run": 3,
        "best_epoch": 2,
        "infer_ms_per_window_b256": 0.1,
        "infer_ms_per_window_b1": 0.2,
        "harness_version": "test",
        "git_sha": "deadbeef",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "weights_path": None,
    }
    if trig:
        row.update(trig)
    row.update(overrides)
    return row


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


FAKE_LEADERBOARD_MD = """# Leaderboard: phase 2

Classification metrics against fixed gbb labels; NOT evidence of trading edge (Constitution III).

## Table 1: pooled leaderboard (seed counts per config: RIDGE=1)

| rank | arch | config_id | MCC |
|---|---|---|---|
| 1 | RIDGE | ridge0000000 | 0.500 |

## Table 2: per-window MCC and macro-F1 (mean ± seed-std)

| arch | config_id | MCC A |
|---|---|---|
| RIDGE | ridge0000000 | 0.500 |

## Table 3: predicted-class share (pooled pred_counts / n_test)

| config_id | arch | SELL share | HOLD share | BUY share |
|---|---|---|---|---|
| ridge0000000 | RIDGE | 0.1 | 0.8 | 0.1 |
"""

FAKE_METHODS_MD = """# Methods notes

## Label fidelity (T026)

IDENTICAL (n=1), exit 0.
"""

FAKE_PHASE_GATE_MD = """# Gate decision: phase 2

Classification metrics against fixed gbb labels; NOT evidence of trading edge (Constitution III).

## Viable set

- cfgabcdef123
"""

FAKE_REVIEW_MD = """# Gate-2 review: fixture

## 1. Sanity checks

All good.

## 2. Decision and recommendations

- Carry LSTM, drop MLP.

## 3. Open questions

1. Is the fixture representative?
"""


@pytest.fixture
def artefact_tree(tmp_path):
    report_dir = tmp_path / "report"
    gates_dir = tmp_path / "gates"
    results_dir = tmp_path / "results"
    report_dir.mkdir()
    gates_dir.mkdir()
    results_dir.mkdir()

    (report_dir / "methods.md").write_text(FAKE_METHODS_MD, encoding="utf-8")
    (report_dir / "leaderboard_2.md").write_text(FAKE_LEADERBOARD_MD, encoding="utf-8")
    (gates_dir / "phase_2.md").write_text(FAKE_PHASE_GATE_MD, encoding="utf-8")
    (gates_dir / "review_2.md").write_text(FAKE_REVIEW_MD, encoding="utf-8")

    results_path = results_dir / "results.jsonl"
    rows = [
        make_row(
            "cfgabcdef123", "B", 0, arch="LSTM", phase=3, mcc=0.6,
            trig={
                "trig_buy_n": 100, "trig_buy_base_rate": 0.5, "trig_buy_prec_argmax": 0.55,
                "trig_buy_auroc": 0.6, "trig_buy_mcc": 0.05,
                "trig_sell_n": 80, "trig_sell_base_rate": 0.4, "trig_sell_prec_argmax": 0.41,
                "trig_sell_auroc": 0.52, "trig_sell_mcc": 0.01,
                "trig_rule_mcc": 0.64,
            },
        ),
        make_row(
            "cfgxyz789012", "C", 0, arch="MLP", phase=4, mcc=0.55,
            trig={
                "trig_buy_n": 90, "trig_buy_base_rate": 0.45, "trig_buy_prec_argmax": 0.47,
                "trig_buy_auroc": 0.58, "trig_buy_mcc": 0.03,
                "trig_sell_n": 70, "trig_sell_base_rate": 0.42, "trig_sell_prec_argmax": 0.43,
                "trig_sell_auroc": 0.51, "trig_sell_mcc": 0.02,
                "trig_rule_mcc": 0.62,
            },
        ),
    ]
    write_jsonl(results_path, rows)

    quickstart_path = tmp_path / "quickstart.md"
    quickstart_path.write_text(
        "# Quickstart\n\n```bash\npytest research/clfstudy/tests -q\n```\n\nDone.\n",
        encoding="utf-8",
    )

    return {
        "report_dir": report_dir,
        "gates_dir": gates_dir,
        "results_path": results_path,
        "quickstart_path": quickstart_path,
    }


# ---------------------------------------------------------------------------
# extract_leaderboard_tables
# ---------------------------------------------------------------------------


def test_extract_leaderboard_tables_slices_between_table1_and_table3():
    out = R.extract_leaderboard_tables(FAKE_LEADERBOARD_MD)
    assert out.startswith("## Table 1")
    assert "## Table 2" in out
    assert "## Table 3" not in out
    assert "ridge0000000" in out


def test_extract_leaderboard_tables_missing_table1_returns_whole_text():
    out = R.extract_leaderboard_tables("# no tables here\n")
    assert "no tables here" in out


# ---------------------------------------------------------------------------
# trigger-rule table
# ---------------------------------------------------------------------------


def test_build_trigger_rule_section(artefact_tree):
    section = R.build_trigger_rule_section(artefact_tree["results_path"])
    assert "Trigger-rule MCC per window" in section
    # window B has one row at trig_rule_mcc=0.64, window C one row at 0.62
    assert "| B | 0.640 | 1 |" in section
    assert "| C | 0.620 | 1 |" in section
    # phase 3/4 within-trigger table: LSTM (phase 3, window B) and MLP (phase 4, window C)
    assert "| 3 | LSTM | B |" in section
    assert "| 4 | MLP | C |" in section


def test_build_trigger_rule_section_empty_results(tmp_path):
    empty = tmp_path / "results.jsonl"
    empty.write_text("", encoding="utf-8")
    section = R.build_trigger_rule_section(empty)
    assert "No `trig_*` rows found" in section


# ---------------------------------------------------------------------------
# narrative extraction
# ---------------------------------------------------------------------------


def test_build_narrative_section_finds_decision_heading(artefact_tree):
    out = R.build_narrative_section(artefact_tree["gates_dir"], r"decision|recommend", "<!-- EMPTY -->")
    assert "Decision and recommendations" in out
    assert "Carry LSTM, drop MLP." in out
    assert "review_2.md" in out


def test_build_narrative_section_open_questions(artefact_tree):
    out = R.build_narrative_section(artefact_tree["gates_dir"], r"open questions?", "<!-- EMPTY -->")
    assert "Open questions" in out
    assert "representative" in out


def test_build_narrative_section_no_match_returns_stub(tmp_path):
    gates_dir = tmp_path / "gates_empty"
    gates_dir.mkdir()
    (gates_dir / "review_9.md").write_text("# Review\n\n## Nothing relevant\n\ntext\n", encoding="utf-8")
    out = R.build_narrative_section(gates_dir, r"decision|recommend", "<!-- EMPTY -->")
    assert out == "<!-- EMPTY -->"


# ---------------------------------------------------------------------------
# gates section
# ---------------------------------------------------------------------------


def test_build_gates_section_includes_phase_and_review(artefact_tree):
    out = R.build_gates_section(artefact_tree["gates_dir"])
    assert "Phase 2 gate decision" in out
    assert "Gate decision: phase 2" in out
    assert "Review memo: `review_2.md`" in out
    assert "Gate-2 review: fixture" in out


def test_build_gates_section_no_files(tmp_path):
    empty_gates = tmp_path / "no_gates"
    empty_gates.mkdir()
    out = R.build_gates_section(empty_gates)
    assert "No `gates/phase_*.md` files found" in out


# ---------------------------------------------------------------------------
# reproduction commands
# ---------------------------------------------------------------------------


def test_build_reproduction_commands_section(artefact_tree):
    out = R.build_reproduction_commands_section(artefact_tree["quickstart_path"])
    assert "```bash" in out
    assert "pytest research/clfstudy/tests -q" in out


def test_build_reproduction_commands_section_missing_file(tmp_path):
    out = R.build_reproduction_commands_section(tmp_path / "nope.md")
    assert "Not found" in out


# ---------------------------------------------------------------------------
# assemble / write_report end to end
# ---------------------------------------------------------------------------


def test_assemble_contains_full_section_skeleton(artefact_tree):
    text = R.assemble(
        report_dir=artefact_tree["report_dir"],
        gates_dir=artefact_tree["gates_dir"],
        results_path=artefact_tree["results_path"],
        quickstart_path=artefact_tree["quickstart_path"],
    )
    assert "# NNC Classifier Study Report" in text
    assert "not evidence of trading edge" in text.lower()
    assert "## 1. Scope & methods" in text
    assert "IDENTICAL (n=1)" in text  # methods.md verbatim
    assert "## 2. Leaderboards" in text
    assert "### Phase 2" in text
    assert "ridge0000000" in text
    assert "## 3. Gate decisions" in text
    assert "Phase 2 gate decision" in text
    assert "## 4. Trigger-rule reference" in text
    assert "## 5. Recommendation" in text
    assert "## 6. Label concerns" in text
    assert "## 7. Open questions" in text
    assert "## 8. Reproduction commands" in text
    assert "pytest research/clfstudy/tests -q" in text


def test_assemble_missing_calibration_and_quickstart_does_not_raise(artefact_tree):
    text = R.assemble(
        report_dir=artefact_tree["report_dir"],
        gates_dir=artefact_tree["gates_dir"],
        results_path=artefact_tree["results_path"],
        quickstart_path=artefact_tree["gates_dir"] / "does_not_exist.md",
    )
    assert "Not found" in text
    assert "## 8. Reproduction commands" in text


def test_assemble_missing_gates_dir_does_not_raise(artefact_tree, tmp_path):
    empty_gates = tmp_path / "another_empty_gates"
    empty_gates.mkdir()
    text = R.assemble(
        report_dir=artefact_tree["report_dir"],
        gates_dir=empty_gates,
        results_path=artefact_tree["results_path"],
        quickstart_path=artefact_tree["quickstart_path"],
    )
    assert "No `gates/phase_*.md` files found" in text


def test_write_report_writes_file(artefact_tree):
    out_path = R.write_report(
        report_dir=artefact_tree["report_dir"],
        gates_dir=artefact_tree["gates_dir"],
        results_path=artefact_tree["results_path"],
        quickstart_path=artefact_tree["quickstart_path"],
    )
    assert out_path.name == "report.md"
    assert out_path.is_file()
    assert "# NNC Classifier Study Report" in out_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# scan_for_forbidden_terms (SC-009)
# ---------------------------------------------------------------------------


def test_scan_for_forbidden_terms_finds_hits(tmp_path):
    report_path = tmp_path / "report.md"
    report_path.write_text(
        "Nothing about backtest here except this scope note.\nAn innocuous line.\n",
        encoding="utf-8",
    )
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "mod.py").write_text("# drawdown is out of scope\nx = 1\n", encoding="utf-8")

    hits = R.scan_for_forbidden_terms(report_path, pkg_dir)
    files_hit = {h[0] for h in hits}
    assert str(report_path) in files_hit
    assert any("mod.py" in f for f in files_hit)
    assert len(hits) == 2


def test_scan_for_forbidden_terms_missing_report_does_not_raise(tmp_path):
    pkg_dir = tmp_path / "pkg2"
    pkg_dir.mkdir()
    (pkg_dir / "mod.py").write_text("x = 1\n", encoding="utf-8")
    hits = R.scan_for_forbidden_terms(tmp_path / "does_not_exist.md", pkg_dir)
    assert hits == []
