# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""Tests for `research/clfstudy/campaign.py` (T0xx, unattended forward_mae campaign).

Never trains anything: every stage's config-building/selection/completion logic is pure
python + pandas over synthetic `results.jsonl`-shaped rows, or a toy 3-window `SplitSet`
standing in for `splits/v1.json` (same pattern as `test_run_smoke.py`). Real subprocess
spawns (`phase._run_isolated`) are never exercised here -- only the argv it *would* build
(`phase._isolated_cmd`, monkeypatch-free) is checked for the label-carrying regression.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from NNNC.NNNClassifierMLX import ClassifierTypeMLX
from research.clfstudy import campaign as C
from research.clfstudy import phase as phase_mod
from research.clfstudy import registry
from research.clfstudy import run as run_mod
from research.clfstudy import splits as S

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _toy_splitset(window_ids=("A", "B", "C")) -> S.SplitSet:
    """A `SplitSet` with one `Window` per `window_ids`, real enough to build (embargo /
    val-fraction arithmetic all hold) but with no data behind it -- only `.windows` /
    `.window_id` is ever read by the code under test."""
    dates = pd.date_range("2024-01-01", periods=20000, freq="15min", tz="UTC")
    step = 4000
    windows = []
    for i, wid in enumerate(window_ids):
        start = i * step
        w = S._make_window(
            wid, dates[start], dates[start + 3000], dates[start + 3500], distant=True,
        )
        windows.append(w)
    return S.SplitSet(
        split_set="v1", embargo_bars=S.EMBARGO_BARS, val_fraction=S.VAL_FRACTION,
        min_history_days=1, timeframe="15m", pairs=["FOO/USDT"], windows=windows,
    )


@pytest.fixture(autouse=True)
def _patch_splitset(monkeypatch):
    """Every campaign function that needs windows goes through
    `run_mod.load_or_build_splitset` -- point it at the toy 3-window split set so no test
    here touches the real `splits/v1.json`."""
    toy = _toy_splitset()
    monkeypatch.setattr(run_mod, "_SPLITSET_CACHE", {})
    monkeypatch.setattr(run_mod, "load_or_build_splitset", lambda split_set: toy)


@pytest.fixture
def tmp_results_dir(monkeypatch, tmp_path):
    """Point `campaign.RESULTS_DIR` (selection files, `campaign_state.json`) and
    `phase_mod.RESULTS_PATH` (`results.jsonl`) at a scratch directory."""
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(phase_mod, "RESULTS_PATH", tmp_path / "results.jsonl")
    return tmp_path


def make_row(config_id, window_id, seed, *, arch, status="ok", mcc=0.3, macro_f1=0.3,
             n_params=100, input_set="raw8", seq_len=64, hp=None, phase=33, tags=None,
             split_set="v1"):
    """One synthetic `RunResult`-shaped row: only the fields `aggregate()`/selection
    actually touch."""
    return {
        "config_id": config_id, "split_set": split_set, "window_id": window_id,
        "seed": seed, "phase": phase, "arch": arch, "hp": hp or {},
        "input_set": input_set, "seq_len": seq_len, "tags": tags or [], "status": status,
        "mcc": mcc, "macro_f1": macro_f1, "bal_acc": 0.3, "n_params": n_params,
    }


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Stage A: enumerated from the enum, EXCLUDED_ARCHS honoured
# ---------------------------------------------------------------------------


def test_stage_a_enumerates_every_enum_member_plus_baselines():
    configs = C.stage_a_configs()
    archs = {c.arch for c in configs}

    all_members = {m.name for m in ClassifierTypeMLX}
    expected = (all_members - set(C.EXCLUDED_ARCHS)) | set(registry.BASELINE_ARCHS)
    assert archs == expected
    # every excluded arch really is excluded, not just "happens to be missing"
    for excluded in C.EXCLUDED_ARCHS:
        assert excluded in all_members  # sanity: it really is a registry member
        assert excluded not in archs


def test_stage_a_configs_use_the_campaign_label_and_screening_settings():
    configs = C.stage_a_configs()
    assert configs, "stage A produced no configs"
    for cfg in configs:
        assert cfg.label == C.CAMPAIGN_LABEL
        assert cfg.hp == {}
        assert cfg.input_set == "raw8"
        assert cfg.seq_len == C.SCREEN_SEQ_LEN
        assert cfg.seeds == C.STAGE_A_SEEDS
        assert cfg.train["lr"] == C.BASE_TRAIN["lr"]
        assert cfg.train["stride"] == C.BASE_TRAIN["stride"]


def test_stage_a_enumeration_is_driven_by_the_enum_at_call_time(monkeypatch):
    """Regression: stage A must enumerate `ClassifierTypeMLX` at CALL time, not from a
    hardcoded list -- shrink the enum campaign.py sees to a strict subset of real
    (already `registry`-valid) members and check stage A's output shrinks to match. If
    `stage_a_configs` instead read from a separately cached/hardcoded list, this
    monkeypatch would have no effect."""
    real_members = list(ClassifierTypeMLX)
    subset = [m for m in real_members if m.name in ("LSTM", "GRU", "MLP")]
    monkeypatch.setattr(C, "ClassifierTypeMLX", subset)

    archs = {c.arch for c in C.stage_a_configs()}
    assert archs == {"LSTM", "GRU", "MLP"} | set(registry.BASELINE_ARCHS)


# ---------------------------------------------------------------------------
# Window-A restriction: the bug-prone mechanic
# ---------------------------------------------------------------------------


def _cfg(arch="LSTM", seeds=(0, 1)):
    return registry.ExperimentConfig(
        arch=arch, hp={}, input_set="raw8", seq_len=64, split_set="v1",
        train={}, phase=33, seeds=list(seeds), label=dict(C.CAMPAIGN_LABEL),
    )


def test_window_restriction_yields_only_window_a_across_multiple_configs():
    configs = [_cfg("LSTM"), _cfg("GRU"), _cfg("MLP")]
    runs, skipped = C._expand_runs_for_windows(configs, existing={}, windows={"A"})

    assert skipped == 0
    assert runs  # something was produced
    assert {w for (_cfg_, w, _seed) in runs} == {"A"}
    # every config x every seed, exactly once, window A only (not 3x from B/C too)
    assert len(runs) == len(configs) * 2
    seen_archs = {cfg.arch for (cfg, _w, _s) in runs}
    assert seen_archs == {"LSTM", "GRU", "MLP"}


def test_window_restriction_none_yields_every_window_for_every_config():
    configs = [_cfg("LSTM"), _cfg("GRU")]
    runs, skipped = C._expand_runs_for_windows(configs, existing={}, windows=None)

    assert skipped == 0
    assert {w for (_cfg_, w, _seed) in runs} == {"A", "B", "C"}
    assert len(runs) == len(configs) * 3 * 2  # configs x windows x seeds


def test_window_restriction_respects_existing_statuses_and_retry_failed():
    cfg = _cfg("LSTM")
    cid = registry.config_id(cfg)
    existing = {
        (cid, "v1", "A", 0): "ok",
        (cid, "v1", "A", 1): "failed",
    }
    runs, skipped = C._expand_runs_for_windows([cfg], existing, windows={"A"}, retry_failed=False)
    assert skipped == 2  # both ok (skip) and failed-without-retry (skip)
    assert runs == []

    runs, skipped = C._expand_runs_for_windows([cfg], existing, windows={"A"}, retry_failed=True)
    assert skipped == 1  # ok still skips
    assert runs == [(cfg, "A", 1)]  # only the failed key comes back


# ---------------------------------------------------------------------------
# Selection: ranking + tie-break, baselines skipped, non-ok rows excluded
# ---------------------------------------------------------------------------


def test_select_stage_a_ranks_by_mcc_then_macro_f1_then_fewer_params():
    configs = [_cfg("LSTM"), _cfg("GRU"), _cfg("MLP")]
    cids = {c.arch: registry.config_id(c) for c in configs}
    rows = [
        # LSTM and GRU tie on MCC; GRU wins on macro_f1
        make_row(cids["LSTM"], "A", 0, arch="LSTM", mcc=0.5, macro_f1=0.4, n_params=100),
        make_row(cids["LSTM"], "A", 1, arch="LSTM", mcc=0.5, macro_f1=0.4, n_params=100),
        make_row(cids["GRU"], "A", 0, arch="GRU", mcc=0.5, macro_f1=0.6, n_params=50),
        make_row(cids["GRU"], "A", 1, arch="GRU", mcc=0.5, macro_f1=0.6, n_params=50),
        make_row(cids["MLP"], "A", 0, arch="MLP", mcc=0.1, macro_f1=0.9, n_params=10),
        make_row(cids["MLP"], "A", 1, arch="MLP", mcc=0.1, macro_f1=0.9, n_params=10),
    ]
    df = pd.DataFrame(rows)

    top = C.select_stage_a(df, configs)
    assert top == ["GRU", "LSTM", "MLP"]


def test_select_stage_a_tie_break_on_params_when_mcc_and_f1_match():
    configs = [_cfg("LSTM"), _cfg("GRU")]
    cids = {c.arch: registry.config_id(c) for c in configs}
    rows = [
        make_row(cids["LSTM"], "A", 0, arch="LSTM", mcc=0.5, macro_f1=0.4, n_params=500),
        make_row(cids["LSTM"], "A", 1, arch="LSTM", mcc=0.5, macro_f1=0.4, n_params=500),
        make_row(cids["GRU"], "A", 0, arch="GRU", mcc=0.5, macro_f1=0.4, n_params=50),
        make_row(cids["GRU"], "A", 1, arch="GRU", mcc=0.5, macro_f1=0.4, n_params=50),
    ]
    df = pd.DataFrame(rows)

    assert C.select_stage_a(df, configs) == ["GRU", "LSTM"]  # fewer params wins the tie


def test_select_stage_a_skips_baselines_and_failed_rows():
    configs = [_cfg("LSTM"), _cfg("MAJORITY")]
    cids = {c.arch: registry.config_id(c) for c in configs}
    rows = [
        # MAJORITY scores higher MCC but must never be promoted
        make_row(cids["MAJORITY"], "A", 0, arch="MAJORITY", mcc=0.9, macro_f1=0.9, n_params=0),
        make_row(cids["MAJORITY"], "A", 1, arch="MAJORITY", mcc=0.9, macro_f1=0.9, n_params=0),
        make_row(cids["LSTM"], "A", 0, arch="LSTM", mcc=0.4, macro_f1=0.4, n_params=100, status="failed"),
        make_row(cids["LSTM"], "A", 1, arch="LSTM", mcc=0.6, macro_f1=0.6, n_params=100),
    ]
    df = pd.DataFrame(rows)

    top = C.select_stage_a(df, configs)
    assert top == ["LSTM"]  # MAJORITY excluded; the failed LSTM row is dropped, not averaged in


def test_select_stage_b_returns_arch_input_set_cells():
    cfg_raw = C._make_cfg("LSTM", {}, "raw8", 64, [0, 1], "B", ["LSTM", "raw8"])
    cfg_ind = C._make_cfg("LSTM", {}, "ind24", 64, [0, 1], "B", ["LSTM", "ind24"])
    rows = [
        make_row(registry.config_id(cfg_raw), "A", 0, arch="LSTM", input_set="raw8", mcc=0.3),
        make_row(registry.config_id(cfg_raw), "A", 1, arch="LSTM", input_set="raw8", mcc=0.3),
        make_row(registry.config_id(cfg_ind), "A", 0, arch="LSTM", input_set="ind24", mcc=0.7),
        make_row(registry.config_id(cfg_ind), "A", 1, arch="LSTM", input_set="ind24", mcc=0.7),
    ]
    df = pd.DataFrame(rows)

    top = C.select_stage_b(df, [cfg_raw, cfg_ind])
    assert top[0] == {"arch": "LSTM", "input_set": "ind24"}
    assert top[1] == {"arch": "LSTM", "input_set": "raw8"}


def test_select_stage_c_parses_hp_back_to_a_dict():
    cfg = C._make_cfg("LSTM_FAM", {"hidden": 128, "depth": 2}, "raw8", 32, [0, 1], "C", ["x"])
    rows = [
        make_row(registry.config_id(cfg), "A", 0, arch="LSTM_FAM", hp={"hidden": 128, "depth": 2}, seq_len=32, mcc=0.5),
        make_row(registry.config_id(cfg), "A", 1, arch="LSTM_FAM", hp={"hidden": 128, "depth": 2}, seq_len=32, mcc=0.5),
    ]
    df = pd.DataFrame(rows)

    cells = C.select_stage_c(df, [cfg])
    assert len(cells) == 1
    assert cells[0]["hp"] == {"hidden": 128, "depth": 2}
    assert cells[0]["seq_len"] == 32
    assert cells[0]["arch"] == "LSTM_FAM"


# ---------------------------------------------------------------------------
# HP_GRIDS: verified keys, MLP_FAM historical width, PATCHTST seq_len guard, fallback
# ---------------------------------------------------------------------------


def test_hp_grids_only_cover_types_with_a_tunable_hp_schema():
    valid_archs = {m.name for m in ClassifierTypeMLX}
    assert set(C.HP_GRIDS) <= valid_archs


def test_mlp_fam_grid_includes_the_historical_width_64_variant():
    assert {"width": 64} in C.HP_GRIDS["MLP_FAM"]


def test_stage_c_fallback_for_arch_absent_from_hp_grids():
    assert "LSTM" not in C.HP_GRIDS
    top_cells = [{"arch": "LSTM", "input_set": "raw8"}]
    configs = C.stage_c_configs(top_cells)

    # one cell per seq_len (STAGE_C_SEQ_LENS), no hp variation
    assert len(configs) == len(C.STAGE_C_SEQ_LENS)
    assert all(cfg.hp == {} for cfg in configs)
    assert sorted(cfg.seq_len for cfg in configs) == sorted(C.STAGE_C_SEQ_LENS)


def test_stage_c_hp_grid_is_crossed_with_seq_lens_for_a_covered_arch():
    top_cells = [{"arch": "MLP_FAM", "input_set": "raw8"}]
    configs = C.stage_c_configs(top_cells)
    assert len(configs) == len(C.HP_GRIDS["MLP_FAM"]) * len(C.STAGE_C_SEQ_LENS)


def test_patchtst_hp_grid_never_violates_the_4x_seq_len_rule():
    top_cells = [{"arch": "PATCHTST", "input_set": "raw8"}]
    configs = C.stage_c_configs(top_cells)
    assert configs  # got something
    for cfg in configs:
        patch_len = cfg.hp.get("patch_len", 16)
        assert cfg.seq_len >= 4 * patch_len, (cfg.seq_len, cfg.hp)


def test_check_patchtst_constraint_raises_on_the_default_patch_len_at_seq_len_32():
    with pytest.raises(ValueError, match="seq_len=32"):
        C._make_cfg("PATCHTST", {}, "raw8", 32, [0, 1], "C", ["x"])


def test_check_patchtst_constraint_passes_at_seq_len_64_with_default_hp():
    cfg = C._make_cfg("PATCHTST", {}, "raw8", 64, [0, 1], "A", ["x"])
    assert cfg.arch == "PATCHTST"  # did not raise


# ---------------------------------------------------------------------------
# Stage D: promotion cells + RIDGE/tripwires, dedup
# ---------------------------------------------------------------------------


def test_stage_d_adds_ridge_and_tripwires_with_reference_cell_input_set():
    top_cells = [
        {"arch": "LSTM_FAM", "input_set": "ind24", "hp": {"hidden": 128}, "seq_len": 32},
        {"arch": "GRU_FAM", "input_set": "raw8", "hp": {}, "seq_len": 64},
    ]
    configs = C.stage_d_configs(top_cells)
    archs = [cfg.arch for cfg in configs]
    assert archs.count("LSTM_FAM") == 1
    assert archs.count("GRU_FAM") == 1
    assert "RIDGE" in archs
    assert "MAJORITY" in archs
    assert "PERSISTENCE" in archs

    ridge_cfg = next(cfg for cfg in configs if cfg.arch == "RIDGE")
    # reference = top_cells[0] (the best-ranked cell)
    assert ridge_cfg.input_set == "ind24"
    assert ridge_cfg.seq_len == 32
    assert ridge_cfg.seeds == C.STAGE_D_SEEDS


def test_stage_d_does_not_duplicate_ridge_if_already_promoted():
    top_cells = [
        {"arch": "RIDGE", "input_set": "raw8", "hp": {}, "seq_len": 64},
    ]
    configs = C.stage_d_configs(top_cells)
    assert [cfg.arch for cfg in configs].count("RIDGE") == 1


def test_stage_d_runs_every_window():
    top_cells = [{"arch": "LSTM_FAM", "input_set": "raw8", "hp": {}, "seq_len": 64}]
    configs = C.stage_d_configs(top_cells)
    runs, _ = C._expand_runs_for_windows(configs, existing={}, windows=C.STAGE_WINDOWS["D"])
    assert {w for (_c, w, _s) in runs} == {"A", "B", "C"}


def test_stage_d_raises_on_empty_top_cells():
    with pytest.raises(RuntimeError):
        C.stage_d_configs([])


# ---------------------------------------------------------------------------
# Stage dependency chaining + selection files (audit trail + resumability marker)
# ---------------------------------------------------------------------------


def test_stage_configs_raises_when_prior_selection_missing(tmp_results_dir):
    with pytest.raises(RuntimeError, match="stage A"):
        C.stage_configs("B")


def test_restart_reads_written_selection_instead_of_recomputing(tmp_results_dir):
    """A restart must re-read an already-written selection file rather than recompute it
    against a (possibly since-changed) results.jsonl -- write an arbitrary selection by
    hand and check `stage_configs('B')` uses exactly that, ignoring any df-derived pick."""
    C._write_selection("A", rule="manual for test", selected=["GRU", "MLP"])

    configs = C.stage_configs("B")
    archs = {cfg.arch for cfg in configs}
    assert archs == {"GRU", "MLP"}  # exactly what was written, not re-derived


def test_write_stage_selection_is_idempotent(tmp_results_dir):
    results_path = tmp_results_dir / "results.jsonl"
    configs = [_cfg("LSTM")]
    cid = registry.config_id(configs[0])
    write_jsonl(
        results_path,
        [make_row(cid, "A", 0, arch="LSTM", mcc=0.5), make_row(cid, "A", 1, arch="LSTM", mcc=0.5)],
    )

    C._write_stage_selection("A", configs)
    first = json.loads(C._selection_path("A").read_text())

    # even if the underlying results.jsonl changes, a second call must not recompute
    write_jsonl(results_path, [make_row(cid, "A", 0, arch="LSTM", mcc=0.9)])
    C._write_stage_selection("A", configs)
    second = json.loads(C._selection_path("A").read_text())

    assert first == second


# ---------------------------------------------------------------------------
# Stage completion detection + resume-from-partial-state
# ---------------------------------------------------------------------------


def test_stage_complete_false_until_every_key_present_and_selection_written(tmp_results_dir):
    results_path = tmp_results_dir / "results.jsonl"
    configs = C.stage_a_configs()

    assert C.stage_complete("A") is False  # nothing run yet

    # write every expected (window A only) key
    keys = C._expected_keys(configs, C.STAGE_WINDOWS["A"])
    rows = [
        make_row(cid, w, seed, arch=next(c.arch for c in configs if registry.config_id(c) == cid))
        for (cid, _ss, w, seed) in keys
    ]
    write_jsonl(results_path, rows)

    # runs are all present but the selection file hasn't been written yet
    assert C.stage_complete("A") is False

    C._write_stage_selection("A", configs)
    assert C.stage_complete("A") is True


def test_resume_from_partial_state_skips_only_whats_done(tmp_results_dir):
    results_path = tmp_results_dir / "results.jsonl"
    configs = [_cfg("LSTM"), _cfg("GRU")]
    cid_lstm = registry.config_id(configs[0])
    write_jsonl(results_path, [make_row(cid_lstm, "A", 0, arch="LSTM")])  # only 1 of 4 keys

    existing = phase_mod._load_existing_statuses()
    runs, skipped = C._expand_runs_for_windows(configs, existing, windows={"A"})

    assert skipped == 1
    assert len(runs) == 3  # the other 3 (config, window, seed) keys still need to run


def test_retry_pass_flag_lets_stage_complete_despite_a_permanently_missing_row(tmp_results_dir):
    """A run the watchdog kills twice leaves NO row, ever (research: no in-progress
    status is ever persisted) -- once this stage's retry pass has already been attempted,
    completion must not block on that row forever (operating rules: "move on
    regardless")."""
    results_path = tmp_results_dir / "results.jsonl"
    configs = [_cfg("LSTM", seeds=(0, 1))]
    cid = registry.config_id(configs[0])
    write_jsonl(results_path, [make_row(cid, "A", 0, arch="LSTM")])  # seed 1 never got a row
    C._write_selection("A", rule="test", selected=["LSTM"])

    assert C.stage_complete("A") is False  # 1/2 present, no retry pass recorded yet

    C._mark_retry_pass_done("A")
    assert C.stage_complete("A") is True


# ---------------------------------------------------------------------------
# Child-argv-carries-label regression test
# ---------------------------------------------------------------------------


def test_campaign_configs_carry_the_forward_mae_label_into_the_child_argv():
    """Regression: a previous bug had the child process silently train on the default
    gbb label although its config_id said otherwise (phase.py, 2026-09-19). Every
    campaign-generated config must carry `label` through `phase._isolated_cmd`'s inline
    `--config` JSON unchanged."""
    for cfg in (C.stage_a_configs()[0], C._make_cfg("LSTM", {}, "raw8", 64, [0], "D", [])):
        cmd = phase_mod._isolated_cmd(cfg, "A", 0)
        payload = json.loads(cmd[cmd.index("--config") + 1])
        assert payload["label"] == C.CAMPAIGN_LABEL
        assert payload["label"]["name"] == "forward_mae"


# ---------------------------------------------------------------------------
# --status / --dry-run smoke (never trains; read-only over the toy split set)
# ---------------------------------------------------------------------------


def test_dry_run_stage_a_returns_zero_and_lists_every_run(tmp_results_dir, capsys):
    rc = C.main(["--dry-run", "--stage", "A"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run][A]" in out


def test_status_reports_not_ready_for_downstream_stages(tmp_results_dir, capsys):
    rc = C.main(["--status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "stage A:" in out
    assert "NOT READY" in out  # stage B has no selection yet


def test_main_raises_systemexit_on_unknown_stage_argument():
    # argparse's `choices=STAGE_ORDER` rejects this before campaign code ever runs;
    # argparse.error() itself calls sys.exit(2) (a SystemExit, not a return value).
    with pytest.raises(SystemExit):
        C.main(["--stage", "Z"])


# --- stage D: baselines are added per distinct shape, not once ------------------


def _cell(arch, input_set, seq_len, hp=None):
    return {"arch": arch, "input_set": input_set, "seq_len": seq_len, "hp": hp or {}}


def test_stage_d_adds_one_ridge_per_distinct_shape():
    """The viability rule is 'beat RIDGE by a margin', so a cell trained on ind24 must be
    compared with a RIDGE on ind24 -- not with one fitted on raw8."""
    cells = [
        _cell("LSTM_FAM", "raw8", 64),
        _cell("TRANSFORMER_FAM", "ind24", 64),
        _cell("GRU_FAM", "raw8", 32),
        _cell("TCN_FAM", "ind24", 64),          # same shape as cell 2 -> no second RIDGE
    ]
    cfgs = C.stage_d_configs(cells)
    ridge_shapes = sorted(
        (c.input_set, c.seq_len) for c in cfgs if c.arch == "RIDGE"
    )
    assert ridge_shapes == [("ind24", 64), ("raw8", 32), ("raw8", 64)]


def test_stage_d_skips_ridge_where_it_already_promoted():
    cells = [_cell("RIDGE", "raw8", 64), _cell("LSTM_FAM", "ind24", 64)]
    cfgs = C.stage_d_configs(cells)
    ridge = [(c.input_set, c.seq_len) for c in cfgs if c.arch == "RIDGE"]
    # raw8/64 came in as a promoted cell; only ind24/64 is added as a reference.
    assert sorted(ridge) == [("ind24", 64), ("raw8", 64)]
    assert ridge.count(("raw8", 64)) == 1


def test_stage_d_tripwires_once_per_seq_len_not_per_input_set():
    """MAJORITY/PERSISTENCE read no features, so input_set cannot change their score;
    seq_len can, because it changes how many windows are emitted."""
    cells = [
        _cell("LSTM_FAM", "raw8", 64),
        _cell("TRANSFORMER_FAM", "ind24", 64),   # same seq_len, different inputs
        _cell("GRU_FAM", "raw8", 32),
    ]
    cfgs = C.stage_d_configs(cells)
    for arch in ("MAJORITY", "PERSISTENCE"):
        seq_lens = sorted(c.seq_len for c in cfgs if c.arch == arch)
        assert seq_lens == [32, 64], f"{arch} got {seq_lens}"


def test_stage_d_runs_every_window_and_three_seeds():
    cfgs = C.stage_d_configs([_cell("LSTM_FAM", "raw8", 64)])
    assert all(c.seeds == C.STAGE_D_SEEDS for c in cfgs)
    assert len(C.STAGE_D_SEEDS) == 3
