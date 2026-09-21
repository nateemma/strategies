# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Unattended architecture-search campaign driver for the `forward_mae` label.

See `report/campaign_plan.md` for the goal, the funnel and the operating rules; this
module only encodes the mechanics: which configs each stage runs, how each stage's
top-N selection is computed and recorded, and how the whole thing resumes cleanly after
being killed and restarted at any point (it is meant to run unattended for weeks).

Funnel (campaign_plan.md "Funnel"): stages A, B and C are screened on evaluation window
A only; stage D pays for the full three-window, three-seed protocol.
`phase.expand_runs` is config-major (`for cfg: for window: for seed`), so capping its
output with a global `--max-runs` does NOT give window A across several configs -- it
spends the budget on the first config's three windows. `_expand_runs_for_windows` below
is this module's own window-restricted run expansion, built directly on
`phase._load_existing_statuses`/`phase._should_skip` (never re-implementing the skip
logic itself).

Every actual training run goes through `phase._run_isolated` unchanged: one fresh
subprocess per (config, window, seed), whose inline `--config` JSON carries every
`ExperimentConfig` field including `label` (`phase._isolated_cmd`'s docstring records the
2026-09-19 bug this guards against: a config trained on the default gbb label although
its id said otherwise). A crashed/OOM-killed/hung run therefore costs one row, never the
campaign; `campaign.sh` runs the external watchdog that kills a stalled trainer.

Selection reuses `leaderboard.load_results`/`aggregate` (never re-implements seed
averaging or pooling): rows are restricted to `status == "ok"` for exactly the stage's
own config_ids *before* calling `aggregate`, then ranked by `pooled_mcc_mean` desc,
`pooled_macro_f1_mean` desc, `pooled_n_params_mean` asc. `registry.is_baseline` (the
training-free `MAJORITY`/`PERSISTENCE` pseudo-archs) is excluded from every
architecture-promotion ranking; `RIDGE`/`ROCKET` are real `ClassifierTypeMLX` members and
are not excluded by this rule (`ROCKET` is excluded from the whole campaign separately,
see `EXCLUDED_ARCHS`).

Every stage's chosen set is written to `results/campaign_stage_<X>_selection.json`: both
the audit trail (data-model.md-style "how did we get here") and the resumability marker
-- a restart with that file already on disk re-reads it rather than recomputing the pick
against a `results.jsonl` that may have grown since (e.g. from a differently-tagged phase
run touching the same config_ids). `results/campaign_state.json` separately records, per
stage, whether the one retry-equivalent pass at the end of that stage has already run --
needed because "every run key has a results.jsonl row" can never be guaranteed as a
resumable completion signal on its own: a run the watchdog kills twice in a row (once in
the main pass, once in the retry pass) leaves NO row at all, ever, and the operating
rules say to move on regardless rather than retry forever.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from research.clfstudy import RESULTS_PATH, assert_interpreter
from research.clfstudy import leaderboard
from research.clfstudy import phase as phase_mod
from research.clfstudy import registry
from research.clfstudy import run as run_mod
from research.clfstudy.registry import ExperimentConfig
from NNNC.NNNClassifierMLX import ClassifierTypeMLX

RunKey = Tuple[str, str, str, int]  # (config_id, split_set, window_id, seed)

RESULTS_DIR = RESULTS_PATH.parent

# ---------------------------------------------------------------------------
# Fixed campaign-wide settings (campaign_plan.md "THE LABEL" / "Base train block")
# ---------------------------------------------------------------------------

CAMPAIGN_LABEL: Dict[str, Any] = {
    "name": "forward_mae",
    "horizon": 12,
    "min_gain": 0.015,
    "min_loss": 0.015,
    "max_drawdown": 0.1,
}
BASE_TRAIN: Dict[str, Any] = {"lr": 0.0003, "stride": 2}
SPLIT_SET = "v1"
SCREEN_SEQ_LEN = 64

STAGE_ORDER: Tuple[str, ...] = ("A", "B", "C", "D")

#: window restriction per stage (campaign_plan.md "Funnel"): A/B/C screen on window A
#: alone; D (`None`) runs every window in the split set (A, B, C).
STAGE_WINDOWS: Dict[str, Optional[Set[str]]] = {
    "A": {"A"}, "B": {"A"}, "C": {"A"}, "D": None,
}

STAGE_A_SEEDS: List[int] = [0, 1]
STAGE_B_SEEDS: List[int] = [0, 1]
STAGE_C_SEEDS: List[int] = [0, 1]
STAGE_D_SEEDS: List[int] = [0, 1, 2]

STAGE_B_INPUT_SETS: Tuple[str, ...] = ("raw8", "ind24", "both32")
STAGE_C_SEQ_LENS: Tuple[int, ...] = (32, 64)

TOP_A = 8
TOP_B = 6
TOP_C = 4

#: phase numbers used only to tag/echo these configs in results.jsonl (never part of
#: config_id) -- one past the highest phase_N.yaml file in configs/ (phase_32).
STAGE_PHASE_NUM: Dict[str, int] = {"A": 33, "B": 34, "C": 35, "D": 36}

#: architectures excluded from the whole campaign on cost/OOM grounds
#: (campaign_plan.md "Exclusions"), measured against this machine's 64 GB physical RAM:
#:   ROCKET   71,842 MB peak Metal -- EXCEEDS physical RAM; also already has a `failed`
#:            and a `collapsed` row in results.jsonl. The worst offender; not on the plan
#:            doc's original exclusion list, added after measurement.
#:   Mamba    48,467 MB peak, up to 4,062 s/run (~68 min) -- ~76% of RAM.
#: Mamba2 (13,299 MB) and TSMamba (17,826 MB) are in the SAME band as LSTM_FAM
#: (17.3-19.9 GB) and are NOT excluded. Peak memory is roughly flat in seq_len (LSTM_FAM
#: measured 19.9 GB @ sl16, 17.3 GB @ sl64, 32.8 GB @ sl256), so the sl16/sl64
#: measurements above are a fair guide at this campaign's seq_len 64.
EXCLUDED_ARCHS: Tuple[str, ...] = ("Mamba", "ROCKET", "TSMamba")

# TSMamba added 2026-09-21 from a stage-A measurement: peak memory is NOT flat in
# seq_len for the Mamba family, which is what the original include-decision assumed.
# Mamba2 measured 13,299 MB at seq_len 16 but 46,653 MB at seq_len 64 (3.5x). TSMamba
# is bidirectional Mamba2 and ran 1.34x Mamba2 at seq_len 16, which projects to ~62 GB
# at seq_len 64 on a 64 GB machine. Excluded before it ran rather than after.
#: `patch_len` for every PATCHTST hp variant below is capped so `4 * patch_len <=
#: seq_len` holds at BOTH stage-C seq_len values (32 and 64) -- `_PatchTSTModel.__init__`
#: raises `ValueError` otherwise (verified against the code 2026-09-21). The type's own
#: default (`patch_len=16`) is fine at stage A/B (seq_len 64 only) but would violate the
#: 4x rule at seq_len 32, so it must never appear in a variant used past stage B; see
#: `_check_patchtst_constraint`.
_PATCHTST_MIN_SEQ_LEN_FACTOR = 4

#: hp variants for stage C's refinement grid (campaign_plan.md stage C), keyed by the
#: `ClassifierTypeMLX` name whose `create_model` each key was verified against. An arch
#: reaching stage C that has no entry here contributes a single cell at its own default
#: hp (`{}`) -- every other registry member has a fixed architecture and no tunable hp at
#: all, so `{}` is its only sensible cell.
HP_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "LSTM_FAM": [
        {},
        {"hidden": 128, "depth": 2},
        {"pool": "mean", "head": "bottleneck"},
    ],
    "GRU_FAM": [
        {},
        {"hidden": 128, "depth": 2},
        {"pool": "mean", "head": "bottleneck"},
    ],
    "TCN_FAM": [
        {},
        {"channels": 128, "kernel": 5},
        {"residual": True, "dilation_base": 2},
    ],
    "TRANSFORMER_FAM": [
        {},
        {"d_model": 128, "heads": 8},
        {"pos": "rope", "layers": 3},
    ],
    "MLP_FAM": [
        # width=64 first: the type's own default is width=256, but every MLP_FAM run in
        # this study so far used width=64 -- keeping that variant makes stage C
        # comparable with the existing results instead of silently drifting off them.
        {"width": 64},
        {"width": 128, "depth": 2},
        {"width": 256, "depth": 3},
    ],
    "TSMIXER": [
        {},
        {"blocks": 4, "ff_mult": 4},
    ],
    "ITRANSFORMER": [
        {},
        {"d_model": 128, "heads": 8, "layers": 3},
    ],
    "XLSTM": [
        {},
        {"hidden": 128, "depth": 2, "heads": 8},
    ],
    "TRANSFORMER_MODERN": [
        {},
        {"d_model": 128, "heads": 8, "layers": 3},
    ],
    "PATCHTST": [
        {"patch_len": 8, "stride": 4},
        {"patch_len": 8, "stride": 4, "d_model": 128, "heads": 8},
    ],
}


def _check_patchtst_constraint(cfg: ExperimentConfig) -> None:
    """Raise `ValueError` if `cfg` is a PATCHTST config that would violate
    `_PatchTSTModel`'s `seq_len >= 4 * patch_len` constraint. `patch_len` defaults to 16
    (the registry type's own default) when `cfg.hp` doesn't override it."""
    if cfg.arch != "PATCHTST":
        return
    patch_len = cfg.hp.get("patch_len", 16)
    if cfg.seq_len < _PATCHTST_MIN_SEQ_LEN_FACTOR * patch_len:
        raise ValueError(
            f"PATCHTST config seq_len={cfg.seq_len} violates seq_len >= "
            f"{_PATCHTST_MIN_SEQ_LEN_FACTOR} * patch_len (patch_len={patch_len})"
        )


def _make_cfg(
    arch: str,
    hp: Dict[str, Any],
    input_set: str,
    seq_len: int,
    seeds: List[int],
    stage: str,
    extra_tags: List[str],
) -> ExperimentConfig:
    """One validated `ExperimentConfig` for this campaign: fixed `label`/`train`/
    `split_set`, everything else from the caller. Raises on the first violated rule
    (`ExperimentConfig.validate`) or PATCHTST's seq_len constraint -- a stage's config
    list must never silently include a mis-specified experiment."""
    cfg = ExperimentConfig(
        arch=arch,
        hp=dict(hp),
        input_set=input_set,
        seq_len=seq_len,
        split_set=SPLIT_SET,
        train=dict(BASE_TRAIN),
        phase=STAGE_PHASE_NUM[stage],
        seeds=list(seeds),
        tags=["campaign", f"stage_{stage}"] + list(extra_tags),
        label=dict(CAMPAIGN_LABEL),
    )
    cfg.validate()
    _check_patchtst_constraint(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Stage config builders
# ---------------------------------------------------------------------------


def stage_a_configs() -> List[ExperimentConfig]:
    """Census (campaign_plan.md stage A): every `ClassifierTypeMLX` member -- enumerated
    at RUNTIME off the enum, so a new type registered later is picked up with no edit
    here -- minus `EXCLUDED_ARCHS`, plus `registry.BASELINE_ARCHS`. `hp={}` for every
    arch (its own defaults), `input_set='raw8'`, `seq_len=64`, 2 seeds."""
    archs = [m.name for m in ClassifierTypeMLX if m.name not in EXCLUDED_ARCHS]
    archs += list(registry.BASELINE_ARCHS)
    return [
        _make_cfg(arch, {}, "raw8", SCREEN_SEQ_LEN, STAGE_A_SEEDS, "A", [arch])
        for arch in archs
    ]


def stage_b_configs(top_archs: List[str]) -> List[ExperimentConfig]:
    """Input-set sweep (campaign_plan.md stage B): `top_archs` (stage A's selection)
    crossed with every `STAGE_B_INPUT_SETS` value, `seq_len=64`, 2 seeds."""
    return [
        _make_cfg(arch, {}, input_set, SCREEN_SEQ_LEN, STAGE_B_SEEDS, "B", [arch, input_set])
        for arch in top_archs
        for input_set in STAGE_B_INPUT_SETS
    ]


def stage_c_configs(top_cells: List[Dict[str, str]]) -> List[ExperimentConfig]:
    """Refinement (campaign_plan.md stage C): `top_cells` (stage B's selection, each
    `{"arch": ..., "input_set": ...}`) crossed with `HP_GRIDS.get(arch, [{}])` and
    `STAGE_C_SEQ_LENS`, 2 seeds."""
    configs = []
    for cell in top_cells:
        arch, input_set = cell["arch"], cell["input_set"]
        for hp in HP_GRIDS.get(arch, [{}]):
            for seq_len in STAGE_C_SEQ_LENS:
                configs.append(
                    _make_cfg(
                        arch, hp, input_set, seq_len, STAGE_C_SEEDS, "C",
                        [arch, input_set, f"sl{seq_len}"],
                    )
                )
    return configs


def stage_d_configs(top_cells: List[Dict[str, Any]]) -> List[ExperimentConfig]:
    """Promotion (campaign_plan.md stage D): `top_cells` (stage C's selection, each
    `{"arch", "input_set", "hp", "seq_len"}`) plus a `RIDGE` cell and the `MAJORITY`/
    `PERSISTENCE` tripwires, run on all 3 windows with 3 seeds.

    The top-4 cells can legitimately span more than one `input_set`/`seq_len` (stage C
    ranks per cell, not per input set), so the additions are made PER DISTINCT SHAPE
    rather than once against the best cell:

    * `RIDGE` is added for every distinct `(input_set, seq_len)` among `top_cells`.
      The study's viability rule is "beat the RIDGE baseline by a margin", so a single
      RIDGE fitted on one input set would make that margin a comparison of architecture
      AND inputs at once for every cell that used a different set. One RIDGE per shape
      keeps each comparison within its own input set. RIDGE is cheap; at most four are
      added.
    * `MAJORITY` and `PERSISTENCE` read no features at all, so `input_set` cannot change
      their score -- but `seq_len` changes how many windows are emitted, so they are
      added once per distinct `seq_len`, using the first cell with that `seq_len` for
      the (immaterial) `input_set`.

    If `RIDGE` already promoted into `top_cells` on its own merits, it is not added again
    for that same shape.
    """
    if not top_cells:
        raise RuntimeError(
            "stage D needs at least one stage-C cell (top_cells is empty) -- no "
            "reference input_set/seq_len for the RIDGE/MAJORITY/PERSISTENCE additions"
        )
    configs = [
        _make_cfg(
            cell["arch"], cell.get("hp", {}), cell["input_set"], cell["seq_len"],
            STAGE_D_SEEDS, "D", [cell["arch"], cell["input_set"], f"sl{cell['seq_len']}"],
        )
        for cell in top_cells
    ]
    # RIDGE: one per distinct (input_set, seq_len), skipping shapes where RIDGE already
    # promoted on its own merits. dict.fromkeys preserves stage-C rank order.
    promoted_ridge_shapes = {
        (c["input_set"], c["seq_len"]) for c in top_cells if c["arch"] == "RIDGE"
    }
    shapes = list(dict.fromkeys((c["input_set"], c["seq_len"]) for c in top_cells))
    for input_set, seq_len in shapes:
        if (input_set, seq_len) in promoted_ridge_shapes:
            continue
        configs.append(
            _make_cfg("RIDGE", {}, input_set, seq_len, STAGE_D_SEEDS, "D",
                      ["RIDGE", "ridge_reference", input_set, f"sl{seq_len}"])
        )

    # Tripwires: feature-blind, so one per distinct seq_len is enough.
    seen_seq_lens: Dict[int, str] = {}
    for input_set, seq_len in shapes:
        seen_seq_lens.setdefault(seq_len, input_set)
    for arch in registry.BASELINE_ARCHS:
        for seq_len, input_set in seen_seq_lens.items():
            configs.append(
                _make_cfg(arch, {}, input_set, seq_len, STAGE_D_SEEDS, "D",
                          [arch, "tripwire", f"sl{seq_len}"])
            )
    return configs


def stage_configs(stage: str) -> List[ExperimentConfig]:
    """Dispatch to the right `stage_<x>_configs` builder, reading any prior stage's
    selection file it depends on. Raises `RuntimeError` if that prior selection hasn't
    been written yet (i.e. the prior stage hasn't been run to completion)."""
    if stage == "A":
        return stage_a_configs()
    if stage == "B":
        selected = _read_selection("A")
        if selected is None:
            raise RuntimeError(
                "stage B requires stage A's selection "
                f"({_selection_path('A')}); run stage A first"
            )
        return stage_b_configs(selected)
    if stage == "C":
        selected = _read_selection("B")
        if selected is None:
            raise RuntimeError(
                "stage C requires stage B's selection "
                f"({_selection_path('B')}); run stage B first"
            )
        return stage_c_configs(selected)
    if stage == "D":
        selected = _read_selection("C")
        if selected is None:
            raise RuntimeError(
                "stage D requires stage C's selection "
                f"({_selection_path('C')}); run stage C first"
            )
        return stage_d_configs(selected)
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")


# ---------------------------------------------------------------------------
# Selection: results.jsonl -> ranked configs -> the next stage's input
# ---------------------------------------------------------------------------


def _rank(df, config_ids: List[str], skip_baselines: bool = True):
    """`leaderboard.aggregate()`'s output for exactly `config_ids`'s `status == "ok"`
    rows, sorted by `pooled_mcc_mean` desc, `pooled_macro_f1_mean` desc,
    `pooled_n_params_mean` asc (the campaign's tie-break rule). Never re-implements
    `aggregate`'s own per-window seed averaging / pooling -- only restricts its input
    (to `status == "ok"` and this stage's own config_ids) and re-sorts its output.
    Returns an empty DataFrame if there is nothing to rank yet.
    """
    if df is None or df.empty or "status" not in df.columns or "config_id" not in df.columns:
        return df.iloc[0:0] if df is not None else df
    ok = df[(df["status"] == "ok") & (df["config_id"].isin(config_ids))]
    agg = leaderboard.aggregate(ok)
    if agg.empty:
        return agg
    if skip_baselines:
        agg = agg[~agg["arch"].apply(registry.is_baseline)]
    sort_cols = [
        c for c in ("pooled_mcc_mean", "pooled_macro_f1_mean", "pooled_n_params_mean")
        if c in agg.columns
    ]
    ascending = [False, False, True][: len(sort_cols)]
    return agg.sort_values(by=sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)


def select_stage_a(df, configs: List[ExperimentConfig]) -> List[str]:
    """Top `TOP_A` arch names by mean MCC (`_rank`'s rule), baselines skipped."""
    ranked = _rank(df, [registry.config_id(c) for c in configs])
    if ranked.empty:
        return []
    return ranked["arch"].head(TOP_A).tolist()


def select_stage_b(df, configs: List[ExperimentConfig]) -> List[Dict[str, str]]:
    """Top `TOP_B` `(arch, input_set)` cells by mean MCC (`_rank`'s rule), baselines
    skipped."""
    ranked = _rank(df, [registry.config_id(c) for c in configs])
    if ranked.empty:
        return []
    top = ranked.head(TOP_B)
    return [{"arch": r.arch, "input_set": r.input_set} for r in top.itertuples()]


def select_stage_c(df, configs: List[ExperimentConfig]) -> List[Dict[str, Any]]:
    """Top `TOP_C` `(arch, input_set, hp, seq_len)` cells by mean MCC (`_rank`'s rule),
    baselines skipped. `hp` comes back as a real dict (`aggregate`'s echoed `hp` column is
    a JSON string, for CSV/parquet -- parsed back here)."""
    ranked = _rank(df, [registry.config_id(c) for c in configs])
    if ranked.empty:
        return []
    top = ranked.head(TOP_C)
    cells = []
    for r in top.itertuples():
        hp = r.hp
        hp = json.loads(hp) if isinstance(hp, str) else dict(hp or {})
        cells.append(
            {
                "arch": r.arch,
                "input_set": r.input_set,
                "hp": hp,
                "seq_len": int(r.seq_len),
                "config_id": r.config_id,
            }
        )
    return cells


# ---------------------------------------------------------------------------
# Selection files (audit trail + resumability marker)
# ---------------------------------------------------------------------------


def _selection_path(stage: str) -> Path:
    return RESULTS_DIR / f"campaign_stage_{stage}_selection.json"


def _write_selection(stage: str, rule: str, selected: Any) -> None:
    path = _selection_path(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "rule": rule,
        "selected": selected,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_selection(stage: str) -> Optional[Any]:
    path = _selection_path(stage)
    if not path.exists():
        return None
    return json.loads(path.read_text())["selected"]


def _write_stage_selection(stage: str, configs: List[ExperimentConfig]) -> None:
    """Compute and write `stage`'s selection file from the current `results.jsonl`, if
    it doesn't already exist (idempotent -- a restart never recomputes a decision that
    was already made)."""
    if _selection_path(stage).exists():
        return
    # NOT `leaderboard.load_results()` with no argument: that function's default
    # parameter value is bound once, at `leaderboard.py` import time, off ITS OWN
    # `RESULTS_PATH` global -- monkeypatching `campaign.RESULTS_DIR` in a test would
    # never reach it. Building the path from `RESULTS_DIR` here (looked up fresh on
    # every call) keeps this in step with the selection/state files above.
    df = leaderboard.load_results(RESULTS_DIR / "results.jsonl")
    if stage == "A":
        selected = select_stage_a(df, configs)
        rule = (
            f"top {TOP_A} archs by mean MCC (status=ok rows, window A, baselines "
            "skipped); ties: macro-F1 desc, then n_params asc"
        )
    elif stage == "B":
        selected = select_stage_b(df, configs)
        rule = (
            f"top {TOP_B} (arch, input_set) cells by mean MCC (status=ok rows, window "
            "A, baselines skipped); ties: macro-F1 desc, then n_params asc"
        )
    elif stage == "C":
        selected = select_stage_c(df, configs)
        rule = (
            f"top {TOP_C} (arch, input_set, hp, seq_len) cells by mean MCC (status=ok "
            "rows, window A, baselines skipped); ties: macro-F1 desc, then n_params asc"
        )
    else:
        raise ValueError(f"stage {stage!r} has no downstream selection")
    _write_selection(stage, rule, selected)
    print(f"campaign[{stage}]: selection written to {_selection_path(stage)} ({len(selected)} entries)")


# ---------------------------------------------------------------------------
# Window-restricted run expansion (the funnel's critical mechanic)
# ---------------------------------------------------------------------------


def _expand_runs_for_windows(
    configs: List[ExperimentConfig],
    existing: Dict[RunKey, str],
    windows: Optional[Set[str]],
    retry_failed: bool = False,
) -> Tuple[List[Tuple[ExperimentConfig, str, int]], int]:
    """Like `phase.expand_runs`, but restricted to `windows` (a set of window ids, or
    `None` for every window in each config's split set). This is the fix for
    `phase.expand_runs` being config-major with no window filter: stages A/B/C need
    window A swept across EVERY config, not window A of the first config followed by
    windows B/C of that same config. Reuses `phase._should_skip` (never re-implements
    the skip logic). Returns `(runs, skipped_count)`.
    """
    runs: List[Tuple[ExperimentConfig, str, int]] = []
    skipped = 0
    for cfg in configs:
        cid = registry.config_id(cfg)
        ss = run_mod.load_or_build_splitset(cfg.split_set)
        for w in ss.windows:
            if windows is not None and w.window_id not in windows:
                continue
            for seed in cfg.seeds:
                key: RunKey = (cid, cfg.split_set, w.window_id, seed)
                status = existing.get(key)
                if phase_mod._should_skip(status, retry_failed):
                    skipped += 1
                    continue
                runs.append((cfg, w.window_id, seed))
    return runs, skipped


def _expected_keys(configs: List[ExperimentConfig], windows: Optional[Set[str]]) -> List[RunKey]:
    """Every (config_id, split_set, window_id, seed) key `configs` x `windows` implies,
    with no skip logic -- used only to measure completion, never to decide what to run.
    """
    keys: List[RunKey] = []
    for cfg in configs:
        cid = registry.config_id(cfg)
        ss = run_mod.load_or_build_splitset(cfg.split_set)
        for w in ss.windows:
            if windows is not None and w.window_id not in windows:
                continue
            for seed in cfg.seeds:
                keys.append((cid, cfg.split_set, w.window_id, seed))
    return keys


def _runs_done(
    configs: List[ExperimentConfig], windows: Optional[Set[str]], existing: Dict[RunKey, str]
) -> Tuple[int, int]:
    """`(n_present, n_expected)` -- `run_one` only ever appends a row on completion (no
    in-progress status is ever persisted, research R9), so "present" == "attempted"."""
    keys = _expected_keys(configs, windows)
    present = sum(1 for k in keys if k in existing)
    return present, len(keys)


# ---------------------------------------------------------------------------
# Stage-progress state (retry-pass bookkeeping; NOT the run ledger, that is always
# results.jsonl itself)
# ---------------------------------------------------------------------------


def _state_path() -> Path:
    return RESULTS_DIR / "campaign_state.json"


def _load_state() -> Dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {"stages": {}}
    return json.loads(path.read_text())


def _save_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _mark_retry_pass_done(stage: str) -> None:
    state = _load_state()
    state.setdefault("stages", {}).setdefault(stage, {})["retry_pass_done"] = True
    state["stages"][stage]["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)


def _retry_pass_done(stage: str) -> bool:
    return bool(_load_state().get("stages", {}).get(stage, {}).get("retry_pass_done", False))


def stage_complete(stage: str) -> bool:
    """A stage is complete once every one of its (config, window, seed) keys has a row
    in `results.jsonl` -- OR, if some rows are permanently missing (e.g. a run the
    watchdog killed twice running), once this stage's one retry-equivalent pass has
    already been attempted (operating rules: "a failing individual run must not abort
    the stage ... then move on regardless"). A/B/C additionally require their own
    selection file to exist (it is what the next stage is built from); D has no
    downstream stage, so its completion is runs-only.
    """
    try:
        configs = stage_configs(stage)
    except RuntimeError:
        return False  # a prior stage hasn't been selected yet
    existing = phase_mod._load_existing_statuses()
    present, expected = _runs_done(configs, STAGE_WINDOWS[stage], existing)
    runs_ok = (expected > 0 and present == expected) or _retry_pass_done(stage)
    if stage == "D":
        return runs_ok
    return runs_ok and _selection_path(stage).exists()


# ---------------------------------------------------------------------------
# Running a stage
# ---------------------------------------------------------------------------


def _run_batch(runs: List[Tuple[ExperimentConfig, str, int]], stage: str) -> Tuple[int, int]:
    """Run every `(cfg, window_id, seed)` in `runs` via `phase._run_isolated`. A single
    run raising or exiting badly is recorded as failed and never aborts the batch
    (operating rules)."""
    ran = failed = 0
    n = len(runs)
    for i, (cfg, window_id, seed) in enumerate(runs, start=1):
        cid = registry.config_id(cfg)
        print(
            f"campaign[{stage}] [{i}/{n}] config_id={cid} arch={cfg.arch} "
            f"input_set={cfg.input_set} hp={cfg.hp} seq_len={cfg.seq_len} "
            f"window={window_id} seed={seed}"
        )
        try:
            returncode, status = phase_mod._run_isolated(cfg, window_id, seed)
        except Exception as exc:  # pragma: no cover - defensive, a run must never abort the stage
            print(f"campaign[{stage}]: run raised {exc!r}", file=sys.stderr)
            ran += 1
            failed += 1
            continue
        ran += 1
        if returncode not in (0, 1):
            print(f"campaign[{stage}]: child exited with unexpected code {returncode}", file=sys.stderr)
        if returncode != 0 or status not in phase_mod._SKIP_STATUSES:
            failed += 1
    return ran, failed


def run_stage(stage: str) -> int:
    """Run every not-yet-attempted (config, window, seed) for `stage`, then (if this
    stage's retry pass hasn't already run) one retry pass over anything left `failed`,
    then write the next stage's selection file (A/B/C only) once the runs are done.
    Returns 0 if `stage_complete(stage)` afterwards, 1 otherwise -- a `--run-all` caller
    stops there rather than promoting an incomplete stage.
    """
    configs = stage_configs(stage)
    windows = STAGE_WINDOWS[stage]

    existing = phase_mod._load_existing_statuses()
    runs, skipped = _expand_runs_for_windows(configs, existing, windows, retry_failed=False)
    print(f"campaign[{stage}]: {len(runs)} runs to attempt, {skipped} already done")
    ran, failed = _run_batch(runs, stage)
    print(f"campaign[{stage}]: main pass ran={ran} failed={failed}")

    if not _retry_pass_done(stage):
        existing = phase_mod._load_existing_statuses()
        retry_runs, _ = _expand_runs_for_windows(configs, existing, windows, retry_failed=True)
        if retry_runs:
            print(f"campaign[{stage}]: retry pass for {len(retry_runs)} failed runs")
            _run_batch(retry_runs, stage)
        _mark_retry_pass_done(stage)

    if stage != "D":
        _write_stage_selection(stage, configs)

    complete = stage_complete(stage)
    print(f"campaign[{stage}]: {'complete' if complete else 'incomplete'}")
    return 0 if complete else 1


def run_all() -> int:
    """Resume at the first incomplete stage and run every stage in order, stopping (a
    caller should re-invoke later) if a stage does not complete this pass."""
    for stage in STAGE_ORDER:
        if stage_complete(stage):
            print(f"campaign: stage {stage} already complete, skipping")
            continue
        print(f"campaign: running stage {stage}")
        rc = run_stage(stage)
        if rc != 0:
            print(f"campaign: stage {stage} did not complete this pass; stopping (resume later)")
            return rc
    print("campaign: all stages complete")
    return 0


def print_status() -> None:
    existing = phase_mod._load_existing_statuses()
    for stage in STAGE_ORDER:
        try:
            configs = stage_configs(stage)
        except RuntimeError as exc:
            print(f"stage {stage}: NOT READY ({exc})")
            continue
        present, expected = _runs_done(configs, STAGE_WINDOWS[stage], existing)
        sel = "n/a" if stage == "D" else ("written" if _selection_path(stage).exists() else "pending")
        retry = "yes" if _retry_pass_done(stage) else "no"
        print(
            f"stage {stage}: {present}/{expected} run keys present, "
            f"retry_pass_done={retry}, selection={sel}, complete={stage_complete(stage)}"
        )


def dry_run(stage: str) -> int:
    """Print every (config, window, seed) `stage` would run, without running anything.
    Also builds each run's isolated-subprocess command (`phase._isolated_cmd`) to prove
    every config actually serialises -- catches a bad hp block or an unpicklable value
    before the campaign spends days acting on it."""
    configs = stage_configs(stage)
    windows = STAGE_WINDOWS[stage]
    existing = phase_mod._load_existing_statuses()
    runs, skipped = _expand_runs_for_windows(configs, existing, windows, retry_failed=False)
    for cfg, window_id, seed in runs:
        cid = registry.config_id(cfg)
        phase_mod._isolated_cmd(cfg, window_id, seed)  # raises on a bad config
        print(
            f"[dry-run][{stage}] config_id={cid} arch={cfg.arch} hp={cfg.hp} "
            f"input_set={cfg.input_set} seq_len={cfg.seq_len} window={window_id} seed={seed}"
        )
    print(
        f"[dry-run][{stage}] {len(configs)} configs, {len(runs)} runs would execute, "
        f"{skipped} already done"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    try:
        assert_interpreter()
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 3

    parser = argparse.ArgumentParser(prog="python -m research.clfstudy.campaign")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--stage", choices=STAGE_ORDER, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.status:
            print_status()
            return 0
        if args.dry_run:
            if args.stage is None:
                parser.error("--dry-run requires --stage")
            return dry_run(args.stage)
        if args.stage is not None:
            return run_stage(args.stage)
        if args.run_all:
            return run_all()
        parser.error("specify one of --run-all, --stage, --status, --dry-run")
    except Exception as exc:  # any problem here must be a nonzero exit, never a swallowed warning
        traceback.print_exc()
        print(f"campaign: FATAL: {exc}", file=sys.stderr)
        return 2
    return 0  # unreachable (parser.error() always raises SystemExit)


if __name__ == "__main__":
    sys.exit(main())
