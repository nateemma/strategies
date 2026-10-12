---
description: "Task list for 029 NNNC / SimpleTrigger merge"
---

# Tasks: NNNC / SimpleTrigger Merge

**Input**: `specs/029-nnnc-simpletrigger-merge/` (spec.md, plan.md, research.md R1–R10, data-model.md,
contracts/interfaces.md I1–I5, quickstart.md)

**Tests**: Included. The spec requires each of these to be shown:
- byte-identical reproduction (SC-001);
- fingerprint neutrality (SC-002);
- stop invariants (SC-003);
- liveness of each switch (SC-004);
- no training (SC-005).

**Conventions**:
- All paths are relative to `user_data/strategies/`.
- Python is `../../.venv/bin/python`.
- Backtests run from the repo root with `set -a; source ~/.freqtrade_env; set +a`, as ONE sequential
  chain, under fresh tags.
- Commit to master, staging named files only.
- **No NNNC, SimpleTrigger or shared-mixin code changes before T009 is committed (the
  baseline).** T002 (report-only invariants in `utils/StopInvariants.py`) is the one exception;
  T003 proves it inert.
- **STOP gate at T008:** any stop-invariant violation in arm D's probe stops the feature before the
  refactor. Mark T010–T030 `[-]` with the reason, then do T031–T034 with the report as the defect write-up.
- **No model is trained.** Every model plan borrows arm D's per-fold weights; a missing model
  fails loudly.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [ ] T001 [P] Create `scripts/nnnc_fingerprints.py` per contract I4.
  - It enumerates the class named by each `NNNC/*.py` file stem and resolves its fingerprint with
    `evaluation.fingerprint` (`load_strategy_class`, `load_params_json`, `resolve`, config
    `config/config_wf.json`).
  - It writes `{class: sha16 | "LOAD_ERROR: <Exc>: <msg>" | "NOT_A_STRATEGY"}`. `NOT_A_STRATEGY`
    marks a stem with no same-named `IStrategy` subclass (mixins, classifiers).
  - With `--compare <before.json>` it prints added, removed, changed and error-changed classes, and
    exits 1 on any change.
  - It never instantiates a class.
  - Smoke test: run it once and check the counts (strategies, load errors, not-a-strategy).
- [ ] T002 [P] Add the report-only mode to `utils/StopInvariants.py` per contract I3.
  - `FT_STOP_INVARIANTS=report` records each violation to
    `$FT_STOP_INVARIANTS_SINK/<class>_<pair>_stopinv.json` (append to a list) and returns.
  - Unset or `halt` keeps today's behaviour.
  - The mode is read from the environment at every check, never cached at import (I3).
  - Tests in `utils/tests/test_stop_invariants.py`: report mode doesn't raise, writes the record
    and returns the parent value; halt mode is unchanged; switching the env var mid-process takes
    effect.
- [ ] T003 Prove T002 inert. Run `pytest utils/tests SimpleTrigger/tests` (all pass), and re-run
  `st_gbb` f00 under a fresh tag (`st_gbb_si029_f00`, plan derived from `plan_st_gbb.json`). Its
  `trades_hash` must equal the recorded `st_gbb` f00.
  - If it doesn't, the record may be stale (it predates 028's `BaseStrategy` change).
  - Attribute it before going on: re-run f00 under a second fresh tag with T002's change
    stashed. Equal hashes with and without T002 mean T002 is inert, and the stale record is noted
    in `baseline.md`.

---

## Phase 2: Foundational — the baseline (blocks everything after)

- [ ] T004 Create the baseline plans in `evaluation/folds/` by copying the source plans with only
  `arm_tag` (and the fold subset) changed:
  - `plan_wf_aro_b029.json` from `plan_NNNC_A_Raw_Open.json` (`wf_aro_b029_fNN`);
  - `plan_wf_aro_b029r.json` (f00 only, `wf_aro_b029r_f00`);
  - `plan_wf_trgate_b029.json` from `plan_wf_trgate.json` (`wf_trgate_b029_fNN`);
  - `plan_<arm>_b029.json` for the 8 SimpleTrigger reference arms (`st_gbb`, `st_bgo_mx`,
    `st_book_rev2`, `st_reg_volreg2`, `st_coex_win4`, `st_coex_cap3`, `st_pre_volreg2`,
    `st_prom_c2`), with tags `<tag>_b029_fNN`.

  Validate each, run `run_folds --dry-run` on each, and check that every fold's `model_dir` exists.
- [ ] T005 Create the probe class `NNNC/NNNC_SIProbe.py`.
  - `class NNNC_SIProbe(NNNC_TriggerGate)` with
    `get_entry_conditions = NNNC_A_Raw_Open.get_entry_conditions`: the twin's borrowed-weights
    machinery and StopInvariantMixin, with arm D's MODEL entries.
  - Docstring: baseline probe, report-only.
  - Its params JSON is a copy of `NNNC_A_Raw_Open.json` with `strategy_name` changed.
  - Build `plan_nnnc_si_probe.json` by **copying `plan_NNNC_A_Raw_Open.json`, keeping
    `thresholds` and `params_snapshot`**. Do NOT use `foldplan derive`: derived plans carry no
    thresholds, and the probe would not enter as arm D does (R4).
    - Set `base_strategy = "NNNC_SIProbe"` and `model_source = "NNNC_ARO"`.
    - Per fold: `strategy_class = "NNNC_SIP_fNN"`, `arm_tag = "nnnc_si_probe_fNN"`, and
      `model_dir` unchanged (`saved_data/NNNC_ARO_fNN`).
  - Dry-run it. Generate one fold's files and check that `NNNC_SIP_f00.json`'s
    `prediction_threshold` equals `NNNC_ARO_f00.json`'s. Delete the generated pair if the dry-run
    doesn't write them.
- [ ] T006 Hash every file under `saved_data/NNNC_ARO_f*/` (sha256) into `baseline.json`
  `model_artifacts`. Then run the baseline chain sequentially:
  - arm D, arm D f00 again, and the twin;
  - the 8 SimpleTrigger `_b029` arms (about 26 min);
  - the probe, with `FT_STOP_INVARIANTS=report FT_STOP_INVARIANTS_SINK=<scratch>/si029`.

  Confirm:
  - all records have returncode 0;
  - `run_folds` printed "model already present — skipping training" for every model fold and
    never "training on" (SC-005);
  - `git diff --stat NNNC/NNNC_ARO_f* NNNC/NNNC_TGT_f*` is empty (regenerated per-fold files
    unchanged), or the differences are recorded and explained before going on.
- [ ] T007 Run `scripts/nnnc_fingerprints.py --out specs/029-nnnc-simpletrigger-merge/fingerprints_before.json`.
- [ ] T008 Write `specs/029-nnnc-simpletrigger-merge/baseline.md` and `baseline.json` (data-model).
  - Per-fold hashes for arm D, its f00 re-run and the twin.
  - **Determinism:** the f00 re-run equals the first run.
  - The twin vs the fresh `st_gbb_b029` baseline, which must be 15/15. It is also reported
    against the 10 Oct `st_gbb` record.
  - Each SimpleTrigger `_b029` arm vs its record, per fold. Differences are reported and
    explained (they predate this feature), and are never a pass or fail condition here.
  - Arm D baseline vs the old `wf_aro_*` records, with any differences listed and explained.
  - The probe:
    - its trades must equal arm D's baseline (15/15);
    - the violation count by check.
  - The fingerprint scan counts.
  - **STOP gate:** any of these stops the feature (see Conventions):
    - violations > 0;
    - probe trades ≠ arm D's baseline;
    - a non-deterministic re-run;
    - twin ≠ `st_gbb_b029`.
- [ ] T009 Commit the baseline: scripts, plans, probe class, records, baseline files and
  fingerprints. Record the commit sha in `baseline.md`.

**Checkpoint**: baseline committed; the refactor may start.

---

## Phase 3: User Story 1 — One framework, no behaviour change (P1) 🎯 MVP

**Goal**: NNNC and SimpleTrigger derive from `EntryFrameworkMixin`; nothing any arm does changes.

**Independent Test**: arm D and the twin reproduce 15/15; 8 SimpleTrigger arms reproduce; all
fingerprints are unchanged.

### Tests first (they must FAIL until T015–T019 land)

- [ ] T010 [P] [US1] `utils/tests/test_entry_framework.py`, on stub strategies over `BaseStrategy`
  with synthetic dataframes:
  - `"triggers"` mode: same mask and tags as today's `SimpleTriggerStrategy` (golden from
    `SimpleTrigger/tests`);
  - `"model"` mode: the mask is the parent's `get_entry_conditions` unchanged, tagged
    `<family>:model` when `MODEL_ENTRY_FAMILY` is set, and left `"model_entry "` when None;
  - `"trigger_and_model"`: mask = model & OR(triggers);
  - refusals per I1;
  - the `FAMILY_EXIT_OVERRIDES` MRO merge (025 G1 cases);
  - the regime gate in model mode drops reversion-tagged rows in the "off" state.
- [ ] T011 [P] [US1] `utils/tests/test_label_families.py`:
  - every table row's labeler condition matches its family (R3), checked by reading each
    labeler's thresholds/condition (unit check on a synthetic frame where possible);
  - consistency with same-named triggers (`gbb`, `breakout_gbb`, `breakout`, `breakout_bb`,
    `breakout_vol`, `breakout_squeeze`, `geometry`);
  - `gbb_path` → reversion; `forward_mae` → None.
- [ ] T012 [P] [US1] `NNNC/tests/test_nnnc_merge.py`:
  - every loadable NNNC class (file-stem scan) has `EntryFrameworkMixin` and `StopInvariantMixin`
    exactly once in its MRO, before `BaseStrategy`;
  - `NNNC_A_Raw_Open` has `ENTRY_MODE == "model"`, `MODEL_ENTRY_FAMILY == "reversion"` and
    `FAMILY_EXITS == {}`;
  - the 9 classes that override `get_entry_conditions` are listed with whether they call `super()`;
  - a guard that no `Framework/` file is modified by this feature (`git diff --name-only` against
    the T009 commit contains no `Framework/` path).

### Implementation

- [ ] T013 [US1] Re-verify each R3 row against its labeler in `Framework/TrainingSignals.py`
  (read only). Drop any row that doesn't hold to None, and record the check in `research.md` R3.
- [ ] T014 [P] [US1] Create `utils/LabelFamilies.py` per contract I2: the `LABEL_FAMILY` table
  with reasons, and `label_family()`.
- [ ] T015 [US1] Move `REGIME_CONTEXT_PAIR` from `SimpleTrigger/presets.py` to
  `utils/RegimeState.py`. `presets.py` re-exports it, so existing imports keep working.
- [ ] T016 [US1] Create `utils/EntryFramework.py` with `EntryFrameworkMixin` per I1.
  - **Move verbatim from `SimpleTrigger/SimpleTriggerStrategy.py`:**
    - `_tags`, the rule-B block, `_apply_regime_gate`, `_regime_state`, `informative_pairs`,
      `_validate_regime_gate`;
    - the pre-emption `custom_exit`, the tagging `populate_entry_trend`, `_sink_coexistence`;
    - the trigger evaluation and causality refusal.
  - **Add:** `ENTRY_MODE`, the model source via `super().get_entry_conditions`,
    `MODEL_ENTRY_FAMILY` resolution (AUTO → `label_family(cls.TRAINING_TYPE)`), the
    `FAMILY_EXIT_OVERRIDES` merge as `_resolve_family_exits`, and the I1 refusals.
  - No reference to NNNC, SimpleTrigger or presets.
- [ ] T017 [US1] Thin `SimpleTrigger/SimpleTriggerStrategy.py` to presets + trigger source:
  - `class SimpleTriggerStrategy(EntryFrameworkMixin, BaseStrategy)` with
    `ENTRY_MODE = "triggers"`;
  - `_apply_preset` calls `cls._resolve_family_exits(preset_families)`;
  - remove the moved code (no duplicates left);
  - keep the module's public names importable.
- [ ] T018 [US1] `NNNC/NNNCStrategy.py`:
  - `class NNNCStrategy(EntryFrameworkMixin, BaseNNStrategy)`;
  - `ENTRY_MODE = "model"`, `MODEL_ENTRY_FAMILY = AUTO`;
  - family exits resolved from an empty base;
  - `_FAMILY_VALIDATE_ON_SUBCLASS` handled so all classes validate after their attributes are
    final.
- [ ] T019 [US1] `NNNC/NNNC_TriggerGate.py`: drop the explicit `StopInvariantMixin` (R5). Extract
  the borrowed-weights methods into `NNNC/BorrowedModelMixin.py`, used by the twin and by
  `NNNC_SIProbe`; behaviour unchanged.
- [ ] T020 [US1] `evaluation/fingerprint.py`: add `ENTRY_MODE` and `MODEL_ENTRY_FAMILY` to the
  family block only when the block is non-empty, or when `ENTRY_MODE` differs from the class
  family's default (R7).
- [ ] T021 [US1] Make T010–T012 pass. Then run
  `pytest NNNC/tests SimpleTrigger/tests utils/tests evaluation Framework/tests research/outcomes/tests research/exit_coupling/tests`.
  Only the 3 known training-pipeline goldens may fail.
- [ ] T022 [US1] Run `scripts/nnnc_fingerprints.py --out .../fingerprints_after.json --compare .../fingerprints_before.json`.
  It must exit 0. Also confirm that the 10 shas pinned in
  `evaluation/tests/test_fingerprint_switches.py` pass.
- [ ] T023 [US1] Create the after-change plans:
  - `plan_wf_aro_a029.json` and `plan_wf_trgate_a029.json`;
  - `plan_<arm>_a029.json` for `st_gbb`, `st_bgo_mx`, `st_book_rev2`, `st_reg_volreg2`,
    `st_coex_win4`, `st_coex_cap3`, `st_pre_volreg2` and `st_prom_c2`. Each copies its source
    plan with only `arm_tag` changed (`<tag>_a029_fNN`).
- [ ] T024 [US1] Run the after-change chain sequentially: arm D, the twin, then the 8 SimpleTrigger
  arms, about 32 min. Then:
  - re-hash `saved_data/NNNC_ARO_f*`; every hash must equal T006's;
  - `run_folds` never prints "training on" (SC-005).
- [ ] T025 [US1] Write `specs/029-nnnc-simpletrigger-merge/verification.md`:
  - per arm and fold: the `_b029` baseline hash vs the `_a029` hash (the pass condition), with
    the record shown alongside;
  - arm D 15/15 and twin 15/15 vs baseline; twin vs `st_gbb_a029` 15/15;
  - each SimpleTrigger arm vs its `_b029` baseline;
  - model artefact hashes unchanged;
  - the fingerprint diff (none);
  - the test-suite result.

  Any mismatch blocks the feature: diagnose by first divergent trade, fix, and re-run that arm.

**Checkpoint**: merge is invisible (SC-001, SC-002, SC-005, SC-006).

---

## Phase 4: User Story 2 — Stop invariants on every NNNC arm (P2)

- [ ] T026 [US2] Confirm that arm D's after-change run (T024) carried the halting check (the MRO
  contains `StopInvariantMixin`, from T012) and completed. Record that in `verification.md`.
- [ ] T027 [P] [US2] Add a planted-violation test to `NNNC/tests/test_nnnc_merge.py`:
  - an NNNC stub whose parent `custom_stoploss` returns a positive ratio halts with
    `StopInvariantError` naming the trade;
  - the same stub in report mode does not halt.

---

## Phase 5: User Story 3 — Features by declaration (P3)

**Goal**: each switch is live when declared and inert when not (FR-008, SC-004).

- [ ] T028 [P] [US3] Create the liveness classes in `NNNC/`, each
  `class X(BorrowedModelMixin, NNNC_A_Raw_Open)` borrowing `NNNC_ARO` weights, with a params JSON
  copied from `NNNC_A_Raw_Open.json`:
  - `NNNC_Live_TrigGate` (`ENTRY_MODE="trigger_and_model"`, `TRIGGER_NAME="gbb"`);
  - `NNNC_Live_C2` (`FAMILY_EXIT_OVERRIDES={"reversion": {"take_profit_atr_mult": 2.8970711921574543}}`);
  - `NNNC_Live_Gate` (`REVERSION_REGIME_GATE` = `SimpleTrigger_RegGateVolreg2`'s, and
    `startup_candle_count = 4096`).
- [ ] T029 [US3] Build the f00 plans `plan_nnnc_live_{triggate,c2,gate[,gateoff]}.json` by
  copying `plan_NNNC_A_Raw_Open.json` (fold f00 only), **keeping `thresholds`** (I5).
  - Set `model_source = "NNNC_ARO"`, with `strategy_class = "<Class>_f00"` and `model_dir`
    unchanged.
  - Check the threshold carry-over as in T005, then run them sequentially with the sink enabled.
- [ ] T030 [US3] Write `specs/029-nnnc-simpletrigger-merge/liveness.md`. Each switch is compared
  with arm D's after-change f00 (control):
  - **TrigGate:** entries ≤ control; every entry's signal bar has gbb true (checked from the
    archive and the analysed frames); dropped count from the sink.
  - **C2:** exit-reason counts differ, and `trades_hash` differs.
  - **Gate:** gate counts in the sink, and entries in the "off" state dropped.
  - Each live (≠ control) is recorded, plus the inert case (arm D f00 after = baseline, from T025).
  - C2 stands for every per-family exit: it goes through the same `FAMILY_EXIT_OVERRIDES` →
    `FamilyExitMixin` → tag path any family exit uses (R9).
  - **Startup note for the gate:** the 4096-bar warm-up is part of the declaration. The control
    for liveness is `NNNC_Live_Gate` with the gate set to None at the same startup.
    - If needed, add `NNNC_Live_GateOff` (gate None, startup 4096) and run it too, so the
      comparison isolates the gate.

---

## Phase 6: User Story 4 — Report (P4) and polish

- [ ] T031 [US4] Write `specs/029-nnnc-simpletrigger-merge/report.md`:
  - **SUMMARY:** SC-001…SC-006 with values;
  - the baseline and its commit; determinism; the old-record vs baseline differences;
  - the probe result;
  - the fingerprint table counts;
  - the 9 overriding variants (which reach the switches);
  - the label-family table, with the arm D correction (`gbb_path` → reversion);
  - liveness;
  - "how to declare each switch on an NNNC arm";
  - follow-ups: C2 for NNNC, trigger-and-model for arm D, a per-bar family for arm D's
    out-of-trigger entries, and the dead `use_forward_peak_profit_label` flag;
  - **the next feature (agreed 2026-10-12): model naming by a resolved model fingerprint** (type,
    label params, gain/loss targets, features, training window, seed) with a manifest, replacing
    class-name model paths. That removes the self-training hazard and `BorrowedModelMixin`.
- [ ] T032 Update `AGENT_GUIDE.md` (if it describes the NNNC/SimpleTrigger structure) and the
  `SimpleTrigger/` / `NNNC/` module docstrings that call them "siblings".
- [ ] T033 Update memory: `project_029_nnnc_merge.md`, the MEMORY.md current-state line, and
  `index_entry_gates.md` / `index_models.md` pointers.
- [ ] T034 Mark tasks `[X]` in this file. Commit code, tests, plans, records and docs (named files
  only).

---

## Dependencies & Execution Order

- **Setup:** T001 ∥ T002 → T003.
- **Baseline:** T004 ∥ T005 → T006 → T007 → T008 (**STOP gate**) → T009.
- **US1:**
  - T010 ∥ T011 ∥ T012 (tests first);
  - T013 → T014;
  - T015 → T016 → T017 → T018 → T019 → T020 → T021 → T022 → T023 → T024 → T025.
- **US2:** T026 after T024; T027 ∥ any US1 task after T016.
- **US3:** T028 after T021 → T029 → T030.
- **US4:** T031 → T032 → T033 → T034.
- **Backtest chains (one at a time):** T003, T006, T024, T029.

## Parallel Example

```text
T001 fingerprint scanner ∥ T002 report-only invariants
T010 mixin tests ∥ T011 label-family tests ∥ T012 MRO/merge tests
```

## Implementation Strategy

- **MVP:** Phases 1–3. Baseline, refactor, then byte-identical verification. That is the merge.
- **Then:** US2 confirms the invariant coverage; US3 proves each switch can be declared.
- **The STOP gate (T008)** protects the "no behaviour change" promise. A defective arm D gets
  fixed in its own change first.
