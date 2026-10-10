---

description: "Task list for feature 022 — per-fold guard tuning"
---

# Tasks: Per-Fold Guard Tuning

**Input**: Design documents from `/specs/022-perfold-guard-tuning/`
**Prerequisites**: spec.md, plan.md, research.md, data-model.md, contracts/guard-cli.md, quickstart.md

## Format: `[ID] [P?] [Story] Description`

**Owner tags**: `@model-implementer` Python and tests · `@experiment-runner` running the grid and
the arms · `@gate-reviewer` reviewing the trackability verdict before it is accepted · `@main`
the provenance audit, the write-up, and anything touching shared harness code.

## Path Conventions

Repository root is `user_data/strategies/`. Commands run from `/Users/philprice95/projects/freqtrade`
with `PYTHONPATH=user_data/strategies` and `.venv/bin/python`.

**Standing constraints, every phase**: the exit is held FIXED (FR-009) — no task may change an
exit parameter. **No task may fit a model** (FR-020). `entry_enable_guards` stays `True` in every
arm (FR-008). `Framework/` is not touched, which is what keeps the live dry-run safe.

**ORDERING PRINCIPLE: the cheapest kill comes first.** Phases 3 and 4 can each terminate the
feature with a complete result before any expensive phase begins. That is Principle VII, not
convenience.

---

## Phase 1: Setup

- [ ] T001 @main Create `evaluation/tests/__init__.py` if absent and confirm pytest collects
  `evaluation/tests/`, so the new test files in later phases are actually run rather than
  silently skipped.
- [ ] T002 [P] @model-implementer **Pin the grid in a committed artefact BEFORE anything is
  scored**: write `evaluation/folds/guard_grid_022.json` enumerating the 4 levels for each of
  `entry_atr_pct`, `entry_adx_threshold`, `entry_rvol_threshold` (64 configurations), each with
  a stable `config_id`, and `entry_enable_guards: true` on every one. FR-005 requires the
  candidate set to be finite and inspectable and FR-019 requires pre-registration to be
  auditable; a grid chosen during the search satisfies neither. Levels bracket the historical
  tuned values (atr_pct 0.013, adx 10.0, rvol 1.9) and the open values.
- [ ] T003 [P] @main Record the four reused arm tags and their measured results in
  `specs/022-perfold-guard-tuning/arms.md`: A `wf_gbbo` +2.64pp 7/15, B `wf_gbbg` +18.76pp 12/15,
  D `wf_aro` +39.98pp 15/15, and C `wf_gbbgrid` (pending). FR-014 forbids re-running A, B and D.

---

## Phase 2: Foundational (Blocking Prerequisites)

Shared harness code. Nothing in Phases 3+ may start until T006 passes, because a regression here
affects features other than this one.

- [ ] T004 @model-implementer **Write the regression test FIRST** in
  `evaluation/tests/test_guards_plan_key.py`: assert that for a plan with **no** `guards` key,
  `run_folds.generate_fold` produces output **byte-identical** to current behaviour. Generate the
  expected fixture from the CURRENT code before T005 modifies it — a fixture generated afterwards
  would record whatever the new code does and assert the new code does it.
- [ ] T005 @main Add the optional `guards` plan key to `evaluation/run_folds.py`, mirroring the
  existing `thresholds` key exactly (read at `:194`, passed to `generate_fold` at `:237`). Values
  are written into the fold's **generated** `.json` derived from the frozen params snapshot,
  never into a live params file. `entry_enable_guards` is forced `True` on write regardless of
  input (FR-008). Owner is `@main` because other features' plans read this file.
- [ ] T006 @model-implementer Run T004 against the modified `run_folds.py` and confirm plans
  without a `guards` key are byte-identical. If any differ, fix `run_folds.py` — do not
  regenerate the fixture.
- [ ] T007 [P] @model-implementer Create `SimpleStrategies/GbbGuardGridOpen.py` as a thin
  `TriggerOnly` subclass with `TRIGGER_NAME = "gbb"`, whose guard values come from the generated
  per-fold params file. It must set no exit field of its own (FR-009) and must not override
  `entry_enable_guards` to anything but `True` (FR-008).
- [ ] T008 [P] @model-implementer Add `SimpleStrategies/GbbGuardGridOpen.py` to the allowlist in
  `Framework/tests/test_exit_policy_single_source.py` **only if** the duplication test flags it,
  and classify it as *configuring* with a one-line reason. Feature 021's experience: a new
  TriggerOnly subclass that names no exit parameter is not flagged, so this task may legitimately
  be a no-op — verify rather than assume.

**Checkpoint**: shared harness code is extended with a proven-inert-when-absent plan key, and the
no-model strategy exists.

---

## Phase 3: User Story 1 — Can the motivating number be trusted? (Priority: P1) 🎯 MVP

**Goal**: establish whether arm B's +18.76pp is out-of-sample. No compute.

**Independent test**: a written provenance verdict with evidence and an overlap answer, produced
without running any arm.

- [ ] T009 [US1] @main Trace the origin of the pre-2026-09-06 tuned guard values
  (`entry_atr_pct` 0.013, `entry_adx_threshold` 10.0, `entry_bb_width_threshold` 0.094,
  `entry_close_norm_threshold` 0.6, `entry_guard_threshold` 0.1, `entry_rvol_threshold` 1.9)
  through `git log -S` on `Framework/BaseStrategy.py` and any `.fthypt` records. Record commit
  ids and dates.
- [ ] T010 [US1] @main Compare the recovered tuning window to the 15 fold **test** ranges from
  `evaluation/folds/plan_NNNC_A_Raw_Open.json` and record `overlaps_folds` as `yes`/`no`/`unknown`
  per data-model.md's `ProvenanceVerdict`.
- [ ] T011 [US1] @main Verify the incumbent arm D's configuration from the strategy class and its
  params file — **not** from prose and **not** from any recorded provenance field. Feature 021
  twice trusted a field that asserted rather than measured (`volume_filter_active` was a
  hardcoded literal; `guards` recorded a CLI flag). Confirm guards-open and a per-fold model
  threshold, or report that the premise is contradicted.
- [ ] T012 [US1] @main Write `specs/022-perfold-guard-tuning/provenance.md` with the verdict and
  set `arm_b_status` to `trustworthy`, `in_sample` or `untrustworthy`. Per data-model.md, an
  unrecoverable window ⇒ `unknown` ⇒ `untrustworthy`. **This file must exist before any arm runs**
  (FR-001) and arm B's figure must carry this status wherever it is quoted (FR-002).

**Checkpoint**: US1 is complete and independently valuable whatever its answer. `untrustworthy`
costs C-vs-B its interpretation but leaves C-vs-A intact.

---

## Phase 4: Audit the proxy — the gate is not licensed without this

**Goal**: establish that the table-based scoring proxy can rank what it will be asked to rank.

**Why its own phase**: research.md R2 — this same instrument reproduced live mean `profit_ratio`
to 3.6e-5 on one population and had the **wrong sign** on another. Its fidelity is a measured
property, not an assumption, and the gate cannot be trusted on an unaudited proxy.

- [ ] T013 @model-implementer Write `evaluation/pick_guards.py`'s scoring core: given a
  `GuardConfig` and a window, produce a `GridScore` per data-model.md — admitted-bar mask →
  concurrency selection (`research/outcomes/leaderboard.py::_select_concurrent`, `max_open` 10) →
  pooled mean, carrying `n_admitted`, `n_taken`, `exit_config_hash` and
  `below_floor = n_taken < 25` (FR-012, verbatim threshold).
- [ ] T014 @model-implementer **Write the leakage test BEFORE the loading path** in
  `evaluation/tests/test_pick_guards.py`: assert `pick_guards` loads fold data only through
  `pick_threshold.TrainWindow.from_fold`, and that requesting a **test**-window score without the
  explicit opt-in flag **raises** (FR-007). A test-window score must not be able to leak into
  selection.
- [ ] T015 @experiment-runner Run `pick_guards.py --audit-proxy --against wf_gbbo,wf_gbbg`:
  score arm A's open configuration and arm B's tuned configuration with the proxy and compare to
  their measured harness results (+2.64pp, +18.76pp).
- [ ] T016 @gate-reviewer Review T015. **PASS** requires the proxy to separate A from B in the
  right direction. On **FAIL**, apply research.md's recorded fallback — reduce to two axes
  (`atr_pct` × `adx`, 16 points) scored by real backtests on training windows — and record that
  the 64-point grid was not licensed. Do not proceed on an unaudited proxy.

**Checkpoint**: the scoring instrument is either trusted with evidence, or replaced by the
pre-recorded fallback. Either way the next phase is licensed.

---

## Phase 5: User Story 2 — Is the guard optimum TRACKABLE? (Priority: P1) **BLOCKING GATE**

**Goal**: Principle VII. Decide whether adaptivity is a candidate at all.

**Independent test**: a `TrackabilityResult` with a verdict and a paired interval, produced in one
working session (SC-008).

**Expected outcome, recorded before running**: `not_trackable`. `GUARD_ABLATION.md` shows every
live guard's effect flipping sign across all three eras and four adaptive schemes have already
lost to static values in this project. FR-006 makes that a complete finding.

- [ ] T017 [US2] @model-implementer **Write the exit-code test FIRST** in
  `evaluation/tests/test_guard_trackability.py`: assert `guard_trackability.py` exits **0** when
  the verdict is `not_trackable`. That verdict is a complete finding under FR-006 and must not
  read as a failed run — an exit code of 1 would make the expected outcome look like breakage.
- [ ] T018 [US2] @model-implementer Add a test asserting `n_distinct_choices` is reported
  **separately** from the verdict, so a merely-varying optimum can never be rendered as
  trackable. Principle VII: variation is "necessary but nowhere near sufficient".
- [ ] T019 [US2] @model-implementer Implement `evaluation/guard_trackability.py` per research.md
  R4, verbatim: `c_i = argmax_c` train score; `c* = argmax_c Σ_i S_i(c)` chosen **in hindsight**;
  `regret_i(adaptive) = max_c S_i(c) − S_i(c_i)`; `regret_i(constant) = max_c S_i(c) − S_i(c*)`.
  Regret is against `max_c` over **the enumerated candidates that actually existed** (FR-005) —
  never a value outside the grid.
- [ ] T020 [US2] @model-implementer Compute the paired interval on the per-fold regret difference
  by reusing `evaluation/bootstrap.py::bootstrap_delta` (FR-015). Do not write a second paired
  bootstrap.
- [ ] T021 [US2] @experiment-runner Score the 64 configurations on all 15 **training** windows via
  `pick_guards.py` and write `evaluation/folds/guards_gbbgrid.json`. Sequential, one at a time.
  Confirm every `GridScore` carries `exit_config_hash == 5a18305d1fafa83e`.
- [ ] T022 [US2] @experiment-runner Run `guard_trackability.py` and write
  `evaluation/folds/trackability_gbbgrid.json`.
- [ ] T023 [US2] @gate-reviewer Review the verdict before it is accepted: check that regret is
  measured against real candidates, that the constant was chosen in hindsight, that
  `n_distinct_choices` is reported apart from the verdict, and that the paired interval is on the
  per-fold difference. Produce a short memo; do not edit results.

**🚦 GATE — FR-006**: if the verdict is `not_trackable`, **Phase 6 and Phase 7 MUST NOT run.**
Proceed directly to T033 (the write-up). Phase 7 is likewise unreachable. This is the expected branch.

---

## Phase 6: User Story 3 — Does per-fold guard tuning trade? (Priority: P2)

**Runs only if T023 accepted a `trackable` verdict.**

**Goal**: arm C — per-fold tuned guards, no model, 15 folds.

**Independent test**: a pooled wallet return, folds-positive count, and a paired interval against
arm A.

- [ ] T024 [US3] @model-implementer **Write the inertness test FIRST** in
  `evaluation/tests/test_guard_axes_bite.py`: given two arm records differing in exactly one grid
  axis, assert their `trades_hash` values differ; and given two records differing only in
  `entry_guard_threshold`, assert the hashes are **IDENTICAL**.
- [ ] T025 [US3] @experiment-runner Run the FR-013 proof on one fold: vary each of the three axes
  across its levels and collect `trades_hash` from each `regime/arms/` record.
- [ ] T026 [US3] @experiment-runner **Run the negative control**: two runs on the same fold
  varying **only** the known-inert `entry_guard_threshold`, and confirm `trades_hash` is
  UNCHANGED. `GUARD_ABLATION.md` records this parameter as producing byte-identical trades on all
  three windows, so it is a free proof that the inertness check can actually detect inertness. A
  check that cannot fail is worthless — feature 021 established this by planting a duplicate. **If
  the hash CHANGES here, the check itself is broken and T025's result means nothing.**
- [ ] T027 [US3] @gate-reviewer Review T025/T026. Any axis whose hash did not change is INERT and
  MUST be reported as a defect (FR-013): the grid was searching a dimension that does not exist.
- [ ] T028 [US3] @experiment-runner Build `evaluation/folds/plan_GbbGuardGrid.json` from the
  control's fold plan with `base_strategy: GbbGuardGridOpen` and the `guards` map from T021, then
  run all 15 folds sequentially via `run_folds.py`. One backtest at a time.
- [ ] T029 [US3] @main **Verify `exit_config_hash` equality across arms A, B, C and D** as its own
  task (SC-004). A mismatch voids every comparison in this feature and is a hard failure, not a
  warning.
- [ ] T030 [US3] @experiment-runner Pool arm C (`evaluation/pool.py`), compare to arm A paired via
  `compare_pooled.py`/`bootstrap_delta` (FR-015), report power (`power.py`), and report every
  effect with its **early-half and late-half** values (FR-016, SC-007). Exclude any fold whose
  selected configuration admitted fewer than 25 trades, reporting it as **unpowered** rather than
  as a 0.0pp observation (FR-017).

---

## Phase 7: User Story 4 — Does the model add anything on top? (Priority: P3)

**Runs only if arm C cleared the pre-registered decision rule (FR-020).**

- [ ] T031 [US4] @experiment-runner Build arm E reusing the incumbent's existing per-fold
  artefacts at `saved_data/NNNC_ARO_f{00..14}/*.safetensors`. **No model is fitted** (FR-020);
  confirm no training is triggered before running.
- [ ] T032 [US4] @experiment-runner Run arm E over the 15 folds and compare to arm C paired on the
  symmetric difference, with early/late halves.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T033 @main **The terminal write-up, and it is a DELIVERABLE not an abandonment.** Write
  `specs/022-perfold-guard-tuning/report.md` recording: the provenance verdict; the proxy audit;
  the trackability verdict with its regret figures; and, if the gate failed, the explicit
  statement that per-fold guard tuning is **not a candidate for adaptivity** and that the feature
  is **complete**. State it as the fifth adaptive scheme to lose to a static value in this
  project, and that the gate cost one session rather than fifteen hyperopts.
- [ ] T034 [P] @main Record in the report that the grid's joint 3-D search is a documented
  deviation from Principle VIII, with super-additivity as the reason, and that the joint winner's
  per-guard marginal effect MUST NOT be read off the grid.
- [ ] T035 [P] @model-implementer Run the full suite and report the count, confirming no
  pre-existing failure was introduced. Attribute any failure against a pristine pre-022 worktree
  rather than asserting it is pre-existing.
- [ ] T036 [P] @main Update `specs/021-label-exit-coordination/report.md`'s follow-up section with
  this feature's outcome, so 021's open puzzle (+39.98pp explained by nothing measured) carries
  the answer or the remaining gap.
- [ ] T037 [P] @main If `arm_b_status` came out `untrustworthy`, add a one-line caveat beside
  every existing quotation of arm B's +18.76pp in feature 021's artefacts, so the figure never
  travels without its status (FR-002).

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (setup)
    └─> Phase 2 (shared harness) ── BLOCKING, strictly ordered:
            T004 (fixture from PRE-change code) ─> T005 (add key) ─> T006 (prove unchanged)
    ├─> Phase 3 (US1 provenance) ── independent, no compute, always completes
    └─> Phase 4 (proxy audit) ── licenses Phase 5
            └─> Phase 5 (US2 GATE) ── 🚦 FR-006 decides what follows
                    ├── not_trackable ─> T033 (write up, STOP)
                    └── trackable ────> Phase 6 (US3 arm C)
                                            └─> Phase 7 (US4 arm E), only if C clears its rule
```

### User Story Dependencies

- **US1** is fully independent — it can run in parallel with Phases 2 and 4 and always completes.
- **US2** depends on Phase 4 (an audited proxy) and on Phase 2 only for `GridScore` plumbing.
- **US3** depends on US2's verdict being `trackable`. **It is expected not to run.**
- **US4** depends on US3 clearing the decision rule.

### Parallel Opportunities

- T002 and T003 together (different files).
- **Phase 3 (US1) runs in parallel with Phases 2 and 4** — it needs no code and no compute, so the
  provenance verdict can be ready before the gate reports.
- T007 and T008 together.
- T034–T037 together once T033 exists.

## Parallel Example: Phase 3 alongside Phase 4

```
@main            T009 → T010 → T011 → T012      (provenance, no compute)
@model-implementer   T013, T014                 (proxy scoring core + leakage test)
@experiment-runner        T015                  (needs T013)
```

## Implementation Strategy

**MVP is Phases 1–5**, and the MVP may be the whole feature. The gate is expected to return
`not_trackable`, in which case T033 is the deliverable and Phases 6–7 never run. That is a
complete result costing one session, which is precisely what Principle VII's "kill or confirm in
an afternoon" buys.

Phases 6–7 are the contingent half. Do not pre-build them; the plan is written so that the
expensive work is only ever reached with evidence that it is worth doing.
