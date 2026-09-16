---

description: "Task list for feature 016 — labelfree-jump-model-gate"
---

# Tasks: labelfree-jump-model-gate

**Input**: Design documents from `specs/016-labelfree-jump-model-gate/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/jump_gate.md, quickstart.md

**Tests**: REQUIRED. The spec makes causality, leak measurement and gate activity testable
preconditions, and the brief mandates that calibration and leakage tests are the first tasks.

**Organization**: grouped by user story. **All runs are sequential** (FR-018); the `[P]` marker
means only that a task touches a different file and has no unmet dependency.

## Format: `[ID] [P?] [Story] Description`

---

## Phase ordering rationale — read before executing

Story phases do **not** run in bare priority order, and the reason is a hard dependency the spec
does not encode:

- **US3 (causality) runs first.** Every other story reads the state sequence. If the states are not
  causal, US1's block counts, US2's leak size and US4's drawdown are all void. Verifying them
  afterwards would mean discovering the whole run was meaningless at the end.
- **US1 (the gate fires) runs next**, before anything expensive. It is the cheapest kill gate in
  the feature and needs no backtest.
- **US2 (the leak) runs before US4**, because the brief requires the smoothed number to exist
  before the honest one is read.

Resulting order: **Setup → Foundational → US3 → US1 → US2 → US4 → Polish**.

---

## Phase 1: Setup

- [X] T001 Create the package skeleton `research/labelfree/jump/__init__.py` and the entry point `research/labelfree/jump_gate.py` named by the brief, dispatching the subcommands in `specs/016-labelfree-jump-model-gate/quickstart.md`
- [X] T002 Install the cross-check dependency into the venv and record the resolved version in `research/labelfree/jump/__init__.py`: `/Users/philprice95/projects/freqtrade/.venv/bin/python -m pip install jumpmodels` (expected 0.1.1)
- [X] T003 [P] Add the interpreter guard to `research/labelfree/jump/__init__.py`: raise unless `sys.executable` resolves under `freqtrade/.venv` — the pyenv shim is a different 3.12.13 with no pandas or scipy
- [X] T004a Save the pre-implementation `/speckit-analyze` report as `specs/016-labelfree-jump-model-gate/analyze.md`, matching features 014 and 015 (SC-015)
- [X] T004 [P] Create the artefact directory `research/labelfree/jump/out/` and add it to `.gitignore`

---

## Phase 2: Foundational (BLOCKING — no user story may begin until complete)

**Purpose**: the state machinery every story consumes.

- [X] T005 Implement `build()` in `research/labelfree/jump/features.py` — causal `rvol_realised`, `rvol_downside`, `sign_persistence` per pair and for BTC, on closed bars only, warm-up rows dropped rather than back-filled (FR-004)
- [X] T006 Implement the `JumpFit` dataclass and `fit()` in `research/labelfree/jump/model.py` — three-block coordinate descent over centroids, states and sparse feature weights (research R1)
- [X] T007 Implement volatility state ordering in `research/labelfree/jump/model.py` — `state_order` sorts states by fitted realised-volatility centroid so "high-vol" names the same thing after every refit (FR-006)
- [X] T008 Implement `calibrate_lam()` in `research/labelfree/jump/model.py` — called **once** on fold 0's training window against the source study's switching rate, then frozen for the entire run (research R5, Principle VIII)
- [X] T008a Implement `fit_ensemble()` and `seed_agreement()` in `research/labelfree/jump/model.py` — ≥5 seeds, interquartile mean, stratified-bootstrap interval, never a single fit and never the best seed. `fit()`'s k-means initialisation is seed-dependent, so everything downstream inherits that randomness (FR-022, FR-023, research R13)
- [X] T009 Implement `filtered_states()` in `research/labelfree/jump/online.py` — forward recursion only, `s_t = argmin_k V_t(k)`, no traceback (FR-004)
- [X] T010 Implement `smoothed_states()` in `research/labelfree/jump/online.py` — forward pass to T then **back-track**; this is the leak reference and the only difference from T009 (FR-007)
- [X] T011 Implement `schedule()` in `research/labelfree/jump/refit.py` — 90-day refit points on an expanding window, **continuing through each fold's test range** (FR-005, clarification Q2=A)
- [X] T012 Implement `run()` in `research/labelfree/jump/refit.py` — per-refit standardisation fitted on the expanding window **up to that refit point only**, then volatility remap (FR-005, FR-006)
- [X] T013 Implement the `StateSequence` artefact writer in `research/labelfree/jump/refit.py` — the per-`(pair, timestamp)` file the backtest will actually read (data-model.md § StateSequence)
- [X] T014a Implement `register_trials()` in `research/labelfree/jump/report.py` wrapping `research/labelfree/trials.py::pre_register` — this feature's trials MUST enter the log the deflated Sharpe ratio is computed over (`00-evaluation-protocol.md` §5 gate 4)
- [X] T014 Wire every quickstart subcommand into `research/labelfree/jump_gate.py`: `seeds`, `crosscheck`, `redundancy`, `leaktest`, `activity`, `check-weights`, `evaluate`, `drawdown`, `report`

**Checkpoint**: states can be produced. Nothing may be believed about them yet.

---

## Phase 3: User Story 3 — Regimes inferred causally (P3) 🎯 runs first, see rationale

**Goal**: a bar's regime label provably uses no information after that bar.

**Independent test**: truncate input history at many cut points; filtered states over the
overlapping region are bit-identical, and smoothed states are not.

### Tests (write first; they MUST fail before T021)

- [X] T015 [P] [US3] `test_filtered_prefix_stable` in `research/labelfree/tests/test_jump_causality.py` — `filtered_states(x[:m], fit) == filtered_states(x, fit)[:m]` for many `m`
- [X] T016 [P] [US3] `test_smoothed_prefix_CHANGES` in `research/labelfree/tests/test_jump_causality.py` — **asserts the failure** of the same prefix equality. A smoothed sequence that passes means the traceback never ran and the leak reference is not a leak reference
- [X] T017 [P] [US3] `test_refit_uses_only_past` in `research/labelfree/tests/test_jump_causality.py` — no bar at or after a refit point enters that fit
- [X] T018 [P] [US3] `test_standardisation_no_leak` in `research/labelfree/tests/test_jump_causality.py` — feature means and scales come from the expanding window only (brief 07's recorded leak)
- [X] T019 [P] [US3] `test_fit_deterministic`, `test_state_order_stable`, `test_lam_frozen` in `research/labelfree/tests/test_jump_model.py` — identical output for a given `(x, k, lam, seed)`; labels do not swap across refits; `lam` and `k` identical across every fit in a run

### Implementation

- [X] T019a [P] [US3] `test_seed_agreement_threshold` and `test_ensemble_not_single_fit` in `research/labelfree/tests/test_jump_model.py` — cross-seed state agreement below 0.95 raises; a caller passing a single `JumpFit` where an ensemble is required raises (FR-022, FR-023)
- [X] T020 [US3] Make T015–T019a pass by correcting `research/labelfree/jump/online.py`, `model.py` and `refit.py` — **not** by relaxing an assertion
- [X] T021 [US3] Run the deletion test against the **written artefact** from T013 rather than an in-memory recomputation, and record the result in `research/labelfree/jump/out/causality.json` (research R7)
- [X] T022 [US3] Implement `audit_package_inference()` in `research/labelfree/jump/crosscheck.py` — read the `jumpmodels` inference path and confirm it is **filtered, not smoothed**; its default is not assumed (FR-017)
- [X] T023 [US3] Implement `agreement()`, `matched_parameter_agreement()` and `describe_divergence()` in `research/labelfree/jump/crosscheck.py`. The **in-repo model is primary and carries the verdict**; the package corroborates and never overrides (FR-014). **FR-016a hard gate**: matched-parameter agreement below 0.95 stops the feature. **FR-016b advisory**: independent agreement is reported with its cause named, never thresholded (FR-015, SC-004a)

**Checkpoint**: the states are causal and independently corroborated. SC-002, SC-004, SC-004a met.

---

## Phase 4: User Story 1 — A gate that demonstrably gates (P1) 🎯 MVP and first kill gate

**Goal**: show the gate changes which trades happen, before any performance number is read.

**Independent test**: per-fold block counts are non-zero and non-total, and a measurable share of
blocks are not ones the incumbent's rvol guard already made.

### Tests

- [X] T024 [P] [US1] `test_inert_gate_blocks_reporting` in `research/labelfree/tests/test_jump_gate.py` — a gate blocking nothing yields verdict `inert` and any attempt to compute a performance statistic from it raises
- [X] T025 [P] [US1] `test_total_gate_refused` in `research/labelfree/tests/test_jump_gate.py` — a gate blocking everything is refused via `research/labelfree/admissibility.py`; abstention is not risk reduction
- [X] T026 [P] [US1] `test_per_fold_counts` in `research/labelfree/tests/test_jump_gate.py` — activity is reported per fold, not only pooled

### Implementation

- [X] T027 [US1] Implement `decide()` in `research/labelfree/jump/gate.py` — block new entries while the filtered state is the high-vol state; exits and open positions untouched (FR-001, research R6)
- [X] T028 [US1] Implement `activity()` in `research/labelfree/jump/gate.py` returning `active | inert | total` with per-fold counts, and make it a precondition callers must check (FR-001, FR-002, FR-003)
- [X] T029 [US1] Implement `overlap()` in `research/labelfree/jump/redundancy.py` — contingency of jump-gate blocks against the incumbent's `rvol >= 2.0` rejections, reporting `novel_block_rate`; verdict is `redundant_with_rvol` below **0.20**, else `distinct` (FR-024, SC-019, Principle IX, research R12)
- [X] T030 [US1] Wire the `total` refusal through `research/labelfree/admissibility.py` rather than a local check (FR-003)
- [X] T031 [US1] Run activity and redundancy across all 15 folds sequentially; write `research/labelfree/jump/out/activity.json` (SC-001)
- [X] T031a [US1] Repeat T031 across the ≥5-seed ensemble; report the per-fold block rate as an interquartile mean with a stratified-bootstrap interval into `research/labelfree/jump/out/seeds.json`, and record `across_seed_spread` against the gated-vs-ungated effect (FR-022, SC-017)
- [X] T032 [US1] **KILL CHECKPOINT.** If the gate is `inert`, if `novel_block_rate < 0.20` (`redundant_with_rvol`), if the gate is `active` in fewer than 10 of 15 folds (SC-020), or if `across_seed_spread` exceeds the gated-vs-ungated effect (SC-017), record it as a **defect under Principle IV** — "the gate never fired", explicitly not "regime gating does not help" — and jump to Phase 7 (SC-012)

**Checkpoint**: the gate is known to fire, and to fire on something rvol had not already removed.

---

## Phase 5: User Story 2 — The leak is measured, not merely avoided (P2) — **NOT RUN**

> **Stopped by T032 on 2026-09-15.** The tasks below are deliberately unchecked: they were
> never executed, and the kill checkpoint is the reason. A gate that blocks what the incumbent's
> rvol guard already rejects (14/15 folds) and whose blocked set moves more with the seed than
> the gate moves anything (spread 0.115 vs block rate 0.104) cannot produce an interpretable
> leak size. Running them would have manufactured a number, not measured one.

**Goal**: publish the smoothed-state result beside the honest one as the size of the avoided leak.

**Independent test**: the smoothed arm beats the online one by ≥ 7.633 pp, and the gap is reported as a number.

- [ ] T033 [US2] Implement `generate()` in `research/labelfree/jump/arms.py` — write the gated fold class and params file, overriding `get_model_path()` to return the **ungated** fold's directory (research R9)
- [ ] T034 [US2] Implement `assert_reuses_weights()` in `research/labelfree/jump/arms.py` — raise unless `saved_data/<ungated>/*.safetensors` exists and no `saved_data/<gated>/` was created. Without this a class trains inside its own backtest, silently, with returncode 0 (`evaluation/run_folds.py` module docstring)
- [ ] T035 [US2] Run the `gated_smoothed` arm across all 15 folds sequentially with no retraining, recording each fold to `regime/arms/*.json` and the pooled set to `research/labelfree/jump/out/arm_gated_smoothed.json` (FR-007)
- [ ] T036 [US2] Compute `gap_pp` against the ungated baseline read from `evaluation/pooled_nnnc.json` and write `research/labelfree/jump/out/leak.json` (FR-008, SC-003)
- [ ] T037 [US2] **STOP CHECKPOINT.** If the smoothed arm does not exceed the online arm by **≥ 7.633 pp** (`evaluation.bootstrap.BASELINE_HALF_WIDTH_PP`), record `suspect_online_leaks` in `research/labelfree/jump/out/leak.json` and halt — the online path is itself leaking. This is not a reassuring result (FR-009, SC-014)

**Checkpoint**: the leak size is published. SC-003 met.

---

## Phase 6: User Story 4 — Drawdown, with intervals (P4) — **NOT RUN**

> **Stopped by T032**, same reason as Phase 5. The 30 gated backtests were not run. Protocol §5
> gates 1-6 are therefore unevaluated, which is the pre-registered consequence of the kill and
> is recorded as such in `briefs/results/08-2026-09-15.md`, not as an omission.

**Goal**: max drawdown and time-underwater with bootstrap intervals, beside return, for the gated
and ungated incumbent.

**Independent test**: every reported statistic carries an interval, and the drawdown claim is
counted per fold.

### Tests

- [ ] T038 [P] [US4] `test_additive_not_compounded` in `research/labelfree/tests/test_jump_drawdown.py` — the equity curve is additive in pp; a compounding implementation fails on a constructed concurrent-trade case (brief 013's recorded inversion)
- [ ] T039 [P] [US4] `test_block_len_in_trades` in `research/labelfree/tests/test_jump_drawdown.py` — passing a bar-panel constant from `stress.bootstrap.BLOCK_LENGTHS` raises; block length is in **trades**
- [ ] T040 [P] [US4] `test_drawdown_on_known_curve` in `research/labelfree/tests/test_jump_drawdown.py` — max drawdown and time-underwater on hand-computed curves

### Implementation

- [ ] T041 [US4] Implement `equity_pp()`, `max_drawdown()` and `time_underwater()` in `research/labelfree/jump/drawdown.py` — additive `profit_abs × scale_pp_per_abs`, close-ordered within a fold
- [ ] T042 [US4] Implement `bootstrap_ci()` in `research/labelfree/jump/drawdown.py` reusing `research/labelfree/stress/bootstrap.py::block_index`, with the trade-unit block length recorded in the output (research R10)
- [ ] T043 [US4] Run the `gated_online` arm across all 15 folds sequentially with no retraining, pooling to `research/labelfree/jump/out/arm_gated_online.json`
- [ ] T044 [US4] Compute per-fold drawdown and time-underwater with CIs for ungated and `gated_online`; write `research/labelfree/jump/out/drawdown.json` (FR-011, SC-011)
- [ ] T045 [US4] Compare pooled return against the 3.350 pp/yr floor all four mandatory baselines of `00-evaluation-protocol.md` §3, **and Phase C's best fixed choice** — defined at `regime/phase_c_trackability.py:177` as `max(take-every-marginal-trade, the consistent rule)`, i.e. `max(ungated incumbent, cash)`, computable from the existing baseline with no new run — net of full costs using `evaluation/` as-is; write `research/labelfree/jump/out/returns.json` (FR-010, FR-012, SC-005, SC-007)
- [ ] T046 [US4] Check persistence — sign positive in at least 10 of 15 folds and in each of W1/W2/W3 — appending the per-window breakdown to `research/labelfree/jump/out/returns.json` (SC-006, Principle II)
- [ ] T047 [US4] Compute PBO via CSCV and the deflated Sharpe ratio with `research/labelfree/stats.py`; re-check the return delta at 1.5× fees (SC-008, SC-009)
- [ ] T047a [US4] Compute the overfitting differential with `research/labelfree/stats.py::overfitting_differential()` — if the in-sample W2 advantage exceeds the W1 advantage by more than the floor, that is overfitting: stop and record (`00-evaluation-protocol.md` §5 kill list, SC-018)
- [X] T048 [US4] Apply the verdict rules and write `research/labelfree/jump/out/verdict.json`: accept only if return is unchanged within the floor (`|Δ annualised return| < 3.350 pp/yr`) **and** drawdown falls with CI excluding zero in ≥10 of 15 folds; a drawdown gain bought by surrendering return is a KILL reported as a trade-off; frame any acceptance as a **deployment** modification, never alpha (FR-013, SC-011, SC-013)

**Checkpoint**: the verdict exists with its intervals.

---

## Phase 7: Polish & Cross-Cutting

- [X] T049 Implement `write()` in `research/labelfree/jump/report.py` and produce `research/label-free-study/briefs/results/08-2026-09-15.md` — **whatever the outcome**, including an early stop at T032 or T037 (FR-021, SC-016)
- [X] T049a Complete every pre-registered trial via `research/labelfree/trials.py::complete` (or `abandon` on an early stop) so the log T051 audits and T047's DSR reads is actually written to
- [X] T050 Reproduce the headline result from `research/labelfree/jump/out/verdict.json` in a separate process, following the pattern of `research/labelfree/reproduce.py` (SC-010)
- [X] T051 [P] Run the trial-log audit per `research/label-free-study/00-evaluation-protocol.md` §6
- [X] T052 [P] Update the status table in `research/label-free-study/README.md` with brief 08's verdict
- [X] T053 Run `specs/016-labelfree-jump-model-gate/quickstart.md` end to end, sequentially
- [X] T054 Run `/speckit-analyze` again and save the report as `specs/016-labelfree-jump-model-gate/analyze-post.md` (FR-020, SC-015)
- [X] T055 Stage **only** the files this feature created or modified — never `git add -A`; check `git status` first and leave anything unexpected alone (FR-019)

---

## Dependencies & Execution Order

### Phase dependencies

- **Phase 1 Setup** — no dependencies
- **Phase 2 Foundational** — depends on Phase 1; **blocks every story**
- **Phase 3 US3** — depends on Phase 2; blocks US1, US2, US4 (states must be causal first)
- **Phase 4 US1** — depends on US3; carries a kill checkpoint (T032) that can skip to Phase 7
- **Phase 5 US2** — depends on US1 passing T032; carries a stop checkpoint (T037)
- **Phase 6 US4** — depends on US2 completing T036
- **Phase 7 Polish** — reachable from T032, T037 or T048; the report is written on every path

### Critical gates

| Task | Gate | If it fails |
|---|---|---|
| T020/T021 | states are causal | fix the code, never the assertion |
| T023 | implementations agree | stop; do not average |
| T019a/T031a | the fit is not initialisation noise | **KILL**; seed spread exceeding the effect is not a finding |
| T032 | fires, novel ≥0.20, active in ≥10/15 | KILL as a **defect**, → Phase 7 |
| T037 | the leak is real | stop; the online path is leaking |
| T047a | no overfitting differential | stop and record |
| T048 | acceptance | KILL is the pre-registered expectation |

### Parallel opportunities

`[P]` marks different files with no unmet dependency. **Evaluation runs are never parallel**
(FR-018): T035, T043 and every fold within them execute sequentially.

- T003, T004 together
- T015–T019a together (six test functions, all failing first)
- T024–T026 together
- T038–T040 together
- T051, T052 together

---

## Implementation Strategy

### MVP — the cheapest path to a defensible answer

1. Phase 1 + Phase 2 → states exist
2. Phase 3 (US3) → states are trustworthy
3. Phase 4 (US1) → **stop here if the gate is inert**

That is the whole first kill gate and it needs **no backtest**. Given that the incumbent's rvol
guard already rejects ~97% of candidates, reaching T032 and finding `redundant_with_rvol` is a
genuinely likely outcome — and it is a complete, reportable result.

### Incremental delivery

Each phase ends in a recordable state. An early stop at T032 or T037 still produces the §6 report
(T049), because a negative result that satisfies the constitution's workflow gates is a successful
outcome and is recorded with the same weight as a positive one.

### Pre-registered expectation

**KILL**, consistent with `regime/VERDICT.md`. The result that would be a *defect* rather than a
finding is an inert gate — which is exactly what T029 and T031 exist to catch before it is
mistaken for a clean null.
