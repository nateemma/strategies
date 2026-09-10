---

description: "Task list for 009-labelfree-direct-policy"
---

# Tasks: Label-free Direct Policy Optimisation

**Input**: Design documents from `specs/009-labelfree-direct-policy/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Test tasks ARE included. The specification mandates three of them as gates — the
calibration test (SC-001), the lookahead sanity ceiling (SC-002) and byte-identical reproduction
(SC-008) — so they are requirements, not optional scaffolding.

**Organization**: Tasks are grouped by user story. Phase 3 (US1) is a hard blocking gate: no policy
code may be written until it passes.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel — **file-authoring tasks only**.
- **[Story]**: US1–US5, mapping to spec.md user stories.

> ⚠️ **Repo-specific meaning of [P].** AGENT_GUIDE's operational rules require every *job*
> (training, backtest, fold run) to execute **sequentially** — two concurrent jobs measured slower
> than serialising and can cross-contaminate the arm archive. So `[P]` here marks only tasks that
> write different files with no shared dependency. **No task that executes a run is ever marked
> `[P]`**, and tasks prefixed **RUN** must be executed one at a time.

## Path Conventions

Repository root is `user_data/strategies/`. New code lives in `research/labelfree/`; the single
deployable wrapper is `NNDP/NNDP_MLX.py`. Nothing is added to `Framework/` (constitution).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the directories and confirm the inputs the whole feature depends on.

- [X] T001 Create package skeleton `research/labelfree/__init__.py` and `research/labelfree/tests/` per plan.md structure
- [X] T002 [P] Create study results directory `research/label-free-study/briefs/results/` (currently absent)
- [X] T003 Write prerequisite check in `research/labelfree/preflight.py` asserting: 15m feather data present, feature scalers present in `saved_data/`, `regime/wf/pooled_nnnc.json` readable with 15 folds, and `config/config_wf.json` parsed for `max_open_trades`/`stake_amount`/`tradable_balance_ratio`
- [X] T004 RUN `research/labelfree/preflight.py` and record per-pair coverage as **days of actual data inside the training range** (never by start date), so the W3 universe is a measured input rather than an assumption

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The accounting primitives and bookkeeping every later phase needs.

**⚠️ CRITICAL**: No user story work begins until this phase is complete.

- [X] T005 [P] Implement fold and window access in `research/labelfree/folds.py` — read pinned windows from `regime/regimedata.py`, the anchored plan from `regime/wf/foldplan.py`, and emit per-fold `pair_universe` with purge/embargo fields (data-model §4, FR-015, FR-015a, FR-018)
- [X] T006 [P] Implement cost model in `research/labelfree/costs.py` — taker fee per side at the real tier, Abdi–Ranaldo half-spread floored at 1 bp, volatility-scaled slippage, and Corwin–Schultz as a sensitivity path (FR-004, FR-004a, research R2)
- [X] T007 [P] Implement append-only trial log in `research/labelfree/trials.py` — write-before-result with `result: null`, unique `trial_id`, abandoned-run records never deleted (FR-019, contracts/trial-log.md)
- [X] T008 [P] Implement archive reader in `research/labelfree/archive.py` — extract a fixed entry/exit trade series from `user_data/backtest_results/*.zip` for the frozen control signal (FR-008)
- [X] T008a [P] Implement feature ingestion in `research/labelfree/features.py` — build per-pair feature tensors via `utils/DataframePopulator.py` and the `Framework/FeatureNormalizer.py` scalers from `saved_data/`, so inputs are byte-for-byte what the control sees (FR-001)
- [X] T008b [P] Implement include-list validation in `research/labelfree/features.py` rejecting raw price, raw volume and any column proportional to price, with the assertion covered in `research/labelfree/tests/test_features.py` (FR-002, constitution Principle X)
- [X] T009 [P] Implement statistics helpers in `research/labelfree/stats.py` — interquartile mean, stratified bootstrap CI, identical-trade deduplication (FR-016)

**Checkpoint**: Accounting primitives ready. US1 can begin.

---

## Phase 3: User Story 1 - Trustworthy simulator (Priority: P1) 🎯 MVP · BLOCKING GATE

**Goal**: A simulator whose P&L reproduces the arbiter's on a fixed, already-realised trade series.

**Independent Test**: Replay the arbiter's own realised entry/exit timestamps for the frozen
control signal over W1; per-trade profit within 1 bp, total within 0.10pp, trade counts identical.

**Blocking**: Per FR-009 and SC-012, **no policy code may be written** until T017 passes. Budget is
5 fix-and-retest attempts.

### Tests for User Story 1

> Write these FIRST and confirm they FAIL before implementing.

- [X] T010 [P] [US1] Calibration test in `research/labelfree/tests/test_sim_calibration.py` asserting all three SC-001 conditions against the arbiter
- [X] T011 [P] [US1] Contract invariant tests in `research/labelfree/tests/test_sim_contract.py` — decision uses only bars ≤ t, fill at t+1 open, exposure ∈ [0,1], cap ≤ 10% of bar quote volume, concurrent positions ≤ `max_open_trades` (contracts/simulator.md)

### Implementation for User Story 1

- [X] T012 [US1] Implement accounting core in `research/labelfree/sim.py` — decision at bar t, fill at bar t+1 open, fee + half-spread + slippage applied to executed delta (FR-003, FR-004)
- [X] T013 [US1] Implement liquidity cap and carry in `research/labelfree/sim.py` — hard `min` in `mode=eval`, soft-min in `mode=train`, unfilled remainder carried to the next bar, `requested_delta` and `unfilled_carry` emitted (FR-005, research R3)
- [X] T014 [US1] Implement portfolio layer in `research/labelfree/sim.py` — `max_open_trades=10`, `stake_amount=900`, `tradable_balance_ratio=0.99` from `config/config_wf.json` (FR-006)
- [X] T015 [US1] Implement `replay()` in `research/labelfree/sim.py` — consumes a fixed trade series and applies the accounting layer only, never consulting a policy; MUST NOT implement take-profit, adaptive stop or grace (FR-008a, contracts/simulator.md)
- [X] T016 [US1] Implement trade-list emission in `research/labelfree/sim.py` matching the backtest-archive shape (`pair, open_date, close_date, profit_abs`) plus `exit_reason` for SC-018 (FR-007, data-model §3)
- [X] T017 [US1] Implement `research/labelfree/calibrate.py` — runs `replay()` against the arbiter and attributes any residual to fee application, fill timing or stake sizing rather than reporting a bare total mismatch (FR-008, research R11, Principle VI)
- [X] T018 [US1] RUN calibration against the arbiter on W1; iterate to pass SC-001 within the 5-attempt budget, logging `attempt_number` each time (SC-012)
- [X] T019 [US1] Write calibration report to `research/label-free-study/briefs/results/01-calibration-2026-09-10.md` per data-model §5 — window, frozen signal, both P&L figures, trade counts, max per-trade difference, component attribution, verdict (FR-033)

**Checkpoint**: 🚦 **Gate.** If T018 fails after 5 attempts, stop the feature and record the discrepancy (SC-011). Otherwise the simulator is now reusable by briefs 2, 4, 5 and 6.

---

## Phase 4: User Story 2 - Leakage and gradient sanity ceiling (Priority: P2)

**Goal**: Prove the learning path can exploit deliberate lookahead, so a later null result means "no edge" rather than "no gradient".

**Independent Test**: Train one configuration with features shifted +1 bar and one without, same fold and seed; the leaky run clears 5× the control's W1 return and the clean run does not.

### Tests for User Story 2

- [ ] T020 [P] [US2] Lookahead sanity test in `research/labelfree/tests/test_lookahead_sanity.py` asserting the leaky run clears ≥ 5× the control's W1 return and the clean run falls far below it (SC-002)
- [ ] T021 [P] [US2] Determinism test in `research/labelfree/tests/test_determinism.py` asserting byte-identical trades from two separate processes at the same seed (SC-008)

### Implementation for User Story 2

- [X] T021a [P] [US2] Add an assertion to `research/labelfree/tests/test_sim_contract.py` that the training path exposes no take-profit, adaptive-stop or grace exit — training exits are band crossings only (FR-014a)
- [ ] T022 [P] [US2] Implement policy in `research/labelfree/policy.py` — `mlx.nn.LSTM(64)` over `seq_len=16`, sigmoid exposure head, optional band head emitting `lo_t`/`hi_t` (FR-011, FR-012, research R4)
- [ ] T023 [P] [US2] Implement objectives in `research/labelfree/losses.py` — minibatch-unbiased mean-variance utility and robust pooled+SoftMin worst-window Sharpe (FR-014, research R5)
- [ ] T024 [US2] Implement training loop in `research/labelfree/train.py` — one policy per fold on that fold's training range only, truncated BPTT with 96-bar burn-in, `MLX_SEED` seeding, early stopping on the training objective only (FR-015, FR-017, research R7, R9)
- [ ] T025 [US2] Implement deliberately-leaky mode (features shifted +1 bar) in `research/labelfree/sim.py`, reachable only via an explicit flag recorded as `leaky: true` in the trial log (FR-010)
- [ ] T026 [US2] RUN `research/labelfree/train.py` twice on fold f08 at seed 1 (with and without `--leaky`); confirm SC-002 (leaky ≥ 5× control, clean far below)
- [ ] T027 [US2] RUN `research/labelfree/tests/test_determinism.py` across two separate processes; if trades are not byte-identical, escalate as a finding — do **not** relax SC-008 to a tolerance (research R9)

**Checkpoint**: Learning path proven live. The variant sweep is now interpretable.

---

## Phase 5: User Story 3 - Variant sweep with full trial accounting (Priority: P3)

**Goal**: Train the grid — 2 objectives × 2 turnover controls × 3 cost scales — at ≥5 seeds on one screening fold, with every configuration logged before it runs.

**Independent Test**: Run the grid on fold f08; the trial log holds one pre-written record per configuration, and every reported figure is an interquartile mean across seeds rather than a best-seed pick.

### Implementation for User Story 3

- [ ] T028 [P] [US3] Implement both turnover controls in `research/labelfree/policy.py` — learned no-trade band via `clamp(w_{t-1}, lo_t, hi_t)`, and plain sigmoid with an explicit `λ·|Δw|` penalty (FR-013, research R6)
- [ ] T029 [P] [US3] Wire the training cost scale γ ∈ {0.5, 1.0, 1.5} through `research/labelfree/costs.py` and the objective, with evaluation pinned to full or pessimistic cost (FR-004)
- [ ] T030 [US3] Implement grid driver in `research/labelfree/sweep.py` — appends each configuration's trial record with `result: null` **before** starting it, then attaches the result (FR-019, SC-015)
- [ ] T031 [US3] Implement seed aggregation in `research/labelfree/stats.py` — interquartile mean with stratified bootstrap CI; seeds producing identical trades counted once (FR-016, SC-016)
- [ ] T031a [US3] Pre-register the carry-forward rule in `research/labelfree/sweep.py` **before** T032 runs — the deterministic rule deciding which configurations advance when the screen cannot resolve them (see T032a). Recording it after seeing results voids it (plan.md Complexity Tracking)
- [ ] T032 [US3] RUN `research/labelfree/sweep.py` for the 12-configuration × 5-seed grid on **fold f08** (71 control trades, the most of any fold; deliberately not f07, which is the most *profitable*), sequentially (FR-019, FR-020)
- [ ] T032a [US3] Bootstrap the between-configuration deltas from the screen in `research/labelfree/stats.py` and state explicitly whether the top-2 selection is resolvable against the ±7.6pp single-window noise floor (AGENT_GUIDE; plan.md Complexity Tracking)
- [ ] T033 [US3] Select the configurations that advance: by rank if T032a shows the screen resolves them, otherwise by the pre-registered rule from T031a. Record which path was taken and the rationale in `research/labelfree/trials.jsonl` (FR-020)

**Checkpoint**: Two candidates chosen, every attempt logged and counted.

---

## Phase 6: User Story 4 - Verdict against every mandatory baseline (Priority: P4)

**Goal**: Evaluate the candidates through the pooled walk-forward harness against all four baselines, with cost stress and overfitting statistics.

**Independent Test**: The candidate and all four baselines run through the pooled harness; the report states the return delta with its CI against the 3.35pp/yr floor, and a sub-floor delta reads "not resolved".

### Implementation for User Story 4

- [ ] T034 [P] [US4] Implement baselines in `research/labelfree/baselines.py` — equal-weight buy-and-hold of the whitelist, and the **pinned** 20/60-bar EMA sign rule with band ±0.1, frozen before any run and contributing no trial-log records (FR-021, FR-021a)
- [ ] T035 [US4] Implement the deployable wrapper `NNDP/NNDP_MLX.py` — `BaseNNStrategy` subclass whose `get_predictions` returns exposure, entry on crossing the band's upper edge, exit on the lower edge; ATR stop and grace left unchanged; parameter dictionaries **merged** from the parent, never redefined (FR-027, FR-028, contracts/strategy-wrapper.md)
- [ ] T036 [US4] Implement `research/labelfree/export_policy.py` writing policy weights to `saved_data/` and pinning `get_model_path` explicitly in the wrapper (FR-029, AGENT_GUIDE silent-failure note)
- [ ] T037 [US4] RUN the pooled 15-fold walk-forward for both candidates via `regime/wf/run_folds.py` — the long job, ~113 min per arm, strictly sequential (FR-020, FR-030)
- [ ] T038 [US4] RUN `regime/wf/pool.py` then `regime/wf/compare_pooled.py --arm <arm> --control regime/wf/pooled_nnnc.json`; report the delta with bootstrap CI against the 3.35pp/yr floor (FR-020, SC-003)
- [ ] T039 [US4] RUN `research/labelfree/baselines.py` plus the control for all four baselines on the identical pair universe within each window (FR-021, SC-005)
- [ ] T040 [US4] RUN single-window evaluations for W1/W2/W3 via `regime/wf/run_folds.py` using universes from `research/labelfree/folds.py`, reporting each window's pair-universe size and running the control on that identical universe; state explicitly that the distant window excludes only XRP and that six of its nine pairs train on partial history (FR-015a, SC-004, research R10)
- [ ] T040a [US4] Compute the in-sample overfitting differential in `research/labelfree/stats.py` — the W2 advantage minus the W1 advantage over the control; if it exceeds 3.35pp/yr, record overfitting and stop the configuration (SC-010)
- [ ] T041 [US4] Produce the SC-018 reconciliation in `research/labelfree/reconcile.py` — simulator P&L, arbiter P&L, their difference, and the share of trades exited by the deployment stack rather than the band, read from the `exit_reason` field (FR-014b, SC-018)
- [ ] T042 [US4] RUN `research/labelfree/stress.py` at 1.5× and 2× fees and with the liquidity cap at 5%; report the fee level at which the edge crosses zero (FR-023, SC-007, SC-014)
- [ ] T043 [US4] Compute the Deflated Sharpe Ratio and PBO via CSCV over the **full** trial log in `research/labelfree/stats.py`; PBO > 0.2 fails the feature regardless of point estimate (FR-024, SC-006)
- [ ] T044 [US4] Extend `research/labelfree/stats.py` to emit trades per year and mean holding time beside every return figure, flagging turnover above 3× the control's (FR-022, SC-014)

**Checkpoint**: A verdict exists — accept, not resolved, or kill.

---

## Phase 7: User Story 5 - Reproduction, report and audit (Priority: P5)

**Goal**: Prove the result reproduces from its saved configuration, then produce the deliverable.

**Independent Test**: A fresh process loads the saved config and seed and produces identical trades; the trial-log audit confirms every configuration was logged before its result.

### Implementation for User Story 5

- [ ] T045 [US5] Implement `research/labelfree/reproduce.py` — reload a saved `PolicyConfiguration` in a fresh process and re-emit its trades (FR-026, data-model §2)
- [ ] T046 [US5] RUN `research/labelfree/reproduce.py` for the accepted configuration from a separate process; assert byte-identical trades (SC-008)
- [ ] T047 [US5] Implement `research/labelfree/audit_trials.py` asserting every configuration is logged, each written before its result, `trial_id` unique, the DSR/PBO trial count equals the record count, and the pinned trend baseline contributes no records (SC-015, contracts/trial-log.md)

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Close the workflow. Per the brief, the **final substantive tasks are the protocol §6 report and the trial-log audit**.

- [ ] T048 Re-run `/speckit-analyze` on the completed feature and save the report as `specs/009-labelfree-direct-policy/analyze.md`; fix every CRITICAL finding and re-run until clean (FR-032, SC-017)
- [ ] T049 RUN the trial-log audit `research/labelfree/audit_trials.py`; a failed audit invalidates the overfitting statistics and therefore the verdict (SC-015)
- [ ] T050 Write the result report `research/label-free-study/briefs/results/01-<date>.md` per protocol §6 — hypothesis, config path and seed list, trials logged, pooled return with CI, folds positive, trades/yr, hold time, four-baseline table, cost stress, PBO/DSR, SC-018 reconciliation, per-window universe sizes, verdict, and what changes for the next brief. When the verdict is "not resolved", the report MUST also record that the calibrated simulator passes to brief 2 and that this family stops after it. **Written even when the verdict is kill or not-resolved** (FR-025, FR-033, SC-009, SC-013, data-model §6)
- [ ] T051 Stage **only** the files this feature created and commit to `master` in `user_data/strategies`; never `git add -A`; the outer freqtrade checkout receives nothing (FR-031, constitution)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
   ↓
Phase 2 (Foundational)  ← blocks everything
   ↓
Phase 3 (US1 Simulator) ← 🚦 HARD GATE: no policy code exists until T018 passes
   ↓
Phase 4 (US2 Sanity)    ← proves the learner works before the grid is trusted
   ↓
Phase 5 (US3 Sweep)
   ↓
Phase 6 (US4 Verdict)
   ↓
Phase 7 (US5 Reproduction)
   ↓
Phase 8 (Analyze → Audit → Report)
```

### User Story Dependencies

Unlike a typical feature, these stories are **strictly sequential, not independent**. That is the
point of the design: each is a gate on the credibility of the next.

- **US1** depends only on Phase 2. It is the MVP and has standalone value — briefs 2, 4, 5 and 6 all consume the calibrated simulator even if this feature's policy fails.
- **US2** depends on US1 (needs a trustworthy simulator to detect leakage against).
- **US3** depends on US2 (an unproven gradient path makes a null sweep uninterpretable).
- **US4** depends on US3 (needs candidates).
- **US5** depends on US4 (needs an accepted configuration).

### Within Each User Story

Tests are written before implementation and must fail first. Implementation proceeds
models → accounting → drivers → runs. Every **RUN** task is sequential.

### Parallel Opportunities

File-authoring only, never runs:

- Phase 2: T005, T006, T007, T008, T008a, T008b, T009 — seven independent modules
- Phase 3 tests: T010, T011
- Phase 4: T020, T021, T021a (tests) and T022, T023 (policy, losses)
- Phase 5: T028, T029
- Phase 6: T034 alongside T035/T036 authoring

T012–T016 all edit `research/labelfree/sim.py` and are therefore **not** parallel despite sitting in one story.

## Parallel Example: Phase 2

```
# Five independent modules, no shared files:
T005  research/labelfree/folds.py
T006  research/labelfree/costs.py
T007  research/labelfree/trials.py
T008  research/labelfree/archive.py
T008a research/labelfree/features.py      (ingestion)
T008b research/labelfree/features.py      (validation — same file, so serialise after T008a)
T009  research/labelfree/stats.py
```

## Implementation Strategy

### MVP (User Story 1 only)

Phases 1–3, ending at T019. This delivers the calibrated simulator and its report. It is a genuine
standalone deliverable: it is the shared dependency of four sibling briefs, and it is the protocol's
blocking gate. If calibration cannot be made to pass in 5 attempts, the correct outcome is to stop
here with a recorded discrepancy — that is a successful negative result, not a failed feature.

### Incremental Delivery

1. **Phases 1–3** → calibrated simulator + calibration report (MVP, unblocks four briefs)
2. **Phase 4** → proven learning path; the leakage ceiling makes later nulls meaningful
3. **Phase 5** → screened candidates with honest trial accounting
4. **Phase 6** → verdict against baselines with cost stress and overfitting statistics
5. **Phases 7–8** → reproduction, audit, report

Each increment is independently valuable, and each can end the feature early with a recorded
result rather than a silent abandonment.

### Budget

The brief allows 3–4 sessions. Phase 3 is budgeted at 5 calibration attempts (SC-012). Phase 6's
pooled runs are the wall-clock cost — roughly 113 minutes per arm, sequential.

## Notes

- **Sequential runs are mandatory.** `[P]` marks file authoring only; no **RUN** task is ever parallel.
- **Stage only created files.** Repository files are edited concurrently by the user; `git add -A` is prohibited.
- **A kill verdict still completes Phase 8.** The negative result is the deliverable, and the spec trail is what makes it citable later.
- **Never weaken a gate.** If SC-008's byte-identical requirement or SC-001's 1 bp tolerance proves hard, escalate as a finding; the study forbids loosening protocol §5.
- **The W3 universe caveat is mandatory reporting**, not an optional footnote: the distant window trains on 9 of 10 pairs (only XRP excluded, 911-day delisting gap), six of them on partial history of 345-533 days rather than 729. Report the asymmetry in data *quantity*; ZEC is present, so there is no missing-pair confound.
