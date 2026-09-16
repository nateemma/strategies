---

description: "Task list for feature 017 — signal-context-timing"
---

# Tasks: signal-context-timing

**Input**: Design documents from `specs/017-signal-context-timing/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/signalctx.md, quickstart.md

**Tests**: REQUIRED. Causality, reconstruction fidelity and the matched-control rule are all
testable preconditions, and three of this study's recorded failures were probes that passed while
measuring nothing.

## Format: `[ID] [P?] [Story] Description`

**All runs are sequential** (FR-016); `[P]` means only that a task touches a different file with
no unmet dependency.

---

## Phase ordering rationale — read before executing

Story phases do **not** run in bare priority order:

- **US1 (P1) first** — it is the gate everything else waits on, and it costs **zero backtests**.
- **US3 (P3) and US4 (P4) before US2 (P2)** — both are cheap, both are independent of US1's
  verdict (FR-023 keeps US3 alive either way), and US4's masked-signal outcomes inform whether
  the expensive exit-side work is worth starting.
- **US2 (P2) and US5 (P5) last** — the only phases that spend backtests, and US2 is explicitly
  gated on US1 clearing the confound.

Resulting order: **Setup → Foundational → US1 → US3 → US4 → US2 → US5 → Polish**.

---

## Phase 1: Setup

- [X] T001 **Preserve the side-cars FIRST.** Copy **only** `/tmp/lf/es_sink` (30 side-cars, **366 MB**) to `user_data/sidecars/es_sink/` and **add that path to `.gitignore` in the same task**, then point `research/signalctx/__init__.py` at the copy. **Do NOT copy `/tmp/lf/es_panels`** — it is 2.1 GB and rebuilds from side-cars in 3.4 s per fold (~1.7 min for all 30), so preserving it would add 2.1 GB to the repository to save under two minutes. A `/tmp` clear otherwise turns this feature from free into ~44 minutes of regeneration
- [X] T002 Create `research/signalctx/__init__.py` with the pinned thresholds (`NOISE_FLOOR_PP = 7.633`, `RETURN_FLOOR_PP_YR = 3.350`, `FIDELITY_MIN = 0.95`, `ALPHA` from `context.HYPOTHESES`) and an interpreter guard using `sys.prefix` — **never** `Path(sys.executable).resolve()`, which follows the venv symlink out to pyenv and rejects the correct interpreter
- [X] T003 [P] Create `research/signalctx/out/` and add it to `.gitignore`; verify `git status` shows no large artefact staged from T001
- [X] T004 [P] Save the pre-implementation `/speckit-analyze` report as `specs/017-signal-context-timing/analyze.md` (SC-016)

---

## Phase 2: Foundational (BLOCKING — no user story may begin until complete)

- [X] T005 Implement `reconstruct()` in `research/signalctx/signals.py` — apply `research/labelfree/es/panel.py::entries()` per fold and **collapse contiguous `enter_long` runs to one Signal event** (R2: f08 has 139 bars but 120 events)
- [X] T006 Implement `fidelity()` in `research/signalctx/signals.py` — match recorded trades' entry bars against reconstructed Signals; refuse any fold below `FIDELITY_MIN`. Measured f08 reference: 70/71 = 0.986, the miss being the test range's first bar (data-model.md § Signal)
- [X] T007 Implement `unrealised()` in `research/signalctx/paths.py` — per-bar unrealised P&L from entry price and OHLCV, **additive in percentage points**, never compounded across concurrent trades (brief 013's recorded inversion)
- [X] T008 Implement `reconcile()` in `research/signalctx/paths.py` — the path's final value must match the archive's recorded `profit_ratio` to within **`RECONCILE_TOL = 0.002`** (roughly twice a round-trip fee); this is the cheap check that the reconstruction is right at all
- [ ] T009 Implement `build()` in `research/signalctx/context.py` — the four context features with their **windows frozen**: `drawdown_from_high` against a **96-bar (1-day)** trailing high, and `burst_position` treating signals more than **24 bars (6h)** apart as separate bursts. An undefined window is a silent free parameter inside a test claiming exactly four hypotheses. Every value from bars at or before the signal (FR-004)
- [ ] T010 Declare `HYPOTHESES` (exactly 4) and `ALPHA = 0.05/4 = 0.0125` in `research/signalctx/context.py` **before any test is run** (FR-006, research R5)
- [ ] T011 Implement `register_trials()` in `research/signalctx/report.py` wrapping `research/labelfree/trials.py::pre_register`, so this feature's trials enter the log the deflated Sharpe reads

- [X] T011a **FIDELITY GATE, all 15 folds.** Run `fidelity()` across every fold and write `research/signalctx/out/fidelity.json`. Any fold below 0.95 is unusable; if several fail the instrument is dead and Phases 3-7 are wasted effort. It costs seconds and is the cheapest possible kill — f08 alone (0.986) is not evidence about the other fourteen

**Checkpoint**: signals, paths and context exist, and the instrument is validated on every fold.

---

## Phase 3: User Story 1 — The effect is not just "losers lose" (P1) 🎯 MVP, zero backtests

**Goal**: establish whether a repeat signal predicts a worse outcome **beyond** what unrealised P&L already predicts.

**Independent test**: the effect survives conditioning on unrealised P&L at signal time, with an effect size in pp.

### Tests (write first; they MUST fail before T016)

- [ ] T012 [P] [US1] `test_reconstruction_fidelity` in `research/labelfree/tests/test_signalctx_signals.py` — recorded trades appear as Signals at ≥ 0.95; a fold below that raises
- [ ] T013 [P] [US1] `test_runs_collapse_to_events` in `research/labelfree/tests/test_signalctx_signals.py` — adjacent signal bars count as ONE event
- [ ] T014 [P] [US1] `test_path_reconciles_with_archive` and `test_path_is_additive_not_compounded` in `research/labelfree/tests/test_signalctx_paths.py`
- [ ] T015 [P] [US1] `test_matched_controls_same_holding_age` in `research/labelfree/tests/test_signalctx_confound.py` — a control drawn from all trades rather than those alive at the same age raises (FR-002)
- [ ] T015a [P] [US1] `test_winrate_effect_cannot_reach_the_return_floor` in `research/labelfree/tests/test_signalctx_confound.py` — passing `effect_winrate_pp` to the resolution gate raises; only `effect_return_pp_yr` may be compared to 7.633
- [ ] T016 [P] [US1] `test_planted_null_is_not_detected` in `research/labelfree/tests/test_signalctx_confound.py` — on data where the repeat flag is assigned at random conditional on unrealised P&L, the test must NOT report an effect. Without this the machinery could manufacture one

### Implementation

- [ ] T017 [US1] Make T012–T016 pass by correcting `signals.py`, `paths.py` and `confound.py` — **not** by relaxing an assertion
- [X] T018 [US1] Implement `strata()` in `research/signalctx/confound.py` — decile bins of unrealised P&L at the moment the repeat signal arrives
- [X] T019 [US1] Implement `cmh()` in `research/signalctx/confound.py` — Cochran-Mantel-Haenszel across strata, the **primary** test (research R3)
- [X] T020 [US1] Implement `logit()` in `research/signalctx/confound.py` — logistic regression of final win on `unrealised_pnl + received_repeat_signal`, the **secondary** test, reporting coefficient and interval
- [X] T020a [US1] Implement `effect_sizes()` in `research/signalctx/confound.py` returning **both currencies**: `effect_winrate_pp` (the win-rate difference, reported only) and `effect_return_pp_yr = Δ mean profit_ratio × 9.0 × trades_per_year`, where 9.0 is pp per unit `profit_ratio` under the fixed stake (FR-007)
- [X] T021 [US1] Implement `verdict()` in `research/signalctx/confound.py` — returns `survives` or `proxy_for_drawdown`, gating on **`effect_return_pp_yr`** only. It MUST NOT return `survives` for a return consequence inside **7.633 pp/yr** however small the p-value, and MUST **raise** rather than silently gate on the win-rate figure: the motivating 14.8 win-rate points would clear that floor while carrying no return information (Principle III, SC-015)
- [X] T022 [US1] Run the confound test across all 15 folds sequentially — the repeat-signal effect re-tested conditioned on unrealised P&L at signal time (FR-001) — and write `research/signalctx/out/confound.json` with effect size beside every p (SC-001, FR-007)
- [X] T023 [US1] **GATE.** If `verdict == proxy_for_drawdown`, record "the repeat signal restates being underwater", **stop the repeat-signal arm** (FR-003, SC-012), and write `exploratory: true` into `research/signalctx/out/confound.json`. Downstream stages **read the flag from disk**, never inherit it in memory, so it cannot be lost between processes (FR-023). Phases 4 and 5 still run; Phase 6 does not

**Checkpoint**: the motivating effect is either real beyond the confound, or named as a proxy.

---

## Phase 4: User Story 3 — First-signal context (P3) — **NOT RUN (open, not killed)**

> Kept alive as EXPLORATORY by FR-023 after the T023 confound kill, with the flag persisted in
> `out/confound.json`. Work stopped here **by decision on 2026-09-16, not by a gate**. Whether
> first-signal context predicts signal quality remains an untested question.

- [ ] T024 [P] [US3] `test_prev_signal_won_is_null_while_open` in `research/labelfree/tests/test_signalctx_context.py` — **the likeliest leak in the feature**: the obvious join takes the previous signal's outcome even when that trade is still open and its result is not yet knowable
- [ ] T025 [P] [US3] `test_context_truncation_stable` and `test_trade_future_never_enters_context` in `research/labelfree/tests/test_signalctx_context.py` — `build(signals[:m])` equals `build(signals)[:m]` for many `m`, and no context value changes when a trade's own later bars are deleted (FR-004, FR-005, SC-002)
- [ ] T026 [US3] Make T024–T025 pass
- [ ] T027 [US3] Implement `test_hypotheses()` in `research/signalctx/sweep.py` — one row per hypothesis with statistic, p, **effect size in pp**, and pass/fail at `ALPHA = 0.0125`; stamp every row EXPLORATORY by **reading `exploratory` from `out/confound.json`** (FR-023)
- [ ] T028 [US3] Run the sweep across all 15 folds; write `research/signalctx/out/sweep.json` (SC-003)
- [ ] T029 [US3] For any hypothesis clearing `ALPHA`, measure its overlap with the incumbent guard chain **before** any return figure is read (FR-011). Overlap that large is inertness under Principle IV, not a result (FR-012, SC-013)

---

## Phase 5: User Story 4 — Masked signals observed honestly (P4) — **NOT RUN (open, not killed)**

> Same status as Phase 4. The census measured **266 masked events, 41.3%**, but what they would
> have been worth was never observed. Stopped by decision, not by a gate.

- [ ] T030 [P] [US4] `test_masked_outcome_cannot_reach_pnl` in `research/labelfree/tests/test_signalctx_confound.py` — `assert_not_pnl` raises when a MaskedOutcome reaches a P&L path (FR-009)
- [ ] T031 [US4] Implement `observe()` in `research/signalctx/masked.py` — run the stacked backtest per fold and label **every row** *not reproducible in dry/live* (FR-008, FR-010)
- [ ] T032 [US4] Implement `assert_not_pnl()` in `research/signalctx/masked.py` and call it from `report.py` before any return figure is written
- [ ] T033 [US4] Run the census and masked-signal outcomes across all 15 folds; write `research/signalctx/out/masked.json`. Compare masked vs admitted **signal quality** (win rate, excursion), never returns (SC-004, research R6)

---

## Phase 6: User Story 2 — The repeat signal as exit-side information (P2) — **NOT RUN**

> **Stopped by T023 on 2026-09-16.** The repeat signal is a proxy for being underwater:
> conditioning on unrealised P&L removes 83% of the raw effect and the logistic coefficient on
> the repeat indicator is -0.0070 (p=0.973). There is no information left to act on, so an
> exit-side rule built on it would be acting on the position's own mark under another name.
> The tasks below are deliberately unchecked: they were never executed, and the gate is why.

- [ ] T034 [P] [US2] `test_entries_identical` in `research/labelfree/tests/test_signalctx_confound.py` — an arm whose entries differ from the control raises (FR-020)
- [ ] T035 [P] [US2] `test_liveness_null_case` in `research/labelfree/tests/test_signalctx_confound.py` — an arm reporting zero triggers MUST be byte-identical to its control; zero triggers on a differing arm means **the probe is broken**, not that the rule is inert (research R9)
- [ ] T036 [US2] Implement the exit-side mixin in `NNNC/_signal_context.py` — on a repeat `enter_long` while a position is open, act on the exit stack only. Follows `NNNC/_hold_window.py` and `NNNC/_position_adjust.py`. **Pre-registered primary rule: tighten the current stop distance by a frozen factor of 0.5** (FR-019a). The magnitude is not searched — "tighten the stop" without a number is a free parameter, and tuning it alongside the trigger is the joint search that took brief 05 to PBO 0.7128. An outright exit is a labelled sensitivity and cannot produce the verdict (Principle VIII)
- [ ] T037 [US2] Implement `generate()` and `assert_reuses_weights()` in `research/signalctx/arms.py` — override `get_model_path()` to the **ungated** fold's directory and refuse unless those weights exist and no new `saved_data/` appears (research R8)
- [ ] T038 [US2] Implement `assert_entries_identical()` and `liveness()` in `research/signalctx/arms.py`, returning `n_triggered` **and** the byte-identity flag. Liveness MUST be proven before any null is read as a finding (FR-014)
- [ ] T039 [US2] Measure overlap between the rule's triggers and the existing `stoploss_grace` / ATR-adaptive stop — the share of triggers on trades where the existing stack fired anyway **within 8 bars** — **before any performance figure** (FR-022). The motivating effect was itself measured *through* that stack, so this is the likeliest inertness (research R7)
- [ ] T040 [US2] **GATE.** Overlap **≥ 0.70**, or `n_triggered == 0`, is **inert** — report as a defect under Principle IV, never as evidence about signal context (FR-015, SC-014, SC-019, SC-020). A zero-trade or zero-change arm is inconclusive, never a zero result. Skip to Phase 8

---

## Phase 7: User Story 5 — A rule that survives the harness (P5) — **NOT REACHED**

> No candidate rule survived to evaluate.

- [ ] T041 [US5] Run the primary `tighten_on_repeat` arm across all 15 folds sequentially, no retraining; pool to `research/signalctx/out/arm_tighten.json`
- [ ] T042 [US5] **Only if the primary arm clears T048**, run the `exit_on_repeat` sensitivity; write `research/signalctx/out/arm_exit.json` **labelled as a sensitivity that cannot produce the verdict**. Deferred deliberately: 15 backtests for something pre-registered as unable to decide anything are not spent speculatively
- [ ] T043 [US5] Compute per-fold return **and** max drawdown with bootstrap intervals for control and arm, stating the count of affected trades per fold **before** either figure (FR-021, SC-018); write `research/signalctx/out/performance.json`
- [ ] T044 [US5] Compare against the 3.350 pp/yr floor and all four mandatory baselines of `research/label-free-study/00-evaluation-protocol.md` §3, net of full costs (FR-013, SC-005, SC-007)
- [ ] T045 [US5] Check persistence — positive in ≥10 of 15 folds and in each of W1/W2/W3 (SC-006, Principle II)
- [ ] T046 [US5] Compute PBO via CSCV and the deflated Sharpe with `research/labelfree/stats.py`; re-check the delta at 1.5× fees (SC-008, SC-009)
- [ ] T047 [US5] Compute the overfitting differential with `research/labelfree/stats.py::overfitting_differential()` — W2 advantage exceeding W1 by more than the floor is overfitting: stop and record (SC-011)
- [ ] T048 [US5] Apply the verdict rules and write `research/signalctx/out/verdict.json`: anything inside 7.633 pp is **not resolved**, never "no effect" (SC-015); a de-risk trigger that buys drawdown at flat return is a success, one that costs return is a trade-off (FR-021)

---

## Phase 8: Polish & Cross-Cutting

- [X] T049 Implement `write()` in `research/signalctx/report.py` and produce `research/signalctx/RESULTS.md` — **whatever the outcome**, including an early stop at T023 or T040 (FR-019, SC-017)
- [ ] T050 Complete every pre-registered trial via `research/labelfree/trials.py::complete` (or `abandon` on an early stop) so the log T052 audits is actually written to
- [ ] T051 Reproduce the headline result from `research/signalctx/out/verdict.json` in a separate process, re-applying the rules rather than reading the conclusion (SC-010)
- [ ] T052 [P] Run the trial-log audit per `research/label-free-study/00-evaluation-protocol.md` §6
- [X] T053 [P] Record the outcome in `docs/WALKFORWARD_TODO.md` item 3, which is where this thread originated
- [ ] T054 Run `specs/017-signal-context-timing/quickstart.md` end to end, sequentially
- [ ] T055 Run `/speckit-analyze` again and save it as `specs/017-signal-context-timing/analyze-post.md` (FR-018, SC-016)
- [ ] T056 Stage **only** the files this feature created or modified — never `git add -A`; check `git status` first and leave anything unexpected alone (FR-017)

---

## Dependencies & Execution Order

### Phase dependencies

- **Phase 1 Setup** — T001 blocks everything (side-car preservation)
- **Phase 2 Foundational** — blocks every story
- **Phase 3 US1** — blocks US2; carries the T023 gate
- **Phase 4 US3** — depends on Phase 2 only; runs whatever T023 decides
- **Phase 5 US4** — depends on Phase 2 only
- **Phase 6 US2** — requires T023 to have cleared; carries the T040 gate
- **Phase 7 US5** — requires T040 to have cleared
- **Phase 8 Polish** — reachable from T023, T040 or T048; the report is written on every path

### Critical gates

| Task | Gate | If it fails |
|---|---|---|
| T006 | reconstruction fidelity ≥ 0.95 | the instrument is wrong; nothing downstream is evidence |
| T008 | path reconciles with the archive | the unrealised-P&L series is wrong |
| T016 | a planted null is NOT detected | the machinery manufactures effects |
| **T023** | effect survives conditioning | **KILL the repeat-signal arm**; US3 continues EXPLORATORY |
| T011a | fidelity ≥ 0.95 on **all 15** folds | the instrument is dead; stop before Phase 3 |
| T029/T039 | overlap < 0.70 with existing guards/stops | **inert** under Principle IV |
| **T040** | the rule fires, and distinctly | **KILL as a defect** → Phase 8 |
| T048 | acceptance | `effect_return_pp_yr` inside 7.633 is "not resolved" |

### Parallel opportunities

`[P]` marks different files with no unmet dependency. **Evaluation runs are never parallel**
(FR-016): T041, T042 and every fold within them execute sequentially.

- T003, T004 together
- T012–T016 together (six test functions, all failing first)
- T024, T025 together
- T034, T035 together
- T052, T053 together

---

## Implementation Strategy

### MVP — a defensible answer for zero backtests

1. Phase 1 + Phase 2 → signals, paths and context exist and are validated
2. Phase 3 (US1) → **stop here if the effect is a proxy for being underwater**

That is the whole first kill gate and it spends **no backtest time at all**. Given that the
motivating measurement cannot distinguish a real effect from "losing trades lose", reaching T023
and finding `proxy_for_drawdown` is a genuinely likely outcome — and a complete, reportable one.

### Incremental delivery

Each phase ends in a recordable state, and T049 writes the report on every exit path. Phases 4
and 5 remain worthwhile even after a T023 kill: the first-signal sweep and the masked-signal
census are independent questions that have never been measured.

### Pre-registered expectation

**The repeat-signal arm dies at T023.** The second most likely outcome is that it survives and
then proves inert at T039 against the stop stack it was measured through. A result that clears
both and then clears the 7.633 pp floor would be the first adoptable finding in this line of
work — which is precisely why the gates are ordered before the performance run, not after.
