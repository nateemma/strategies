---

description: "Task list for 005 NNMT Operating Point"
---

# Tasks: NNMT Operating-Point Programme

**Input**: Design documents from `specs/005-nnmt-operating-point/`

All commands run from the repository root `/Users/philprice95/projects/freqtrade`.
Paths below are relative to `user_data/strategies/` unless shown otherwise.

## Standing conventions (apply to EVERY evaluation task)

**Guards-off is always the six neutralising values, never `--guards off`** — that
flag also kills the entry `volume > 0` check AND the entire exit-side guard block:

    NEUT="--set entry_rvol_threshold=-1.0 --set entry_atr_pct=-1.0 \
          --set entry_adx_threshold=-1.0 --set entry_bb_width_threshold=-1.0 \
          --set entry_guard_threshold=5.0 --set entry_close_norm_threshold=5.0"

**Per-arm protocol**: confirm `returncode` 0 (FR-027); compare `trades_hash`
against the arm's control — identical means INCONCLUSIVE, not refuted;
judge on W1 only, report W2/W3 labelled (FR-018), and additionally give any
W3 statement its trade count and an explicit flag (R7 — W3 is near-flat on this
family); never accept on a classifier metric (FR-019); every arm carries its own
`.json` (FR-022); every oracle figure is a LOOKAHEAD CEILING and nothing measured
with it is promotable (FR-024); two seeds via `MLX_SEED` before any acceptance
(FR-020, R8 — `train_seed` is INERT for multi-task and would silently produce
identical models); never compare across labels at a fixed confidence threshold
(FR-021).

**Arms run SEQUENTIALLY, never concurrently (FR-025).** Two concurrent jobs
previously produced load 60 on 14 cores and finished slower than serialising.
This applies to every task below that invokes `run_arm.py` or `test_strat.sh`.

**The power threshold is DEFERRED (R6).** `compare.py`'s 1.0pp floor was
calibrated for ~97-trade arms and must NOT be applied blindly to NNMT's counts,
which are unknown. T008 sets it from the observed baseline count; until then no
task may declare a difference significant.

**Parameter dictionaries MUST merge from the parent.** Every new subclass here
redefines `buy_params`/`sell_params` at its own risk: a fresh literal silently
drops every inherited key (recorded failure — a strategy ran with its stoploss
grace disabled, undetected). Build each arm's dict as `{**parent, **overrides}`.

**Hazards**: `get_model_path` keys off `self.__class__.__name__`, so a pure
re-parameterisation with no model of its own TRAINS INSIDE THE BACKTEST, fitting
the evaluation window and returning different numbers with NO error — this
produced 290 trades instead of 431 on NNNC before it was caught. `NNMT_NoModel`
must call the parent's `get_entry_conditions` FIRST or `process_predictions`
never runs and the exit side raises `KeyError`. `test_strat.sh -f` DELETES the
model before retraining. `TaskStop` does not kill process trees — use
`pkill -9 -f`. `Framework/BaseStrategy` MECHANISM code must not be left modified
(a live dry run imports it); parameter DEFAULTS are fine.

**Already settled by research.md — do NOT redo**: R1 (rebuild the control),
R2 (the label also drives the profit head), R3 (the NNNC oracle pattern
transplants), R4 (NNMT has no params file), R5 (the exit-rvol asymmetry is
long-standing), R6 (defer the power threshold), R7 (W3 caveat), R8 (MLX_SEED),
R9 (the oracle cannot settle exit questions).

---

## Phase 1: Build a valid control

**Purpose**: "NNMT today" is a mixture — models trained at label 0.007 running
under a 0.020 config with a changed take-profit — and is not a valid control for
anything (R1).

- [X] T001 Create `NNMT/NNMT_MLX_Base.json` pinning NNMT's PRE-INHERITANCE configuration: label thresholds 0.007, `cexit_take_profit` 0.032, and NNMT's own unchanged guards (rvol 0.25, guard 0.3, adx 38.0, atr_pct 0.006, bb_width 0.06, close_norm -0.1) and `prediction_threshold` 0.5. NNMT has no params file today, which is exactly how the inheritance reached it (R4, FR-022)
- [X] T002 Create `NNMT/NNMT_MLX_Base.py` subclassing `NNMT_MLX` with `MIN_BUY_GAIN_THRESHOLD` and `MIN_SELL_LOSS_THRESHOLD` at 0.007, and a `get_model_path` override aliasing `NNMT_MLX`'s weights so the control does NOT train its own model (FR-023, hazard)
- [X] T003 Verify the model at `saved_data/NNMT_MLX/` was trained against label 0.007, by reading the training metadata/config recorded beside the weights (and cross-checking the file mtime against the framework change). All three of spec Context, research R1 and Phase 1's purpose already assert 0.007, so the expected outcome is CONFIRMED and no retrain. **If it was not**: T002 aliases `NNMT_MLX_Base` onto `saved_data/NNMT_MLX/`, so retraining the control writes to the PRODUCTION weight directory and `-f` DELETES it first — back up `saved_data/NNMT_MLX/` in full before running `MLX_SEED=42 zsh scripts/test_strat.sh -f -t 20220101-20250831 NNMT NNMT_MLX_Base` (FR-007)
- [X] T004 Establish the control baseline on all three windows: `run_arm.py --strategy NNMT_MLX_Base --window W1|W2|W3 --tag nnmt_ctl`, confirming `returncode` 0 on each (FR-018, FR-027)
- [X] T005 Record the control in `specs/005-nnmt-operating-point/baseline.md` with per-window trade counts, return, drawdown, profit factor and win rate — this is the reference every later delta is measured against, and the count that T008's power threshold is derived from (FR-018, SC-001)

**Checkpoint**: a coherent control exists whose model and configuration agree.

---

## Phase 2: User Story 1 — HEADROOM (Priority: P1) 🎯 **HARD GATE**

**Goal**: establish how much a perfect model would be worth on NNMT at its own
gate, before spending anything else.

**Independent Test**: a three-way comparison — no model, real model, perfect
model — exists on all three windows with trade counts, and the headroom figure
is stated.

> ⛔ **HARD GATE ON PHASE 5.** On NNNC this single measurement retrospectively
> invalidated years of model-side work: a perfect model was worth 0.68pp, so
> nothing could ever have paid. If NNMT's headroom is not materially larger than
> the resolution implied by its observed trade count, **that is the feature's
> ANSWER** (FR-004). Report it and re-scope. **Re-scoping means specifically (FR-004): Phase 3 and Phase 4 still
> run — Phase 3 EXPLAINS the headroom result and Phase 4 is a defect fix that
> stands regardless of it — while Phase 5 does NOT run as written.**

- [X] T006 Create `NNMT/NNMT_Oracle.py` + `.json` following the `NNNC/NNNC_Oracle.py` pattern: subclass `NNMT_MLX_Base`, override ONLY `get_entry_conditions` to set `predict_buy`/`predict_sell` from `get_training_labels`, and override `get_model_path` to alias the control's weights. This bypasses the multi-task predict path, `apply_task_filters` and the forward-peak-regressor filter, which is correct — it measures what a perfect ENTRY signal is worth (R3, FR-001)
- [X] T007 Create `NNMT/NNMT_NoModel.py` + `.json` following `NNNC/NNNC_NoModel.py`: accept every bar so the guards alone select. It MUST call the parent's `get_entry_conditions` first — skipping it means `process_predictions` never runs and the exit side raises `KeyError` (FR-002, hazard)
- [X] T008 Run all three arms on all three windows at the control configuration (`nnmt_oracle`, `nnmt_nomodel`, `nnmt_ctl`), then STATE THE POWER THRESHOLD as a function of the observed baseline trade count — do not import NNNC's 1pp floor (R6, FR-001, FR-002, FR-003)
- [X] T009 **GATE** Write `specs/005-nnmt-operating-point/headroom.md`: the three-way comparison with trade counts per window, the headroom figure on W1, the share of the span the real model captures, and an explicit statement of whether headroom exceeds the T008 threshold. If it does not, state that further model-side work on NNMT cannot pay at this gate and re-scope: Phases 3 and 4 still run, Phase 5 does not (FR-004, SC-001, SC-002)

**Checkpoint**: the feature either has a reason to continue, or has its answer.

---

## Phase 3: User Story 3 — Guard attribution (Priority: P2)

**Goal**: determine whether NNMT's guards or its model select the trades.

**Independent Test**: each guard's individual contribution is reported, inert
guards are named, and the sum of individual effects is compared to the joint
effect.

- [ ] T010 [US3] Run a leave-one-out over NNMT's six quality guards at the control configuration, neutralising each in turn (`rvol`/`atr_pct`/`adx`/`bb_width` to -1.0, `guard_metric`/`close_norm` to 5.0), tags `nnmt_no_<guard>`, all three windows, keeping `entry_enable_guards` True so `volume > 0` stays live (FR-009)
- [ ] T011 [US3] Run the joint arm with all six neutralised (`nnmt_gqoff`) on all three windows, so interaction can be separated from addition (FR-011)
- [ ] T012 [US3] Record each guard's delta and trade count in `specs/005-nnmt-operating-point/guards.md`, naming any guard that changes NEITHER trades NOR return as INERT rather than beneficial, and stating whether the individual deltas sum to the joint effect (FR-009, FR-010, FR-011, SC-004)

**Checkpoint**: selection is attributed between guards and model.

---

## Phase 4: User Story 2 — Resolve the inherited changes (Priority: P1)

**Goal**: make NNMT's configuration and models agree, with each inherited change
judged on its own evidence.

**Independent Test**: both changes are measured separately against the control on
two seeds each, one value is chosen for each with a stated reason, and
configuration and models are left consistent.

- [X] T013 [US2] Check whether the PROFIT head carries a non-zero loss weight — it is one of `_FILTER_TASKS` and may be untrained, in which case the label/profit-head coupling (R2) is cosmetic. Record the answer; it decides whether T014's pinning is load-bearing (R2, FR-008, Constitution VIII)
- [ ] T014 [US2] Create the LABEL arm `NNMT/NNMT_Lbl02.py` + `.json` at label 0.020, **pinning the profit-head threshold to 0.007** so the arm stays single-variable — `MIN_BUY_GAIN_THRESHOLD` is also read at `NNMTStrategy` ~438/~510 where it sets the profit head's boundaries at `PROFIT_HORIZON` (R2, FR-008, Constitution VIII)
- [ ] T015 [US2] Train the label arm at two seeds (`MLX_SEED` 42 and 7) — a label change requires retraining, and `train_seed` is inert for multi-task (FR-020, R8)
- [ ] T016 [US2] Re-sweep the confidence threshold for the label arm rather than inheriting the control's: a different label shifts the confidence distribution, so a fixed threshold is confounded (FR-021)
- [ ] T017 [US2] Measure the TAKE-PROFIT change (0.032 → 0.022) separately against the control via `--set-sell cexit_take_profit`, both seeds with the spread reported, all three windows, run sequentially. This needs NO retraining and MUST be measured on the real model — an oracle null is not evidence for an exit change (R9, FR-030)
- [ ] T018 [US2] Record both verdicts in `specs/005-nnmt-operating-point/inherited.md`, each justified on realised return and never on a classifier metric, each supported by BOTH seeds with the spread reported explicitly, then leave NNMT's configuration and models consistent with the chosen values, marking any model trained against a superseded label as stale. Note in the document that the LABEL comparison is optimum-vs-optimum, not single-variable — FR-021 requires the threshold re-swept per label, which is deliberate and is what makes the comparison fair. **The chosen label MUST be implemented as an NNMT class-level override, NEVER by reverting `Framework/TrainingConfig` — NNNC now depends on the framework value** (FR-005, FR-006, FR-007, FR-008, FR-020, FR-030, SC-003, SC-007)

**Checkpoint**: NNMT is internally consistent and each inherited change is
attributed.

---

## Phase 5: User Stories 4, 5, 6 — Operating point and exit side (Priority: P2/P3)

**Entry condition**: Phase 2 found headroom. If it did not, this phase is
re-scoped, not executed.

- [ ] T019 [US4] FIRST choose the gate: pick the guard configuration for the sweep from Phase 3's attribution by a STATED rule (e.g. neutralise every guard Phase 3 found inert, and any whose removal did not cost W1 return), and record it — the operating point is the threshold AND the guard settings together, so leaving the gate implicit leaves half the operating point unchosen (FR-031). THEN sweep `prediction_threshold` across all three windows at that gate, reporting per-window optima EVEN WHERE THEY DISAGREE — NNNC's disagreed (0.82/0.80/0.75) and its value is explicitly not transferable to NNMT, which uses 0.5 today (FR-012, FR-014)
- [ ] T020 [US4] Choose a threshold by a STATED decision rule over the per-window results rather than by the best single window, supported by both seeds with the spread reported, and record it as a fixed choice never to be tuned per era. Report the chosen gate alongside it as the other half of the operating point (FR-013, FR-020, FR-031, SC-005, SC-007)
- [ ] T021 [US5] Sweep `exit_rvol_threshold` including a fully-open setting via `--set-sell`, ON THE REAL MODEL — never the oracle, which only enters bars that already reach the target and so cannot bound how the exit harvests (R9). It sits at 2.0 — NNNC's ENTRY value — while NNMT's own entry rvol is 0.25, an eight-fold asymmetry that has existed for the life of the family and only became testable on 2026-09-06 (R5, FR-015)
- [ ] T022 [US5] Record the exit-gate result, naming it INERT if it changes nothing rather than calling it confirmed-correct (FR-015)
- [ ] T023 [US6] Evaluate `enable_exit_signal` ON THE REAL MODEL (R9) at BOTH the chosen exit-gate value and with that gate fully open, so a null cannot be an artefact of the gate — on NNNC a naive test showed almost nothing purely because the exit path had its own unopened guards (FR-016)
- [ ] T024 [US6] State explicitly whether the prior NNMT claim of +10.4pp for the exit signal at an open gate reproduces (FR-017, SC-006)

**Checkpoint**: the operating point and both exit-side questions are settled on
NNMT's own evidence.

---

## Phase 6: User Story 7 — Conclusion (Priority: P3)

- [ ] T025 [US7] Write the conclusion into `regime/NNMT_OPERATING_POINT.md`: NNMT's headroom, the windows it was measured on, the chosen configuration, and what the result forecloses for future model and augmentation work on this family (FR-028, SC-010)
- [ ] T026 [US7] If a configuration change is recommended, measure its capacity curve by raising `QUOTE_VOLUME_HEADROOM_MULT` on a subclass — the comparable NNNC configuration retained only 57% of return at a 4x book against production's 74% (FR-029)

---

## Phase 7: Polish & Cross-Cutting

- [ ] T027 Verify `git status --short Framework/` is clean — no shared mechanism modified, and the live dry run that imports `BaseStrategy` is unaffected (FR-026, SC-009)
- [ ] T028 [P] Verify every oracle figure in the feature's documents is labelled a lookahead ceiling and that no oracle configuration has been promoted, AND audit every verdict recorded by this feature for a classifier metric used as an acceptance input — SC-008 has both halves and only the first is self-evident (FR-019, FR-024, SC-008)
- [ ] T029 [P] Commit the arm records under `regime/arms/` — with the findings documents they are the durable evidence, and each carries the `returncode` that makes it admissible (FR-027)
- [ ] T030 Run the `quickstart.md` scenarios and confirm SC-001..SC-010, recording the verification
- [ ] T031 [P] Record anything that generalises beyond NNMT in memory and the relevant `regime/` documents

---

## Dependencies

    Phase 1 (T001-T005)  — the control. T001/T002 before T003; T003 before T004.
        ↓
    Phase 2 (T006-T009)  — HEADROOM. T009 is a HARD GATE.
        ↓                   T006 and T007 are [P] with each other (different
        ↓                   files); both precede T008; T008 precedes T009 and
        ↓                   also SETS THE POWER THRESHOLD every later phase uses.
        ├──────────────→ Phase 3 (T010-T012) — guard attribution
        ↓
    Phase 4 (T013-T018)  — the inherited changes. T013 before T014 (it decides
        ↓                   whether the pinning is load-bearing). T014→T015→T016.
        ↓                   T017 is INDEPENDENT of T014-T016 and needs no
        ↓                   retraining, so it may run first if convenient.
        ↓
    Phase 5 (T019-T024)  — CONTINGENT on Phase 2 finding headroom.
        ↓                   T019→T020; T021→T022; T023 requires T021's chosen
        ↓                   value; T024 requires T023.
        ↓
    Phase 6 (T025-T026)  — needs Phases 2-5
        ↓
    Phase 7 (T027-T031)  — polish

**The only hard gate is T009, and it gates PHASE 5 ONLY.** This is FR-004's
definition of re-scoping, and it is stated identically in the Phase 2 gate
callout, in T009, and in plan.md: Phases 3 and 4 run even if headroom is small —
Phase 3 EXPLAINS why it is small, and Phase 4 fixes a live inconsistency that
stands regardless. Phase 5 is genuinely contingent and is re-scoped, not
executed, on a negative result.

---

## Priority vs execution order

Execution order differs from spec priority, deliberately:

- **US3 (P2) runs before US2 (P1).** US2 requires retraining and its
  interpretation depends on knowing whether the guards or the model select the
  trades. Running the free diagnostic first means the expensive arms are aimed.
- **US2 is P1 despite being fourth** because it is a DEFECT FIX — an
  inconsistency introduced by work done for another family — not a research
  question. It must not be dropped if the feature is cut short.
- **US1 (P1) is first and is a kill switch**, not merely the highest priority.

---

## Implementation Strategy

**MVP = Phases 1 and 2.** A valid control plus the headroom measurement is a
complete, valuable deliverable: it says whether anything else on this family can
pay. On NNNC the equivalent measurement was the most consequential of the entire
programme.

**Cheapest-first.** Phase 2 and Phase 3 are backtests only, and so is Phase 1 in
the expected case. The first UNCONDITIONAL retrain is T015, reached only after
the diagnosis has justified it; T003 retrains only in the contingency where the
control's model does not match its label, which all present evidence says it
does.

**A negative headroom result is the ANSWER.** T009 and FR-004 both make this
explicit so that a null is reported as a finding rather than treated as a failed
feature. On such a result the work does NOT stop: Phases 3, 4, 6 and 7 still
run, and Phase 5 is re-scoped. Only Phase 5 is contingent.
