---
description: "Task list for feature 003 — tail-targeted breakout labelling"
---

# Tasks: Tail-Targeted Breakout Labelling

**Input**: Design documents from `specs/003-breakout-tail-labelling/`
**Prerequisites**: plan.md, spec.md, research.md (R1–R9 resolved), data-model.md, contracts/study-cli.md, quickstart.md

**Tests**: No unit-test tasks. This is a measurement study; correctness is
established by REPRODUCTION (T007's hash gate) and by the per-arm liveness and
configuration checks, not by a test suite. This is a deliberate, recorded choice.

**Organization**: Grouped by user story. Note the deliberate deviation below.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelizable (different files, no dependency on incomplete work)
- **[Story]**: US1–US5 from spec.md
- All paths are relative to `user_data/strategies/` unless stated

---

## ⚠️ Priority vs execution order — read first

Spec priorities are US1/US2/US3 = P1, US4 = P2, US5 = P3. **Execution order is
deliberately different**, and the phases below follow execution order:

- **US4 (screening, P2) runs FIRST** because FR-002 requires candidates to be
  excluded before a retrain is spent on them. It is P2 by *value* (it saves cost)
  but a prerequisite by *sequence*.
- **US3 (liveness, P1) is cross-cutting.** Its harness is built in Foundational
  and its check is applied inside every evaluation task in US1, US2 and US5. It
  has no standalone phase because it is a property of every comparison.
- **US1 before US2** because US1's candidate needs no exit change and therefore
  isolates the label effect (FR-004); US2 adds the hold window on top.

## ⚠️ Already settled by research.md — DO NOT redo

R1 (distribution shape), R2 (current-label economics), R3 (three guard_metric
cells), R4 (hold-window conflict + the per-strategy-override decision), R5 (path
reachability), R6 (gate absorption base rate), R7 (metric untrustworthiness), R8
(control-config decision), R9 (power). Tasks below CONSUME these; none re-derives
them. FR-001 is therefore **partly pre-satisfied** — only two candidates remain
unscreened.

## ⚠️ Operational hazards — encoded in the tasks that can trigger them

1. `test_strat.sh -f` **DELETES the model before retraining**. Back up first.
   This destroyed a model with no backup earlier in this study.
2. `TaskStop` does **not** kill the process tree. Kill by pattern
   (`pkill -9 -f <name>`), then verify with `ps`.
3. A strategy without its own `<Strategy>.json` **silently runs class defaults**.
   This produced a spurious 12pp result on a prior comparison.
4. A live dry run imports `Framework/BaseStrategy.py`. No experimental edit may
   be left there (FR-010).
5. **Run everything SEQUENTIALLY.** Two concurrent jobs produced load 60 on 14
   cores and ran slower than serialising.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: the candidate registry and the screening harness. No training.

- [X] T001 Create `regime/tail/` package directory with `__init__.py` for this feature's study code, mirroring the `regime/exit/` layout used by feature 002
- [X] T002 Create the candidate registry in `regime/tail/candidates.py` implementing the Candidate entity from data-model.md — fields `id`, `horizon`, `min_gain`, `basis`, `mae_bound`, `requires_hold_change`, `motivating_measurement`, `excluded_reason` — with a `validate()` that raises when `motivating_measurement` is empty (FR-008)
- [X] T003 Populate `regime/tail/candidates.py` with the five screened cells from research.md R3 as pre-recorded entries, marking `H=48/0.03` and `H=96/0.05` EXCLUDED with reason "below the ~5.5% density floor" so they are never trained (FR-002), and each carrying its motivating measurement (FR-008)
- [X] T004 Add the two UNSCREENED candidates to `regime/tail/candidates.py` — the MAE-constrained variant (FR-006) and the `bb_position` basis variant (FR-007) — with `excluded_reason` empty and screening pending. Record that any `bb_position` threshold MUST be expressed on the **[−1, 1]** scale, not [0, 1]
- [X] T005 Implement `regime/screen_candidates.py` per `contracts/study-cli.md` §1 — emits one ScreeningRecord per candidate with all six measures, sets `excluded_reason` below the density floor, sets `requires_hold_change` when time-to-peak exceeds the 12h forced-exit window, loads no model and trains nothing (FR-001, FR-002, FR-003), exit code 0 even when candidates are excluded

**Checkpoint**: registry and screening harness exist; no measurement taken yet.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: a control every arm can be compared against, and the comparison
harness that decides whether a comparison is even valid.

**⚠️ CRITICAL**: T007 is a HARD GATE. Comparing arms against an unverified
control would invalidate every downstream number in this feature.

- [X] T006 Write `NNNC/NNNC_BreakoutOpen.json` pinning the host's CURRENT effective configuration per research.md R8 — buy space from the class (`entry_enable_guards` true, dip guards neutralised at 5.0, `prediction_threshold` 0.50) and sell space as currently inherited from `BaseStrategy` including `cexit_take_profit` **0.032**. Set `strategy_name` to `NNNC_BreakoutOpen`. Do NOT adopt NNNC's tuned 0.024 — that is an unmeasured change and is explicitly out of scope (Constitution VIII, FR-016)
- [X] T007 **HARD GATE** — re-baseline the control: run `regime/run_arm.py --strategy NNNC_BreakoutOpen --window W1|W2|W3 --tag ctl003` and verify the resulting `trades_hash` for EACH window is byte-identical to the corresponding `regime/arms/brkfix_W{1,2,3}.json`. Expect W1 97tr −0.74%, W2 119tr +0.34%, W3 77tr +1.71%. **`exit_config_hash` WILL CHANGE and that is correct** — it is computed from the params file's sell/roi/stoploss/trailing sections, and the host previously had NO params file, so `bb3239ab7e52df1b` is the hash of an EMPTY exit config. Pinning the configuration necessarily produces a new value (`203297442969c7da`); expecting the old one would be expecting the pin not to happen. The gate is **trade-hash reproduction only**. Record the new hash as the value every subsequent arm must match. Confirm the arm record shows realistic execution constraints active, i.e. `volume_filter_active` true (FR-018). **If any hash differs, STOP** — the params file changed behaviour and every historical breakout comparison is invalidated; resolve before any further task (FR-016, FR-018, SC-007)
- [X] T008 [P] Implement the comparison harness in `regime/tail/compare.py` rendering the five-part report from `contracts/study-cli.md` §4 in order: exit-config match, per-window trade-hash comparison, W1 marked OUT-OF-SAMPLE, W2/W3 marked IN-SAMPLE and excluded from the verdict, then the verdict (FR-013, FR-015, FR-016, FR-017)
- [X] T009 [P] In `regime/tail/compare.py`, implement the verdict rule so that **W1 `trades_hash` identical to the control yields `inconclusive_absorbed`, which OUTRANKS every other classification** regardless of returns, and is never reported as refutation (FR-015, SC-003). Encode the R9 power threshold so a W1 delta below ~1pp on ~97 trades renders as "indistinguishable", not "improved" (FR-017)
- [X] T010 [P] In `regime/tail/compare.py`, assert that classifier metrics are never an input to the verdict — they may be printed as context only (FR-014). Cite R7: EV pointed the wrong way by 43pp, precision tripled while returns halved, MCC fell while returns rose

**Checkpoint**: a verified control exists and every comparison is guaranteed to
report liveness and in-sample status before any number is read.

---

## Phase 3: User Story 4 — Screen cheaply before spending retrains (Priority: P2, runs FIRST)

**Goal**: every candidate admitted or excluded on recorded numbers, with zero
training spent on a candidate that cannot work.

**Independent Test**: run `regime/screen_candidates.py` and confirm every
candidate has a complete ScreeningRecord and that excluded candidates carry a
reason — with no model loaded and nothing trained.

- [X] T011 [US4] Screen the MAE-constrained candidate at tail settings using `regime/tail_path.py`'s method, recording density, median gain, stop-clearing share, reward:risk, time-to-peak and stopped-before-peak into its ScreeningRecord (FR-001, FR-006)
- [X] T012 [P] [US4] Screen the `bb_position`-basis candidate (method 26) at tail settings the same way, using the **[−1, 1]** scale for the position threshold (FR-001, FR-007)
- [X] T013 [US4] Run `regime/screen_candidates.py` over the full registry and write the consolidated screening table to `regime/tail/README.md`, confirming the two below-floor cells remain excluded and flagging which admitted candidates have `requires_hold_change` true. Assert explicitly that at least one ADMITTED candidate reaches reward:risk above 1.0 while staying above the density floor; if none does, the feature's premise fails at screening and that is itself the result (FR-002, SC-001, SC-002)
- [X] T014 [US4] Verify FR-003 held: confirm no model file was read and no training occurred during screening (check `saved_data/` mtimes are unchanged)

**Checkpoint**: the admitted candidate set is fixed and justified. SC-001 met.

---

## Phase 4: User Story 1 — Isolate the label (Priority: P1) 🎯 MVP

**Goal**: measure the label effect alone, with no exit-side change to confound
it.

**Independent Test**: the arm trains, its `exit_config_hash` matches the control,
and its verdict is one of improved / unchanged / worsened /
inconclusive_absorbed with the trade-hash comparison recorded.

- [X] T015 [US1] Create `NNNC/NNNC_Brk_L1.py` subclassing `NNNC_BreakoutOpen` with `MIN_BUY_GAIN_THRESHOLD` and `MIN_SELL_LOSS_THRESHOLD` at the admitted no-exit-change setting and `HORIZON` unchanged, with a docstring citing its ScreeningRecord (FR-004, FR-008)
- [X] T016 [US1] Create `NNNC/NNNC_Brk_L1.json` as a copy of `NNNC/NNNC_BreakoutOpen.json` with `strategy_name` updated, so the arm runs the control's configuration rather than class defaults (FR-016)
- [X] T017 [US1] Back up `saved_data/NNNC_BreakoutOpen/` before any training in this phase — `test_strat.sh -f` deletes before retraining and this destroyed an un-backed-up model earlier in this study (hazard 1)
- [X] T018 [US1] Train the arm: `cd <repo root> && PATH=.venv/bin:$PATH zsh user_data/strategies/scripts/test_strat.sh -f -t 20220101-20250831 NNNC NNNC_Brk_L1`. Run SEQUENTIALLY. Record the label counts from the training log and confirm they match the candidate's screened density (FR-012)
- [X] T019 [US1] Evaluate on all three windows: `regime/run_arm.py --strategy NNNC_Brk_L1 --window W1|W2|W3 --tag l1`
- [X] T020 [US1] Compare against the control with `regime/tail/compare.py`: verify `exit_config_hash` matches, compare `trades_hash` per window, and classify. **If W1 trades are identical to the control, record `inconclusive_absorbed`** — the gate absorbed the change and the returns are not evidence either way (FR-015, SC-003)
- [X] T021 [US1] Record the W1 result as the verdict with trade counts, and W2/W3 as IN-SAMPLE context excluded from acceptance. State explicitly whether the W1 delta clears the R9 power threshold (FR-013, FR-017, SC-004)

**Checkpoint**: the label effect is measured in isolation. If this is
conclusively negative, FR-022 permits stopping here with a recorded negative
result — see Implementation Strategy.

---

## Phase 5: User Story 2 — Make the hold window consistent with the label (Priority: P1)

**Goal**: test the tail candidate whose payoff peaks beyond the forced-exit
window, with the label and hold-window contributions separated.

**Independent Test**: three arms exist — hold-window change alone, label alone
(from US1), and both — so each contribution is attributable; and no shared
framework file is left modified.

**Entry condition**: US1 not conclusively negative.

**Deliberately NOT built**: an arm carrying the tail label WITHOUT the hold-window
override. research.md R4 establishes that configuration is knowably compromised —
its payoff peaks at 17.0h against a 12h forced exit, so it would be harvested
early by construction and would produce a false negative rather than a test. The
consequence is accepted and stated: if L2 improves, this feature cannot say
whether the tail label alone would have sufficed.

- [X] T022 [US2] Implement the hold-window extension as a **per-strategy override** of the method containing the forced-exit rule, on an arm subclass in `NNNC/`. Do NOT edit `Framework/BaseStrategy.py` — a live dry run imports it, and an override needs no revert at all (research.md R4, FR-009, FR-010)
- [X] T023 [US2] Create `NNNC/NNNC_Brk_H1.py` + `.json` applying ONLY the hold-window override on the control's existing label, so the hold-window contribution is measurable independently of the label (FR-011)
- [X] T024 [US2] Evaluate the hold-window-only arm on all three windows with tag `h1` and compare. Note its `exit_config_hash` is EXPECTED to differ from the control here — this is the one permitted exception, because the hold window is the parameter under test (FR-016)
- [X] T025 [US2] Create `NNNC/NNNC_Brk_L2.py` + `.json` applying BOTH the tail label (H and threshold at the admitted best-economics cell) AND the hold-window override (FR-005)
- [X] T026 [US2] Train `NNNC_Brk_L2` sequentially on the pinned window, having backed up any model it would overwrite (hazard 1)
- [X] T027 [US2] Evaluate `NNNC_Brk_L2` on all three windows with tag `l2` and compare against the control, applying the full per-arm protocol including the liveness gate (FR-013, FR-015, FR-017)
- [X] T028 [US2] Report the separated contributions using arms that share a label: **hold window alone = H1 − control** (T024) and **tail label given the hold window = L2 − H1** (T027 − T024). Do NOT subtract US1's T021 result here — US1's arm carries the no-exit-change label, NOT the tail label, so a three-way decomposition against it would be arithmetically invalid. US1's result stands as a separate answer about a different label (FR-011, SC-005)
- [X] T029 [US2] Verify no shared framework file is modified: `git status Framework/` must be clean (FR-010, SC-006)

**Checkpoint**: the package is measured and its two components are attributable.

---

## Phase 6: User Story 5 — Re-tune the risk side (Priority: P3, CONTINGENT)

**Goal**: re-examine settings that were fitted against the OLD label, but only if
a new label is adopted.

**Entry condition**: a label from US1 or US2 is adopted. If none is, this phase
does not run.

**Independent Test**: any proposed setting is accepted only on W1 return against
that label's own control, never the historical one.

- [ ] T030 [US5] Re-sweep the ATR stop band against the adopted label's control using subclasses that override `get_model_path()` to reuse the adopted arm's weights, so no retrain is needed. `--stoploss` is INERT for NN strategies; the operative knobs are `atr_stoploss_multiplier`/`_floor`/`_cap` (contracts/study-cli.md §2)
- [ ] T031 [US5] Re-sweep `prediction_threshold` against the adopted label's control, noting the prior sweep found win rate FROZEN at 58.2% across a 3.5× range at the OLD label — that result does not transfer (FR-019 permits this only at a new label)
- [ ] T032 [US5] Report both sweeps against the adopted label's control with the full per-arm protocol, accepting nothing on a classifier metric (FR-014)

**Checkpoint**: risk side re-tuned or explicitly declined.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T033 Write the consolidated findings to `regime/tail/README.md` — screening table, every arm's verdict including absorbed cases, separated contributions, and the explicit answer to whether label mis-specification was material. **A negative outcome is a valid deliverable and must be recorded as such** (FR-022, SC-008)
- [ ] T034 [P] Delete throwaway arm classes, their `.json` files and their `saved_data/` directories. **Leave `NNNC/NNNC_BreakoutOpen.json` in place** — it pins the control (quickstart.md Cleanup)
- [ ] T035 [P] Commit the arm records under `regime/arms/` — they are the only durable evidence once the throwaway models are deleted
- [ ] T036 Run the `quickstart.md` validation end to end and confirm every success criterion SC-001…SC-008 is either met or explicitly recorded as not applicable. Verify no arm introduced per-pair special-casing — every candidate is defined structurally over all traded pairs (FR-021) — and that realistic execution constraints were active in every recorded arm (FR-018)
- [ ] T037 If any finding generalises beyond this host, record it in memory and the relevant `docs/` file — but promote NOTHING to `Framework/` defaults without measuring on a second family (FR-020)

---

## Dependencies & Execution Order

### Phase dependencies

- **Phase 1 (Setup)** — no dependencies.
- **Phase 2 (Foundational)** — depends on Phase 1. **T007 BLOCKS EVERYTHING
  DOWNSTREAM.** No arm may be compared before the control reproduces its hashes.
- **Phase 3 (US4)** — depends on Phase 1; can overlap Phase 2 since screening
  needs no control. Must complete before Phase 4 (FR-002).
- **Phase 4 (US1)** — depends on Phase 2 (control) and Phase 3 (admitted set).
- **Phase 5 (US2)** — depends on Phase 4's outcome not being conclusively
  negative.
- **Phase 6 (US5)** — contingent on a label being adopted.
- **Phase 7 (Polish)** — depends on whichever phases ran.

### Critical path

```text
T001-T005 (setup) → T006-T007 (control, HARD GATE) → T011-T014 (screen)
   → T015-T021 (US1, MVP) → T022-T029 (US2 package) → T030-T032 (US5)
   → T033-T037 (polish)
```

### Parallel opportunities

Genuinely limited, and that is deliberate — **all training and backtesting is
SEQUENTIAL** (hazard 5).

- T008, T009, T010 — different concerns in the same new file; write together.
- T012 — screening the second basis is independent of T011.
- T034, T035 — cleanup and record-commit are independent.

**Never parallel**: any two tasks that train or backtest.

---

## Implementation Strategy

### MVP scope

**Phases 1, 2 and 3, then Phase 4 (US1).** That yields a verified control plus one
trained arm measuring the label effect in isolation — enough to answer the
feature's central question for the cheapest candidate.

### Stopping early is a legitimate outcome

If T020 returns `inconclusive_absorbed`, the gate prevented the test and the
candidate is NOT refuted — Phase 5 should still run, because the package changes
the exit machinery too and may reach the trades where the label alone did not.

If T021 shows a clear W1 regression that clears the power threshold, FR-022
permits concluding at Phase 4: "the label was not the problem" is a valid and
useful result, and six of six comparable prior interventions changed nothing.

### Cost

Screening: minutes. Control re-baseline: 3 backtests, no retrain. Each arm:
one retrain (8–45 min) plus 3 backtests. MVP is roughly one retrain; the full
run through Phase 5 is three.

---

## Notes

- `[P]` = different files, no dependency on incomplete work.
- Every evaluation task applies the per-arm protocol from plan.md; the harness
  built in T008–T010 enforces it rather than relying on discipline.
- Requirement coverage: FR-001…FR-022 and SC-001…SC-008 each map to at least one
  task, verified mechanically. FR-001 is partly pre-satisfied by research.md
  R1–R5 and completed by T011–T013.
- **US3 has no tasks of its own by design.** It is the liveness property applied
  inside every evaluation (T020, T024, T027, T032); its enforcement lives in the
  harness built by T008–T010. A story with no phase is unusual and is called out
  here so it does not read as an omission.
