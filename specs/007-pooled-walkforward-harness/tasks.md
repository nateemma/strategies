---
description: "Task list for feature 007 — Pooled Walk-Forward Evaluation Harness"
---

# Tasks: Pooled Walk-Forward Evaluation Harness

**Input**: spec.md (24 FRs, 8 SCs, 4 user stories), plan.md, research.md (R1-R12), data-model.md,
contracts/harness-cli.md, quickstart.md (S1-S8)

## Format: `[ID] [P?] [Story] Description`

`[P]` = parallelisable **as authoring work only**. All *execution* is sequential (FR-016, R12).

## Path Conventions

Repo root = the directory containing `user_data/`. Strategy repo root =
`user_data/strategies`. All commands run from the repo root with
`PYTHONPATH=user_data/strategies`.

---

## Standing conventions — apply to EVERY execution task

These are the cross-cutting FRs. They are cited here ONCE and are binding on every task below;
T029 verifies them explicitly at the end. **Coverage of exactly these has been asserted
prematurely in four prior features of this project (003, 004, 005 twice) and the repeat offender
is always this block.**

- **FR-015 / SC-008**: no entry gate, exit parameter, label or model configuration may change.
  This feature alters only how results are measured.
- **FR-016 / R12**: strictly sequential. Two concurrent jobs put load 60 on 14 cores and ran
  *slower*. To kill work use `pkill -9 -f 'freqtrade backtesting'` — **never** bare `freqtrade`,
  which matches the live dry run.
- **FR-017**: state the compute budget before running; report actual elapsed against it.
- **FR-018**: `regime/run_arm.py`'s record schema is frozen — 480+ stored records depend on it.
- **FR-019**: `Framework/` is untouchable; a live dry run imports `BaseStrategy`.
- **FR-020**: an incomplete fold set is reported **PARTIAL**; never present a partial floor as the
  full plan's.
- **FR-021**: era subsets must stay recomputable, so Principle II remains checkable.
- **FR-022 / SC-005**: same inputs ⇒ same floor and same verdicts (`BOOTSTRAP_SEED = 0`).

Per-run protocol, every execution task:
1. Verify the **model artefact** exists (`saved_data/<CLS>/*.safetensors`) — **never** trust the
   exit code; `test_strat.sh` returns 0 when training crashed (R5, FR-004).
2. Confirm `run_arm.py` returncode 0. A failed or stale record is not evidence.
3. Zero-trade fold ⇒ **zero-trade**, never zero-return (FR-012).
4. Identical `trades_hash` ⇒ **inconclusive**, never refuted.

---

## Phase 1: Setup

- [ ] T001 Create the package skeleton `user_data/strategies/regime/wf/` with `__init__.py` and
      `folds/`, per plan.md's structure decision
- [ ] T002 Add `NNNC/_wf/` to `user_data/strategies/.gitignore` — generated fold classes and
      `.json` files are build artefacts, not source (R4)
- [ ] T003 [P] Record the fixed baseline constants in `regime/wf/__init__.py`:
      `BASELINE_HALF_WIDTH_PP = 7.633`, `BASELINE_N_TRADES = 99`, `BASELINE_PAIRED_PP = 6.1`,
      `BOOTSTRAP_SEED = 0`. **R1 settled these — reuse, never recompute**

## Phase 2: Foundational (BLOCKING for all user stories)

- [ ] T004 Implement fold-plan construction in `regime/wf/foldplan.py`: anchored expanding
      training, `--min-train-days 730`, `--test-days 90`, data 2021-01-01..2026-09-07, first test
      2023-01-01 (FR-001; **R3 settled the geometry — do not re-derive or re-open rolling**)
- [ ] T005 Implement plan validation in `regime/wf/foldplan.py`: refuse any fold whose test range
      intersects its own training range, naming the fold; also enforce unique `strategy_class` and
      unique `model_path` across folds (FR-002, SC-003)
- [ ] T006a **Snapshot the params file ONCE, at plan time**, into
      `regime/wf/folds/params_<run_id>.json`, and record its SHA in the run manifest.
      `NNNC/NNNC_MLX.json` is **untracked working state that the user edits**; reading it per fold
      would let a mid-run edit give later folds a different entry gate, making the pooled set
      silently heterogeneous with no error (FR-015, FR-022)
- [ ] T006 Implement fold-class generation in `regime/wf/run_folds.py`: emit `NNNC_WF_f{NN}`
      subclassing `NNNC_MLX` into `NNNC/_wf/`, **each with its own `.json` generated from the
      T006a SNAPSHOT (never re-read from `NNNC/NNNC_MLX.json`) with `strategy_name` rewritten to
      the class name** (FR-003; **R4** — without the `.json` the fold silently inherits
      `NNNCStrategy`'s all-guards-open params, and a mismatched `strategy_name` aborts with
      `Invalid parameter file provided`)
- [ ] T007 Implement the artefact check in `regime/wf/run_folds.py` — a fold is `trained` only if
      `saved_data/<CLS>/*.safetensors` exists (FR-004; **R5** — the exit code lies)
- [ ] T008 Implement per-fold evaluation in `regime/wf/run_folds.py` by invoking `run_arm.py` with
      the fold's `--timerange` and a fold-specific `--tag` (FR-005; **R6** — reuse, do not touch
      its schema or add windows to `regimedata.py`)
- [ ] T009 Make `run_folds.py` resumable and add `--dry-run` (prints the fold sequence and
      generated paths, trains nothing) — the cheap collision check before a 3-hour run

**Checkpoint**: the harness can plan, validate, generate identities and run a fold. No compute spent.

---

## Phase 3: User Story 1 — Pool many genuinely out-of-sample folds (P1) 🎯 MVP

**Goal**: a validated fold plan executes, every test fold strictly out-of-sample, trades pooled
with fold identity retained.

**Independent test**: quickstart S1-S4 — plan validates, a bad fold is refused, identities are
unique, and a full run pools ≥350 trades.

### Phase 3a — no compute

- [ ] T010 [US1] Build and validate the 15-fold NNNC plan; confirm ~15 folds, first test
      2023-01-01, `VALIDATED` (quickstart S1, FR-001)
- [ ] T011 [US1] **Negative control**: hand-edit one fold's `test_start` into its training range,
      re-validate, and confirm it is **REFUSED** by `fold_id` with a non-zero exit. A validator
      that silently accepts this invalidates every downstream number (quickstart S1, FR-002)
- [ ] T012 [US1] Run `--dry-run` and confirm every fold has a distinct class **and** a distinct
      model path, and that each generated `.json` carries the production entry gate rather than
      class defaults (quickstart S2, FR-003, **R4**)

### Phase 3b — first compute

- [ ] T013 [US1] Run fold `f00` end-to-end; confirm training, artefact check, `run_arm.py`
      invocation with the fold timerange, and an arm record carrying `result_file`
      (quickstart S3, FR-004, FR-005)
- [ ] T014 [US1] **Verify f00's test range is strictly later than its training range.** If the
      model directory was empty yet the backtest "succeeded", the model trained **inside the
      backtest** and the fold is void (**R4** — the single most dangerous failure mode here)

### Phase 3c — ⚠️ BLOCKING: the full run

- [ ] T015 [US1] **BLOCKING — ~3.1 h, sequential, NNNC only, seed 1, 15 folds.** Nothing after
      this phase can run until it completes. Record actual elapsed against the 3.1 h budget
      (FR-016, FR-017; **R11** — do NOT add NNMT or a second seed here)
- [ ] T016 [US1] Implement pooling in `regime/wf/pool.py`: read each fold's arm record, extract
      trades via **`compare.py`'s existing `_trades()` accessor** keyed off `result_file`, tag each
      trade with `fold_id`, write the pooled set (FR-006, FR-013; **R7** — archives are the only
      per-trade source)
- [ ] T016a [US1] **Smoke-test the coupling to `compare.py._trades()`**: assert it returns rows for
      a known archive before the pooling run. It is a PRIVATE symbol in a module this feature must
      not modify, so this assertion is the only thing that will fail loudly if it changes
      (FR-013; decision: keep `compare.py` untouched rather than promote the symbol, since it is
      the statistic of record and was only just rewritten)
- [ ] T017 [US1] Handle zero-trade folds as `zero_trade` (completed, excluded from expectancy) and
      mark the pooled output `partial` if any planned fold is missing (FR-012, FR-020)

**Checkpoint**: a pooled trade set exists with fold identity retained.

---

## Phase 4: User Story 2 — Report the achieved noise floor (P1) — THE FEATURE'S ANSWER

**Goal**: the measured floor, against the 7.633pp baseline.

**Independent test**: quickstart S4/S5 — the report states a floor computed from pooled per-trade
P&L, and reruns reproduce it.

> **This phase produces the feature's ANSWER, not its success condition.** A floor that fails to
> reach 4.0pp, or that fails to fall as 1/sqrt(n) predicts, is a **RESULT to report** (R9) — the
> most valuable one available, because it would mean the power ceiling is lower than data volume
> implies. It is never grounds to retry with different geometry until the number looks better.

- [ ] T018 [US2] Implement the bootstrap floor in `regime/wf/power.py`: pooled 95% CI and
      half-width, seeded (FR-007, FR-022). **`power.py` computes the floor of ONE set;
      `compare.py` computes the delta between TWO — do not bootstrap the same quantity twice (R8)**
- [ ] T019 [US2] Report the achieved floor against `BASELINE_HALF_WIDTH_PP = 7.633`, the
      improvement ratio, and the **minimum resolvable effect size** stated explicitly (FR-008,
      SC-004)
- [ ] T020 [US2] Compute and report the 1/sqrt(n) prediction, any shortfall, **and the inter-fold
      return correlation explaining it** (FR-009, SC-007; **R9** — report, never hide)
- [ ] T021 [US2] Report the per-fold breakdown: trade count and return per fold (FR-010)
- [ ] T022 [US2] Satisfy **SC-001 (mandatory)** by reporting the measured floor and improvement
      ratio against 7.633pp; then record whether **SC-001a (target, ≤ 4.0pp)** and SC-002
      (≥ 350 pooled trades) were met. The estimate is 3.98pp — a 0.02pp margin — so **missing
      SC-001a is a reportable RESULT, not a failure**, and never grounds to retune the geometry
      (quickstart S4, research R9)
- [ ] T023 [US2] Confirm reproducibility: rerun `power.py` on the same pooled file and require a
      byte-identical floor (quickstart S5, FR-022, SC-005)
- [ ] T024 [US2] Record per-fold val metrics and per-trade expectancy against training length —
      **the confound detector**: a monotone relationship means early short-trained folds are
      weaker and the instrument is measuring itself (plan.md risk, FR-010)

**Checkpoint**: the achieved floor is known and stated.

---

## Phase 5: User Story 3 — Feed the existing comparison (P2)

**Goal**: arm-vs-control verdicts come from the pooled set.

**Independent test**: a pooled comparison cites the pooled trade count and CI.

- [ ] T025 [US3] Route pooled arm-vs-control through `regime/tail/compare.py` unchanged; confirm
      the verdict reports the pooled CI and p (FR-013; **R8** — feed, do not fork)
- [ ] T026 [US3] Confirm an arm differing in fewer than `MIN_DIFF_TRADES=20` returns
      `inconclusive_underpowered`, not a win (FR-013)

---

## Phase 6: Seed and era subsets

- [ ] T027 [US1] Run seed 7 on folds `f00,f04,f08,f12` (~0.8 h); for each, report either differing
      trades or **identical** trades — identical is ONE measurement, not two (FR-011; **R11**)
- [ ] T028 [US2] Recompute the floor over a recent-era subset via `--folds-subset` and confirm era
      sensitivity is inspectable (FR-021, quickstart S7)

---

## Phase 7: Polish & Cross-Cutting

- [ ] T029 **Verify the standing-conventions block, one named check per requirement.** A single
      blanket sweep is what let this slip through in 003, 004 and 005 (twice), so each line below
      must be evidenced individually and recorded in the conclusion doc:
      - FR-015 / SC-008: `git diff --stat NNNC/NNNC_MLX.json Framework/` is EMPTY, and the run
        manifest's snapshot hash equals the file's hash at run start (no gate moved mid-run)
      - FR-016: no two `freqtrade backtesting` PIDs ever overlapped — confirm from the run log's
        per-fold start/end timestamps, which must be strictly non-overlapping
      - FR-017: report actual elapsed minutes against the 3.1 h / ~12.5 min-per-training budget
      - FR-018: `git status --short regime/run_arm.py` EMPTY, and one pre-existing arm record still
        parses against the current reader
      - FR-019: `git status --short Framework/` EMPTY
      - FR-020: assert a deliberately truncated fold set yields `status: partial` in the report
      - FR-021: `--folds-subset recent-third` returns a different, smaller n than the full set
      - FR-022 / SC-005: two consecutive `power.py` runs on one pooled file return byte-identical
        floors
- [ ] T030 Write `regime/WALKFORWARD_POWER.md`: the achieved floor, the baseline, the improvement
      ratio, the per-fold breakdown, the sqrt(n) shortfall and inter-fold correlation, and — **not
      optional** — the statement that **1-2pp parameter effects remain unresolvable** and that the
      win is the 3-7pp band (**R2**; without this the harness will be over-read)
- [ ] T031 Record a memory note with the achieved floor and what it does and does not make
      measurable, superseding the "stop tuning parameters" guidance with the new bound
- [ ] T032 Update `AGENT_GUIDE.md`'s noise-floor section: replace the "PENDING — the harness does
      not exist yet" paragraph with the achieved floor and the new decision rule
- [ ] T033 Commit: stage only this feature's files (`regime/wf/`, the spec directory, the doc, the
      `.gitignore` line). **Never `git add -A`** — files are edited concurrently

---

## Deferred (NOT in this run — do not schedule)

- Phase E pair expansion (US4, FR-014, SC-006, ~3.1 h): **R10** — an open question, and the floor
  must be demonstrated at the current universe first.
- Phase F NNMT (~3.1 h): **R11** — NNNC first; NNMT fold classes must have **no** `.json`, matching
  `NNMT_MLX`.
- Full 2-seed coverage over all 15 folds (6.3 h): not justified before the floor is shown.

---

## Dependencies & Execution Order

- **Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5/6 → Phase 7**, strictly.
- **T015 is BLOCKING**: T016-T028 all require the full fold set. Its ~3.1 h is the feature's
  critical path.
- T011 (negative control) MUST pass before T015 — a validator that accepts an overlapping fold
  invalidates the entire run.
- T012 (dry-run identity check) MUST pass before T015 — catching a class collision after 3 hours
  wastes the run.
- T013/T014 (one fold) MUST pass before T015 — prove the pipeline on one fold before fifteen.
- T018 before T019-T023; T016 before T018.

### Parallel opportunities

Authoring only: T003 is independent; T004/T005 and T006/T007/T008 may be written together within
their files. **No execution task may run in parallel with another** (FR-016).

## Implementation Strategy

**MVP = Phases 1-4** (T001-T024): the harness plus the measured floor. That alone answers whether
pooling delivers power, which is the feature's entire purpose. Phases 5-7 deliver it into the
decision path and write it down.

**Stop conditions**: if T011 fails, fix the validator before anything else. If T014 shows a fold
trained inside its backtest, stop — every number after it is void. If T022 shows the floor missing
SC-001, **report it** and proceed to T030; do not tune the geometry to chase the number.
