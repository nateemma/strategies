---

description: "Task list for labelfree-es-policy-search (brief 05)"
---

# Tasks: labelfree-es-policy-search

**Input**: Design documents from `specs/013-labelfree-es-policy-search/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/es-cli.md, quickstart.md

**Tests**: INCLUDED. The spec's acceptance scenarios are measurements ("reported as a number
rather than asserted"), and Principle IV plus this study's history of silently-inert parameters
make wiring checks insufficient on their own.

**Organization**: grouped by the five user stories from spec.md.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependencies)
- **[Story]**: US1–US5 from spec.md
- All commands use `freqtrade/.venv/bin/python` — `~/.pyenv/shims/python3` is a different 3.12.13
  that has `cma` but **not** pandas or scipy.
- All runs are sequential (FR-021). Stage only files this feature touches (FR-022).

---

## Phase 1: Setup

**Purpose**: package skeleton and the interpreter guard.

- [ ] T001 Create `research/labelfree/es/__init__.py` with the package docstring naming brief 05, the nine search dimensions and the blocking calibration gate
- [ ] T002 Add an import-time interpreter check in `research/labelfree/es/__init__.py` that raises with a clear message if `pandas`, `scipy` or `cma` is missing, naming `freqtrade/.venv/bin/python` as the interpreter to use
- [ ] T003 [P] Record `cma==4.5.0` in `research/labelfree/es/REQUIREMENTS.txt` with the install date and the venv path, so a later reader can reproduce the environment

**Checkpoint**: `from research.labelfree.es import *` succeeds under the venv and fails loudly elsewhere.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: the space, the trial record and the objective. No user story can start until these exist.

**⚠️ CRITICAL**: T005 (write-before-score) is the invariant the whole leakage audit rests on.

- [ ] T004 Implement the nine-dimension `Space` in `research/labelfree/es/space.py` — bounds and incumbent values per data-model.md §1, reading the incumbent from `NNNC/NNNC_MLX.json` and `Framework/BaseStrategy.py` rather than hard-coding them
- [ ] T005 Implement `encode`/`decode` in `research/labelfree/es/space.py` mapping the unbounded z-space through the bounds, with `atr_stoploss_floor < atr_stoploss_cap < 0` enforced as **inadmissibility, not clipping** (clipping maps distinct candidates to one point and corrupts the trial count)
- [ ] T006 Implement `candidate_id` in `research/labelfree/es/space.py` as a deterministic hash of the **decoded** values, and the `on_bound` flag listing parameters sitting on a bound (FR-019)
- [ ] T007 [P] Write `tests/test_es_space.py`: `decode(encode(x)) == x` over 10,000 random points; the floor/cap ordering is rejected not clipped; two candidates rounding to the same `cexit_max_days` share a `candidate_id`; `on_bound` fires at each bound
- [ ] T008 Implement the sidecar in `research/labelfree/es/history.py` — `write(evaluation) -> eval_id` appends with `score: null` **before** scoring, `complete(eval_id, result)` fills it in (FR-013), one JSON object per line in `research/labelfree/es/history.jsonl`
- [ ] T009 [P] Write `tests/test_es_history.py`: an evaluation written then abandoned still appears with `score: null`; `complete` on an unknown `eval_id` raises; the file is append-only and survives a simulated crash between write and complete
- [ ] T010 Implement `research/labelfree/es/objective.py` — cost-aware net return using the brief-1 cost model, plus the incumbent-distance and turnover regularisation of FR-009, with the regularisation weights and their units stated in the module docstring
- [ ] T011 Wire `research/labelfree/admissibility.py` into `objective.py` so a candidate with no trades is **inadmissible with its score withheld** (FR-004) and a candidate losing to doing nothing is never reported as an improvement (FR-005) — reuse the shared gate unchanged, do not reimplement it
- [ ] T012 [P] Reuse the keyed panel cache from feature 012 in `research/labelfree/es/panel.py` so each fold's prices, ATR and rvol load once per run rather than once per evaluation

**Checkpoint**: the space round-trips, the trial record is crash-honest, and the objective refuses to score an abstention.

---

## Phase 3: User Story 1 - An evaluation that means something (P1) 🎯 MVP

**Goal**: a candidate score that is fast enough to run thousands of times **and** faithful to the
exit behaviour those parameters actually produce.

**Independent Test**: score a set of candidates and confirm the ranking matches full, honest
backtests of the same candidates.

### Tests for User Story 1 ⚠️

> Write these FIRST and confirm they fail.

- [ ] T013 [P] [US1] Write `tests/test_es_exits.py::test_incumbent_exit_mix` — replay the incumbent parameters over a fold and compare the surrogate's exit-reason mix against the archive's (`take_profit` ≈ 52%, the ATR+grace stack ≈ 26%). A surrogate that cannot reproduce the incumbent's own exit mix cannot rank anything.
- [ ] T014 [P] [US1] Write `tests/test_es_exits.py::test_parameters_change_the_trade_series` — the same exposure path under two different parameter sets must produce **different** trades (FR-001). This is the test that would have caught brief 1's simulator being unusable here.
- [ ] T015 [P] [US1] Write `tests/test_es_exits.py::test_grace_gates_the_stop` — inside `stoploss_grace_hours` the stop sits at `stoploss_grace_level` anchored to the entry price and does **not** trail; after it, the ATR stop applies
- [ ] T016 [P] [US1] Write `tests/test_es_exits.py::test_volume_confirmation_tightening` — with rvol > 1 the stop tightens by `1/sqrt(rvol)`, clamped at `atr_stoploss_cap`; with rvol ≤ 1 it is untouched

### Implementation for User Story 1

- [ ] T017 [US1] Implement `simulate(exposure, prices, atr_pct, rvol, params, costs)` in `research/labelfree/es/exits.py` per contracts/es-cli.md — bar-by-bar position tracking, exiting on the **first** of the ATR-adaptive stop, `cexit_take_profit`, `cexit_max_days`, or the replayed `model_exit`/`unclog_12h`
- [ ] T018 [US1] Implement the ATR stop path in `exits.py` at the highest fidelity — `-atr_stoploss_multiplier * atr_pct` clamped to `[atr_stoploss_floor, atr_stoploss_cap]`, volume-confirmation tightened, grace-gated — mirroring `Framework/BaseStrategy.py:1032-1100`. Per research.md R2 this stack governs 26.2% of exits and the entire loss side; it is not a rare branch.
- [ ] T019 [US1] Replay `model_exit` and `unclog_12h` from the archive in `exits.py` rather than recomputing them, and emit a module-level constant naming their combined 19% share so `report.py` can state the limitation rather than discover it
- [ ] T020 [US1] Apply purge/embargo via `apply_leakage_control` in `research/labelfree/es/panel.py`, and assert it is actually called — `purge_bars=0` with the function never invoked is exactly how this study's leakage control was silently absent before
- [ ] T021 [US1] Run `tests/test_es_exits.py` and iterate `exits.py` until green; record the achieved exit-mix agreement in the test's assertion message so a later regression names its own number
- [ ] T022 [US1] Measure one surrogate evaluation's wall-clock on a fold's training range and record it; the plan's budget assumes ≤ 50 ms. If it is materially slower, report the measured figure and the revised budget rather than silently running longer.

### The blocking gate

- [ ] T023 [US1] Implement `research/labelfree/es/calibrate.py` per contracts/es-cli.md — select 40 candidates **spanning** the space rather than clustered near the incumbent, score each with the surrogate and with a real freqtrade backtest, write `calibration.json`
- [ ] T024 [US1] Add the freshness gate to `calibrate.py`: freqtrade exits 0 on a fatal config error, so `rc` is not a failure signal — verify each result zip's mtime is newer than the run's start before reading it
- [ ] T025 [US1] Compute and report `spearman_rho` and `win_disagreements` in `calibrate.py`, on **rank** agreement not level (research.md R4), and exit non-zero when the two orderings disagree on which candidate wins (SC-017a)
- [ ] T026 [US1] Add a `require_calibration()` guard imported by `liveness.py`, `run.py` and `report.py` that refuses to run without a passing `calibration.json`
- [ ] T027 [US1] **RUN THE GATE**: `zsh -c 'source ~/.freqtrade_env && .venv/bin/python -m research.labelfree.es.calibrate --n 40 --folds f05..f14 --out calibration.json'` (≈2 h, sequential). Report `spearman_rho` and any win disagreements.

**⚠️ If T027 fails, the feature stops here.** Write the result report saying so (FR-020) rather
than spending 48,000 evaluations on an unvalidated evaluator. Do not proceed to Phase 4.

### Principle IV over the search space

- [ ] T028 [US1] Implement `research/labelfree/es/liveness.py` — sweep each of the nine dimensions alone across its bounds with the other eight at the incumbent, and report the objective's span in pp per dimension with a `live`/`inert` verdict and the threshold used
- [ ] T029 [US1] **RUN** `.venv/bin/python -m research.labelfree.es.liveness --folds f05..f14 --out liveness.json` and record which dimensions are inert. Four parameters in this study were configured, logged, believed active and reaching nothing; this probe is what catches a fifth.

**Checkpoint**: the evaluator is calibrated and its search space is known to be live. US2 may begin.

---

## Phase 4: User Story 2 - Search, and a control that spends the same budget (P2)

**Goal**: an evolutionary search and a uniform random control at an **identical evaluation
budget**, so the optimiser's contribution is separable from the budget's.

**Independent Test**: run both at the same budget and compare their out-of-sample results.

### Tests for User Story 2 ⚠️

- [ ] T030 [P] [US2] Write `tests/test_es_cmaes.py::test_finds_known_optimum` — the wrapper drives `cma` to the sphere function's optimum, proving the ask/tell loop, the bounds mapping and the seeding are wired
- [ ] T031 [P] [US2] Write `tests/test_es_cmaes.py::test_identical_budgets` — both methods at budget N consume exactly N evaluations, asserted on the sidecar's line count, not on a counter the code keeps (FR-007)
- [ ] T032 [P] [US2] Write `tests/test_es_cmaes.py::test_seeded_reproducibility` — the same seed produces the same candidate sequence for both methods

### Implementation for User Story 2

- [ ] T033 [US2] Implement `research/labelfree/es/cmaes.py` as a thin wrapper over `cma` 4.5.0 — ask/tell against `objective.py`, bounds via the space's encode/decode, explicit seeding. Use the reference implementation so a negative result cannot be blamed on the optimiser.
- [ ] T034 [P] [US2] Implement `research/labelfree/es/random_search.py` — uniform draws from the **same bounded space**, so the control is a genuine null and not a differently-shaped search
- [ ] T035 [US2] Implement the training-side driver in `research/labelfree/es/run.py` — `--fold`, `--method`, `--budget`, `--seed`; searches the fold's **training range only** (FR-002); budget is a count of evaluations, not wall-clock (research.md R5)
- [ ] T036 [US2] Assert in `run.py` that the two methods' evaluation counts for a fold are equal, and record each count in the sidecar's per-run summary
- [ ] T037 [US2] Count numerically identical candidates as **one evaluation for scoring, two for the trial count** (spec edge case) — the deflation counts attempts, so a cache hit must still append a row
- [ ] T038 [US2] Run `tests/test_es_cmaes.py` and iterate until green
- [ ] T039 [US2] **RUN** the search and control across all 15 folds sequentially at budget 1600, seed 7, per quickstart.md Step 5. Report the sidecar's line count against the expected 48,000.

**Checkpoint**: both methods have run at equal budgets and every evaluation is on disk.

---

## Phase 5: User Story 3 - The test fold is touched exactly once (P3)

**Goal**: each fold's test range evaluated **once**, after its search ends, provable from the record.

**Independent Test**: audit the record and confirm exactly one test-range evaluation per fold.

- [ ] T040 [US3] Add the single test evaluation to `run.py` — after the search concludes, evaluate the best candidate on the fold's test range exactly once and stop (FR-010)
- [ ] T041 [US3] Implement `research/labelfree/es/history.py --audit` per contracts/es-cli.md, checking all four invariants **from the sidecar alone** without trusting the search code: one test evaluation per fold; every test evaluation later than every train evaluation for that fold; every evaluation written before its score; the shared log's summary row matching the sidecar
- [ ] T042 [US3] Assert in `run.py` that no selection decision consumes a test-range quantity (FR-012) — the best candidate is chosen from training scores only, and the test evaluation happens after that choice is frozen
- [ ] T043 [US3] **RUN** `.venv/bin/python -m research.labelfree.es.history --audit` and report the result. A non-zero exit invalidates Phase 4's run and it must be repeated, not patched.

**Checkpoint**: the leakage property the whole design exists to protect is proven, not asserted.

---

## Phase 6: User Story 4 - Overfitting is the headline (P4)

**Goal**: PBO over the **entire** evaluation history — every candidate, every generation, both
methods — as the primary reported number.

**Independent Test**: the reported figure is computed over the full history with the evaluation
count stated beside it.

- [ ] T044 [US4] Implement the PBO path in `research/labelfree/es/report.py` using `stats.pbo_cscv` over the complete sidecar, including **abandoned and discarded** candidates (FR-014)
- [ ] T045 [US4] Populate `block_returns` (16 blocks) on every evaluation via `trials.block_returns`, so DSR/PBO have the time axis four consecutive features reported as unevaluable. This is the first feature able to use it.
- [ ] T046 [US4] Deflate using the **total** evaluation count, never the count of survivors (FR-015), and print the count beside every overfitting figure (FR-016) — `stats.pbo_cscv`'s own docstring records CSCV is biased low below ~50 configurations, so the count is what makes the figure interpretable
- [ ] T047 [US4] Write the shared-log summary row via `research/labelfree/trials.py` — one row per search run carrying the sidecar's path, line count and sha256 (SC-018), so the shared log points at the full record rather than duplicating 20 MB into git
- [ ] T048 [US4] Teach the existing trial-log audit about **both** files, so it cannot certify a record it never examined (SC-018)

**Checkpoint**: PBO is computed over everything and the count that makes it readable is beside it.

---

## Phase 7: User Story 5 - Did it actually move? (P5)

**Goal**: determine whether the found optimum differs from the incumbent by more than the
sweep's own noise.

**Independent Test**: the distance from the incumbent is reported against the 4.93 pp band.

- [ ] T049 [US5] Compute `delta_vs_incumbent_pp` and `within_sweep_noise` in `report.py` against the **4.93 pp** band; inside it the wording is "not resolved", never "improvement" (FR-018)
- [ ] T050 [US5] Report any parameter sitting on a bound as **a space drawn too narrowly** (FR-019), listing which and at which end
- [ ] T051 [US5] Implement the derived `status` in `report.py` per data-model.md §7 — `KILL` on PBO > 0.2, `KILL` when the control matches the search, `KILL` on inadmissibility, `NOT RESOLVED` inside the noise band — derived, never typed by hand
- [ ] T052 [US5] Handle "the search converged to the incumbent" as a legitimate informative outcome reported as **"no better operating point found"**, not as a failure to run (spec edge case)

**Checkpoint**: the verdict is derived from the gates rather than chosen.

---

## Phase 8: Polish & Cross-Cutting

- [ ] T053 Attach trade count, trades per year and mean holding time to **every** reported return (FR-017) — a bare percentage is not admissible anywhere in this feature's output
- [ ] T054 Re-test the return delta at **1.5× fees** (SC-005) and report both figures
- [ ] T055 Implement the reproduction path (SC-006) — a separate process reproduces the accepted result from its saved configuration alone
- [ ] T056 Write `research/label-free-study/briefs/results/05-<date>.md` per protocol §6 with **PBO as the headline number**, the evaluation count beside it, the liveness report's inert dimensions, the calibration `spearman_rho`, and the replayed-mechanism limitation from T019. Written whether the verdict is accept or kill (FR-020).
- [ ] T057 [P] Lead the result report with the exit-reason census (research.md R2) and the correction it carries — `trailing_stop_loss` is the ATR+grace stack, not the native trailing stop, which is off — since anyone reading the exit mix without that will draw the opposite conclusion
- [ ] T058 [P] Update `research/label-free-study/README.md`'s verdict table with brief 05's outcome
- [ ] T059 Run `quickstart.md` end to end as written and fix any step whose stated expectation does not match what actually happens
- [ ] T060 Stage only the files this feature created or modified (FR-022); check `git status` first and leave anything unexpected alone

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup** → no dependencies
- **Phase 2 Foundational** → blocks everything; T005 and T008 carry the invariants
- **Phase 3 US1** → depends on Phase 2. **T027 is a hard gate**: if calibration fails, stop and report.
- **Phase 4 US2** → depends on T027 passing and T029 complete
- **Phase 5 US3** → depends on Phase 4 (there is no history to audit before it)
- **Phase 6 US4** → depends on Phase 5 (a leaked history must not be reported on)
- **Phase 7 US5** → depends on Phase 6
- **Phase 8 Polish** → depends on Phase 7

Unusually for this template, the stories are **strictly sequential**. Each one consumes the
previous one's output: there is nothing to search until the evaluator is calibrated, nothing to
audit until the search has run, and nothing to report until the audit passes.

### Parallel Opportunities

Within phases only: T007/T009/T012; T013–T016 (all new test functions); T030–T032; T057/T058.
Everything else is sequential, both by data dependency and by FR-021.

---

## Implementation Strategy

**MVP is Phase 1 + Phase 2 + Phase 3 (US1).** That is the feature's entire risk: an evaluator
that is fast enough and faithful enough. If T027's calibration gate fails, the MVP has still
delivered a real finding — that this search space cannot be evaluated cheaply and faithfully —
and the report says so.

**The expected deliverable is a quantified negative.** A pass requires three conditions together:
beat the random control, sit more than 4.93 pp from the incumbent, and clear the 3.35 pp/yr
floor — with PBO ≤ 0.2. Briefs 01–04 delivered their value as quantified negatives; this one is
budgeted the same way.

**Total**: 60 tasks — US1 17, US2 10, US3 4, US4 5, US5 4, setup/foundational 12, polish 8.
