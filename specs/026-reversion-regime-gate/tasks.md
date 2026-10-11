---
description: "Task list for 026 regime gate on the reversion family"
---

# Tasks: Regime Gate on the Reversion Family

**Input**: `specs/026-reversion-regime-gate/` (spec.md, plan.md, research.md R1–R6, data-model.md,
contracts/interfaces.md C1–C5, quickstart.md)

**Tests**: Included. The spec requires causality verification (SC-001) and an early-only rule
choice. The acceptance tests for US2 are walk-forward reproductions.

**Conventions**:
- All paths are relative to `user_data/strategies/`.
- Python is `../../.venv/bin/python`.
- Backtests need `set -a; source ~/.freqtrade_env; set +a`, and run as ONE sequential chain with
  fresh tags (`run_folds` skips existing records).
- Commit to master, staging named files only.
- Every result is labelled PROVISIONAL (gate 2).
- **US1 is a hard STOP gate.** Phase 4 (US2) tasks run ONLY if T008 reports at least one PASS. On a
  STOP, skip to Phase 5 and mark T009–T017 `[-]` (not applicable) with the reason.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [ ] T001 Create `specs/026-reversion-regime-gate/baseline.md`. Record:
  - the commit sha;
  - fingerprints for arm D, `SimpleTrigger_Gbb` (`303456c1fa322b4c`) and
    `SimpleTrigger_FamCombined` (`27617f430beca1cd`);
  - the `st_fam_combo` f00 `trades_hash` (`f9e7a4531e21cce3`);
  - the BTC_USDT 15m span (2022-01-01 → 2026-10-10) and the 10-pair whitelist span.

  Create the `research/regime_gate/__init__.py` and `research/regime_gate/tests/__init__.py`
  packages.

---

## Phase 2: Foundational — instruments (blocking)

- [ ] T002 [P] Write `utils/tests/test_regime_state.py` (contract C1, research R5). Cases:
  - (a) each of `breadth`, `volreg`, `persist` and `btctrend`, computed on real 15m data (SOL,
    LTC, AAVE, XRP, ZEC + BTC, a slice of 2023-10-01 ± 40 days), is causal under truncation at 16
    cut points: values at or before each cut are identical between the full and truncated
    computations (NaN==NaN);
  - (b) a planted control (a centred rolling mean of close) FAILS the same check;
  - (c) `breadth` is NaN where fewer than 6 pairs have data, and lies in [0, 1] elsewhere;
  - (d) `persist` ≈ 1 on a synthetic random walk (|median − 1| < 0.1) and > 1 on a synthetic
    trending series;
  - (e) the output is indexed by UTC 15m dates;
  - (f) no instrument function reads a pair name except as a dict key (inspect source for
    literal pair strings).
- [ ] T003 Create `utils/RegimeState.py` (contract C1, data-model RegimeInstrument, the research R1
  constructions):
  - `breadth`, `volreg`, `persist`, `btctrend`;
  - `INSTRUMENTS`;
  - `compute(id, frames, btc)`.

  Use only backward rolling and EWM operations. T002 must then pass.

---

## Phase 3: User Story 1 — feasibility gate (Priority: P1) 🎯 MVP, STOP gate

**Goal**: decide, without backtests, whether any instrument separates reversion's losing trades in
both halves.

**Independent Test**: `feasibility.md` with a PASS/STOP per instrument and overall.

- [ ] T004 [P] [US1] Write `research/regime_gate/tests/test_feasibility.py` (contract C2, research
  R2/R3), using synthetic trades and instrument series. Cases:
  - (a) the rule choice (3 percentiles × 2 directions) reads only f00–f07 trades: an instrument
    whose values perfectly separate ONLY late losers is not chosen to separate them (its early
    choice is computed blind to late);
  - (b) options whose "on" state keeps < 50% of early trades are excluded;
  - (c) the chosen option minimises early "off" P&L, with ties going to the higher on-share;
  - (d) PASS requires `off.early < 0` AND `off.late < 0` AND `on.early > 0` AND on-share ≥ 0.5,
    with each condition failed once in a fixture;
  - (e) NaN instrument values count as "on";
  - (f) off-bar share and off-run counts on a known series;
  - (g) exit 3 when no instrument passes, exit 0 when one passes, exit 2 on a missing input.
- [ ] T005 [US1] Create `research/regime_gate/feasibility.py` (contract C2):
  - load the 10 whitelist pairs and BTC (`research.clfstudy.data.load_pair`);
  - compute the instruments (`utils.RegimeState`);
  - pool `st_fam_combo` and `st_gbb` (`evaluation.pool`);
  - take reversion trades (`utils.FamilyExits.family_of`; all of `st_gbb`);
  - look up each instrument at the signal bar (`open_date − 15min`, as-of, causal);
  - choose rules (R2), score (R3), and write `feasibility.md` and `feasibility.json`.

  The P&L unit is `profit_abs × scale_pp_per_abs`. T004 must then pass.
- [ ] T006 [US1] Causality spot-check on the live data path: compute every instrument once on the
  full data and once truncated at the last f07 date. Assert equality up to that date. Record in
  `baseline.md`.
- [ ] T007 [US1] Run `python -m research.regime_gate.feasibility --primary st_fam_combo --cross
  st_gbb --out specs/026-reversion-regime-gate/feasibility.md`.
- [ ] T008 [US1] **STOP gate.** Record the per-instrument verdicts in `baseline.md`, along with
  off-bar shares and off-run counts (calendar-switch check: a PASS whose late "off" is one
  contiguous run covering > 80% of late bars is flagged as a calendar switch and treated as
  STOP for that instrument).
  - If no instrument passes, mark T009–T017 `[-]` with the reason and go to Phase 5.
  - Otherwise, list the passing instruments and their rules (from `feasibility.json`) for Phase 4.

**Checkpoint**: the feature either STOPS here (a recorded negative) or proceeds with named rules.

---

## Phase 4: User Story 2 — gated combined arm (Priority: P2) — ONLY if T008 passed

- [ ] T009 [P] [US2] Write `SimpleTrigger/tests/test_regime_gate.py` (contract C3), using a
  synthetic frame plus a stubbed `dp` (whitelist frames and BTC):
  - `REVERSION_REGIME_GATE=None` gives tags and mask identical to 025;
  - a gate with `off_when="ge"` drops reversion-only rows where the state ≥ threshold;
  - overlap rows in "off" are tagged momentum;
  - momentum rows are never dropped;
  - NaN state counts as "on";
  - sink `regime_gate_refused` / `regime_gate_off_bars` counts;
  - validation refuses an unknown instrument, a bad `off_when`, or no enabled reversion trigger.
- [ ] T010 [US2] Edit `SimpleTrigger/SimpleTriggerStrategy.py` (contract C3):
  - `REVERSION_REGIME_GATE`;
  - `informative_pairs()` (BTC when needed);
  - the state is computed via `utils.RegimeState` over `dp.current_whitelist()` frames, cached per
    last date;
  - drop/retag in `get_entry_conditions` after the 025 rule-B step;
  - sink counts;
  - validation.

  T009 and every existing SimpleTrigger test must then pass. Do NOT edit while a backtest runs.
- [ ] T011 [P] [US2] Edit `evaluation/fingerprint.py` (contract C4): `REVERSION_REGIME_GATE` joins
  the optional family members when non-None. Test that existing shas are unchanged and that gated
  configs are distinct.
- [ ] T012 [P] [US2] Edit `evaluation/stage.py` (contract C5): add `--bar <tag>` (default
  `--previous`). The late judgement is chosen vs the bar, and the block records both the bar
  verdict and the vs-previous verdict. Extend `evaluation/tests/test_stage.py`: adoption follows
  the `--bar` verdict only.
- [ ] T013 [US2] Create `SimpleTrigger/SimpleTrigger_RegGateOff.py` (`REVERSION_REGIME_GATE = None`,
  subclass of `SimpleTrigger_FamCombined`) and derive plan `st_reg_off`. Run it. It must reproduce
  `st_fam_combo` in 15/15 folds by `trades_hash` (FR-006). **STOP on divergence.**
- [ ] T014 [US2] For each PASSING instrument, create `SimpleTrigger/SimpleTrigger_RegGate<Inst>.py`
  with the rule from `feasibility.json`, and derive plan `st_reg_<inst>`. Run the chain.
  Liveness: non-zero `regime_gate_refused` in each arm, plus the momentum family's trade count.
  Record both in `baseline.md`.
- [ ] T015 [US2] Pool the gated arms. Run `evaluation.contention` against `st_gbb` / `st_bgo_mx`
  for each.
- [ ] T016 [US2] Run `python -m evaluation.stage --stage regime --previous st_fam_combo --bar
  st_bgo_mx --candidates <st_reg_* tags> --singles reversion=st_gbb,momentum=st_bgo_mx --out
  specs/026-reversion-regime-gate/decisions.md`. Record the adopted arm, or none.
- [ ] T017 [US2] Compare the stage winner against `st_bgo_mx`, `st_fam_combo` and `st_gbb`
  (`compare_pooled` + `halves`, `PYTHONHASHSEED=0`). Add per-family contributions by half and
  regime refusal counts by half.

---

## Phase 5: User Story 3 — report & polish (Priority: P3)

- [ ] T018 [US3] Write `specs/026-reversion-regime-gate/report.md`:
  - SUMMARY;
  - the feasibility table (every instrument: rule, per-state per-half P&L, off-bar shares,
    verdict);
  - the US2 results, or the STOP reason;
  - the SC-001…SC-005 table;
  - an explicit SC-004 answer;
  - every figure labelled PROVISIONAL;
  - the substitution caveat on bucket splits.
- [ ] T019 Run the test gate: `pytest -q Framework/tests evaluation/tests SimpleTrigger/tests
  research/outcomes/tests research/regime_gate/tests utils/tests NNNC/tests`. Only the 3 known
  pre-existing training-pipeline failures may remain. Then ruff on the added lines (024 method),
  and the exit single-source test, classifying any new files.
- [ ] T020 If US2 ran: run a fresh `run_arm` f00 re-check of `st_fam_combo` (`f9e7a4531e21cce3`)
  and `st_gbb` (`aa058ba815f48627`) under `t026_*` tags, and check fingerprints unchanged
  (SC-005). If US1 stopped, only the fingerprint check is needed (no strategy code changed).
- [ ] T021 Update memory (`project_026_reversion_regime_gate.md` plus the MEMORY.md hook; an
  `index_*` hub line for a reusable regime finding). Mark the tasks. Commit, staging named files
  only.

---

## Dependencies & Execution Order

- T001 → T002/T003 → T004/T005 → T006 → T007 → **T008 (STOP gate)**.
- Phase 4 runs only if T008 passes:
  - T009→T010;
  - T011 ∥ T012 (separate files);
  - T013 (neutral reproduction) before T014;
  - T014 → T015 → T016 → T017.
- Phase 5 always runs.

### Parallel opportunities

- T002 ∥ T004 (test files).
- T011 ∥ T012.
- T010 must not run during any backtest.

## Implementation Strategy

**MVP = Phases 1–3:** instruments plus the feasibility verdict, with no backtests at all. That is
the cheapest possible answer to "is there a regime state worth gating on?". US2 is built only on a
PASS.
