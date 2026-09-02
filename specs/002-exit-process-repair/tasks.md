---

description: "Task list for 002-exit-process-repair"
---

# Tasks: Exit-Process Characterisation and Repair

**Input**: Design documents from `specs/002-exit-process-repair/`

**Prerequisites**: spec.md, plan.md, research.md, data-model.md, contracts/, quickstart.md — all present. Constitution v1.0.0.

**Tests**: No unit-test tasks. This is a measurement study; correctness is established by the FR-014 baseline-reproduction gate (T014) and the per-arm inertness check (FR-023), both functional requirements rather than test scaffolding.

**Organization**: Grouped by user story. Phase 4 runs only for candidates that survive the Phase 3 screen.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: [US1] [US2] [US3] [US4]
- Paths relative to `user_data/strategies/` unless stated

## Path Conventions

- **Study code**: `regime/exit/` (new)
- **Shared runner**: `regime/run_arm.py` (already extended with `--set-sell` / `--roi` / `--stoploss`)
- **Execution**: from the freqtrade root as `PYTHONPATH=. .venv/bin/python user_data/strategies/regime/exit/<script>.py`

---

## ⚠️ Standing constraints (every task)

| Constraint | Detail |
|---|---|
| **Nothing retrains** | The model at `saved_data/NNNC_MLX/NNNC_MLX.safetensors` is reused unchanged. Phases 1–3 do not load it at all. |
| **Entry is frozen** | Validated by feature 001. No task may change an entry parameter. |
| **The JSON drifts** | `NNNC/NNNC_MLX.json` is untracked and was overwritten by a hyperopt mid-session, which confounded one test before a pinned run caught it. Baseline comes from `regime/exit/baseline.py`, never from that file (FR-028b). |
| **Framework edits are temporary** | `Framework/BaseStrategy.py` is committed and clean. Use edit → measure → `git checkout`. A live dry run (`MomentumRegimeBasket15mFast`) imports it — never leave an experimental edit in place. |
| **Pinned windows** | W1 `20250901-20260831` (headline), W2 `20240601-20250531`, W3 `20230101-20231231` (decides persistence). |
| **Shell** | zsh does not word-split unquoted parameter expansions. Pass runner args as an array: `"${BASE[@]}"`. |

### Per-arm protocol — every backtest arm (Phase 4)

| Gate | Requirement | Req |
|---|---|---|
| Pinned literal timerange | Never `-n`/`-o`, which compute from today | FR-017 |
| Three windows | W1 headline, W3 decides | FR-017, FR-020, FR-021 |
| Realistic execution | Volume/liquidity constraints active | FR-018 |
| Judge on | Realized profit, Calmar, Sortino, drawdown — never a model metric | FR-019 |
| Liveness | `trades_hash` equal to baseline ⇒ **INERT**, a defect, not a result | FR-023 |
| Trade count | Reported; a zero- or few-trade arm is inconclusive | FR-022 |
| Exit mix | Reported, so category-shifting is visible | edge case |
| Entry untouched | `entry_config_hash` matches baseline | FR-024 |

---

## Phase 1: Setup

- [ ] T001 Create `regime/exit/` with `__init__.py` and a `regime/exit/README.md` stub naming the feature, the three pinned windows, and the baseline exit configuration
- [ ] T002 Create `regime/exit/baseline.py` holding `BASELINE` as **literals** — grace 5.6h, grace_level −0.27 (marked INERT), take_profit 0.024, max_days 10, ROI 3%, stoploss −0.05, exit signals off — plus the three archive paths from `regime/arms/applied_W{1,2,3}.json`. On import it MUST assert the archives resolve and the per-window trade counts are 51 / 96 / 20, raising rather than silently characterising a different config (FR-028, FR-028b, data-model §1)

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ CRITICAL**: no user story work begins until this completes.

- [ ] T003 Create `regime/exit/candidates.py` — the `CandidateIntervention` registry per data-model §4. Each entry carries `name`, `motivating_measurement` (required, non-empty), `kind` (`adjust_existing` | `enable_dormant`), `requires_retrain` (must be False). Pre-populate the four **excluded** entries with their reasons so they are never re-tested: grace *level* (inert — enabling `_ft_stop_uses_after_fill` gives byte-identical results), `return 1.0` → `None` (refuted, −9.24pp), `trailing_*` under the current architecture (inert, byte-identical at 0.01 vs 0.03), hyperopt exit config (rejected, W3 sign reversal) (FR-009..FR-012, FR-027)
- [ ] T004 Verify `regime/run_arm.py`'s sell-space support end to end: run the pinned baseline on all three windows and confirm it reproduces the characterised baseline with **identical trade hashes** (51 / 96 / 20 trades; +3.95 / +5.752 / +1.724%). Already demonstrated once this session — this task re-establishes it as a repeatable check (FR-028b)

**Checkpoint**: baseline pinned and reproducible; candidate registry seeded.

---

## Phase 3: User Story 1 — Know what the stop is actually doing (Priority: P1) 🎯 MVP

**Goal**: A stated verdict on whether the stop cuts noise or cuts losers, from what price actually did after each stopped trade.

**Independent Test**: Read the verdict from the Phase 1 report. No strategy change, no backtest.

> **Substantially pre-satisfied by research.md.** R1 established the mechanism, R2 refuted the obvious correction, R3 catalogued which settings are live. T005–T008 add only the per-trade counterfactual, which research did not do.

- [ ] T005 [P] [US1] Implement exit-mix and stop-attribution reporting in `regime/exit/characterise.py`, writing `exit_mix.csv` and `stopped_trades.csv` per `contracts/study-cli.md`. Load recorded trades with `freqtrade.data.btanalysis.load_backtest_data` on the T002 archives. **Assert `initial_stop_pct` is −5.00% on every row** — any other value means the grace level became reachable and contradicts research R1; halt rather than report (FR-001, FR-005, data-model §2)
- [ ] T006 [P] [US1] Add the counterfactual replay to `characterise.py`: for every stopped trade, measure best and worst reachable outcome over ~1/3/6/12/24h extensions from the actual exit price, and whether the take-profit or ROI level would have been reached. Write `counterfactual.csv`. Trades with insufficient subsequent history are **excluded and counted**, never treated as "did not recover" (FR-002, FR-003, FR-008, SC-002)
- [ ] T007 [US1] Split every table by `ended_in_profit`. The 37.4% that exit in profit and the 62.6% that exit at a loss are separate populations — 50.6% of stopped trades end with the stop **above** entry, which is a different phenomenon from falling through it. The report MUST state whether one intervention can serve both (FR-004, SC-003)
- [ ] T008 [US1] Run `characterise.py` and record the verdict — noise or losers — with the aggregate figure behind it, in `regime/exit/README.md`. Confirm no model was loaded and nothing trained (FR-006, FR-007, FR-025, SC-001, SC-007)

**Checkpoint**: the question "is the exit broken?" is answered. If the stop is cutting genuine losers, the feature may stop here with that finding (SC-010).

---

## Phase 4: User Story 2 — A ranked set of candidate repairs (Priority: P2)

**Goal**: The interventions worth testing, enumerated from Phase 1 evidence, each citing its motivating measurement.

**Independent Test**: Review the list; every entry cites a measurement, and every excluded candidate carries a reason.

- [ ] T009 [US2] Populate the seven live candidates in `regime/exit/candidates.py`, each with its motivating measurement: **(a) reimplement the ratchet deliberately with a tunable distance** — 50% of exits, currently the constant `1.0`, and dependent on two uncontracted freqtrade behaviours; **(b) native trailing stop** via `use_custom_stoploss=False`, a second implementation of the same profit-lock with real distance and offset parameters; **(c) grace duration** — the one remaining live grace dimension; **(d) ROI ladder** — 22% of exits land at exactly +3.00%; **(e) `cexit_take_profit` upward** — downward is known harmful; **(f) re-enable model exit signals** — no retrain, `kind = enable_dormant`; **(g) time-based exits** — only 6 of 167, low ceiling (FR-009, FR-010, FR-012, SC-005)
- [ ] T010 [US2] Cross-check each live candidate against research R3's inert list and reject any that touches `stoploss_grace_level`, `trailing_*` under the current architecture, `exit_guard_threshold` or `exit_close_norm_threshold` — four of nine exit settings do nothing, and tuning one yields a guaranteed null that reads like a finding (FR-027)
- [ ] T011 [US2] Record in `regime/exit/README.md` which anticipated candidates were excluded and why, so none is silently omitted (FR-011, FR-028a)

**Checkpoint**: candidate list is evidence-backed and inert-free.

---

## Phase 5: User Story 3 — Cheap screening before expensive testing (Priority: P3)

**Goal**: Score candidates by re-deriving outcomes from price history, so only survivors consume a backtest.

**Independent Test**: Rank two or more candidates against the recorded baseline with no backtest run.

- [ ] T012 [US3] Implement `regime/exit/replay.py` — re-derive each recorded trade's outcome under a candidate exit rule from price history, modelling the full exit stack (ratchet, anchored stop, take-profit, ROI, time-based) (FR-013)
- [ ] T013 [US3] Implement `--validate-only`, which replays the **current** rule and compares against the recorded baseline outcomes (FR-014)
- [ ] T014 [US3] ⛔ **GATE — run `replay.py --validate-only` and do not proceed until it reproduces the recorded baseline.** Exit code 2 on failure. A screen that cannot reproduce the known answer will mis-score every candidate **invisibly** — the candidates would simply rank wrongly, with nothing to signal it. Nothing may be scored before this passes (FR-014, SC-004)
- [ ] T015 [US3] Score all live candidates, writing `screen_results.csv` with the change in total outcome versus the recorded baseline and a PASS/REJECTED verdict per candidate (FR-015)
- [ ] T016 [US3] Reject failing candidates **without** consuming a backtest, and record at least one such rejection as evidence the screen earns its place (FR-016, SC-006)
- [ ] T017 [US3] Record the screen's known blind spots in `regime/exit/README.md`: it models the exit stack but not position limits, one-open-trade-per-pair blocking, or capital freed by an earlier exit. Its disagreement with Phase 4 is expected and is itself a reported quantity (FR-013, SC-004, Assumptions, edge case)

**Checkpoint**: the candidate set is filtered for the cost of a calculation.

---

## Phase 6: User Story 4 — Confirm survivors under real execution (Priority: P4)

**Goal**: Full evaluation across all three windows for candidates the screen endorsed.

**Independent Test**: Each survivor is evaluated against the pinned baseline over the same windows and accepted or rejected on its own.

> Ordered by screen rank, not by this list. The ratchet reimplementation is expected to lead on prior evidence, but the screen decides.

- [ ] T018 [US4] For each survivor, run three arms via `regime/run_arm.py` with the pinned baseline args plus that candidate's override, passing args as a zsh array. Record `n_trades`, `exit_mix`, wallet return, drawdown, PF, Calmar, Sortino, `trades_hash`, `entry_config_hash` (FR-017..FR-022, data-model §5)
- [ ] T019 [US4] Check every arm for inertness: `trades_hash` equal to the baseline's means the change **did not take effect** — report it as a defect and fix the wiring, never as a result about the intervention (FR-023, SC-008)
- [ ] T020 [US4] Verify `entry_config_hash` matches the baseline on every arm — entry is frozen by feature 001 (FR-024, FR-026)
- [ ] T021 [US4] For candidates requiring a `Framework/BaseStrategy.py` change (the ratchet reimplementation, the native-trailing variant), use edit → measure → `git checkout` per arm. **Never leave an experimental edit in place** — a live dry run imports that file (FR-029)
- [ ] T022 [US4] Reject any candidate whose risk-adjusted return reverses sign in W3, regardless of its W1 result. This test has already rejected two candidates in this project (`rvol 2.5`, the hyperopt exit config) (FR-021, SC-009)
- [ ] T023 [US4] Compare each survivor's screen prediction against its measured result and report the gap — it quantifies how far the screen can be trusted for future work (SC-004, edge case, Assumptions)

**Checkpoint**: an accepted candidate, or a stated finding that none improves the exit.

---

## Phase 7: Polish & Close-out

- [ ] T024 Consolidate `regime/exit/README.md`: the mechanism (research R1–R3), the characterisation verdict, the candidate table with screen and confirmation results, and the excluded list with reasons (FR-011, SC-001)
- [ ] T025 If a candidate is accepted, apply it to `Framework/BaseStrategy.py` and/or `NNNCStrategy.sell_params` and verify by re-running all three windows with **no overrides**, confirming the applied defaults reproduce the accepted arm's trade hashes — the check that caught a stale-config error in feature 001 (FR-030, FR-023)
- [ ] T026 If no candidate is accepted, record that the exit is already well-calibrated, with the evidence. This is a successful completion of the feature, not a failure to deliver (SC-010)
- [ ] T027 Verify the whole study re-runs from a clean checkout with no manual path or parameter edits (SC-011)
- [ ] T028 Stage only the files this feature touched (`regime/exit/`, `regime/run_arm.py`, and on acceptance the strategy file) and commit to master. **Never `git add -A`** — the user edits concurrently
- [ ] T029 Record the outcome in project memory: the mechanism finding, which settings are inert, and the verdict — so the `return 1.0` question is not re-opened from scratch (SC-010)

---

## Dependencies

```
Phase 1 Setup (T001-T002)
   └─> Phase 2 Foundational (T003-T004)
          └─> Phase 3 US1 characterise (T005-T008)      ← MVP; may end the feature
                 └─> Phase 4 US2 candidates (T009-T011)
                        └─> Phase 5 US3 screen (T012-T017)
                               └─ T014 GATE ─> T015-T017
                                      └─> Phase 6 US4 confirm (T018-T023)   [survivors only]
                                             └─> Phase 7 Polish (T024-T029)
```

**Hard ordering**

- T002 blocks everything — no measurement before the baseline is pinned
- T004 blocks Phase 5 — the screen is validated against a baseline that must first be reproducible
- **T014 blocks T015–T017 absolutely** — nothing is scored before the screen reproduces the known answer
- T016 blocks T018 — only screen survivors get a backtest
- T019 blocks T022 — an INERT arm has no result to judge

---

## Parallel Execution Opportunities

**Phase 3** — the two analyses read the same trades but write different artifacts:

```
T005 [P] exit mix + stop attribution  -> exit_mix.csv, stopped_trades.csv
T006 [P] counterfactual replay        -> counterfactual.csv
```

**Phase 6** — the three windows for a given candidate are independent backtest runs and can run together. Candidates themselves stay **sequential**, because each may require a different `Framework/BaseStrategy.py` edit and those must not overlap.

**Not parallelisable**: anything crossing the T014 gate; the Framework-editing arms in T021.

---

## Implementation Strategy

**MVP = User Story 1 (T001–T008).** It answers the question that was actually asked — is the exit broken, and how — from recorded trades alone. No backtests, no model load. If the stop turns out to be cutting genuine losers, the feature is complete there.

**Cost profile.** Phases 1–3 read 167 trades and price history; they run in seconds and cost nothing. Phase 6 is the only expensive step, and the T014 gate plus the T016 screen exist to keep the number of survivors small.

**What research already settled**, so it is not re-derived: the firing mechanism, the inert-parameter catalogue, the refutation of the `None` correction, and the rejection of the hyperopt exit config. Four candidates enter the registry pre-excluded with reasons.

**The strongest remaining lever** is the ratchet reimplementation. It governs 50% of exits, has no tunable distance today, and rests on freqtrade behaviour that is not contracted — so it carries a correctness argument independent of whether it also improves P&L.
