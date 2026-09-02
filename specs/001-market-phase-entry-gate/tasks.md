---

description: "Task list for 001-market-phase-entry-gate"
---

# Tasks: Market-Phase Framework and Phase-Adaptive Entry Gate

**Input**: Design documents from `specs/001-market-phase-entry-gate/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md — all present. Constitution v1.0.0.

**Tests**: No unit-test tasks. The spec does not request TDD, and this is a measurement study — correctness is established by the lookahead validation gate (T012–T013, FR-007) and by the per-arm liveness proof (FR-031), both of which are functional requirements rather than test scaffolding.

**Organization**: Tasks are grouped by user story. User Stories 3 and 4 are **gated behind the Phase C verdict** and may never run.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: [US1] [US2] [US3] [US4]
- All paths are relative to `user_data/strategies/` unless stated otherwise

## Path Conventions

- **Study code**: `regime/` (new, standalone — mirrors `btc_corr/`)
- **Shared indicators**: `utils/custom_indicators.py` (the ONLY existing file modified in Phases A–C)
- **Strategy code**: `utils/PhaseGate.py`, `NNNC/` — Phase D only
- **Execution**: everything runs from the freqtrade root (`/Users/philprice95/projects/freqtrade`) as `PYTHONPATH=. .venv/bin/python user_data/strategies/regime/<script>.py` — the system Python has no pandas

---

## ⚠️ Standing constraints (apply to every task)

| Constraint | Detail | Req |
|---|---|---|
| **Live dry run** | `MomentumRegimeBasket15mFast` has been running since 2026-08-23. Phases A–C touch no strategy code. Phase D must not modify shared `Framework/` files while it runs. | — |
| **The JSON override** | `NNNC/NNNC_MLX.json` is untracked user working state and **overrides** class `buy_params`. Never modify it; write a per-arm copy. A threshold set only in Python silently does nothing — the single most likely false null in this feature. | R5 |
| **Single host** | Every measurement in this feature runs on `NNNC_MLX`, the non-GAN base. Never an `NNNC_DDPM_` variant — that prefix activates a GAN. No host change mid-feature. | FR-033 |
| **Staging** | Stage only files your change touched. Never `git add -A` — the user edits concurrently. | — |

### Per-arm protocol — **every backtest arm, Phases C and D alike**

Applies to T013, T018–T020 (User Story 2) and every Phase 6/7 arm. Not optional, and not restated per task.

| Gate | Requirement | Req |
|---|---|---|
| **Pinned windows** | W1 `20250901-20260831`, W2 `20240601-20250531`, W3 `20230101-20231231`. Always `test_strat.sh -t`; **never** `-n`/`-o`, which compute from today's date. | FR-026 |
| **Three windows, one distant** | All three, every arm. W1 and W2 are three months apart and cannot establish persistence between them; W3 is the window that can. | FR-027 |
| **Headline verdict** | W1 is the deployment regime and carries the headline result. Full-history or cross-window aggregates are context, never the headline. | FR-027a |
| **Realistic execution** | Volume/liquidity constraints active. Fills exceeding available liquidity inflate results even when lookahead is clean. | FR-028 |
| **Judge on P&L only** | Accept or reject on realized profit and risk-adjusted return. **No model-fit metric** — MCC, validation loss, correlation — may decide an arm, in any phase. | FR-029 |
| **Liveness** | Report model-condition pass count, full-chain pass count, trade count. Chain count equal to baseline ⇒ **INERT**, results discarded, never recorded as a null. | FR-031 |
| **Trade count** | Reported every arm. A zero-trade arm is inconclusive, never a zero-return result. | FR-032 |
| **Exit invariance** | `exit_config_hash` byte-identical to the baseline for the same window. No arm alters exit behaviour. | FR-024, SC-010 |
| **Sign holds in W3** | An improvement that reverses in the distant era is era-specific and rejected. | SC-009 |
| **Paired seeds** | Any accepted delta under one percentage point requires paired repeated runs first. | FR-030 |

---

## Phase 1: Setup

**Purpose**: Create the study skeleton.

- [ ] T001 Create the study directory `regime/` with an empty `__init__.py` and a `regime/README.md` stub stating the feature name, the three pinned windows from research.md R3, and a "results pending" placeholder
- [ ] T002 [P] Create output subdirectories `regime/labels/` and `regime/arms/` with `.gitkeep` files, per the artifact paths in `contracts/artifacts.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared machinery every user story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase completes.

- [ ] T003 Add `variance_ratio(series, q, window)`, `hurst_from_vr(series, q_grid, window)`, and `rolling_autocorr(series, lag, window)` to `utils/custom_indicators.py` — additive only, matching the existing flat dataframe-to-series style (`zema`, `RMI`, `T3`). Lo–MacKinlay `VR(q) = Var(r_q)/(q·Var(r_1))` on log returns; Hurst from the log-regression of `VR(q)` against `q` as `H = 1 + slope/2`; fixed 480-bar window, never adaptive (research R1, R2; Constitution VII)
- [ ] T004 Create `regime/regimedata.py` — panel loading from `user_data/data/binanceus/*-15m.feather`, universe assembly across the 78 pairs, `log_returns`, and the market-level coverage rule (≥20 pairs each with ≥200 bars of prior history, else `UNKNOWN`). Follow the `btc_corr/btcdata.py` pattern: `__file__`-relative with an absolute DATA path, re-runnable from a clean checkout (research R4, R10; FR-006)
- [ ] T005 Create `regime/instruments.py` — the `PhaseInstrument` catalogue per `data-model.md` §1, with the schema of `contracts/artifacts.md` `instrument_catalogue.csv`. Enforce: unique names, `UNKNOWN` in every state vocabulary, exactly one `is_baseline`, `window_bars` ≥ 20× the largest estimator horizon, and no instrument definition referencing a specific trading pair (FR-025, Constitution X)
- [ ] T006 Create `regime/run_arm.py` per `contracts/study-cli.md` — resolves `--window` to a literal pinned range passed via `-t`, routes `--threshold` through a **per-arm copy** of `NNNC/NNNC_MLX.json` (never the user's file), keeps volume/liquidity constraints active, adds `--export trades` with a per-arm path, and emits the three liveness counts plus `exit_config_hash`. Do not modify `scripts/test_strat.sh` (research R5, R7, R8; FR-028, FR-031)

**Checkpoint**: shared machinery ready — User Story 1 can begin.

---

## Phase 3: User Story 1 — A reusable, actionable phase framework (Priority: P1) 🎯 MVP

**Goal**: A three-level, lookahead-clean phase classifier whose instruments each carry a SURVIVING or ELIMINATED status with a measured reason, plus a straight answer on whether the prior taxonomy's risk discrimination is real.

**Independent Test**: Run the classifier over full history and inspect `regime/actionability.csv` and `regime/instrument_catalogue.csv`. Value is delivered when every instrument has a status and a quoted reason — no strategy backtest needed for the framework itself.

### Instruments (independent of each other)

- [ ] T007 [P] [US1] Implement the **BTC-level** baseline instrument in `regime/instruments.py`, reproducing the exp4 taxonomy **verbatim**: `talib.EMA(close, 50)` with `pct_change(24)` for direction × `talib.ADX(high, low, close, 14)` for strength, cut at ADX<20 Sideways / 20–30 Weak / ≥30 Strong. Add `btc_vol24` and `btc_adx` as state variables — BTC's predictive content is in its state, not its returns (FR-004, research R10)
- [ ] T008 [P] [US1] Implement the **asset-level** instruments in `regime/instruments.py`: variance ratio at `q ∈ {2,4,8,24}`, Hurst from the VR curve, rolling autocorr(1) — all via T003 — plus the existing free indicators (`adx`, `atr_pct_roll`, `bb_width`, `dwt_trend`, `dwt_slope`, `dwt_nseq_up/dn`, `dc_dist`) (FR-003)
- [ ] T009 [P] [US1] Implement the **market-level** instruments in `regime/instruments.py`: cross-sectional return dispersion, breadth (share of universe above its own EMA200), and median pairwise correlation — all under the T004 coverage rule, emitting `UNKNOWN` below threshold (FR-005)

### Classification and the lookahead gate

- [ ] T010 [US1] Create `regime/phase_a_classify.py` per `contracts/study-cli.md` — `--level`, `--instruments`, `--timerange`, `--out`. Emits one label artifact per instrument per pair plus `regime/instrument_catalogue.csv`. The `date` column must be tz-aware UTC and align bar-for-bar with the source feather (a naive timestamp produces an all-NaN merge downstream) (FR-001, FR-002)
- [ ] T011 [US1] Create `regime/phase_a_lookahead.py` per `contracts/study-cli.md` — recompute each instrument over history truncated at three cut points and assert label identity over the overlapping region. Write `regime/lookahead_validation.csv` with `n_differing` and `first_diff_date`. Exit code 2 on any non-zero `n_differing` (FR-007, research R9)
- [ ] T012 [US1] **GATE** — run T011 and drive `n_differing` to 0 for every instrument. Use `first_diff_date` to localise: divergence at the end of the overlap ⇒ centred window; divergence propagating backwards ⇒ `ffill`/`bfill` across the boundary; market-level only ⇒ cross-sectional operation over a panel containing future rows (the highest-risk case here). An instrument that cannot be fixed is dropped. **No instrument proceeds to Phase B with `n_differing > 0`** (SC-001)

### Actionability characterisation

- [ ] T013 [US1] Run the baseline arm to obtain the **host's own** measured distributions: `run_arm.py --strategy NNNC_MLX --window W1 --threshold 0.69 --tag baseline`. Record (a) trade-duration median and 25th percentile — the dwell kill threshold; and (b) the trade **rate**, from which derive `N`, the minimum trades per arm for a decision block to qualify in Phase C. Do **not** reuse the 1.4–31h figures from the momentum dry run — that is a different strategy family (research R6, R12)
- [ ] T014 [US1] Create `regime/phase_b_characterize.py` per `contracts/study-cli.md` — state distribution, median dwell, and transition matrix per instrument per era. `--dwell-source` is **required with no default**, deliberately, so a duration figure from another strategy family cannot be borrowed by accident (FR-008, FR-009)
- [ ] T015 [US1] Run T014 across W1/W2/W3 with `--dwell-source` from T013 and set every catalogue `status`. Eliminate instruments whose median dwell falls below the 25th percentile of host trade duration (`DROP_DWELL`), or which meet either instability threshold (`DROP_UNSTABLE`): largest absolute state-share change **> 15 percentage points**, or the most-likely next state differing for **≥ half the states**. States holding **< 5%** of a window's bars are reported with their sample count and excluded from verdict-driving comparisons. Every elimination must quote a measured number. **Instability may only be claimed from a W1-vs-W3 difference** — W1 and W2 are three months apart, share a regime, and cannot establish it (FR-010, FR-011, SC-002, SC-003; Constitution II)
- [ ] T016 [US1] Create `regime/phase_b_risk_normalize.py` — recompute the exp4 per-state drawdown normalized for time-in-state, writing `regime/risk_normalized.csv` (FR-012)
- [ ] T017 [US1] Run T016 and record the verdict plainly as `CONFIRMED` or `REFUTED`. The original showed a ~4× spread (28.1% best state vs 118.3% worst) while the worst state held 1.55× the bars. A `REFUTED` result removes the primary motivation for treating phase as a risk-modulation signal and must be stated in `regime/README.md`, not buried in a table (SC-004)

**Checkpoint**: If **zero instruments survive T015**, the feature stops here — record it in `regime/README.md` with the eliminated-instrument list and quoted reasons, then go to Phase 8 (T033) — this is a terminal negative result, not a failure. Otherwise User Story 2 can begin.

---

## Phase 4: User Story 2 — The trackability verdict (Priority: P2)

**Goal**: One unambiguous PASS or FAIL on whether a phase label predicts which entry configuration wins the coming period, better than a fixed configuration.

**Independent Test**: Read `regime/VERDICT.md` — the verdict line plus the per-window comparison table. No gate implementation required.

### Bracketing arms (the three windows are independent runs)

- [ ] T018 [P] [US2] Run the **aggressive** arm across W1/W2/W3: `run_arm.py --strategy NNNC_MLX --window {W1,W2,W3} --threshold 0.55 --tag thr055` → `regime/arms/thr055_*.json` (research R5)
- [ ] T019 [P] [US2] Run the **conservative** arm across W1/W2/W3 at `--threshold 0.80 --tag thr080` → `regime/arms/thr080_*.json`
- [ ] T020 [P] [US2] Run the **constant baseline** at `--threshold 0.69 --tag thr069` for W2 and W3 (W1 already exists from T013) → `regime/arms/thr069_*.json`
- [ ] T021 [US2] Verify all three arms are `LIVE`, not `INERT` — the `chain_pass_count` values must differ across thresholds. If they are identical, the JSON override did not take effect and every downstream number is void. Fix `run_arm.py` before proceeding; do not interpret the results (FR-031; Constitution IV)

### The verdict

- [ ] T022 [US2] Create `regime/phase_c_trackability.py` per `contracts/study-cli.md`, following the harness shape of `Basket/tools/rolling_optimal_lookback.py`. Split each window into non-overlapping decision blocks of length `H ∈ {14d, 30d, 60d}`; score by **mean regret** over qualifying blocks (regret = the block winner's risk-adjusted return minus the chosen configuration's). The bar is `constant_error = min(error(always-aggressive), error(always-conservative))` — **the better of the two fixed choices**; the status-quo 0.69 is a reference column only, never the bar. Exclude and report any block where either arm has fewer than `N` trades; never score it as a tie. The script must **refuse to run against a price series**: only realized trade exports are valid input, which is the substitution that invalidated the prior BTC gate (FR-013, FR-014, FR-018, FR-018a, FR-018b, research R10, R12)
- [ ] T023 [US2] Run T022 against the T018–T020 exports with `--min-trades N` from T013 and `--status-quo 0.69`. Block winners are decided on **risk-adjusted return**, so a configuration that raises return and drawdown proportionally does not count as a win. Write `regime/trackability.csv` and `regime/VERDICT.md`. A PASS at exactly one horizon in exactly two windows is the weakest admissible result and must be reported as such, not as a clean positive (FR-018, FR-018a, FR-018b, SC-005, research R12)
- [ ] T024 [US2] Validate the verdict before believing it: every `trackability.csv` row's `source` points at a trade export and never a price series; no row supporting a PASS has `n_trades = 0`; a PASS rests on ≥2 windows including a temporally distant pair (W1+W2 alone is **not** a PASS); one window passing and another failing is **FAIL**, not "mixed" (FR-015, FR-016, FR-032)

**Checkpoint**: the verdict exists. Proceed to the stop gate.

---

## Phase 5: ⛔ THE STOP GATE

**This is an explicit decision point, not a formality.** Exactly one of T026 or T027 executes.

- [ ] T025 Read `regime/VERDICT.md` and record the outcome. On PASS, record the winning instrument, level, and horizon `H` — only that instrument proceeds. On FAIL, User Stories 3 and 4 do not happen (FR-017)
- [ ] T026 **[FAIL BRANCH — TERMINAL]** Write the negative result to `regime/README.md`: the full `trackability.csv` comparison table, the eliminated-instrument list with quoted reasons, the T017 risk verdict, and a short statement of what was measured and why it does not support a gate. Include enough detail that the result is reproducible and nobody re-proposes the idea without re-running it — the same convention the tool that killed the dynamic lookback uses. Stage only `regime/` and `utils/custom_indicators.py`, commit, and **stop.** Then go to Phase 8 (T033). **This is a successful completion of the feature, not an abandonment** — per the constitution's experiment gate, a negative result that clears the gates is a finding of equal weight (SC-006)

> **T027 onward executes only on PASS.** If T026 ran, skip to Phase 8 (T033).

---

## Phase 6: User Story 3 — Phase-adaptive gate, no-retrain tier (Priority: P3) — GATED ON PASS

**Goal**: Modulate the existing reversion leg's entry aggressiveness by phase, using levers requiring no retraining, and learn which actually move outcomes.

**Independent Test**: Each mechanism is an independent A/B arm against the common baseline over the same pinned windows; any one can be accepted or rejected on its own.

**Per-arm protocol**: as defined in Standing Constraints above — it applies here unchanged. Note especially FR-029 (no model-fit metric may decide an arm) and SC-009 (an improvement reversing sign in W3 is era-specific and rejected).

- [ ] T027 [US3] Create `utils/PhaseGate.py` — an opt-in mixin defaulting **OFF** (`phase_gate_enable = False`), wrapping the `utils/BtcRegimeGate.py` merge pattern rather than rewriting it: declare in `informative_pairs()`, pull via `dp.get_pair_dataframe()` at the strategy timeframe, merge on the tz-aware `date` column, `ffill`, fail open when data is unavailable. Exposes the winning instrument's phase state to the entry chain. Must not modify shared `Framework/` files while the dry run is active (FR-023, research R10)
- [ ] T028 [US3] **M3 (first, highest leverage)** — create `NNNC/NNNC_MLX_PhaseThr.py`, a thin subclass making `prediction_threshold` a function of phase (relax in favorable, tighten in unfavorable). Run across W1/W2/W3 via `run_arm.py`. This is the same lever T023 validated, so a PASS transfers here most directly (FR-019)
- [ ] T029 [US3] **M1 (second, cheapest, strongest prior)** — create `NNNC/NNNC_MLX_PhaseTrend.py` making `entry_trend_filter_enable` / `entry_trend_ema_period` a function of phase rather than a fixed setting. Run across W1/W2/W3 (FR-020)
- [ ] T030 [US3] **M2 (control arm)** — create `NNNC/NNNC_MLX_PhaseGuards.py` making `entry_guard_threshold` and `entry_close_norm_threshold` phase-conditional. Expected near-inert (these guards filter only ~12% of bars); the T021-style liveness check is what distinguishes "near-inert" from "measured null". Run across W1/W2/W3 (FR-021)

**Checkpoint**: three no-retrain arms measured, each with a liveness proof (SC-007, SC-008).

---

## Phase 7: User Story 4 — Phase as a learned input (Priority: P4) — GATED ON PASS **AND** SIGNAL

**Goal**: Let the model learn when phase matters instead of imposing a hand-set rule.

**Entry condition**: only if M3 (T028) or M1 (T029) showed signal. This is the only path requiring retraining, and prior measurement of market-state features as model inputs showed only a modest lift.

- [ ] T031 [US4] **Orthogonality check first** (Constitution IX) — measure phase-feature return correlation against the existing model input set and record it **before** any performance evaluation. A phase feature strongly correlated with an existing one is that feature in disguise and dilutes the set rather than adding to it
- [ ] T032 [US4] Add the phase columns to the model `include_list`, regenerate scalers via `Framework CreateScalers` **before** any comparison (a comparison against stale normalization is void), retrain, and run arms across W1/W2/W3. Decide on realized trading outcomes only, never on any improvement in model-fit metrics (FR-022, US4 acceptance scenarios 1–2)

---

## Phase 8: Polish & Close-out

Runs after either branch of the stop gate.

- [ ] T033 Consolidate findings in `regime/README.md`: instrument catalogue with statuses and reasons, the T017 risk verdict, the trackability verdict, and — on PASS — the per-mechanism arm table with liveness status. On FAIL this was already written at T026
- [ ] T034 Verify the whole study re-runs from a clean checkout with no manual path or parameter edits, per the `quickstart.md` scenarios (SC-011, FR-006)
- [ ] T035 Stage only the files this feature touched (`regime/`, `utils/custom_indicators.py`, and on PASS `utils/PhaseGate.py` plus the `NNNC/NNNC_MLX_Phase*.py` subclasses) and commit directly to master. **Never `git add -A`** — the user edits concurrently and an unexpected staged file is their work in progress
- [ ] T036 Record the outcome in project memory: the verdict, the surviving/eliminated instruments with reasons, and whether the exp4 risk discrimination was confirmed or refuted — so this is not re-litigated in a later session

---

## Dependencies

```
Phase 1 (T001-T002)
   └─> Phase 2 Foundational (T003-T006)
          └─> Phase 3 US1 (T007-T017)          ← MVP, delivers value alone
                 └─> Phase 4 US2 (T018-T024)
                        └─> Phase 5 STOP GATE (T025)
                               ├─ FAIL ─> T026 (terminal) ─> Phase 8
                               └─ PASS ─> Phase 6 US3 (T027-T030)
                                             └─ signal? ─> Phase 7 US4 (T031-T032)
                                                              └─> Phase 8 (T033-T036)
```

**Hard ordering within phases**

- T003 blocks T008 (asset instruments need the estimators)
- T004 blocks T009 (market instruments need the coverage rule)
- T006 blocks T013, T018–T020, and every Phase 6/7 arm
- T012 (lookahead gate) blocks all of Phase B — no instrument proceeds with `n_differing > 0`
- T013 blocks T015 — the dwell threshold must exist before characterisation
- T021 blocks T023 — an INERT arm makes the verdict meaningless
- T025 blocks Phases 6 and 7 absolutely

---

## Parallel Execution Opportunities

**Phase 3 — the three levels are independent** (different instrument families, no shared state):

```
T007 [P] BTC-level baseline
T008 [P] asset-level instruments
T009 [P] market-level instruments
```

**Phase 4 — the two bracketing arms are independent**, and within each, the three windows are independent backtest runs:

```
T018 [P] aggressive 0.55  (W1, W2, W3 in parallel)
T019 [P] conservative 0.80 (W1, W2, W3 in parallel)
T020 [P] constant 0.69     (W2, W3 in parallel; W1 from T013)
```

**Phase 6 — each mechanism's three windows are independent runs.** The mechanisms themselves are ordered M3 → M1 → M2 by leverage, not by dependency, so they *could* run in parallel; keeping them sequential means a clear M3 result can stop the tier early.

**Not parallelisable**: T003 (single file), the whole of Phase 5, and anything crossing the stop gate.

---

## Implementation Strategy

**MVP = User Story 1 (T001–T017).** It delivers a reusable, lookahead-validated phase framework plus a definitive answer on the exp4 risk-discrimination question, and it is valuable standing alone even if everything after it returns a negative.

**Incremental delivery**

1. **T001–T017** — framework and characterisation. Stop and assess: if zero instruments survive, the feature is complete with a negative result.
2. **T018–T025** — the verdict. Roughly a day. This is the cheapest possible test of the whole idea, and it is deliberately front-loaded ahead of every expensive build (Constitution VII).
3. **T026 or T027+** — one branch only.

**Cost asymmetry to keep in view**: Phases 1–5 involve no model retraining and touch one existing file. Phase 7 alone requires a retrain plus scaler regeneration. The gate exists so that the expensive half is never paid for on an unvalidated premise — and the prior here is real: the same trackability test on a different parameter returned NO at every horizon.
