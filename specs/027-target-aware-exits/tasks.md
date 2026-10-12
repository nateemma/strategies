---
description: "Task list for 027 target-aware exits"
---

# Tasks: Target-Aware Exits

**Input**: `specs/027-target-aware-exits/` (spec.md, plan.md, research.md R1–R8, data-model.md,
contracts/interfaces.md C1–C7, quickstart.md)

**Tests**: Included. The spec requires:
- switch-off reproduction (SC-002);
- stop invariants in every mode (FR-009);
- early-only value choice (FR-004);
- liveness (FR-010).

**Conventions**:
- All paths are relative to `user_data/strategies/`.
- Python is `../../.venv/bin/python`.
- Backtests need `set -a; source ~/.freqtrade_env; set +a`, and run as ONE sequential chain with
  fresh tags (`run_folds` skips existing records).
- Commit to master, staging named files only.
- Every result is labelled PROVISIONAL (gate 2).
- **US1 is a hard STOP gate.** Phases 4–5 run ONLY if T007 reports at least one SUPPORTED coupling.
  On a STOP, skip to Phase 6 and mark T008–T021 `[-]` with the reason.
- **Couplings not SUPPORTED are not built.** In Phases 4–5, each task touches only the SUPPORTED
  couplings. A sub-item for an unsupported coupling is marked `[-]` in place.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Create `specs/027-target-aware-exits/baseline.md`. Record:
  - the commit sha;
  - framework fingerprints for `NNNC_MLX_Open` (arm D), `SimpleTrigger_Gbb`,
    `SimpleTrigger_BreakoutGbb_MomExit` (`st_bgo_mx`), `SimpleTrigger_RevOnlyVolreg2`
    (`st_book_rev2`), `SimpleTrigger_FamCombined` and `SimpleTrigger_RegGateVolreg2`;
  - the per-fold `trades_hash` (all 15) of `st_bgo_mx` and `st_book_rev2`;
  - the reference arms' exit values (research R1 table), read from the resolved classes, not
    copied from the docs.

  Create the `research/exit_coupling/__init__.py` and `research/exit_coupling/tests/__init__.py`
  packages.

---

## Phase 2: Foundational — path metrics and replays (blocking)

- [X] T002 [P] Write `research/exit_coupling/tests/test_paths.py` (contract C1, research R2).
  Synthetic OHLC cases:
  - (a) target touch on the first bar whose HIGH ≥ open×1.02; none if never;
  - (b) MAE before touch = the lowest LOW strictly before the touch bar;
  - (c) a bar whose high touches the target AND whose low breaches the stop in force is flagged
    ambiguous;
  - (d) the post-grace stop in force is the tightest entry-anchored `atr_stop_fraction` since grace
    expiry (ratchet), and is the grace level inside the window;
  - (e) the stop-breach bar is the first bar whose LOW is at or below the stop in force;
  - (f) a horizon of 48 bars is honoured exactly; the max-hold window is 960 bars;
  - (g) replays:
    - C1 (deeper stop) exits later or not at all;
    - C3 "close" exits at bar 49's open;
    - C3 "breakeven" stop at entry from bar 49;
    - C5 lock activates only from the bar AFTER a closed-candle touch, never the touch bar itself
      (research R4);
    - C4 extends the grace window to 48 bars at the given depth;
    - C2 takes profit at `k × atr_pct` of the signal bar, clamped to [1%, 5%] net;
  - (h) replay P&L is net of fees through `ExitPolicy.net_from_price_ratio`;
  - (i) replay order per bar, as freqtrade runs it (research R2):
    - `custom_exit_decision` on the OPEN, with the 1-hour discretionary veto and the effective
      unclog_12h (24 h when a hold window is given);
    - then the stop on the LOW;
    - then ROI 5%, detected on the HIGH and filled at the target, and off when the trade is flagged
      ROI-suppressed;
  - (j) the replay calls `ExitPolicy` for every exit rule: no exit threshold literal appears in
    `paths.py` (source inspection).
- [X] T003 Create `research/exit_coupling/paths.py`:
  - `path_metrics(...)` and `replay(...)` (contract C1), pure numpy;
  - every exit rule via `utils.ExitPolicy` (`custom_exit_decision`, `discretionary_exit_allowed`,
    `atr_stop_fraction`, `price_ratio_for_net`, `ExitParams`), no copy;
  - the C2–C5 rule hooks are inputs to `replay`, not changes to `ExitPolicy` (Phase 3 must not edit
    the framework). The C3 hook keeps the live order (research R4): a take-profit, unclog_12h or
    unclog_1d decision wins; C3 "close" applies only when `custom_exit_decision` returned None or
    max_hold. T008 later pins the live implementation to the same order;
  - make T002 pass.

**Checkpoint**: path machinery tested on synthetic data.

---

## Phase 3: User Story 1 — feasibility gate (Priority: P1) 🎯 MVP, STOP gate

**Goal**: measurements (a)–(d) per family × half, and a SUPPORTED/NOT SUPPORTED verdict per coupling
under the pre-declared rules (research R3).

**Independent Test**: `feasibility.md`/`.json` exist, every rule is evaluated in both halves, and
the exit code reflects the verdict.

- [X] T004 [P] [US1] Write `research/exit_coupling/tests/test_feasibility.py`:
  - (a) grid choices (C1 p, C3 action, C5 λ, C2 k) read ONLY f00–f07 rows: a planted late-only
    effect leaves every choice unchanged;
  - (b) a rule holding in one half only is NOT SUPPORTED;
  - (c) C2 is N/A for a family without a take-profit;
  - (d) ambiguous trades are excluded from (a)/(b) and counted;
  - (e) exit codes 0 / 3 / 2;
  - (f) a null-replay fidelity below 90% in either family exits 2 with no verdict;
  - (g) C2 and C4 need a positive replay gain as well as their descriptive condition (research R3);
  - (h) the C4 depth is chosen from {−27%, early p95 MAE capped at −27%} on early rows only.
- [X] T005 [US1] Create `research/exit_coupling/feasibility.py` (contract C1, research R2/R3).
  Inputs:
  - trades from the archives of `st_bgo_mx` and `st_book_rev2`: open_rate, close_rate, dates,
    exit_reason, fold (via `evaluation.pool` helpers);
  - prices from `research.clfstudy.data.load_pair`, with `atr_pct_roll` and rvol from the same
    populator and `ExitPolicy.rolling_rvol` logic the strategy uses;
  - exit values from `utils.FamilyExits.class_level_exit_params` for each reference class (no
    instance), plus its `HOLD_WINDOW_HOURS` and `minimal_roi`;
  - per trade, the ROI-suppressed flag (research R2).

  It computes (a)–(d) and the C1–C5 replays at each grid value (C2 at its rule value), chooses values on f00–f07,
  applies the R3 rules in both halves, and writes `feasibility.md` + `feasibility.json`. Make T004
  pass.
- [X] T006 [US1] Fidelity gates, inside `feasibility.py` and reported, BEFORE any verdict:
  - **null replay** (research R2): with no coupling, the share of trades whose recorded exit reason
    AND exit bar the replay reproduces, per family, with a disagreement table by exit reason;
  - **stop model**: for archive stop-loss exits, the recomputed stop in force vs the recorded exit
    price (agreement share, median absolute gap in pp).

  Either below 90% in either family: exit 2, STOP, and diagnose the replay before T007 (constitution
  "audit the instrument").
- [ ] T007 [US1] Run:
  ```
  python -m research.exit_coupling.feasibility --momentum st_bgo_mx --reversion st_book_rev2 \
      --out specs/027-target-aware-exits/feasibility.md
  ```
  Record the verdicts. **STOP gate:** with no coupling SUPPORTED, go to Phase 6. Commit.

**Checkpoint**: answer to "is the exit inconsistent with the targets, and where?".

---

## Phase 4: User Story 2 — switches in the shared exit (Priority: P2) — ONLY if T007 supported ≥ 1

**Goal**: each SUPPORTED coupling is a default-off switch in the one exit definition, inert when
off.

**Independent Test**:
- recorded fingerprints are unchanged;
- f00 switch-off reproduction of both reference arms;
- invariants hold;
- each switch has unit-tested behaviour.

- [ ] T008 [P] [US2] Write `utils/tests/test_exit_policy_coupling.py` (contract C2):
  - `ExitParams` validation (domains; C1 ≥ −0.10; a switch without targets refused);
  - `custom_exit_decision` with every new input `None` is identical to today on the existing
    characterisation fixture;
  - C2 `take_profit_override` used and clamped to [0.01, 0.05];
  - C3 returns `horizon_close` only at `bars_open >= horizon` and not touched;
  - C3 order: a take-profit, unclog_12h or unclog_1d decision on the same bar wins over C3, and C3
    wins over max_hold (research R4); the same cases give the same answer through the T003 hook;
  - `simulate_forward` raises if any coupling field is set.
- [ ] T009 [US2] Edit `utils/ExitPolicy.py` (contract C2):
  - add the optional fields, `CLOSE_HORIZON`, the keyword-only decision inputs,
    `coupled_stop_floor` and `lock_price_ratio`, and the `simulate_forward` refusal;
  - leave `SIMULATION_LOGIC_VERSION` unchanged;
  - make T008 pass.
- [ ] T010 [P] [US2] Extend `utils/tests/test_stop_invariants.py` (contract C4):
  - a C1 stop at −10% passes;
  - a −11% stop under a −10% `stop_min_depth` bound raises;
  - a C5 lock / C3 break-even on a trade already below the lock returns `STOP_AT_MARKET_RATIO`;
  - C4's longer grace does not trip the direction check.
- [ ] T011 [US2] Edit `utils/StopInvariants.py`: the bound floor includes `stop_min_depth` when set.
  Make T010 pass.
- [ ] T012 [US2] Edit `Framework/BaseStrategy.py` (contract C3):
  - `_exit_params` maps `EXIT_TARGETS` + coupling fields;
  - `_target_touched` from closed candles of the analysed dataframe (never `trade.max_rate`);
  - in `custom_stoploss`: C4 grace length, C1 floor in `_resolve_normal_stop`, C3 break-even and
    C5 lock (never a positive ratio);
  - in `custom_exit`: `bars_open`, `target_touched`, and C2 `take_profit_override` from the
    signal-bar `atr_pct_roll`.

  Every branch is skipped when its field is `None`. No family or trigger names. Add a
  `utils/tests/test_base_strategy_coupling.py` covering each branch with a stub dataprovider and a
  stub trade, including a lock that must NOT activate on the touch candle itself.
- [ ] T013 [P] [US2] Edit `evaluation/fingerprint.py` (contract C5):
  - `EXIT_TARGETS`/`EXIT_COUPLING` are optional members;
  - new ExitParams fields are dropped from the family hash while `None`.

  Write `evaluation/tests/test_fingerprint_027.py`: every class listed in T001 hashes to its
  baseline sha.
- [ ] T014 [P] [US2] Edit `regime/run_arm.py` (contract C6) to record `exit_coupling`. Create
  `evaluation/coupling_liveness.py` (research R5b):
  - inputs are pooled sets only;
  - per half: paired trades with a different exit, trades unique to either side, folds with a
    different `trades_hash`, `horizon_close` count;
  - a test in `evaluation/tests/test_coupling_liveness.py`: identical sets count 0; one moved exit
    counts 1 in its half.
- [ ] T015 [US2] Switch-off reproduction (SC-002):
  - plans `plan_st_cpl_off_mom.json` / `plan_st_cpl_off_rev.json` run the unchanged reference
    classes under fresh tags, all 15 folds;
  - every fold's `trades_hash` must equal the recorded reference arm's;
  - `coupling_liveness` against the reference must be 0 everywhere;
  - arm D is covered by T013's unchanged sha;
  - run the full related test suites (`utils/tests`, `evaluation/tests`, `SimpleTrigger`,
    `research/exit_coupling`).

  Commit.

**Checkpoint**: switches exist, are inert when off, and are invariant-safe.

---

## Phase 5: User Story 3 — walk-forward judgement (Priority: P3) — ONLY if T007 supported ≥ 1

**Goal**: each SUPPORTED coupling is judged per family, late, against its own reference arm.

**Independent Test**: per arm, non-zero liveness in both halves; the late P&L verdict and risk line
are in `decisions.md`.

- [ ] T016 [US3] Create the arm classes `SimpleTrigger/SimpleTrigger_<Ref>_C<n>.py` (contract C7),
  one per SUPPORTED coupling × applicable family:
  - each sets `EXIT_TARGETS = {"gain": 0.02, "horizon_bars": 48}` and ONE coupling, with the value
    copied from `feasibility.json` (comment citing it);
  - C4 is arm-wide (grace length and depth);
  - `--dry-run` the plans `plan_st_cpl_<ref>_c<n>.json` to confirm the classes.
- [ ] T017 [US3] Run the arms sequentially (one chain), 15 folds each. Run
  `evaluation.coupling_liveness` for each against its reference: zero changed trades in a half →
  INERT, reported and not judged.
- [ ] T018 [US3] Judge each arm with `halves.split` (`PYTHONHASHSEED=0`) against its reference arm:
  - late P&L verdict (adoption = IMPROVED);
  - late risk line;
  - whole-span line;
  - results appended to `specs/027-target-aware-exits/decisions.md`.
- [ ] T019 [US3] Combination: for a family with ≥ 2 adopted couplings, create and run one arm with
  all of them, and judge it the same way (FR-012). Otherwise mark `[-]`.
- [ ] T020 [US3] Per-family conflict check (US3 scenario 2): a coupling adopted for one family and
  WORSENED for the other is recorded as per-family. For C4 (arm-wide grace), the recommendation is
  conditional on both families.
- [ ] T021 [US3] Commit arm classes, plans and `decisions.md`.

---

## Phase 6: User Story 4 — report & polish (Priority: P4)

- [ ] T022 [US4] Write `specs/027-target-aware-exits/report.md`:
  - SUMMARY;
  - feasibility table ((a)–(d) per family × half, the rules and verdicts, the fidelity check);
  - switches built (or the STOP);
  - per-arm verdicts by half with risk lines;
  - the combination;
  - the recommended exit NNNC should inherit, per family;
  - fix-oriented next steps.

  Every figure is labelled PROVISIONAL.
- [ ] T023 [US4] Memory:
  - a new `project_027_target_aware_exits.md`;
  - update the `MEMORY.md` "Current state" line and the `index_exits.md` hub;
  - a feedback memory if a non-obvious lesson emerges (e.g. the closed-candle touch rule).
- [ ] T024 Full test gate: `utils/tests`, `evaluation/tests`, `research/exit_coupling`,
  `research/outcomes/tests`, `SimpleTrigger`, `regime`. Lint the added lines only. Mark tasks `[X]`.
  Final commit.

---

## Dependencies & Execution Order

- T001 → T002 → T003 → T004/T005 → T006 → **T007 (STOP gate)**.
- Phase 4 runs only if T007 supports ≥ 1 coupling:
  - T008 → T009;
  - T010 → T011;
  - T009 + T011 → T012;
  - T013 ∥ T014 (separate files) after T009;
  - T015 last.
- Phase 5 runs after T015: T016 → T017 → T018 → T019 → T020 → T021.
- Phase 6 always runs.

### Parallel opportunities

- T002 ∥ T004 (test files).
- T008 ∥ T010 (test files).
- T013 ∥ T014.
- No code edit (T009–T014) may run during a backtest (T015, T017, T019).

## Implementation Strategy

**MVP = Phases 1–3:** the feasibility verdict with no backtests and no framework edits. That is the
cheapest answer to "does the exit kill trades on their way to the target, and which coupling would
fix it?". The framework is edited only for SUPPORTED couplings, and only after that answer.
