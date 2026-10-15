---
description: "Task list for 035 new-input screen for NNNC"
---

# Tasks: New-Input Screen for NNNC

**Input**: `specs/035-new-input-screen/`: spec.md, plan.md, research.md R1–R13, data-model.md,
contracts/screen_cli.md, contracts/new_input_mixin.md, quickstart.md.

**Tests**: Included. The spec requires each of these, and each needs a test:
- the mixin is byte-identical when off, and existing keys are unchanged (FR-014, FR-022, SC-003);
- causality by truncation, including a planted failing control (FR-005);
- alignment and timestamp-unit rules (R3, R4);
- missing-value and flag rules (R5);
- the S1 statistic and the survivor rule (FR-009, FR-010);
- judge refusal and stage order (FR-020, FR-021).

**Conventions**:
- **Paths and interpreter:** paths are relative to `user_data/strategies/`, and Python is
  `../../.venv/bin/python`. Study tools run as
  `PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.<tool>`.
- **Fold runner:** `run_folds` runs from the repo root as
  `PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/evaluation/run_folds.py --plan user_data/strategies/evaluation/folds/<plan>.json`.
- **Chains:** training, picks and backtests need `set -a; source ~/.freqtrade_env; set +a` and
  run as ONE sequential chain, picks one fold per process. Check `ps` first. On a low-memory kill,
  report and wait for the user.
- **Downloads** go to `user_data/data/external/` (never committed). They are parsed with
  `zipfile`/pandas and never executed, and are not run while a training or backtest chain is
  running.
- **Commits:** to master, staging named files only. Run the full suite and check pytest's exit
  code before every commit:
  `../../.venv/bin/python -m pytest Framework/tests utils/tests SimpleTrigger/tests research/exit_coupling/tests research/held_pair/tests research/candidate_training/tests research/width_relative/tests research/label_screen/tests research/control_sizer/tests research/new_inputs/tests evaluation -q -p no:cacheprovider`.
- **Pre-registration first:** no S1 score is computed before T018 (the pre-registration) is
  committed, and no `ni_*` arm record may exist before it.
- **Late half:** no late-half (f08–f14) P&L is read for any input except the S2 winner (SC-004).
- **Results** are at most PASS (PROVISIONAL), since there is one era.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Create the package `research/new_inputs/__init__.py` and `research/new_inputs/tests/__init__.py`, with `research/new_inputs/tests/conftest.py` holding a small synthetic fixture:
  - two pairs of 15-minute OHLCV with gap-filled bars;
  - matching global klines in both ms and µs units;
  - 5-minute metrics starting mid-span.
- [X] T002 Add `research/new_inputs/tests` to the full-suite command in this file and in `AGENT_GUIDE.md` (the Environment / tests line). *(AGENT_GUIDE already uses the generic `research/<study>/tests` form; no edit needed.)*

---

## Phase 2: Foundational (blocks all stories)

- [X] T003 [P] `research/new_inputs/inputs.py`: the declared input table (data-model "Input"; research R6), with the 17 tags, family, source, windows, norm (`z`/`c`), missing_flag (positioning only) and live_source (R11).
  - Provide `columns(tag)` (`ni_<tag>` plus `_missing` when flagged), `TABLE_SHA` (a hash of the table) and `PAIRS` (the 10 `config_wf` pairs).
- [X] T004 [P] `research/new_inputs/sources.py`: archive parsers and alignment (R3, R4, R5).
  - Kline parsing with ms/µs detection by magnitude, and a header probe for premium-index files.
  - Metrics parsing.
  - `align_kline(src, dates)` aligns on the same open stamp.
  - `align_snapshots(src, dates)` is a backward as-of merge with `create_time + 5m <= date + 15m`.
  - Gap-filled Binance.US bars (volume 0) give NaN order flow.
- [X] T005 [P] Tests in `research/new_inputs/tests/test_sources.py` for:
  - ms and µs files give the same dates;
  - a snapshot stamped at bar close + 1 s is not used for that bar;
  - same-stamp klines align;
  - gap bars give NaN, never 0.
- [X] T006 `research/new_inputs/build.py`: one builder per raw definition in R6, `z` and `c` normalisation (2880-bar trailing window, min_periods 960, clip ±5, /5), and positioning flags.
  - Market state uses `utils/RegimeState.py` `volreg`/`breadth` unchanged, excluding gap-filled pairs per bar with MIN_PAIRS 6.
  - BTC ADX via talib; BTC data from `user_data/data/binanceus/BTC_USDT-15m.feather`.
  - Writes `user_data/data/new_inputs/<PAIR>.parquet` and `build_meta.json` (builder sha, `TABLE_SHA`, manifest sha).
  - A `build_pair(pair, sources, cut=None)` entry point serves the truncation test.
- [X] T007 [P] Tests in `research/new_inputs/tests/test_build.py`:
  - every value is in [−1, 1] or NaN;
  - flags are 1 exactly where the raw positioning is absent;
  - MS_VR equals `RegimeState.volreg` applied to the fixture frames with gap-filled bars removed, then normalised;
  - a positioning flag is 1 during the z warm-up as well as before the source starts;
  - a hand-computed OF_TBG on 4 bars.
- [X] T008 `research/new_inputs/causality.py` (R12):
  - 200 shared cut times, 2021-03..2026-09-07;
  - at each cut, truncate every pair's sources at the cut's close, rebuild all pairs once, and require every input of every pair at the cut bar to equal the full build (atol 1e-12, NaN = NaN);
  - a planted centred OF_TBG variant must FAIL;
  - writes `research/new_inputs/causality.json`.
- [X] T009 [P] Tests in `research/new_inputs/tests/test_causality.py`: the harness passes every fixture input and fails the planted centred control.
- [X] T010 `NNNC/_new_inputs.py` `NewInputMixin` per contracts/new_input_mixin.md:
  - `NEW_INPUTS = ()`, `MODEL_KEY_ATTRIBUTES = ("NEW_INPUTS",)`;
  - `add_additional_indicators` calls `super()` first, then left-merges the declared columns from the cache on `date`;
  - refuses a missing cache or a stale `TABLE_SHA`;
  - never mutates parent lists.
- [X] T011 [P] Tests in `research/new_inputs/tests/test_mixin.py`:
  - **off:** the mixin over `NNNC_A_Raw_Open` with `NEW_INPUTS = ()` returns the parent's frame unchanged, with the same include_list and pre_normalized_columns; every pre-034 model key is unchanged (contract amended: a mixin class cannot share the parent's key under 030 method hashing);
  - **on:** a test subclass with one input has a different key, and `include_list` gains exactly its columns, which are also in `pre_normalized_columns`;
  - `saved_data/main_scaler.pkl`'s 6 feature names are a subset of the non-pre-normalised columns in both cases;
  - the merge keeps the row count on a frame with gaps;
  - the stale-cache refusal;
  - end-to-end causality (constitution V): for one fixture pair, the merged column at a cut equals the value from a cache built on sources truncated at that cut.
- [X] T012 [P] `research/control_sizer/pick.py`: optional `--plan <name>` (default `plan_cs_aro.json`, unchanged behaviour). Add a test in `research/control_sizer/tests/test_control_sizer.py` that the default path is unchanged.
- [X] T013 [P] `research/nnnc_framework/signals.py`: runtime `NI_<TAG>` registration, as `LS_` (034), with no change to the existing registrations. Add a test that existing model names resolve as before.
- [X] T013a [P] `research/new_inputs/trades.py`: `trades()` rebuilds the 412 early width-book trades (f00–f07 only) with 034's S1 instrument, and asserts the count and the control-at-share gap 0.1004 ± 0.002 before returning. Each trade carries its signal date, bb_width and the control's p_buy. Add a test in `research/new_inputs/tests/test_trades.py` that every returned trade's fold is in f00–f07.

**Checkpoint**: tests green; byte identity when off proven (T011).

---

## Phase 3: User Story 1 - Screen new inputs without training (Priority: P1) 🎯 MVP

**Goal**: an S0/S1 table for all 17 inputs and a committed survivor list, with no training.

**Independent test**: `screen.json` has a row per input; the reproduction gate passed;
`survivors.json` is committed.

- [X] T014 [US1] `research/new_inputs/fetch.py` (contract `fetch`; research R7):
  - data.binance.vision spot monthly and daily 15-minute klines (10 pairs + BTC), UM daily metrics 2021-12-01..2026-09-07, and UM monthly premium-index 15-minute klines, each with its `.CHECKSUM` verified;
  - Binance.US `/api/v3/klines` paged from listing at 1000 bars per page, throttled;
  - resumable, with `manifest.json`;
  - refuses while a chain runs.

  Add a test in `research/new_inputs/tests/test_fetch.py`, offline with a fake opener: checksum mismatch → one refetch → coverage gap; already-verified files are skipped.
- [X] T015 [US1] Run `fetch` for every source; record the counts and any coverage gaps in `research/new_inputs/fetch_log.md`.
- [X] T016 [US1] Run `build`, then `causality`. Every input must PASS and the planted control must FAIL. An input that fails is fixed, or declared INCONCLUSIVE in the pre-registration with its reason, before T018.
- [X] T017 [US1] `research/new_inputs/coverage.py` and its run:
  - per input, pair and fold bar coverage, and coverage on the 412 early trades (`research.new_inputs.trades.trades()`, T013a);
  - the R11 live-feasibility record;
  - writes `coverage.json` (S0).
- [X] T018 [US1] Pre-registration `research/new_inputs/preregistration.json` / `.md`:
  - input table and `TABLE_SHA`, alignment rules (R3), missing rules (R5) and normalisation (R6);
  - S1 statistics and cutoffs (coverage ≥ 0.90, below which the input is INCONCLUSIVE; |ρ_bb_width| ≤ 0.5; |partial| ≥ 0.10; sign ≥ 6/8; top 3 by |partial|);
  - S2 rule (gap > 0.1004 with ≤ 2 of 8 folds degenerate; winner = largest gap; margins recorded; S2 selects, S3 decides);
  - S3 rule (judge_one vs `cs_aro` AND `w_aro`, with `wf_aro_g24` reported; small margin under 1 pp needs seed 2), plus the rule maps copied from `research/control_sizer/preregistration.json`;
  - fold sets, the causality and coverage outcomes, the R1 scaler exposure note, and the multiple-testing note (R13);
  - the 035 amendment (FR-013).

  **Commit it before T020.**
- [X] T019 [US1] `research/new_inputs/screen.py` (FR-008, FR-009; research R8):
  - it takes the trades from `research.new_inputs.trades.trades()` (T013a) and attaches every `ni_*` value at the signal bar;
  - pooled and per-fold partial Spearman (rank residuals on [1, rank bb_width, rank p_buy]), Spearman with bb_width and with p_buy, sign_folds, coverage, and split gap;
  - refuses unless the pre-registration is committed and clean;
  - writes `screen.json`.

  Add a test in `research/new_inputs/tests/test_screen.py`: partial Spearman matches a hand-computed case, and is 0 for an input equal to bb_width. The screen must refuse a trade set containing any fold outside f00–f07 (FR-011).
- [X] T020 [US1] Run `screen`.
- [X] T021 [US1] `research/new_inputs/survivors.py` (FR-010): mechanical rule with a reason per input, and low coverage giving INCONCLUSIVE. Add a test in `research/new_inputs/tests/test_survivors.py` covering coverage (INCONCLUSIVE), orthogonality, magnitude, sign and top-3 ranking. Run it and commit `causality.json`, `coverage.json`, `screen.json` and `survivors.json`.

**Checkpoint**: US1 answers "does any new input rank width-book trades beyond width and the model?" with no training.

---

## Phase 4: User Story 2 - Train the survivors into the model (Priority: P2)

**Goal**: per survivor, ARO with that one input added, scored by the S2 gap against 0.1004.

**Independent test**: `separation.json` has a gap per survivor and the S2 winner (or none).

- [X] T022 [US2] `research/new_inputs/build_arms.py`: — **N/A: no S1 survivors (survivors.json empty); not run.**
  - per survivor, generate `NNNC/NNNC_NI_<TAG>.py`/`.json` (`NewInputMixin` + `NNNC_A_Raw_Open`, `NEW_INPUTS`, new include_list and pre_normalized_columns lists; `.json` copied from the control with strategy_name changed) and `NNNC_NI_<TAG>_SZ` (the `NNNC_CS_ARO` bases + mixin);
  - plans `evaluation/folds/plan_ni_<tag>.json` and `plan_ni_<tag>_sz.json`, copied from `plan_cs_aro.json` with class, arm-tag and run-id changes and thresholds removed.

  Add a test in `research/new_inputs/tests/test_build_arms.py`:
  - the twin key equals the entry key and differs from the control's;
  - the params snapshot equals the control's;
  - `HELD_PAIR_ADD_RESOLVED` equals the width book's for `_SZ`;
  - existing model keys are unchanged (034's fixture `research/label_screen/tests/fixtures/model_keys_pre034.json`).
- [X] T023 [US2] `research/new_inputs/train.py`: the 032 driver with a runtime registration of `NI_<TAG>` (as `research/label_screen/train.py`). It refuses late folds unless `verdicts.json` names the tag as the S2 winner. — **N/A: no S1 survivors (survivors.json empty); not run.**
- [X] T024 [US2] Smoke run: train the first survivor on f00 only. Confirm the training frame contains its column(s), non-constant, the new key directory exists, and the control's key directory is untouched. Then run the chain: every survivor on f00–f07. — **N/A: no S1 survivors (survivors.json empty); not run.**
- [X] T025 [US2] Signal tables `NI_<TAG>` f00–f07, then sizer picks and `--verify` per fold (`research.control_sizer.pick --plan plan_ni_<tag>_sz`), one fold per process. — **N/A: no S1 survivors (survivors.json empty); not run.**
- [X] T026 [US2] `research/new_inputs/separation.py` (FR-015): 034's S1 scoring on the `NI_<TAG>` tables with the stored thresholds, the gap per survivor against 0.1004 with ≤ 2 of 8 degenerate folds, and the winner = the largest passing gap. Add a test for the winner rule. Run it, write `separation.json`, and commit. — **N/A: no S1 survivors (survivors.json empty); not run.**

---

## Phase 5: User Story 3 - Backtest the winner as the sizer (Priority: P3)

**Goal**: a judge_one verdict for the winner against `cs_aro` and `w_aro`.

- [X] T027 [US3] If there is an S2 winner: — **N/A: no S1 survivors (survivors.json empty); not run.**
  - `train --tag T --folds f08,...,f14`, its tables, and picks plus verify on f08–f14;
  - `run_folds` for `plan_ni_<tag>_sz` on all 15 folds from the repo root.
- [X] T028 [US3] `research/new_inputs/instrument.py` (FR-019): the control-sizer instrument reading the `NI_<TAG>` tables. Run it. — **N/A: no S1 survivors (survivors.json empty); not run.**
- [X] T029 [US3] `research/new_inputs/judge.py` (FR-018, FR-020, FR-021): staged verdicts for every input; for the winner, judge_one vs `cs_aro` and `w_aro` (both must pass), `wf_aro_g24` reported, and the small-margin rule. It refuses an uncommitted pre-registration or stale `ni_*` records. — **N/A: no S1 survivors (survivors.json empty); not run.**

  Add a test in `research/new_inputs/tests/test_judge.py` for stage order, the two-comparator rule, small margin and refusal. Run it and write `verdicts.json`.
- [X] T030 [US3] If any margin is under 1 pp: the seed-2 repeat (winner at seed 2; ARO seed-2 weights for `cs_aro` on 15 folds, then picks and backtests), then `judge` again. Commit. — **N/A: no S1 survivors (survivors.json empty); not run.**

---

## Phase 6: Polish

- [X] T031 SC-003: re-run `wf_aro_g24` f00 and f07 with `HELD_PAIR_ADD = None` declared, as in 034's T028, into scratch tags. The trade hashes must equal the records. Delete the scratch classes afterwards and record the result.
- [X] T032 Write `research/new_inputs/report.md`: decision table per input (S0, S1, S2, S3), coverage and live feasibility, the scaler exposure note (R1), the multiple-testing note, harness lessons and provenance. Add a pointer section in `research/model_calibration/findings.md`.
- [X] T033 Memory: `project_035_new_input_screen.md`, its MEMORY.md "Current state" line, and the features hub line (`index_features.md`).
- [X] T034 Full suite (exit code checked), mark every task [X], final commit.

---

## Dependencies

- Setup → Foundational (T003–T013a) → US1 (T014–T021) → US2 (T022–T026) → US3 (T027–T030) → Polish.
- **T018 (the pre-registration commit) precedes T020** (the first S1 read).
- T021's commit precedes T022.
- US2 runs only with survivors, and US3 only with an S2 winner. Otherwise their tasks are marked N/A with the reason.

## Parallel opportunities

- **Foundational code and tests** touch separate files: T003, T004, T005, T007, T009, T011, T012, T013 and T013a.
- **Never in parallel:** training, picks and backtests (one chain), and downloads while a chain runs.

## Implementation strategy

MVP = US1: data, causality and the no-training screen answer the core question at the cost of the
downloads alone. US2 and US3 run only if their gate passes.
