---
description: "Task list for 034 label-type screen for NNNC"
---

# Tasks: Label-Type Screen for NNNC

**Input**: `specs/034-label-type-screen/`: spec.md, plan.md, research.md R1–R11, data-model.md,
contracts/candidate_classes.md (C1), contracts/screen_cli.md (C2), quickstart.md.

**Tests**: Included. The spec requires each of these, and each needs a test:
- control model keys unchanged and the classes that exist today byte-identical (FR-008, SC-003);
- declared label parameters accepted and live (FR-003, FR-008a);
- sizer-twin key equals entry-source key (C1);
- 4-decimal thresholds (FR-017);
- the S1 rule against both references and the survivor rule (FR-010, FR-011);
- judge refusal and stage order (FR-014, FR-018).

**Conventions**:
- All paths are relative to `user_data/strategies/`. Python is `../../.venv/bin/python`. Study
  scripts run as `PYTHONPATH=. ../../.venv/bin/python -m research.label_screen.<script>`; the
  harness runs from the repo root as
  `PYTHONPATH=user_data/strategies .venv/bin/python -m evaluation.run_folds --plan ...`.
- Training, picks and backtests need `set -a; source ~/.freqtrade_env; set +a` and run as ONE
  sequential chain, picks one fold per process. Check `ps` for freqtrade, `run_folds`, `run_arm`,
  `pick` and `train` first. Do not run tests that read fold json while a chain runs.
- Any script scoring a fold strategy outside a backtest calls `strat.ft_bot_start()` (032 lesson).
- Commit to master, staging named files only. Run the full suite and check pytest's exit code
  before every commit:
  `../../.venv/bin/python -m pytest Framework/tests utils/tests SimpleTrigger/tests research/exit_coupling/tests research/held_pair/tests research/candidate_training/tests research/width_relative/tests research/label_screen/tests evaluation -q -p no:cacheprovider`.
- **No training of any `NNNC_LS_*` class before T014 (the pre-registration) is committed.** No
  `ls_*` arm record may exist before it.
- **No late-half (f08–f14) P&L or late separation is read for any candidate except the S2 winner
  (SC-004).**
- Results are at most PASS (PROVISIONAL) (one era).

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Create `research/label_screen/__init__.py` and `research/label_screen/tests/__init__.py`.

---

## Phase 2: Foundational (blocking)

- [X] T002 BEFORE any NNNC or shared-script edit, capture `research/label_screen/tests/fixtures/model_keys_pre034.json`: the 030 model keys (via `Framework.ModelKey.resolve` under `config/config_wf.json`) of `NNNC_A_Raw_Open`, `NNNC_GbbFwdMae_Open`, `NNNC_ARO_f00`, `NNNC_ARO_f07`, `NNNC_W_ARO` and `NNNC_WR_ARO`.
- [X] T003 [P] Write `research/label_screen/candidates.py`: the CandidateLabel table of data-model.md as data (tags CONTROL, GFM, FMAE, TB, MMR, FEXP, QF, LMF, BOG; method ids 27, 28, 0, 1, 3, 12, 2, 4, 24; label_params per research R2 — FMAE `{"max_drawdown": 0.04}`, TB `{"pt": 0.02, "sl": 0.05}`, LMF `{"max_drawdown": 0.04}`, others `{}`; reuse `wf_aro_g24` for CONTROL and `sw_gfm_ctl_b030` for GFM; roles ("sz","es") except BOG ("sz",) and CONTROL (); training order FMAE, TB, MMR, FEXP, QF, LMF, BOG). TB also declares `label_params_sell = {"pt": 0.05, "sl": 0.02}` (research R2: the sell mirror of +2% before −5%). Test in `research/label_screen/tests/test_candidates.py`: tags unique; every declared buy key is in its labeler's signature and survives `TrainingSignals._filter_label_kwargs` with strict keys; every declared sell key is in the `allowed` set of its explicit sell branch in `get_train_sell_signals` (strict keys are not honoured there), or, for the fallback, survives strict filtering; a bogus buy key raises; TB's sell labels on a synthetic path equal "−2% before +5%".
- [X] T004 Write `research/label_screen/build_arms.py` (contract C1): from `plan_wf_aro_g24.json` generate per trained tag `NNNC/NNNC_LS_<TAG>.py/.json` (LabelParamsMixin + NNNC_A_Raw_Open, TRAINING_TYPE, LABEL_PARAMS, LABEL_PARAMS_SELL None, 4-decimal `prediction_threshold`) and `NNNC/NNNC_LS_<TAG>_SZ.py/.json` (WidthSizerMixin + WidthBookMixin + NNNC_LS_<TAG>, WIDTH_CUTS from `NNNC_W_ARO`); for GFM, `NNNC_LS_GFM` (4-dp threshold on `NNNC_GbbFwdMae_Open`) and `NNNC_LS_GFM_SZ`, whose plans take the folds and dates of `sw_gfm_ctl_b030` and the ARO control's 24 h-grace params snapshot (research R4); plans `evaluation/folds/plan_ls_<tag>_sz.json` and `plan_ls_<tag>_es.json` copied from the control plan (same folds, config and params snapshot; new classes, model dirs and arm tags `ls_<tag>_<role>`; no thresholds). `--model GBR` builds the same on `NNNC_GbbRaw_Open` / `fs_gbr_gr` / `NNNC_W_GBR` (used only at T027). Tests in `research/label_screen/tests/test_build_arms.py`: fixture keys of T002 unchanged; each candidate's key differs from the control's and from every other candidate's; `_SZ` key == entry-source key on f00 and f07; GFM twin key == GFM's existing f00 and f07 keys; thresholds load at 4 dp; each candidate's feature column list equals the control's.
- [X] T005 [P] Register the screen in the shared drivers at runtime, one line each, as 032/033 did: `LS_PLANS` in `research/nnnc_framework/signals.py` (`LS_<TAG>` → the training plan `ls_<tag>_es`, from `candidates.table_plans()`), plus a `--folds` filter there (screen reads f00–f07 only). Existing `--models` defaults unchanged. *Implementation note:* the `diagnose.MODELS` registration was dropped: `entries.Model` needs arm records, which do not exist before S2, and S1 only needs the parquet tables, so `separation.py` reads them directly.
- [X] T006 Write `research/label_screen/train.py`: the 032 driver (`research/candidate_training/train.py`) pointed at `plan_ls_<tag>_sz` per `--tag`, folds f00–f07 by default, `--seed`, busy-check extended with its own name; refuses a tag whose S0a status in `audit.json` is not PASS and refuses any late fold unless `verdicts.json` records that tag as the S2 winner. Test in `research/label_screen/tests/test_train.py` (refusals, default folds) with the driver stubbed.

**Checkpoint**: classes, plans and drivers exist; control keys unchanged; suite green. Commit.

---

## Phase 3: User Story 1 — Screen label types without a backtest (Priority: P1) 🎯 MVP

**Goal**: S0a, S0 and S1 for every candidate on f00–f07, and a committed survivor list.

**Independent Test**: `separation.json` ranks every candidate against both references; the control
reproduces 0.072; `survivors.json` follows the rule.

- [X] T007 [US1] Write `research/label_screen/audit.py` (FR-008a, research R1/R2/R10): per candidate (including CONTROL and GFM as reference rows, labels only, never gated) and early fold, load the fold's training window as training does, compute buy and sell labels via the candidate class (`get_train_buy_signals` / `get_train_sell_signals`, peak filter included), the raw BUY–SELL overlap, then the classes after `augment_training_signals` conflict resolution; shares of BUY/SELL/HOLD; per declared extra that differs from the labeler default, the count of labels that change when it is removed (f00 only; extras equal to the default are recorded as `equal_default`); feature columns equal to the control's. Status INCONCLUSIVE if kept BUY share < 0.01 in any fold or a non-default declared extra changes nothing. One fold per process with a `--fold` flag plus a merge step. Tests in `research/label_screen/tests/test_audit.py` on a synthetic frame: shares sum to 1; an identical buy/sell pair (the mfe_mae_ratio case) is reported as full overlap; a no-op non-default extra is flagged and an extra equal to the default is not; the floor rule.
- [X] T008 [US1] Run `build_arms` (ARO) and `audit` for all candidates on f00–f07 (labels only, no model). Write `research/label_screen/audit.json`. Record mfe_mae_ratio's overlap and kept shares. Commit the generated classes, plans and `audit.json`.
- [X] T009 [P] [US1] Write `research/label_screen/pick.py`: `--role sz` reuses `research.width_relative.pick` (validation-tail run starts of the width book, 0.5 share, 4 dp) on `plan_ls_<tag>_sz`; `--role es` calls `evaluation/pick_threshold.py`'s `cmd_pick` at target rate 0.014735 on the training window and stores 4 dp; `--tag CONTROL --role sz` scores the control at the same share for the S1 fairness reference (written to `research/label_screen/thresholds_control_share.json`, not to any control plan). One fold per process; `--verify` checks the loaded threshold and tag-to-decision agreement. Tests in `research/label_screen/tests/test_pick.py` (role dispatch, 4-dp rounding, one-fold refusal, control output path).
- [X] T010 [P] [US1] Write `research/label_screen/separation.py` (FR-010, research R6): generalise `research/width_relative/separation.py` to every candidate with tables: width book simulated once per early fold from the control's tables (`width_twin.json` cuts); per candidate agree = p_buy > its sz threshold (amended at T015: no BUY-argmax condition, AGREE_RULE "p_buy"); gap, agree/disagree n and pp per trade, AUC of p_buy against pp > 0, degenerate folds (agree share 0 or 1). References: pre-registered 0.072 and the control at share. Early half only — no option reads the late half. Tests in `research/label_screen/tests/test_separation.py`: identical trades for every candidate; AUC on a hand case; degenerate-fold counting.
- [X] T011 [P] [US1] Write `research/label_screen/survivors.py` (FR-011): candidates with S0 PASS, ≤ 2 degenerate folds and gap above both references; top 3 by gap, ties by AUC; writes `survivors.json` with every candidate's reason. Tests in `research/label_screen/tests/test_survivors.py` (both-reference rule, cap at 3, tie-break, none-survive case).
- [X] T012 [P] [US1] Write `research/label_screen/instrument.py` (FR-009, FR-015): S0 per candidate and fold (weights present, model key distinct, Spearman(label, bb_width) on rule-eligible training bars for buy and sell, thresholds loaded at 4 dp); for S2 arms (once records exist): sz trade identity with `w_aro` on (pair, open_date, close_date) with ≤ 2 folds differing, capital ratio within ±10%, realised agree share and stake ratio, wallet shortfalls, tag mismatches; es entry-signal rate within ±5% of 0.014735 and trade counts vs control. Refuses if the pre-registration is uncommitted. Tests in `research/label_screen/tests/test_instrument.py` on synthetic records.
- [X] T013 [US1] Write `research/label_screen/judge.py` (FR-012–FR-014, FR-018): staged verdicts per candidate (S0a → S0 → S1 → survivor) and per arm (S2 early pp vs `w_aro` and `wf_aro_g24`; small margin < 1 pp → `--seed-repeat 2`; S3 for the winner with `judge_one` vs control and late pp vs width; GBR repeat). A failed stage leaves later stages `not_reached` and reads no later records. Refuses a dirty/uncommitted pre-registration or any `ls_*` record older than its commit (exempt names allowed). Tests in `research/label_screen/tests/test_judge.py` following `research/width_relative/tests/test_judge.py` (stage order, refusal, seed-2 both ways, two roles, BOG sizer-only, winner selection by largest minimum margin).
- [X] T014 [US1] Write and COMMIT `research/label_screen/preregistration.json` and `.md` (data-model Preregistration): candidates with params (from `candidates.py`, hashed), S0a floor 0.01, agree share 0.5, references, survivor rule, comparators, entry rate 0.014735, stakes, bands, small-margin 1 pp, seed 2, judge_one settings and min_control_trades 200 copied from 032, confirmation route, the S0a outcome (which candidates are trainable). No `NNNC_LS_*` training before this commit.
- [X] T015 [US1] f00 smoke (research R5): `train --tag FMAE --folds f00`; signal table `LS_FMAE_f00`; `pick --tag FMAE --fold f00 --role sz` then `--verify`. Confirm the weights land at a new key and no existing weights change. Any fix is an amendment to the pre-registration before T016. If any `ls_*` arm record was written, set it aside by name (`*.smoke.json`) and list it in `exempt_records`.
- [X] T016 [US1] Training chain: every S0a-PASS trained candidate in pre-registered order on f00–f07, one sequential chain (≈ 11 h). Resumable; on a low-memory kill, report and wait for the user before restarting.
- [X] T017 [US1] Signal tables for every trained candidate on f00–f07 (`research.nnnc_framework.signals --models LS_<TAG>`), confirm GFM and ARO tables exist for f00–f07, then sz picks per candidate and fold and the control-at-share picks (one fold per process); write the sizer thresholds into `plan_ls_<tag>_sz.json` only (entry-source thresholds come from T021); run `--verify` per fold.
- [X] T018 [US1] Run `instrument` (S0) and write `research/label_screen/instrument.json`.
- [X] T019 [US1] Run `separation` (S1) and write `separation.json`. SC-002: the control's gap must reproduce 0.072 within ±0.002; if it does not, stop and diagnose the instrument before reading any candidate.
- [X] T020 [US1] Run `survivors`; commit `instrument.json`, `separation.json`, `survivors.json`.

**Checkpoint**: US1 complete — the label-type table exists even if nothing survives.

---

## Phase 4: User Story 2 — Backtest the survivors (Priority: P2)

**Goal**: S2 verdicts for each survivor in both roles (BOG sizer only).

**Independent Test**: early-half pooled pp for every survivor arm, width book and control, with
staged verdicts.

- [X] T021 [US2] Entry-rate picks (`--role es`) per survivor per early fold, one fold per process; write into `plan_ls_<tag>_es.json`; verify 4-dp load and the achieved rate within ±5% of 0.014735. — **N/A: no S1 survivors (survivors.json empty); not run.**
- [X] T022 [US2] Backtests: `evaluation.run_folds --plan plan_ls_<tag>_<role> --folds f00,f01,f02,f03,f04,f05,f06,f07` per survivor arm, one chain. — **N/A: no S1 survivors (survivors.json empty); not run.**
- [X] T023 [US2] Run `instrument` again for the S2 checks (trade identity, capital, tags, rate) and `judge`; write `verdicts.json`. — **N/A: no S1 survivors (survivors.json empty); not run.**
- [X] T024 [US2] If any S2 margin is under 1 pp: seed-2 repeat — `train --tag <T> --seed 2` on f00–f07, picks, backtests into `_s2` tags. First check for ARO control seed-2 records on f00–f07; none existed at planning, so train and run `wf_aro_g24` at seed 2 on those folds (≈ 1.6 h; the width book is seedless). Then `judge --seed-repeat 2`. — **N/A: no S1 survivors (survivors.json empty); not run.**
- [X] T025 [US2] Commit the S2 records summary, `verdicts.json` and plans. — **N/A: no S1 survivors (survivors.json empty); not run.**

---

## Phase 5: User Story 3 — Confirm the winner (Priority: P3)

**Goal**: late-half verdict for the single best S2 arm, and the GBR repeat on a pass.

- [X] T026 [US3] If an S2 winner exists: `train --tag <T> --folds f08,...,f14`, its tables and picks for its role, backtests on f08–f14, `judge` (S3: `judge_one` vs `wf_aro_g24`, late pp vs `w_aro`, small-margin and seed-2 rules). Commit. — **N/A: no S1 survivors (survivors.json empty); not run.**
- [X] T027 [US3] On an S3 PASS: `build_arms --model GBR` for the winning label, train all 15 folds, picks, backtests against `fs_gbr_gr` and `w_gbr`, `judge` (GBR repeat). Commit. — **N/A: no S1 survivors (survivors.json empty); not run.**

---

## Phase 6: Polish & Cross-Cutting

- [X] T028 SC-003 byte identity: re-run `wf_aro_g24` on f00 and f07 into scratch arm tags and compare trades hashes with the recorded records; record the result in the report.
- [X] T029 Write `research/label_screen/report.md` (decision table per candidate and arm, S0a/S0/S1 tables, S2/S3, harness lessons, provenance) and a pointer section in `research/model_calibration/findings.md`.
- [X] T030 Memory: `project_034_label_screen.md` and its MEMORY.md "Current state" line; update the labels hub if a label-level finding results.
- [X] T031 Full suite (exit code checked), mark every task [X], final commit.

---

## Dependencies & Execution Order

- Phase 1 → Phase 2 (T002 first; T003 before T004–T006) → US1.
- US1 order: T007 → T008 → (T009–T013 in parallel) → T014 → T015 → T016 → T017 → T018 → T019 → T020.
- US2 needs T020 with at least one survivor; if none, skip to Phase 6.
- US3 needs an S2 winner in T023/T024.
- Long runs (T008 audit, T016, T017, T022, T024, T026, T027) are strictly sequential: one chain.

## Parallel Example: User Story 1

```text
T009 pick.py   T010 separation.py   T011 survivors.py   T012 instrument.py   (different files)
```

## Implementation Strategy

- **MVP = US1**: it answers "which label types separate width-book trades better than gbb_path"
  without a backtest.
- US2 and US3 run only on survivors and the winner, so their cost is bounded by S1's outcome.
