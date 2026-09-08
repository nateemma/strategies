---

description: "Task list for 006 Oracle-Instrumented Exit Diagnosis"
---

# Tasks: Oracle-Instrumented Exit Diagnosis

**Input**: Design documents from `specs/006-oracle-exit-diagnosis/`

All commands run from the repository root `/Users/philprice95/projects/freqtrade`.
Paths below are relative to `user_data/strategies/` unless shown otherwise.

**Tests**: No unit tests. This is a measurement feature; correctness is enforced
by the Phase 2 instrument check, trade-hash liveness, refusal of failed or stale
records, and paired resolution.

## Standing conventions (apply to EVERY evaluation task)

**Per-arm protocol**: confirm `returncode` 0 — a failed or stale record is not
evidence (FR-024); compare `trades_hash` against the arm's control, identical
means INCONCLUSIVE and a zero-trade arm is inconclusive rather than a zero return
(FR-020); judge on **W1 only** and report W2/W3 labelled in-sample (FR-014) AND
with their **net/gross ratio**, not merely their trade count (FR-015, R7); never
accept on a classifier metric (FR-017); every arm carries its own `.json`
(FR-021); every perfect-entry figure is a **LOOKAHEAD CEILING** and nothing
measured with one is promotable (FR-019).

**Resolution is PAIRED BY TRADE for exit arms (FR-016, R6).** These arms share
their entries and differ only in outcome, so bootstrap the **per-trade P&L
difference** over matched `(pair, open_date)` keys. The membership-based
symmetric-difference statistic returns ZERO differing trades here and is the
WRONG instrument; use it only where the entry set actually changes.

**W3 cannot carry a sub-threshold conclusion (R7).** NNMT net/gross is W1 29.8%,
W2 18.7%, **W3 7.5%** — W3's four worst trades exceed the entire window's profit
and it has disagreed with W1 on three consecutive exit-side changes for that
reason alone (`regime/W3_INSTABILITY.md`).

**Arms run SEQUENTIALLY, never concurrently (FR-022).** Two concurrent jobs
previously produced load 60 on 14 cores and finished slower than serialising.

**No `Framework/` modification (FR-023).** Every lever is a class attribute or a
runtime parameter (R4). A live dry run imports `BaseStrategy`.

**The entry guards are NEVER opened (FR-025).** Measured: NNMT −34.15pp on W1;
NNNC's open gate carries +17.8pp of headroom and still loses 5–6pp of realised
return.

**Two seeds via `MLX_SEED` wherever a MODEL is involved (FR-018)** — Phase 6
only, and using the seed pairs that ALREADY EXIST, so **the feature as planned
spends ZERO retrains (R9)**. The ONE exception is recorded at T022a: NNNC's
closed-gate production has a single model and cannot be seed-paired from disk, so
a promotion there — and only a promotion there — would cost one retrain.

**Hazards**: `get_model_path` keys off `self.__class__.__name__`, so a pure
re-parameterisation with no model of its own TRAINS INSIDE THE BACKTEST, fitting
the evaluation window and returning different numbers with NO error.
`NNMTStrategy.get_trading_classes` writes debug columns into `self.dbg_curr_df`,
fixed 2026-09-07 so the attribute is assigned before its writers — do not
reintroduce. `test_strat.sh` needs the venv on `PATH`. `TaskStop` does not kill
process trees — use `pkill -9 -f`. **Read configuration ONE STRATEGY PER
PROCESS** and call `ft_load_hyper_params()`, or the values are raw defaults
(R10).

**Already settled by research.md — do NOT redo**: R1 (prior exit arms are stale,
re-run rather than rebuild), R2 (compliance is a bounded task), R3 (test the
geometry first), R4 (levers reachable without a Framework change), R5 (joint
bound then one-at-a-time), R6 (paired by trade), R7 (net/gross), R8 (an oracle
win is not evidence for), R9 (zero retrains), R10 (one strategy per process).

---

## Phase 1: Setup

- [X] T001 Record the measured baselines this feature is judged against in `specs/006-oracle-exit-diagnosis/baseline.md`: NNMT real +10.708% (176tr), oracle@0.02 +8.733% (145tr), oracle@0.007 +6.150% (155tr); NNNC real +10.446% (95tr), oracle@0.02 +13.934% (85tr). Add each window's net/gross ratio (FR-014, FR-015)
- [X] T002 [P] Re-read `NNNC/NNNC_Ora_StopWide.py`, `NNNC_Ora_StopMid.py`, `NNNC_Ora_StopTight.py`, `NNNC_Ora_Comb.py` and the `NNNC_Ora_L*/P*` classes, and record in `baseline.md` what each already measured and at which label — feature 004's arms were ALL run at label 0.007 and are stale, not redundant (R1). Do NOT rebuild them and do NOT re-derive 004's conclusions

---

## Phase 2: User Story 1 — VALIDATE THE INSTRUMENT (Priority: P1) 🚧 **BLOCKING**

**Goal**: explain or bound the 84.1% compliance of perfect entries with their own
label, so every later figure rests on a defensible measurement.

**Independent Test**: the compliance rate is reported, and any shortfall is either
attributed to a named cause with a correction or expressed as a bound.

> 🚧 **BLOCKING.** Every ceiling figure in this repository inherits this slop. An
> exit conclusion built on an instrument that is 16% not-what-it-claims would be
> worse than no conclusion. **Nothing in Phases 3-7 is trustworthy until this
> closes.**

- [X] T003 [US1] Recompute compliance against the STRATEGY's dataframe rather than the raw feather — run `add_indicators` on the same pairs and timerange the backtest used, including freqtrade's prepended startup candles, and compare the labelled bars with the arm's actual entries. This is R2's most likely cause (FR-001)
- [ ] T004 [US1] Test the window-edge hypothesis: bars within `HORIZON` (48) of the window end cannot see a full forward window, so the strategy's label differs from a feather read that has data beyond the window. Report what share of the shortfall this accounts for (FR-001)
- [ ] T005 [US1] Test fill timing: `entry_pricing` uses `price_side: "other"` with `order_book_top: 1`, so an unfilled limit could open the trade later than the signal bar. Compare each trade's `open_date` with the nearest preceding labelled bar (FR-001)
- [X] T006 [P] [US1] ELIMINATE the subset operations by measurement, not argument alone: `_filter_trading_by_tasks` (NNMTStrategy) and `filter_peaks_by_future_performance` (BaseNNStrategy) can only REMOVE labels, so they cannot add a non-compliant entry. Confirm the label set is a subset at each stage and record the elimination (FR-001)
- [X] T007 [US1] **GATE** Write `specs/006-oracle-exit-diagnosis/instrument.md`: the compliance rate, the cause if found with the corrected figures restated, or a quantified residual bound to be carried into EVERY number this feature produces. State explicitly which of the two outcomes applies (FR-002, FR-003, SC-001)

**Checkpoint**: the instrument is either trustworthy or its error is bounded.

---

## Phase 3: User Story 2 — THE GEOMETRY, THEN THE BOUND (Priority: P1) 🎯 **HARD GATE**

**Goal**: find out how much the exit leaves on the table — after first checking
whether the payoff geometry already explains everything.

**Independent Test**: a perfect-entry arm exists whose exit waits for the label
horizon, with its return compared to the current exit and to the real model.

> ⛔ **HARD GATE ON PHASES 4-6.** If the bound is not materially larger than the
> study's resolution, **the exit is NOT the constraint — that is the feature's
> ANSWER** (FR-005). Report it and re-scope; do not execute the attribution as
> written.

- [ ] T008 [US2] Compute the reward:risk arithmetic per family from the MEASURED configuration — NNMT label 2.0% / harvest 2.2% / stop −2%..−4% (mult 2.5) → 1:1 to 1:2; NNNC label 2.0% / harvest 2.4% / stop −4%..−6% (mult 4.0) → 1:2 to 1:3. State whether a perfectly-labelled trade is a positive-expectancy bet at all, before costs and after (R3)
- [ ] T009 [US2] Recompute the stop-breach rate PER FAMILY with the correct band. The previously reported 6.6% used NNNC's band on NNMT data and UNDERSTATES the problem; NNMT's floor is −0.04 and cap −0.02 (R3, spec Context item 3)
- [ ] T010 [US2] Create `NNMT/NNMT_Ora_Patient.py` + `.json` — the JOINT BOUND. Subclass the oracle with `use_atr_adaptive_stoploss = False`, import `HoldWindowMixin` from `NNNC/_hold_window.py` to suppress the hardcoded 12-hour unclog (do NOT rebuild it, do NOT edit `BaseStrategy`), and override `get_model_path` to alias the production weights. **This arm is a BOUND and is NEVER promotable under any result** (R4, R5, FR-019, FR-021)
- [ ] T011 [US2] Run the patient arm on all three windows with the remaining exits lifted clear — `--roi` disabled, `--set-sell cexit_take_profit=` above any reachable level, `--set-sell cexit_max_days=` beyond the label horizon — and confirm `returncode` 0 on each (FR-004, FR-024)
- [ ] T012 [US2] **GATE** Write `specs/006-oracle-exit-diagnosis/bound.md`: the opportunity cost as `patient − current` on W1 with paired-by-trade resolution, alongside the real model, with every window carrying its net/gross ratio. If the cost does not exceed its own resolution, state that the exit is not the constraint and RE-SCOPE Phases 4-6 (FR-004, FR-005, SC-002, SC-006)

**Checkpoint**: the feature either has a gap worth attributing, or has its answer.

---

## Phase 4: User Story 3 — ATTRIBUTE THE BOUND (Priority: P2)

**Goal**: find which exit mechanism accounts for the gap.

**Independent Test**: each mechanism's individual effect is reported alongside the
joint effect, with inert mechanisms named.

- [ ] T013 [US3] Re-enable the ATR stop ALONE against the patient arm (strictly one mechanism, Constitution VIII) and report its contribution and the exit-reason mix it moves, via `regime/trade_forensics.py --section exits` (FR-006)
- [ ] T014 [P] [US3] Re-enable the ROI table alone (`--roi`), reporting contribution and exit-reason mix (FR-006)
- [ ] T015 [P] [US3] Re-enable the take-profit alone (`--set-sell cexit_take_profit`), reporting contribution and exit-reason mix (FR-006)
- [ ] T016 [P] [US3] Re-enable the 12-hour unclog alone (drop `HoldWindowMixin`), reporting contribution and exit-reason mix (FR-006)
- [ ] T017 [P] [US3] Re-enable the maximum hold alone (`--set-sell cexit_max_days`), reporting contribution and exit-reason mix (FR-006)
- [ ] T018 [US3] RE-RUN feature 004's existing stop-band classes at the CURRENT label — `NNNC_Ora_StopTight` / `StopMid` / `StopWide`, which scored −13.58 / −17.68 / −18.68% at label 0.007 against a −18.89% baseline. Their bands were compared against a label that lost 19%, so they say nothing about the band at a label that pays (R1)
- [ ] T019 [US3] Record every contribution in `specs/006-oracle-exit-diagnosis/mechanisms.md`, naming any mechanism that changes NEITHER trades NOR return as INERT rather than correct, and stating whether the individual effects SUM to the joint bound — the entry-side equivalent was 4.20× super-additive, so addition must not be assumed (FR-006, FR-007, FR-008, SC-003)

**Checkpoint**: the gap is attributed to a mechanism, or shown to be interactive.

---

## Phase 5: User Story 4 — THE TARGET/HARVEST ORDERING (Priority: P2)

**Goal**: resolve whether a harvest point above the label's target is an ordering
error.

**Independent Test**: the harvest point is placed below, at, and above the target
and the share of trades harvesting the promised move is reported for each.

- [ ] T020 [US4] Run the perfect-entry arm with the harvest point BELOW, AT, and ABOVE the label's 2.0% target via `--set-sell cexit_take_profit`, all three windows (FR-009)
- [ ] T021 [US4] Record in `specs/006-oracle-exit-diagnosis/ordering.md` whether any gain comes from harvesting MORE moves or from taking SMALLER ones, using the exit-reason mix rather than return alone. Feature 004 found raising the take-profit monotonically WORSE (−28.06 / −25.76 / −23.34% at 0.004 / 0.006 / 0.010) and R3 predicts exactly that, because the label only guarantees +2% (FR-009, FR-010, SC-004)

**Checkpoint**: the ordering question is settled on the instrument.

---

## Phase 6: User Story 5 — CONFIRM ON THE REAL MODEL (Priority: P1) — the ONLY path to promotion

**Goal**: establish whether anything found on the instrument survives a real model.

**Independent Test**: each candidate is measured on the real model across two
seeds, and its verdict rests on that measurement alone.

> ⚠️ **An oracle WIN is not evidence FOR an exit change (R8).** A patient exit is
> rewarded on an instrument where every entry is guaranteed to reach its target
> and punished on a real model whose entries often do not. **Expect
> instrument-only results and record them as such** — that is a finding, not an
> anomaly.

- [ ] T022 [US5] Re-measure every candidate from Phases 4-5 on the REAL model, all three windows, using the seed pairs that ALREADY EXIST so no training is spent: NNMT `NNMT_Ctl_s42`/`_s7`, and NNNC's open gate `NNNC_MLX_Open_s1`/`_s7`. Report the spread (FR-011, FR-018)
- [ ] T022a [US5] **NNNC's CLOSED-gate production has only ONE model** (`saved_data/NNNC_MLX/`), so it cannot be seed-paired from existing weights. Measure it single-seed and record that a PROMOTION on that configuration is not licensed until a second seed exists — which would cost the feature's only retrain (~7 min at NNNC's size). Do NOT train speculatively: spend it only if a candidate is otherwise promotable there (FR-018, R9)
- [ ] T023 [US5] Apply the paired-by-trade resolution to each real-model comparison and record which candidates are RESOLVED versus within noise (FR-016)
- [ ] T024 [US5] Record verdicts in `specs/006-oracle-exit-diagnosis/confirmation.md`: any candidate that helps the instrument but not the real model is INSTRUMENT-ONLY and is NOT promoted (FR-012, SC-005)
- [ ] T025 [US5] For any candidate that survives, measure its capacity curve by raising `QUOTE_VOLUME_HEADROOM_MULT` on a subclass BEFORE promotion — MULT 20 ≈ a 2× book, 40 ≈ 4× (FR-013)

**Checkpoint**: promotion candidates are decided by the real model, or none exist.

---

## Phase 7: User Story 6 — Conclusion (Priority: P3)

- [ ] T026 [US6] Write `regime/EXIT_DIAGNOSIS.md`: the exit's opportunity cost, the responsible mechanism, what transferred to the real model, the windows measured on, and what the result FORECLOSES for both families — written so a future exit proposal can be accepted or rejected without new measurement (FR-026, SC-009)

---

## Phase 8: Polish & Cross-Cutting

- [ ] T027 Verify `git status --short Framework/` is clean — no shared mechanism modified and the live dry run that imports `BaseStrategy` is unaffected (FR-023, SC-008)
- [ ] T028 [P] Audit every document this feature produced: each perfect-entry figure labelled a LOOKAHEAD CEILING, no perfect-entry configuration promoted, and no verdict citing a classifier metric as an acceptance input — SC-007 has both halves and only the first is self-evident (FR-017, FR-019, SC-007)
- [ ] T029 [P] Verify every window reported anywhere in the feature carries its net/gross ratio, not only its trade count (FR-015, SC-006)
- [ ] T030 [P] Commit the arm records under `regime/arms/` — with the findings documents they are the durable evidence, and each carries the `returncode` that makes it admissible (FR-024)
- [ ] T031 Verify the standing conventions were actually honoured, not merely stated: no arm was reported as a RESULT where its trades were identical to its control or where it produced zero trades (FR-020); arms were run sequentially throughout (FR-022); and the entry guards were never opened on either family (FR-025). These three bind every task and are otherwise verified nowhere
- [ ] T032 Run the `quickstart.md` scenarios and record the verification of SC-001..SC-009
- [ ] T033 [P] Record anything that generalises beyond these two families in memory and the relevant `regime/` documents (no FR — knowledge capture beyond the feature's own requirements)

---

## Dependencies

    Phase 1 (T001-T002)  — baselines and prior art. T002 is [P] with T001.
        ↓
    Phase 2 (T003-T007)  — INSTRUMENT. T007 is BLOCKING.
        ↓                   T003-T006 in any order (T006 is [P]); all precede
        ↓                   T007. NOTHING below runs until T007 closes.
        ↓
    Phase 3 (T008-T012)  — GEOMETRY then BOUND. T012 is a HARD GATE.
        ↓                   T008-T009 (geometry, nearly free) precede T010-T011
        ↓                   deliberately — they may answer the question alone.
        ↓                   T010 before T011; T011 before T012.
        ↓
        ├──────────────→ Phase 4 (T013-T019) — attribution, CONTINGENT on T012
        ├──────────────→ Phase 5 (T020-T021) — ordering,    CONTINGENT on T012
        ↓
    Phase 6 (T022-T025)  — REAL MODEL. Needs candidates from Phases 4-5.
        ↓                   T022→T023→T024; T025 only if a candidate survives.
        ↓
    Phase 7 (T026)       — conclusion, needs Phases 2-6
        ↓
    Phase 8 (T027-T033)  — polish

**Two gates, and they differ.** T007 is BLOCKING — it gates everything, because a
mis-specified instrument invalidates every later number. T012 is a HARD GATE on
Phases 4-6 specifically: a small bound means the exit is not the constraint, and
the correct response is to re-scope the attribution rather than perform it.

**Phase 4 and Phase 5 are independent of each other** and may run in either
order; both depend on T012.

---

## Priority vs execution order

- **US5 (P1) runs fifth**, after two P2 stories. It is the only step that can
  license a promotion and must not be dropped if the feature is cut short — but
  it has nothing to confirm until Phases 4-5 produce candidates.
- **US1 (P1) is first and is a blocking gate**, not merely the highest priority.
- **US2 (P1) is second and is a kill switch.** A small bound is the answer.

---

## Implementation Strategy

**MVP = Phases 1-3.** A validated instrument plus the bound is a complete,
valuable deliverable: it says whether the exit is the constraint at all.

**Cheapest-first, twice over.** The whole feature is backtests — **ZERO retrains**
(R9). Within Phase 3, the geometry arithmetic (T008-T009) precedes the patient arm
because it costs almost nothing and may answer the question outright: NNMT risks
2-4% to capture 2.2% on a move guaranteed to reach only 2.0%.

**A negative bound is the ANSWER.** T012 and FR-005 both make this explicit so a
null is reported as a finding rather than treated as a failed feature. On such a
result Phases 4-6 are re-scoped — INCLUDING Phase 6, which has nothing to
confirm once Phases 4-5 produce no candidates — and Phases 7-8 still run.

**Expect instrument-only results.** R8 makes the patient exit the intervention
most likely to flatter itself here. Phase 6 exists to catch exactly that.
