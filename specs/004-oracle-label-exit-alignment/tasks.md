---

description: "Task list for 004 Oracle-First Label/Exit Alignment"
---

# Tasks: Oracle-First Label/Exit Alignment for NNNC

**Input**: Design documents from `specs/004-oracle-label-exit-alignment/`

**Prerequisites**: plan.md, spec.md, research.md (R1–R11), data-model.md,
contracts/forensics-cli.md, quickstart.md

All commands run from the repository root
`/Users/philprice95/projects/freqtrade`. Paths below are relative to
`user_data/strategies/` unless shown otherwise.

## Standing conventions (apply to EVERY evaluation task)

**Guards-off is always the six neutralising values, never `--guards off`** (R7) —
this keeps `volume > 0` live:

    NEUT="--set entry_rvol_threshold=-1.0 --set entry_atr_pct=-1.0 \
          --set entry_adx_threshold=-1.0 --set entry_bb_width_threshold=-1.0 \
          --set entry_guard_threshold=5.0 --set entry_close_norm_threshold=5.0"

**Per-arm protocol**: confirm `returncode` 0 (a failed or stale record is not
evidence, FR-026); compare `trades_hash` against the arm's control (identical =>
inconclusive, not refuted, FR-010); judge on W1 only, report W2/W3 labelled —
and for oracle arms note they corroborate a MECHANISM but never confer
acceptance (R10, FR-022); never accept on a classifier metric (FR-008); every arm
carries its own `<Strategy>.json` (FR-023); every oracle figure is reported as a
LOOKAHEAD CEILING and nothing measured with it is promotable (FR-012, SC-009).

**Power rule at high trade counts.** `regime/tail/compare.py` uses
`POWER_THRESHOLD_PP = 1.0`, calibrated for ~97-trade arms. Guards-off arms carry
1760–6015 trades, where a 1pp difference is far outside noise. Do NOT apply the
1pp floor to guards-off arms: for those, the gate in R9 (return > 0 AND
>= 200 trades) is the decision rule, and `compare.py` is used for its liveness
and labelling output, not its verdict line. Any task using `compare.py` on a
guards-off arm must state this.

**Arms run SEQUENTIALLY, never concurrently (FR-025).** Two concurrent jobs
previously produced load 60 on 14 cores and finished slower than serialising.
This applies to every task in Phases 1, 2 and 4 that invokes `run_arm.py`.

**Hazards**: a variant without its own `.json` silently runs class defaults (cost
12pp on a prior comparison); `Framework/BaseStrategy.py` must NOT be left
modified — a live dry run imports it; `TaskStop` does not kill process trees, use
`pkill -9 -f`; `exit_config_hash` cannot see method overrides, so hold-window
variants are verified by the TRADE hash only; guards-off arms are slower than
guards-on arms.

**Already settled by research.md — do NOT redo**: R1 (archives hold everything,
no re-measurement for the diagnosis), R2 (mechanism mapping), R4 (separate
reader), R6 (which knobs need subclasses), R7 (neutralising values), R9 (gate
condition), R10 (window treatment), R11 (cost model).

---

## Phase 1: Complete the baseline

**Purpose**: the baseline against which every variant is judged must be as
complete as FR-014 demands of the variants themselves. Two backtests, no
training (R8).

- [X] T001 Run the missing W2 baseline for the guards-off oracle: `PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/regime/run_arm.py --strategy NNNC_Oracle --window W2 --tag oracle_nog $NEUT` — confirm `returncode` 0 and a recorded `wallet_profit_pct` in `regime/arms/oracle_nog_W2.json` (R8, FR-014, FR-026)
- [X] T002 Run the missing W2 baseline for the guards-off no-model pool: same command with `--strategy NNNC_NoModel --tag gpool` — confirm `returncode` 0 in `regime/arms/gpool_W2.json` (R8, FR-014, FR-026)
- [X] T003 Run the missing W2 baseline for the guards-ON oracle: same command with `--strategy NNNC_Oracle --window W2 --tag oracle` and NO `$NEUT` — completes the guards-on row so the guards-on/guards-off contrast is available on all three windows (R8, FR-014)
- [X] T004 Record the completed baseline table (all six arms × three windows) in `specs/004-oracle-label-exit-alignment/baseline.md`, flagging any arm whose `returncode` is non-zero as not-evidence (FR-026, SC-004)

**Checkpoint**: the baseline is complete on all three windows and every record is
from a successful run.

---

## Phase 2: User Story 1 + User Story 4 — DIAGNOSE (Priority: P1) 🎯 **HARD GATE**

**Goal**: localise the mechanism that turns perfect foreknowledge into a loss,
and separate cost drag from label quality — BEFORE any parameter is varied.

**Independent Test**: a per-exit-reason breakdown exists for the guards-off and
guards-on oracle arms with 100% trade coverage, the dominant mechanism is named
with its trade count and share of destroyed P&L, and the question "does the
guards-off loss survive removal of fees" is answered yes or no.

> ⛔ **HARD GATE ON PHASE 4.** No label or exit parameter may be varied until
> T012 is reported. Constitution Principle VI forbids tuning before a diagnostic
> localises the fault, and this feature exists because that discipline was not
> applied earlier. If Phase 2 cannot identify a dominant mechanism, that itself
> is the reported result and Phase 4 must be re-scoped, not started blind.

- [X] T005 Create `regime/trade_forensics.py` implementing the contract in `contracts/forensics-cli.md`: read-only over `user_data/backtest_results/`, options `--archive`/`--latest`/`--match-trades` + `--strategy`, `--section {exits,costs,pairs,excursions,all}`, `--json` (R1, R4, FR-001)
- [X] T006 In `regime/trade_forensics.py`, implement the R2 MECHANISM MAPPING and print the mapping table with every report: `stop_loss` → ATR adaptive stop (initial), `trailing_stop_loss` → **ATR adaptive stop (moved), NOT the native trailing stop which is disabled**, `take_profit` → cexit_take_profit, `roi` → ROI table, `unclog_*`/`max_hold` → hold-window limits, `rsi_*`/`metric_overbought` → profit checks, `force_exit` → counted in coverage but excluded from mechanism attribution (R2, FR-003)
- [X] T007 In `regime/trade_forensics.py`, implement the `exits` section per data-model.md ExitReasonBreakdown: per mechanism `n_trades`, `sum_pnl`, `mean_pnl`, `mean_duration_h`, `share_of_total_pnl`, ordered by `sum_pnl` ascending; ALWAYS print coverage as `n_attributed / n_trades` and the archive path, strategy and trade count used (R5, FR-001, FR-002, FR-003, SC-002)
- [X] T008 [P] In `regime/trade_forensics.py`, implement the `costs` section per data-model.md CostProfile: `n_trades`, `total_fees` from each trade's own `fee_open`/`fee_close` and `stake_amount`, `net_pnl`, `gross_pnl = net + fees`, `fees_as_share_of_gross`, and the boolean `loss_survives_costs` (R11, FR-018, FR-019)
- [X] T009 [P] In `regime/trade_forensics.py`, implement the `pairs` section: per pair `n_trades`, `sum_pnl`, share of trades, so a changed pair mix between two arms is visible and a guards-off loss is not misread as a difference in bars when it is a difference in instruments (FR-020, Edge Case 3)
- [X] T010 [P] In `regime/trade_forensics.py`, implement the `excursions` section from `max_rate`/`min_rate` normalised to `open_rate`: distribution of favourable and adverse excursion, share of trades whose adverse excursion reached the operative ATR stop band before their favourable excursion reached `cexit_take_profit` (stop-before-payoff), and share whose favourable excursion EXCEEDED the take-profit level (truncation). No path recomputation from candles — the archive already holds the realised excursion (R1, FR-004, FR-005)
- [X] T011 Run the forensics over the four decisive arms on W1 — guards-off oracle (`oracle_nog`, 1760tr), guards-on oracle (`oracle`, 89tr), guards-off real model (`gqoff`/`gsw069`, 1970tr) and guards-on production (`gctl`, 95tr) — resolving each archive by `--match-trades` + `--strategy` and confirming the printed trade count matches the arm record (R5, FR-001, FR-002)
- [X] T012 **GATE** Write the diagnosis to `specs/004-oracle-label-exit-alignment/diagnosis.md`: the dominant mechanism NAMED with its trade count and share of destroyed P&L (FR-003, SC-001), coverage 100% on every arm (SC-002), the guards-on vs guards-off difference showing which mechanisms the guard chain suppresses (FR-002), whether the guards-off loss survives removal of fees (FR-019, SC-006), the pair-mix comparison (FR-020) and the stop-before-payoff and truncation shares (FR-004, FR-005). State explicitly that no parameter has yet been varied (FR-006)

**Checkpoint**: the fault is localised and Phase 4 may be aimed. Until T012 is
written, Phase 4 is blocked.

---

## Phase 3: User Story 2 — The oracle-first gate as standing process (Priority: P1)

**Goal**: a written, reusable rule that screens any label or exit change against
the oracle before a model is trained.

**Independent Test**: `regime/ORACLE_GATE.md` exists stating what is measured, on
which windows, the pass/fail/inconclusive conditions and the no-promotion rule,
and it has been applied to one worked example end to end.

- [ ] T013 [US2] Write `regime/ORACLE_GATE.md`: what is measured (W1 guards-off oracle realised return and trade count), the R9 PASS condition (return **> 0** AND `n_trades >= 200`), INCONCLUSIVE (trades identical to control, or `returncode != 0`), FAIL (anything else, including an improvement that remains negative), and that a FAIL records progress but does NOT license training (R9, FR-007, FR-009, FR-010, SC-003)
- [ ] T014 [US2] In `regime/ORACLE_GATE.md`, record the two rules that keep the gate honest: realised P&L only, never a classifier metric — val_mcc, precision and EV-per-signal have each failed to predict return in this repo (FR-008); and every oracle figure is a LOOKAHEAD CEILING, never an achievable return, with no oracle configuration promotable toward live trading under any result (FR-012, SC-009)
- [ ] T015 [US2] In `regime/ORACLE_GATE.md`, record the 200-trade floor as an explicit JUDGEMENT rather than a measurement, with its rationale — guards-on arms hold ~95 trades, so a guards-off variant collapsing toward that count has re-imposed a gate rather than fixed the label (R9, Edge Case 4)
- [ ] T016 [US2] Apply the gate to one worked example end to end and record it inside `regime/ORACLE_GATE.md` as the reference application, using the cheapest variant that addresses the T012 mechanism; report the verdict with its trade count (FR-011, SC-003)

**Checkpoint**: the gate is documented and demonstrated, and is usable by later
features without re-deriving it.

---

## Phase 4: User Story 3 — Screen variants (Priority: P2)

**Goal**: find a label or exit configuration the oracle can profit from with the
guards off.

**Entry condition**: T012 written. **Variants MUST address the mechanism the
diagnosis identified** — a variant chosen speculatively is out of order (FR-013,
Constitution VI).

**Independent Test**: each screened variant has arm records on all three windows
and a gate verdict, and the best is stated against the −18.89% W1 baseline.

**Cheapest-first ordering (R6)**: `--set-sell`/`--roi` variants on the existing
`NNNC_Oracle` host need NO new class; only after those are exhausted do variants
requiring a subclass follow.

- [ ] T017 [US3] Screen the exit-side variants reachable WITHOUT a new class, one at a time on W1, guards-off, against the T012 mechanism — candidates are `--set-sell cexit_take_profit=<v>`, `--set-sell cexit_max_days=<v>`, `--set-sell cexit_enable_profit_checks=false`, `--roi <table>` — each as its own tag, run SEQUENTIALLY, each confirmed `returncode` 0 (FR-013, FR-014, FR-025, R6)
- [ ] T018 [US3] Promote every W1 survivor of T017 to the full three windows and apply the gate, reporting guards-off return, trade count and drawdown per window and labelling W2/W3 as corroborating a mechanism only (FR-014, R10, SC-004)
- [ ] T019 [US3] If the diagnosis implicates the ATR stop band, create `NNNC/NNNC_Ora_Stop.py` + `NNNC/NNNC_Ora_Stop.json` subclassing `NNNC_Oracle` and overriding `atr_stoploss_multiplier`/`_floor`/`_cap` (class attributes, unreachable via `--set`; `--stoploss` is INERT). Copy the `.json` from `NNNC/NNNC_Oracle.json` and set `strategy_name`, or the arm silently runs class defaults (R6, FR-023)
- [ ] T020 [US3] If the diagnosis implicates the hold window, create `NNNC/NNNC_Ora_Hold.py` + `.json` applying `HoldWindowMixin` from `NNNC/_hold_window.py` with `HOLD_WINDOW_HOURS` set. Verify liveness by the TRADE hash only — `exit_config_hash` cannot see a method override and WILL match the control (R6, FR-010)
- [ ] T021 [US3] Screen the LABEL-side variants separately from every exit-side variant, as subclasses of `NNNC_Oracle` overriding `MIN_BUY_GAIN_THRESHOLD`/`MIN_SELL_LOSS_THRESHOLD`/`HORIZON`/`TRAINING_TYPE` (class attributes; precedent `NNNC/NNNC_Brk_L1.py`), each with its own `.json` — so the label and exit contributions are separately attributable (FR-015, SC-005, Constitution VIII)
- [ ] T022 [US3] Only after T018 and T021 have reported separately, screen any COMBINED label+exit variant, and compare it only against arms differing from it in exactly ONE respect so the decomposition is arithmetically valid (FR-015, SC-005)
- [ ] T023 [US3] Record every screened variant in `specs/004-oracle-label-exit-alignment/screening.md` with its gate verdict, W1 guards-off return, trade count and drawdown on all three windows, and state the best result against the −18.89% baseline; report any passing variant as a CANDIDATE FOR TRAINING, never as an adopted change (FR-014, FR-016, SC-004, SC-007)
- [ ] T024 [US3] If no variant passes the gate, record that as the result together with what it implies about the strategy's dependence on the guard chain — a complete and valuable outcome, not a failure to report (FR-017, SC-007)

**Checkpoint**: the search is complete and its outcome, positive or negative, is
recorded.

---

## Phase 5: User Story 5 — Conclusion (Priority: P3)

**Goal**: the headroom finding governs later decisions instead of being
rediscovered.

- [ ] T025 [US5] Write the model-headroom conclusion into `regime/GUARD_ABLATION.md`: the 0.68pp figure, the windows it was measured on, its dependence on the CURRENT gate (a configuration admitting materially more trades would need it re-measured), and what it forecloses for model, GAN and architecture work on this family (FR-021, SC-010)
- [ ] T026 [US5] State in the same conclusion the test a future model or GAN proposal on this family must pass to be worth running, so the conclusion is sufficient to accept or reject such a proposal WITHOUT new measurement (SC-010)

---

## Phase 6: Polish & Cross-Cutting

- [ ] T027 Verify `git status --short Framework/` is clean — no shared framework file modified, and the live dry run that imports `BaseStrategy` is unaffected (FR-024, SC-008)
- [ ] T028 [P] Verify every oracle figure in `diagnosis.md`, `screening.md`, `ORACLE_GATE.md` and `GUARD_ABLATION.md` is labelled a lookahead ceiling and that no oracle configuration has been promoted (FR-012, SC-009)
- [ ] T029 [P] Commit the arm records under `regime/arms/` — with `regime/trade_forensics.py` and the findings documents they are the durable evidence for this feature (FR-026)
- [ ] T030 Run the `quickstart.md` scenarios end to end and confirm every success criterion SC-001..SC-010 is satisfied, recording the verification
- [ ] T031 [P] Record anything that generalises beyond this host in memory and the relevant `regime/` documents — in particular the R2 mechanism-mapping trap and the oracle-gate process, both of which apply to any strategy family in this repo (FR-021, SC-010)

---

## Dependencies

    Phase 1 (T001-T004)  — baseline, no dependencies
        ↓
    Phase 2 (T005-T012)  — DIAGNOSIS.  T012 is a HARD GATE.
        ↓                   T005 precedes T006-T010 (same file).
        ↓                   T008, T009, T010 are [P] with each other
        ↓                   (independent sections) but all follow T007's
        ↓                   coverage/reporting scaffold.
        ↓                   T011 needs T005-T010; T012 needs T011.
        ├──────────────→ Phase 3 (T013-T016) — the gate. T016 needs T012
        ↓                   (its worked example must address the mechanism).
        ↓                   T013-T015 may be written as soon as T012 exists.
        ↓
    Phase 4 (T017-T024)  — BLOCKED until T012 is written (FR-006).
        ↓                   T017 → T018.  T021 independent of T017-T020.
        ↓                   T022 requires BOTH T018 and T021 reported.
        ↓                   T019/T020 are conditional on what T012 finds.
        ↓
    Phase 5 (T025-T026)  — needs Phase 4's outcome
        ↓
    Phase 6 (T027-T031)  — polish, needs everything above

**The only hard gate is T012.** Phase 3 depends on it only for its worked
example (T016); T013–T015 are documentation of decisions already taken in
research.md and may proceed in parallel with Phase 2 if convenient.

**Conditional tasks**: T019 and T020 execute only if the diagnosis implicates the
ATR stop or the hold window respectively. If it implicates neither, they are
skipped and that is recorded — they are not to be executed speculatively.

---

## Priority vs execution order

Execution order differs from spec priority, deliberately:

- **US4 (P2) is executed inside Phase 2 alongside US1 (P1)**, not later. Both
  read the same archives through the same reader, so splitting them would mean
  building the reader twice or reading the archives twice. More importantly the
  cost answer (FR-019) is a precondition for interpreting the diagnosis at all —
  the feasibility probe suggests fees may exceed the entire net loss, and
  blaming the label before settling that would be the exact error this feature
  was created to correct.
- **US3 (P2) precedes US5 (P3)** as priority implies.
- **US2 (P1) is written after US1's gate** because its worked example must
  address the mechanism the diagnosis identifies; its rules, however, come from
  research.md and are independent.

---

## Implementation Strategy

**MVP = Phase 1 + Phase 2.** The diagnosis alone is a complete, valuable
deliverable: it names the mechanism destroying the oracle's P&L and settles
whether the loss is cost or label. Everything after it is a search that the
diagnosis aims.

**Cheapest-first throughout.** No task in this feature trains a model. An oracle
arm is a backtest; guards-off arms are slower only because they carry more
trades. If the feature is stopped after Phase 3, the repo still gains a
permanent training-free screening gate.

**A negative outcome is a result.** T024 and T017's framing both make explicit
that "no variant makes the oracle profitable" is a finding to record, not a
failure to hide.
