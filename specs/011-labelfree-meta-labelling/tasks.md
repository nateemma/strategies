# Tasks: labelfree-meta-labelling

**Feature**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md) · **Research**: [research.md](research.md)
**Data model**: [data-model.md](data-model.md) · **Contract**: [contracts/metalabel-cli.md](contracts/metalabel-cli.md)
**Validation**: [quickstart.md](quickstart.md)

## Format: `[ID] [P?] [Story] Description`

`[P]` = parallelisable (different files, no dependency on incomplete work). **File authoring
only — RUNS ARE NEVER PARALLEL.** Two concurrent jobs put load 60 on 14 cores and finished
slower than one.

## Path Conventions

All paths are relative to `user_data/strategies/`. Commands run from the **repo root** (the
directory containing `user_data/`) with `PYTHONPATH=user_data/strategies`.

---

## Phase 1: Setup

- [X] T001 Create the package skeleton `research/labelfree/metalabel/` with `__init__.py`, and a test module `research/labelfree/tests/test_metalabel.py`
- [X] T002 [P] Record the pinned constants in `research/labelfree/metalabel/__init__.py`: `EVAL_FOLDS = range(5, 15)`, `WARMUP_FOLDS = range(0, 5)`, `FLOOR_PP_PER_YEAR = 4.681`, `BASE_WIN_RATE = 0.732`, `N_EVAL_TRADES = 287`, `AUC_KILL = 0.55`, with the derivation of the floor (research R2) in the docstring and a note that these are MEASURED, not chosen
- [X] T003 [P] Assert the archive still matches those constants in `research/labelfree/tests/test_metalabel.py` — 378 trades / 15 folds / 287 evaluation trades / 73.2% base rate. **If any has moved, the floor and base rate must be recomputed before anything else runs** (quickstart S1)

---

## Phase 2: Foundational — THE THREE GATES (Blocking Prerequisites)

**Nothing downstream is evidence until all three pass.** Each gate exists because the
corresponding defect has already occurred in this study.

### Gate A — the framework can report what the primary believed

- [X] T004 Add three debug columns to `get_predictions()` in `Framework/BaseNNStrategy.py`: `%predict_prob_sell`, `%predict_prob_hold`, `%predict_prob_buy`. NaN-initialise at full dataframe length, write at the **same `offset`** used to front-pad `predictions`, and place the write **after** the Markov branch (R1)
- [X] T005 [P] Test in `research/labelfree/tests/test_metalabel.py` that the columns are dropped by `DataframeUtils.remove_debug_columns()` — the `%` prefix is what stops the model's own output entering the feature list, and without it this is a predict-your-own-prediction leak (R1)
- [X] T006 [P] Test that a short prediction array is written at the correct offset and that the padded region is **NaN, not 0.0** — 0.0 is a legitimate probability meaning "confidently not this class" and would mark every warm-up bar a confident non-buy (R1)
- [X] T007 Verify the change is inert for existing strategies: run `Framework/test_base_nn_strategy.py` and confirm `predict_buy`/`predict_sell` are byte-identical before and after T004

### Gate B — re-inference reproduces the trades that actually happened

- [X] T008 Implement `research/labelfree/metalabel/archive.py` — read `regime/wf/pooled_nnnc.json` into `TradeRow` records (data-model), one per archived trade (FR-001)
- [X] T009 Implement `research/labelfree/metalabel/reinfer.py` — reload each fold's saved model from `saved_data/NNNC_WF_f*`, replay its test range, and read the `%predict_prob_*` columns (FR-005a). **Refuse to run if any fold's model file is missing**; a silent skip would train on a subset without saying so
- [X] T010 Implement the reproduction gate in `reinfer.py`: the re-inferred `(pair, open_date)` set MUST equal the archived set exactly. On mismatch, print the symmetric difference, write no partial output, and exit non-zero (FR-005b, SC-017)
- [X] T011 RUN `reinfer.py` over all 15 folds and confirm `matches_exactly: true` for each (quickstart S2). **A mismatch halts the feature** — it means the models, their pinned params, or the feather data changed since 2026-09-09, which is a finding, not an inconvenience
- [X] T012 Handle NaN confidence in `reinfer.py`: a row whose entry bar was never scored is **excluded and counted**, never imputed (FR-006), and the count is reported (quickstart S2b)

### Gate C — no feature reads the future, and the audit can prove it

- [X] T013 Implement `research/labelfree/metalabel/features.py` — assemble the entry-time vector by calling the repo's existing feature code, **never reimplementing it** (FR-005). Pair identity, pair group and pair-derived history are **excluded by construction** (FR-005c, R7)
- [X] T014 Implement causality-by-deletion in `research/labelfree/metalabel/audit.py`: recompute each feature with all bars strictly after `open_date` deleted and assert bit-identity (FR-002, FR-003)
- [X] T015 Implement the **planted-leak control** in `audit.py` — a feature equal to the trade's own `profit_abs` — which the audit is REQUIRED to flag. It cannot be disabled (FR-004, SC-011)
- [X] T016 [P] Test that the planted-leak control actually fails the audit. **A control that passes vacuously certifies nothing** — the first version of the analogous control in `research/labelfree/liveness.py` did exactly that by not severing anything
- [X] T017 Implement the permutation test in `audit.py` alongside deletion, reported separately: permutation measures *informativeness*, deletion measures *causality*, and they answer different questions
- [X] T018 RUN `audit.py` and confirm every feature passes causality and the planted leak is flagged (quickstart S3)

**Checkpoint**: T011 and T018 both green. Until then no model may be trained.

---

## Phase 3: User Story 1 - The audited trade table (Priority: P1) 🎯 MVP

**Goal**: one row per archived trade, features known at entry only, with positive evidence of
causality.

**Independent test**: the table exists, the audit passes, and the planted leak is detected.

- [X] T019 [US1] Emit the joined trade table to `research/labelfree/metalabel/rows.jsonl` with the excluded-row count in its summary
- [X] T020 [P] [US1] Test that exactly one row exists per archived trade and that two trades on the same pair and bar are **one decision, not two measurements** (data-model)
- [X] T021 [US1] Document the table's schema in the module docstring so briefs 5 and 8 can reuse it — this table is a standalone deliverable whether or not any veto works

**Checkpoint**: US1 delivers value independently.

---

## Phase 4: User Story 2 - Nested walk-forward veto model (Priority: P2)

**Goal**: a secondary model for fold *k* trained only on earlier trades, with discrimination
measured against the base rate.

**Independent test**: per-fold discrimination reported against 73.2%, with per-fold trade counts.

- [ ] T022 [US2] Implement `research/labelfree/metalabel/nested.py` — fold *k* trains on trades whose **`close_date`** precedes fold *k*'s `test_start` (FR-007, FR-008; R6; stricter than `open_date`, since a trade opening before the boundary but closing after carries an outcome not yet knowable)
- [ ] T023 [P] [US2] Test that no training trade closes after its evaluation fold opens, and that `f05` reports `n_train == 91` (quickstart S4)
- [ ] T024 [US2] Implement `research/labelfree/metalabel/model.py` — LightGBM veto with capacity pinned for the **first** fold's 91 rows, not the last fold's 287 (R4): shallow trees, high min-samples-per-leaf, few estimators, all fixed before any run
- [ ] T025 [US2] Extend `research/labelfree/liveness.py` with probes for `veto_threshold`, `n_estimators`, `max_depth`, `min_child_samples`, each in its **active regime** — a probe outside it reports DEAD on correct code, as `participation_weight` did
- [ ] T026 [US2] Make `model.py` refuse to run until every swept parameter probes LIVE (FR-012, SC-014, quickstart S5). Four parameters in this study were configured, logged, believed active and reaching nothing; 75+ runs void between them
- [ ] T027 [US2] Pre-register every configuration to `research/labelfree/trials.jsonl` **before** training it, carrying the pinned capacity parameters (FR-011)
- [ ] T028 [US2] Report **AUC and precision at the operating point**; never accuracy (FR-009, R3) — a majority-class model scores 73.2%, so accuracy cannot distinguish skill from doing nothing
- [ ] T029 [US2] Report the interquartile mean with a bootstrap interval over ≥5 seeds, never a best seed (FR-010), with `n_train` beside every per-fold figure (FR-014)
- [ ] T030 [US2] RUN the nested veto at thresholds {0.4, 0.5, 0.6} × seeds {1..5}, sequentially (quickstart S6)
- [ ] T031 [US2] Apply SC-009: if AUC < 0.55 for every threshold, **kill the feature** and record that entry-time features carry no information about outcome beyond what the primary already used

---

## Phase 5: User Story 3 - The P&L consequence (Priority: P3)

**Goal**: what the veto does to realised return, against the recomputed floor.

**Independent test**: replay the archive with vetoed trades removed and compare over the
identical span.

- [ ] T032 [US3] Implement `research/labelfree/metalabel/evaluate.py` — replay the archived trades with vetoed trades removed; the **control is the unvetoed primary**, i.e. exactly what the veto would replace
- [ ] T033 [US3] Quote the **recomputed 4.681 pp/yr** floor, never the study-wide 3.350 (R2, FR-013). A delta between the two clears the old floor and is **not resolvable** here
- [ ] T033a [US3] Implement `trade_activity()` in `research/labelfree/metalabel/evaluate.py` — activity computed from the **trade list**: `trades_per_year = retained / years`, `mean_hold_hours = mean(close_date - open_date)`, plus the retained count. **Do NOT reuse `stats.py::activity`**: it takes an exposure path `(n_pairs, T)` and derives trades from contiguous non-flat runs, which this feature does not have. Synthesising an exposure path to feed it would invent a quantity (analyze C1)
- [ ] T033b [P] [US3] Test `trade_activity()` against a hand-checked three-trade fixture, including a trade that opens and closes on the same bar
- [ ] T034 [US3] Attach retained-trade count, trades/yr and mean holding time from `trade_activity()` to **every** return (FR-014, SC-012)
- [ ] T035 [US3] Emit `inconclusive: true` with the **return withheld** when too few trades are retained to resolve (FR-015). A veto can otherwise "win" by discarding almost everything
- [ ] T036 [P] [US3] Test that a fold from which the veto removed every trade reports zero **by construction** and is not presented as a loss avoided (spec edge cases)
- [ ] T037 [US3] Always include the null configuration (accept everything) as the control row; never omit it
- [ ] T038 [US3] RUN the evaluation and report folds-positive out of 10 against SC-002's ≥7 (quickstart S7)

---

## Phase 6: User Story 4 - Is it pair selection? (Priority: P4)

**Goal**: decide whether the gain is within-pair discrimination or the removal of weak pairs.

**Independent test**: recompute the gain with ZEC excluded and within each pair.

- [ ] T039 [US4] Implement `research/labelfree/metalabel/attribution.py` — gain across all pairs, with ZEC excluded, and decomposed within pairs (FR-016, FR-017)
- [ ] T040 [US4] Print a verdict line `PAIR SELECTION` or `WITHIN-PAIR`; the former **stops the feature** (SC-010)
- [ ] T041 [US4] RUN attribution (quickstart S8). Note that the model **cannot** rank pairs by construction (R7), so a large all-pairs/ex-ZEC gap would itself be surprising and worth diagnosing rather than reporting

---

## Phase 7: Persistence and cost stress

- [ ] T041a Add `--probe {W1,W2,W3}` to `research/labelfree/metalabel/model.py` per the contract: it inverts the nested assignment (trains on folds *after* the window) and stamps `anti_causal: true` into its output. Without this the declared Principle II deviation has no way to execute and would silently not happen (analyze H2)
- [ ] T042 RUN the W3 anti-causal persistence probe via `model.py --probe W3 --out research/labelfree/metalabel/probe_w3.json` — trained on **later** folds, tested on 2023 (R6). Label its construction wherever its figure appears; it may never be quoted as an out-of-sample return nor counted toward SC-001 (SC-018, quickstart S9)
- [ ] T043 RUN W1 and W2 single-window out-of-sample and report both against SC-002
- [ ] T043a Add a span mode to `research/labelfree/baselines.py` — `--folds f05..f14` aggregating across folds on the **same annualisation the floor uses** (research R2). Today it accepts `--fold` only (single fold, default `f08`) and cannot express the evaluation span, so SC-003 is unsatisfiable without this (analyze C2)
- [ ] T043b [P] Test the span mode against the single-fold path: a one-fold span must reproduce `--fold` exactly
- [ ] T044 RUN the four mandatory baselines over folds 5–14 via `research/labelfree/baselines.py --folds f05..f14` (SC-003)
- [ ] T045 RUN cost stress at 1.5× and 2× fees and report the fee level at which the edge crosses zero (SC-005)
- [ ] T046 Compute the in-sample overfitting differential via `research/labelfree/stats.py::overfitting_differential`, using the split **pre-registered in research.md R10**: each fold-*k* model's advantage on its own training trades (in-sample) versus its advantage on fold *k* (held-out), aggregated across evaluation folds. Exceeding the recomputed 4.681 pp/yr floor records overfitting and stops the configuration (SC-008). The split is fixed before any result exists — choosing it afterwards is the failure the gate exists to prevent (analyze H1)
- [ ] T047 Compute DSR and PBO over the **full** trial log via `stats.py::trial_log_deflation` (SC-004). **Expect `CANNOT BE EVALUATED`** — the log stores summary scalars, so the only axis is seeds; report it as unevaluable rather than passing on an artefact

---

## Phase 8: User Story 5 - Strategy integration (Priority: P5) — GATED ON ACCEPTANCE

**Do not start unless a configuration cleared SC-001 through SC-006** (FR-019).

- [ ] T048 [GATED] [US5] Write `NNNC/NNNC_Meta.py` — a thin `NNNC_MLX` subclass loading the primary from a **pinned** path, not retraining it (FR-018), merging parameter dictionaries from the parent rather than redefining them (constitution)
- [ ] T049 [GATED] [US5] Consult the secondary model in `confirm_trade_entry()` and decline below the threshold
- [ ] T050 [GATED] [US5] RUN the subclass over the evaluation span and compare its trades to the offline replay; **report any divergence** rather than averaging it away (FR-020, SC-016)

---

## Phase 9: Reproduction and close

- [ ] T051 Give the meta-label package its own reproduction path in `research/labelfree/metalabel/reproduce.py`, re-running an accepted configuration in a **fresh process**. This is a second code path, not a parameter addition: `research/labelfree/reproduce.py`'s `CONFIG_KEYS` are `ArmConfig` fields and its `rerun()` subprocess imports `research.labelfree.arms` and calls `train_arm`, none of which a LightGBM configuration shares (analyze M2). Reuse its comparison/tolerance logic (SC-006)
- [ ] T052 Re-run `/speckit-analyze` and append to `specs/011-labelfree-meta-labelling/analyze.md`; fix every CRITICAL and re-run until clean (FR-023, SC-015)
- [ ] T053 RUN the trial-log audit via `research/labelfree/trials.py::audit` (SC-013, quickstart S10)
- [ ] T053a Apply SC-007 in `research/labelfree/metalabel/report.py`: a delta below the recomputed 4.681 pp/yr floor is rendered as **"not resolved"**, never as "no effect" (FR-024). Assert the phrase "no effect" cannot appear in a generated verdict line
- [ ] T054 Write `research/label-free-study/briefs/results/03-<date>.md` per protocol §6 — hypothesis, config and seeds, trials logged, return with CI, folds positive, trades/yr, hold time, baselines, cost stress, DSR/PBO, verdict, and what changes for the next brief. **Written even on a kill** (FR-025)
- [ ] T054a If the feature killed at SC-009 or SC-010, mark T048–T050 **cancelled with the reason inline** rather than leaving them open. Feature 010 left gated tasks unchecked and they had to be explicitly cancelled at close; an open box reads as outstanding work (analyze M3)
- [ ] T055 Stage ONLY the files this feature created or modified and commit; never `git add -A` (FR-021, FR-022)

---

## Dependencies & Execution Order

```
Phase 1 (Setup)
   ↓
Phase 2 (THE THREE GATES)   ← 🚦 A: columns · B: reproduction · C: leakage
   ↓                            nothing downstream is evidence until all pass
Phase 3 (US1 trade table)   ← MVP, standalone deliverable
   ↓
Phase 4 (US2 veto model)    ← SC-009 kill point
   ↓
Phase 5 (US3 P&L)           ← discrimination is not edge
   ↓
Phase 6 (US4 attribution)   ← SC-010 kill point
   ↓
Phase 7 (persistence, cost, deflation)
   ↓
Phase 8 (US5 integration)   ← GATED on acceptance
   ↓
Phase 9 (analyze → audit → report → commit)
```

### User Story Dependencies

- **US1** depends only on Phase 2. It is the MVP and a genuine standalone deliverable: an
  audited entry-time feature table over 378 archived trades is reusable by briefs 5 and 8
  whether or not any veto works.
- **US2** depends on US1's table.
- **US3** depends on US2's model. Separable: discrimination and P&L are different questions.
- **US4** depends on US3's gain.
- **US5** depends on acceptance, not merely on US4 completing.

### Parallel Opportunities

File authoring only, never runs: T002/T003; T005/T006; T016; T020; T023; T036.

---

## Implementation Strategy

**MVP** = Phases 1–3, ending at T021. Delivers the audited trade table and the three gates.

**Two kill points are reached before any integration work**: SC-009 at T031 (AUC < 0.55) and
SC-010 at T040 (gain is pair selection). Either stops the feature with a reportable finding,
and both land before the expensive integration in Phase 8.

**Expected non-obvious outcome**: T047's DSR/PBO will very likely report `CANNOT BE EVALUATED`,
because the trial log stores summary scalars and the only available axis is seeds. That is
recorded as unevaluable, not passed on an artefact — the same decision feature 010 reached.
Satisfying SC-004 as written needs a trial-log schema change, which is study-wide work and out
of scope here.

## Notes

- **Runs are sequential. Always.**
- Every gate in Phase 2 exists because the corresponding defect has already happened here:
  columns entering the feature list, a reproduction mismatch going unnoticed, and an audit that
  cannot fail.
- A negative result that satisfies the gates is a successful outcome and is recorded with the
  same weight as a positive one.
