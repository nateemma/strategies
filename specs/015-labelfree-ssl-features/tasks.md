---

description: "Task list for labelfree-ssl-features (brief 07)"
---

# Tasks: labelfree-ssl-features

**Input**: Design documents from `specs/015-labelfree-ssl-features/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ssl-cli.md, quickstart.md

**Tests**: INCLUDED. Two are load-bearing rather than hygienic — the planted-leak test (T019),
because a causality check that cannot detect a leak it was handed is not evidence; and the
bar-reproduction test (T009), because the feature's gate is a *measured* quantity and a drifted
bar silently changes every verdict.

**Organization**: grouped by the four user stories.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependencies)
- **[Story]**: US1–US4 from spec.md
- All commands use `freqtrade/.venv/bin/python` from `user_data/strategies`, **sequentially**
  (FR-019). Stage only files this feature touches (FR-020).
- **Nothing is installed and nothing is downloaded** (FR-006, SC-016).

**Build order differs from priority order.** US1 (the screen) is P1 because it is the feature's
gate and its expected end, but it consumes the encoder US3 guards and the control US2 provides.
The bar is built first because everything is compared against it.

---

## Phase 1: Setup

- [ ] T001 Create `research/labelfree/ssl/__init__.py` with the package docstring naming brief 07, the TS2Vec-only scope and why (no `transformers`, no downloads), and the screen-gates-everything design
- [ ] T002 Add the interpreter guard used by features 013/014 — `~/.pyenv/shims/python3` is a different 3.12.13 without pandas or scipy — extended to require **MLX**, since this feature cannot run without it
- [ ] T003 [P] Add `research/labelfree/ssl/embeddings/` to `.gitignore`; per-fold embeddings at width 320 over ~130K bars are large and reproducible from the encoder

**Checkpoint**: the package imports under the venv and fails loudly elsewhere.

---

## Phase 2: Foundational — the bar, before anything is compared to it

**⚠️ Everything downstream is a comparison against this. If it drifts, every verdict changes
silently.**

- [ ] T004 Implement `inputs.py::engineered()` returning the 24-column `FeatureNormalizer.include_list` matrix via `features.build_features`
- [ ] T005 (FR-007a) Implement `inputs.py::ohlcv()` returning raw windows **converted to log returns and a volume ratio first** — `AGENT_GUIDE.md:546` prohibits raw price and raw volume as NN inputs because their ranges differ by pair, and that rule applies to the encoder's input as much as to a strategy's
- [ ] T006 Implement the three-way split in `screen.py::splits()` per data-model.md §3 — encoder on A, ridge on B, score on C, in time order within a fold
- [ ] T007 (FR-003) Apply purge and embargo at **every** boundary, not only A|B: a label at the end of B reaches `horizon` bars into C. Record the count applied.
- [ ] T008 Implement `screen.py::handbuilt_bar()` — the same ridge over the production feature set at the same horizon on the same split, returning ρ with a bootstrap interval
- [ ] T009 Write `test_ssl_bar.py::test_bar_reproduces_the_measured_ceiling` asserting the H=3/6/48 medians land near **0.172 / 0.113 / 0.083** (research.md R1, raw data in `research/labelfree/ceiling_by_horizon.json`). **The bar is the gate**; a drifted bar changes every verdict with nothing failing.
- [ ] T010 [P] Record in `screen.py`'s docstring that **ρ≈0.15 is a horizon-3-to-5 number, not a constant** — predictability decays from ~0.17 at H=3 to ~0.06 at H=96 — and that quoting it at H=48 would set a bar the baseline itself cannot clear

**Checkpoint**: the bar reproduces, per horizon, with an interval.

---

## Phase 3: User Story 3 - The encoder never sees the future (P3)

**Goal**: an encoder whose causality is demonstrated, not argued.

**Independent Test**: a bar's embedding is bit-equal when every later bar is deleted, and a
planted leak is detected.

- [ ] T011 [US3] (FR-004) Implement the dilated **causal** CNN backbone in `ts2vec.py` (MLX), 10 levels, width 320 — TS2Vec defaults, **not searched**; searching them across four screens is the multiple-testing failure the design guards against
- [ ] T012 [US3] Implement the hierarchical contrastive loss — temporal and instance-wise at successive pooling levels — with two random crops per instance
- [ ] T013 [US3] (FR-005) Put **every updated array** in the per-step `mx.eval`. Omitting one lets the graph grow until Metal's ~500K limit trips, a failure this repo has already recorded. Assert the step's evaluated-array count matches the optimiser's state count.
- [ ] T014 [US3] (FR-001) Fit per fold on the **training range only, without labels**; seed via `MLX_SEED`, since a framework-level `train_seed` was measured inert for multi-task models here
- [ ] T015 [P] [US3] `test_ssl_ts2vec.py::test_loss_decreases_and_embedding_is_not_constant` — a collapsed encoder scores ρ≈0 and would read as evidence about the hypothesis when it is encoder failure (FR-018)
- [ ] T016 [P] [US3] `test_ssl_ts2vec.py::test_seeded_reproducibility` under `MLX_SEED`
- [ ] T017 [US3] (FR-002) Implement `causality.py::deletion_test()` — recompute a bar's embedding with every later bar removed; require **bit-equality**
- [ ] T018 [US3] Implement `causality.py::planted_leak()` — inject the forward return as an extra input channel
- [ ] T019 [US3] (SC-003) `test_ssl_causality.py::test_planted_leak_is_detected` — **the load-bearing test.** The deletion test must pass *and* the planted leak must make the screen's ρ jump. A causality check that cannot see a leak it was handed is not evidence, and a screen run on an uncausal embedding is **void, not negative**.
- [ ] T020 [US3] **RUN** the causality pair on one fold; exit non-zero if either fails

**Checkpoint**: the embedding demonstrably cannot see the future, and the check can prove it would notice if it could.

---

## Phase 4: User Story 2 - A control that separates information from capacity (P2)

**Goal**: distinguish "the embedding helped" from "more inputs helped".

- [ ] T021 [US2] (FR-011) Implement `control.py` — a Gaussian block of **exactly the embedding's width**, through the **identical** ridge and splits. The ridge's capacity scales with input width, so a narrower control would measure width rather than information.
- [ ] T022 [P] [US2] `test_ssl_control.py::test_control_matches_embedding_width` — a width mismatch makes the comparison meaningless
- [ ] T023 [US2] (FR-012, SC-002, SC-012) Record a control scoring comparably to the embedding as **capacity, not information**, and make it a KILL

**Checkpoint**: any screen result can be attributed.

---

## Phase 5: Pair-agnosticism — the deployability gate

- [ ] T024 (FR-012a, SC-003a) Implement `pairagnostic.py` at the **pre-registered** ratio of data-model.md §5: reject if any dimension's between-pair location spread exceeds **0.5×** its within-pair scale. A ratio rather than an absolute, because embedding dimensions have arbitrary units.
- [ ] T025 (FR-012b) Report a rejection as a **finding** — an embedding whose scale depends on the pair cannot be deployed here however well it screens (`AGENT_GUIDE.md:545`) — never normalise it away silently
- [ ] T026 [P] `test_ssl_pairagnostic.py::test_threshold_is_the_pre_registered_one` and `::test_a_pair_scaled_embedding_is_rejected` — a gate that cannot fail is not a gate

---

## Phase 6: User Story 1 - The screen (P1) 🎯 MVP

**Goal**: learn, for one session and without a backtest, whether the embedding beats the
hand-built features.

- [ ] T027 [US1] (FR-007) Implement `screen.py::run()` — nested ridge over the three-way split, Spearman ρ on split C, with a bootstrap interval
- [ ] T028 [US1] (FR-008) Fit the ridge on **B only** and score on **C only**. A ridge fitted and scored on the same bars beats any bar, which is how a null becomes a false positive — the spec's own "projection overfits the screen" edge case.
- [ ] T029 [US1] (FR-007a, SC-001a) Run all four combinations — {engineered, ohlcv} × {6, 48} — reporting `rho_embedding`, `rho_handbuilt`, `rho_control` computed **identically** for each
- [ ] T030 [US1] (FR-007b) Mark **`engineered` × 48 as PRIMARY**, fixed before any screen runs, and record the other three as secondary/exploratory. Four screens against one threshold is four chances to clear a bar meant to be cleared once — brief 05 measured that failure at PBO 0.7128.
- [ ] T031 [US1] (FR-009, SC-001) Compare each screen against **its own** bar with an interval, never a bare comparison against a constant
- [ ] T032 [US1] (FR-010, SC-004, SC-011) Exit non-zero and **forbid any backtest** when the primary does not beat its own bar
- [ ] T033 [US1] **RUN THE SCREEN** across f05..f14 and report all four combinations

**⚠️ If T033's primary fails, the feature STOPS.** That is the expected path (research.md R8):
a learned 1D-CNN on raw OHLCV already scored ρ 0.04–0.06 in this repo, and an 85-indicator
battery left ρ invariant. Write the result report and do not proceed to Phase 7.

**Checkpoint**: the hypothesis has a number attached to it.

---

## Phase 7: User Story 4 - Only a passing screen earns a backtest (P4)

> Reached only if T033's primary passes.

- [ ] T034 [US4] (FR-013) Build an `NNNC_MLX` **subclass**; the production class is never modified
- [ ] T035 [US4] (FR-014) Let it **retrain** — nothing pinned, since the input set changed
- [ ] T036 [US4] (FR-015) **Regenerate scalers** after the input set changes, or the validation curve is meaningless
- [ ] T037 [US4] (SC-005, SC-006, SC-007, SC-008, SC-009, SC-010) Run the pooled walk-forward under protocol §5 and report every gate: return vs control + 3.35 pp/yr with a bootstrap interval, ≥10/15 folds and each of W1/W2/W3, all four baselines, PBO ≤ 0.2 with DSR > 0, the delta at 1.5× fees, and reproduction from config
- [ ] T037a [US4] (SC-013) If the screen passes but the pooled walk-forward does not clear SC-005, record the verdict as **"not resolved"**, never "no effect". A screen that found learnability followed by a backtest that found no edge is the study's own Principle III in one result — learnability is not edge — and calling it "no effect" would discard that distinction

---

## Phase 8: Polish & Cross-Cutting

- [ ] T038 (FR-016, SC-015) Write `research/label-free-study/briefs/results/07-<date>.md` per protocol §6, leading with the measured bar and its horizon dependence. Written **whatever the outcome**, and a screen KILL is the expected one.
- [ ] T039 (FR-017) Report ρ for the embedding, the random control and the hand-built baseline **on the same data**, for all four combinations
- [ ] T040 State in the report that the screen is a **learnability** measure and therefore necessary-but-not-sufficient: this repo has recorded a +0.155 MCC producing identical trades, so passing earns a backtest and never an acceptance (Principle III)
- [ ] T041 [P] Record that ρ≈0.15 is horizon-dependent, with the five-fold table, so the next brief does not quote it as a constant
- [ ] T042 [P] (SC-016) Confirm and state that no package was installed and no weights downloaded
- [ ] T043 [P] Update `research/label-free-study/README.md`'s Results table with brief 07's outcome
- [ ] T044 Run `quickstart.md` end to end and fix any step whose stated expectation does not match what happens
- [ ] T045 (FR-020) Stage only files this feature created or modified; check `git status` first and leave anything unexpected alone
- [ ] T046 (FR-021, SC-014) Re-run `/speckit-analyze` and record the clean report — one before implementation and one after

---

## Dependencies & Execution Order

- **Phase 1** → no dependencies
- **Phase 2** → blocks everything; the bar is what all else is compared against
- **Phase 3 (US3)** → depends on Phase 2. **T019/T020 gate the screen**: an uncausal embedding makes it void
- **Phase 4 (US2)** → depends on Phase 3 (needs the embedding's width)
- **Phase 5** → depends on Phase 3
- **Phase 6 (US1)** → depends on Phases 2–5. **T033 is the hard gate**
- **Phase 7 (US4)** → only on a passing primary
- **Phase 8** → depends on Phase 6, except T038 which also runs on a kill

### Parallel Opportunities

Within phases only: T003/T010; T015/T016; T022; T026; T041/T042/T043. Everything else is
sequential by data dependency and FR-019.

---

## Implementation Strategy

**MVP is Phases 1–6.** The screen is the feature. If the primary fails, the MVP has delivered the
brief's expected result — a number closing the door on self-supervised embeddings as extra
inputs — for one session rather than a week of backtests.

**The expected outcome is a KILL at T033**, and the reasons are measured rather than assumed: a
learned 1D-CNN on raw OHLCV scored ρ 0.04–0.06 here, an 85-indicator battery left ρ invariant,
and the study's own conclusion is that ρ≈0.15 at short horizons is an information limit rather
than a modelling problem.

**Total**: 47 tasks — US1 7, US2 3, US3 10, US4 5, setup/foundational 10, pair-agnostic 3,
polish 9.

T037a was added by a coverage check that found **SC-013 had no task at all**: a screen that passes
followed by a backtest that fails must be recorded as *"not resolved"*, never *"no effect"*. That
distinction is the study's own Principle III — learnability is not edge — and losing it would
discard the most interesting outcome this feature can produce.
