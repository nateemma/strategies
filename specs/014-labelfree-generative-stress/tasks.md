---

description: "Task list for labelfree-generative-stress (brief 06)"
---

# Tasks: labelfree-generative-stress

**Input**: Design documents from `specs/014-labelfree-generative-stress/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/stress-cli.md, quickstart.md

**Tests**: INCLUDED. The decisive one (T014) asserts a synthetic bank and a real fold panel score
*identically on the same bars* — without it, a shape drift between the bank and the surrogate
would make every downstream number meaningless while everything still ran.

**Organization**: grouped by the four user stories.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependencies)
- **[Story]**: US1–US4 from spec.md
- All commands use `freqtrade/.venv/bin/python`, run from `user_data/strategies`, **sequentially**
  (FR-018). Stage only files this feature touches (FR-019).

**Build order differs from priority order, deliberately.** US1 (calibration) is P1 because it is
the feature's gate and its only kill, but it cannot be built first — it consumes the bootstrap US2
provides. US3 is built first because it costs minutes, has no dependencies, and is the documented
reason the design has one generator instead of four.

---

## Phase 1: Setup

- [X] T001 Create `research/labelfree/stress/__init__.py` with the package docstring naming brief 06, the single-generator design and the reason for it (research.md R7)
- [X] T002 Add the same interpreter guard feature 013 uses — `~/.pyenv/shims/python3` is a different 3.12.13 without pandas or scipy, so a missing import is far more likely to be the wrong interpreter than a missing install
- [X] T003 [P] Create `research/labelfree/stress/banks/` and add it to `.gitignore` — a 1,000-path bank over ten fold panels is large, and feature 013's 60 MB sidecar was untracked but not ignored until an analyze pass caught it

**Checkpoint**: the package imports under the venv and fails loudly elsewhere.

---

## Phase 2: Foundational (Blocking Prerequisites)

- [X] T004 Implement `provenance()` in `research/labelfree/stress/bank.py` per data-model.md §3 — generator, block length, seed, fold ids, path count, source range, feature columns, `source_sha256`, timestamp
- [X] T005 Implement `source_hash()` over the fold panels feeding a bank, so a bank whose source has moved is **detectable** (FR-002)
- [X] T006 [P] Implement `max_drawdown()` and `expected_shortfall()` in `research/labelfree/stress/calibrate.py`. Do NOT reuse `regime/phase_b_risk_normalize.py`'s drawdown — it operates on a log-return series, while this needs drawdown over a per-trade equity path; check before reusing, and state which was chosen

**Checkpoint**: provenance round-trips and a moved source is detected.

---

## Phase 3: User Story 3 - What the repo's generators can actually do (P3)

**Goal**: state, per generator, whether it can emit coherent sequences, verified against running code.

**Independent Test**: the census cites a `file:line` for every claim.

- [X] T007 [US3] (FR-004, SC-003) Implement `research/labelfree/stress/gencensus.py` per contracts/stress-cli.md — record `gan_type`, observed return shape, `emits_sequences`, `evidence` (file:line), `in_scope`
- [X] T008 [US3] (FR-004, SC-003) Determine capability from each implementation's **return contract in the running code, cited by `file:line`** — not from documentation, and not by instantiating a model. No weights are loaded, so the census cannot be broken by an untrained or missing checkpoint. FR-004 was reworded by the analyze pass to match what this can actually evidence; it previously claimed "actual output", which requires running `generate()`.
- [X] T009 [US3] **RUN** `python -m research.labelfree.stress.gencensus --out census.json`. Expect TabDDPM `(n,1,F)`, WGAN `(n,1,F)`, CTAB-GAN+ `DataFrame` all false; CGAN `(n,seq_len,F)` true but out of scope
- [X] T009a [US3] **FR-016 — the rebuilt implementations are authoritative.** TabDDPM, WGAN and CTAB-GAN+ were rebuilt on 2026-09-10; `docs/GAN_SCORECARD.md` predates that. Every figure this feature quotes from the scorecard MUST be either re-verified against the current code or explicitly labelled historical with its date. Do not carry a stale number forward silently.
- [X] T010 [US3] (FR-005, FR-009, SC-008) Record in the census that this is **not a defect** — they are class-conditional row samplers built for class balancing and they do that job. It is a statement about what they cannot be reused for.

**Checkpoint**: the pivot to a single generator is documented with evidence.

---

## Phase 4: User Story 2 - A path generator that does not need a GAN (P2)

**Goal**: a stationary block bootstrap over real per-pair panels, sampled offline into a bank.

**Independent Test**: sampled paths preserve within-block ordering, and a bank scores identically
to the real panel on the same bars.

### Tests for User Story 2 ⚠️

> Write these FIRST and confirm they fail.

- [X] T011 [P] [US2] `test_stress_bootstrap.py::test_within_block_ordering_preserved` — consecutive bars inside a block come from consecutive source bars; a bootstrap that shuffled within blocks would destroy exactly the autocorrelation the method exists to keep
- [X] T012 [P] [US2] `test_stress_bootstrap.py::test_geometric_block_lengths` — realised block lengths are geometric with mean `L`, not fixed. Fixed blocks make the series non-stationary (a bar's chance of starting a block depends on position mod `L`), which is why Politis & Romano use the geometric variant
- [X] T013 [P] [US2] `test_stress_bootstrap.py::test_all_columns_resampled_together` — prices, guards, cost **and** `p_buy`/`p_sell` come from the same source bar. If they ever decouple, the model's reaction no longer matches the market state and the path is a fiction (research.md R1)
- [X] T014 [US2] `test_stress_bank.py::test_bank_and_fold_panel_score_identically` — **the decisive test.** A degenerate bank (one block covering the whole series, no resampling) must produce exactly the surrogate's real-panel result. Without it, a shape drift between bank and surrogate is invisible.
- [X] T015 [P] [US2] `test_stress_bank.py::test_moved_source_is_refused` — a bank whose `source_sha256` no longer matches is **refused, not warned about**

### Implementation for User Story 2

- [X] T016 [US2] (FR-003) Implement the stationary bootstrap in `research/labelfree/stress/bootstrap.py` — uniform random start, continue with probability `1 − 1/L`, wrap at the series end
- [X] T017 [US2] Resample **every panel column together** by source index, per data-model.md §1 — one index vector drives all columns so they cannot decouple
- [X] T018 [US2] (FR-003) Draw **one block index and apply it to every pair**. The fold panels are date-aligned, so a shared index keeps the cross-section intact: when the index selects a moment, each pair contributes its own bar from that moment, and pairs that crashed together still do. Resampling pairs independently would destroy cross-pair co-movement and **understate joint drawdown** — and `dd_ratio` is one of the three calibration statistics, so the bias would point straight at a gate. Record the cost in the docstring: less path diversity, which is the correct trade.
- [X] T018a [P] [US2] `test_stress_bootstrap.py::test_shared_index_preserves_cross_pair_comovement` — the correlation between two pairs' resampled returns must stay close to their real correlation; independent resampling would drive it toward zero
- [X] T019 [US2] (FR-001, FR-002, SC-004, SC-005) Implement `bank.build()` / `bank.load()` writing `.npz` under `stress/banks/` with full provenance; `load()` recomputes the source hash and refuses on mismatch
- [X] T020 [US2] Run `pytest research/labelfree/tests/test_stress_bootstrap.py test_stress_bank.py` and iterate until green
- [X] T021 [US2] **RUN** `bank --build --folds f05..f14 --block 192 --paths 1000 --seed 7` and record the wall-clock and file size

**Checkpoint**: a reproducible bank exists and scores identically to real on undisturbed bars.

---

## Phase 5: User Story 1 - A generator whose paths are worth trusting (P1) 🎯 MVP

**Goal**: the Tail-GAN criterion — does the incumbent's P&L distribution on synthetic paths match
its distribution on real ones?

**Independent Test**: three numbers against three tolerances fixed before the run.

- [X] T022 [US1] (SC-002) Put the **pre-registered** tolerances in `calibrate.py` as constants with their derivation: `KS_MAX = 0.10` (at n=378 the 95% critical value against an identical distribution is 1.36/√378 = 0.0700, verified), `ES_RATIO = (0.7, 1.4)`, `DD_RATIO = (0.7, 1.6)` — the drawdown lower bound tightened from 0.6 by the analyze pass, to match ES, since "thinner than real" is the dangerous direction for both tail statistics. Read them from the design document — do **not** choose a tolerance after seeing a distance
- [X] T023 [US1] Document the **asymmetry** beside the constants: a synthetic tail fatter than real makes the harness conservative and is tolerable; thinner is the dangerous direction, because it would let a fragile policy through
- [X] T024 [US1] (FR-006, FR-007, SC-001) Implement `calibrate.run()` — score the incumbent across the bank, collect per-trade P&L, and compare against **the same policy scored by the same evaluator on the real panels**, on KS distance, 5% ES ratio and median max-drawdown ratio
- [X] T024a [US1] (FR-006) **Hold the evaluator constant.** Do NOT calibrate against the 378 archived freqtrade trades: that conflates synthetic-vs-real paths with the known surrogate-vs-freqtrade gap (level offset **−5.30 ± 23.46 pp** over 21 candidates, `roi` 22 vs a real 13 on f07). SC-001 is the only gate and a failure is a kill, so an unattributable failure could kill the feature for a reason that has nothing to do with the generator. Report the archive as a **secondary sanity reference**, never gating.
- [X] T025 [US1] (FR-008) Implement `require_calibration()`, imported by `run.py` and `report.py`, refusing to proceed without a passing `calibration.json`; exit non-zero on failure
- [X] T026 [US1] State in the module docstring that calibration is a **plumbing-and-tails check**: the bootstrap resamples real blocks, so the body should match by construction. What it can genuinely catch is spurious gaps at block joins firing stops, and a drawdown distribution wrong because resampling destroys long-range dependence (research.md R2). A near-guaranteed pass must not be reported as validation.
- [X] T027 [US1] **RUN THE GATE** on the L=192 bank and report all three numbers

**⚠️ If T027 fails the feature STOPS.** Clarification Q3 left no second generator, so a
calibration failure is a KILL (SC-007), not a fallback. Write the result report saying so and do
not proceed to Phase 6.

- [X] T028 [US1] (FR-003a, SC-006a) **SWEEP** the block length over {96, 144, 192, 240, 288} bars (1–3 days), rebuilding and recalibrating each, and report calibration **per `L`**. `L` is swept, not chosen — reporting one value as *the* answer would hide the sensitivity (Principle IV applied to this feature's one parameter)

**Checkpoint**: the generator is calibrated, or the feature is dead.

---

## Phase 6: User Story 4 - A stress verdict on a policy (P4)

**Goal**: a distribution over plausible re-orderings of real market conditions.

- [X] T029 [US4] Implement `run.py` — score the policy on every path with feature 013's exit surrogate, reused unchanged (31 ms per evaluation, rank fidelity Spearman +0.9766 against real backtests)
- [X] T030 [US4] Report a **distribution, never a point estimate**: P&L quantiles {5,25,50,75,95}, drawdown quantiles {50,90,95,99}, and the loss-path fraction (FR-011)
- [X] T031 [US4] (FR-013) Count a path on which the policy never trades as **inadmissible, not flat**, routing the check through `research/labelfree/admissibility.py::compare` with the **incumbent-relative exposure floor** feature 013 added — the absolute `MIN_EXPOSURE` marks production itself degenerate on a 90-day window (measured 0.0196 on f00). Do not reimplement the check locally; that habit is why the module was centralised. If inadmissible paths dominate, report the whole bank inadmissible rather than averaging zeros into a distribution.
- [X] T032 [US4] Name the output field `verdict ∈ {VETO, NO VETO, INADMISSIBLE BANK}` — **not** `passed`. "NO VETO" means the harness found no reason to object, not that the policy is good (SC-009, FR-014)
- [X] T033 [US4] Carry the incumbent's figures beside any candidate's (FR-012)
- [X] T034 [US4] (FR-010, SC-006) **RUN** the stress over ≥1,000 paths on the incumbent and record the loss-path fraction

**Checkpoint**: the harness has produced its first verdict.

---

## Phase 7: Polish & Cross-Cutting

- [X] T035 (FR-021, SC-011) Write `research/label-free-study/briefs/results/06-<date>.md` per protocol §6, leading with the generator census and the pivot it forced, then the calibration, then the stress distribution. Written **whatever the outcome**, including a calibration kill (FR-021)
- [X] T035a **FR-017 — assert that nothing was trained on synthetic data.** The brief's entire framing is that a generator cannot add information the real bars lack, so this is the claim most worth making checkable rather than implicit: grep the feature's own modules for any training entry point and record the result in the report.
- [X] T036 State in the report the limitations that are properties of the method, not discoveries: a shared block index preserves cross-pair co-movement but yields less diverse paths; the surrogate's residual divergence (`roi` 22 vs a real 13 on f07) is inherited by any stress verdict; and a resampled path cannot contain a market state the data has never shown
- [X] T037 Record that protocol §5's gates apply to the policies this harness vetoes, **not** to the harness itself (clarification Q1, SC-013)
- [X] T038 [P] (FR-015, SC-012) Note the multi-task variants as explicitly out of scope, nowhere else in the results (SC-012)
- [X] T039 [P] Update `research/label-free-study/README.md`'s Results table with brief 06's outcome
- [X] T040 Run `quickstart.md` end to end as written and fix any step whose stated expectation does not match what actually happens
- [ ] T041 Stage only the files this feature created or modified (FR-019); check `git status` first and leave anything unexpected alone
- [ ] T042 Re-run `/speckit-analyze` after implementation and record the clean report in the feature directory — **FR-020 and SC-010 require one before implementation and one again after**

---

## Dependencies & Execution Order

- **Phase 1 Setup** → no dependencies
- **Phase 2 Foundational** → blocks everything
- **Phase 3 US3** → independent; built first because it is cheap and is the design's justification
- **Phase 4 US2** → depends on Phase 2
- **Phase 5 US1** → depends on Phase 4. **T027 is a hard gate**: failure kills the feature
- **Phase 6 US4** → depends on T027 passing
- **Phase 7 Polish** → depends on Phase 6, except T035 which must also run on a kill

### Parallel Opportunities

Within phases only: T003/T006; T011–T013, T015 and T018a (distinct test functions); T038/T039.
Everything else is sequential by data dependency and by FR-018.

---

## Implementation Strategy

**MVP is Phases 1–5.** That is the feature's whole risk: a generator whose paths are worth
trusting. If T027's calibration fails, the MVP has still delivered a real finding — that a block
bootstrap of this repo's own panels cannot reproduce the incumbent's P&L distribution — and the
report says so.

**The calibration is expected to pass, and that is a weakness rather than a reassurance**
(research.md R2). The feature's informative output is the stress distribution in Phase 6: given
brief 05 measured the incumbent at ≈ +0.17 pp/yr on the pooled test folds under this same
surrogate, the prior is that a large share of sampled paths lose money. **That would be a finding,
not a failure.**

**Total**: 46 tasks — US1 8, US2 12, US3 5, US4 6, setup/foundational 6, polish 9.

Two tasks were added after a coverage check found requirements with no task at all:
**T009a** (FR-016) — the user's own constraint that the rebuilt generators are
authoritative and stale scorecard figures must be re-verified or labelled historical;
and **T035a** (FR-017) — asserting that nothing was trained on synthetic data, which is
the brief's headline claim and was relying on absence rather than a check.
