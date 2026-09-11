---

description: "Task list for 010-labelfree-warmstart-finetune"
---

# Tasks: Label Warm-Start with P&L Fine-Tune

**Input**: Design documents from `specs/010-labelfree-warmstart-finetune/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Test tasks ARE included. Two are blocking gates the specification mandates — leakage
control (SC-012) and warm-start transfer — plus value-asserting regressions on every shared
module this feature extends.

**Organization**: Grouped by user story. Brief 01's calibration is inherited and NOT re-derived;
the blocking gates here are leakage and transfer.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel — **file-authoring tasks only**.
- **[Story]**: US1–US4, mapping to spec.md user stories.

> **Cross-cutting constraints.** FR-001 (reuse brief 01's modules, never fork them), FR-003
> (fix any defect found in shared code in place, with a value-asserting test), FR-023
> (sequential runs) and FR-024 (stage only this feature's files) are continuous obligations,
> not single tasks. They apply to EVERY task here.

> ⚠️ **Repo-specific meaning of [P].** AGENT_GUIDE requires every *job* to run sequentially. `[P]`
> marks only tasks that write different files with no shared dependency. **No task that executes
> a run is ever marked `[P]`**, and tasks prefixed **RUN** execute one at a time.

> ⚠️ **Memory constraint.** Window tensors scale as `bars × seq_len × features × pairs` and reach
> **2.14 GB** on the full train panel (measured). They MUST be built per chunk (0.04 GB), never
> for the whole panel.

## Path Conventions

Repository root is `user_data/strategies/`. New code joins brief 01's in `research/labelfree/`;
`NNNC/NNNClassifierMLX.py` is imported read-only and never modified.

---

## Phase 1: Setup

**Purpose**: Confirm the inherited components are healthy before extending them.

- [X] T001 RUN `research/labelfree/preflight.py` and `pytest research/labelfree/tests/ -q` to confirm brief 01's 16 tests pass and the simulator's inputs are present; a failure here is a brief-01 regression and must be fixed before anything is built on it
- [X] T002 [P] Record brief 01's P&L control figures in `specs/010-labelfree-warmstart-finetune/research.md` — `utility-penalty-g0.5`, mean exposure 0.130, −7.06% held-out — as the pre-existing reference point for this feature's architecture re-run (FR-004b)

---

## Phase 2: Foundational — LEAKAGE GATE (Blocking Prerequisites)

**Purpose**: Every arm here has a forward-looking term. Nothing downstream is evidence until
leakage control is proven applied.

**⚠️ CRITICAL**: No arm may run until T006 passes.

- [X] T003 [P] Implement label construction in `research/labelfree/labels.py` — directional labels at both `MIN_BUY_GAIN_THRESHOLD` values (0.007, 0.02) via `Framework/TrainingSignals.py`, aligned to the panel's shared date index (FR-008, data-model §3)
- [X] T004 [P] Implement the forward-realised-volatility target in `research/labelfree/labels.py` — a magnitude carrying no directional information, computed causally from the same OHLCV the features use (FR-011, research R5)
- [X] T005 Implement purge and embargo in `research/labelfree/labels.py` — purge the label horizon at each fold boundary, embargo one horizon after it, and record the values ACTUALLY applied per fold rather than the values requested (FR-013, FR-014)
- [X] T006 [P] Leakage test in `research/labelfree/tests/test_purge_embargo.py` asserting that for every fold no target value within one horizon of the boundary survives, and that a run with either control at zero is rejected rather than caveated (SC-012)
- [X] T007 RUN `pytest research/labelfree/tests/test_purge_embargo.py -q`; a failure blocks every arm

**Checkpoint**: 🚦 Leakage control proven. Arms may be built.

---

## Phase 3: User Story 1 - Warm-start from the classifier (Priority: P1) 🎯 MVP

**Goal**: Determine whether the classifier's representation is a better starting point than
random initialisation — the question brief 01 could not answer.

**Independent Test**: Train Arm A on the screening fold across ≥5 seeds; compare held-out return
and mean exposure against the P&L control on the same fold and seeds.

### Tests for User Story 1

- [X] T008 [P] [US1] Warm-start transfer test in `research/labelfree/tests/test_warmstart.py` asserting inherited tensor names and shapes match the policy encoder exactly, that post-load weights DIFFER from a randomly-initialised policy of the same architecture, and that the classification head is absent from the inherited set (contracts/warm-start.md)
- [X] T009 [P] [US1] Learning-rate test in `research/labelfree/tests/test_warmstart.py` asserting `encoder_lr * 10 <= head_lr` is enforced rather than merely configured (FR-005)

### Implementation for User Story 1

- [X] T010 [P] [US1] Add the classifier-encoder policy variant to `research/labelfree/policy.py` — resize → conv1 → ln_conv → residual → LSTM → ln_lstm → exposure head, matching `NNNClassifierMLX_LSTM.encode()` tensor-for-tensor so weights load without surgery (FR-004a, research R1)
- [ ] T011 [US1] Implement per-fold classifier training in `research/labelfree/warmstart.py` — trained on the fold's OWN training range; the production model MUST NOT be reused, because it saw data later folds treat as held out (FR-004, research R3)
- [X] T012 [US1] Implement weight inheritance in `research/labelfree/warmstart.py` — transfer the encoder only, never the classification head, and record which tensors actually transferred (data-model §2)
- [ ] T013 [US1] Implement windowed encoding in `research/labelfree/train.py` — build `(n_bars, seq_len, n_features)` windows PER CHUNK, never for the whole panel (2.14 GB vs 0.04 GB, measured) (FR-004a, research R2)
- [ ] T014 [P] [US1] Implement arm dispatch in `research/labelfree/arms.py` — A/B/C as configurations of the existing trainer, differing only in initialisation and objective (FR-002, contracts/arms.md)
- [ ] T015 [US1] RUN the per-fold classifier for f08, then Arm A at seeds 1-5 via `research/labelfree/arms.py --arm warm_start --fold f08`, reported as an interquartile mean with a bootstrap interval, never a best seed (FR-016)
- [ ] T016 [US1] Compute the exposure/label correlation in `research/labelfree/arms.py` and report it as a DIAGNOSTIC — near 1 means the fine-tune changed nothing, near 0 means the prior was discarded; it MUST NOT enter any acceptance decision (FR-006, SC-011, Principle III)
- [ ] T017 [US1] If Arm A underperforms the P&L control in-fold, record it in `research/labelfree/trials.jsonl` and proceed to Phase 4 — do NOT tune Arm A; a classifier tuned to the wrong label can legitimately hurt, and tuning here is fitting the screen (US1 scenario 4, Principle VI)

**Checkpoint**: The initialisation question is answered independently of the other arms.

---

## Phase 4: User Story 2 - Blended objective (Priority: P2)

**Goal**: Test the literature's central claim that the best result sits at a blend of label and
profit objectives rather than at either extreme.

**Independent Test**: Sweep the blend weight across both labels on f08 at ≥5 seeds; confirm every
configuration is logged before running and reported as an interquartile mean.

### Implementation for User Story 2

- [ ] T018 [P] [US2] Implement the blended objective in `research/labelfree/losses.py` — `α·CE(label) + (1−α)·utility`, with the participation floor applied as inherited from brief 01 (FR-007)
- [ ] T019 [US2] Implement loss-term accounting in `research/labelfree/losses.py` — record each term's REALISED contribution and `alpha_realised = label_term/(label_term+profit_term)`, so a blend that has collapsed to one term is detected rather than assumed balanced (FR-009, data-model §4, Principle IV)
- [ ] T020 [US2] RUN the sweep via `research/labelfree/arms.py --arm blended --fold f08 --alphas 0.75,0.5,0.25 --labels 0.007,0.02 --seeds 1,2,3,4,5` — 30 runs, sequential, each reported as an interquartile mean across seeds (FR-016)
- [ ] T021 [US2] Report the two labels SEPARATELY in `research/label-free-study/briefs/results/02-<date>.md`, never pooled, and record that the repo previously measured them as identical in P&L (332 trades, 10.45% both) while 38% apart in label count — agreement here replicates a known finding rather than discovering one (FR-008, research R4)
- [ ] T022 [US2] Record every blend weight as a trial in `research/labelfree/trials.jsonl`; the report MUST NOT identify a best α — the sweep is 6 configurations and enters the multiple-testing correction as such (FR-010, SC-010)

**Checkpoint**: The blend question is answered, with the collapse case detectable.

---

## Phase 5: User Story 3 - Auxiliary non-directional target (Priority: P3)

**Goal**: Separate "labels help because they carry direction" from "any auxiliary supervision
regularises the objective".

**Independent Test**: Train at ≥5 seeds; compare against both the P&L control and the best
blended arm.

### Implementation for User Story 3

- [ ] T023 [P] [US3] Implement the auxiliary head and loss in `research/labelfree/losses.py` — utility plus a forward-volatility prediction from the shared latent, with NO directional label entering the objective at any point (FR-011)
- [ ] T024 [P] [US3] Assertion in `research/labelfree/tests/test_purge_embargo.py` that Arm C's objective contains no directional term, checked structurally rather than by inspection (FR-011)
- [ ] T025 [US3] RUN Arm C via `research/labelfree/arms.py --arm auxiliary --fold f08 --seeds 1,2,3,4,5`, confirming no directional label enters the objective (FR-011, FR-016)
- [ ] T026 [US3] State in `research/label-free-study/briefs/results/02-<date>.md` whether directional content was necessary, by comparing Arm C against the best blended arm (FR-012)

---

## Phase 6: User Story 4 - Verdict against both controls (Priority: P4)

**Goal**: Convert arm results into the directional decision the study needs — whether labels
should be dropped or kept as a prior.

**Independent Test**: Run surviving arms and both controls through the pooled harness; the report
states the delta against each control with its interval, measured against the floor.

### Implementation for User Story 4

- [ ] T027 [US4] Apply the degeneracy screen in `research/labelfree/arms.py` — any arm whose mean exposure falls below the participation floor is reported as INCONCLUSIVE and excluded from carry-forward, never reported as a return (FR-020, SC-015)
- [ ] T028 [US4] Select surviving arms in `research/labelfree/arms.py` by the pre-registered rule inherited from `research/labelfree/sweep.py`, ranking on the HELD-OUT range only (FR-017)
- [ ] T029 [US4] RUN `research/labelfree/warmstart.py` for the pooled run — 15 classifiers, one per fold, each on its own training range; this is the feature's dominant compute cost and is budgeted separately (research R3)
- [ ] T030 [US4] RUN the pooled 15-fold walk-forward for surviving arms via `regime/wf/run_folds.py`, sequential (FR-018, SC-001)
- [ ] T031 [US4] RUN the P&L control re-run under THIS feature's architecture via `research/labelfree/arms.py --arm pnl_control`; comparing against brief 01's recorded numbers would confound the label prior with the architecture change (FR-004b)
- [ ] T032 [US4] RUN `regime/wf/pool.py` then `regime/wf/compare_pooled.py` against BOTH controls — the label control (`regime/wf/pooled_nnnc.json`) and the re-run P&L control (FR-018, SC-001)
- [ ] T033 [US4] RUN single-window evaluations for W1/W2/W3 via `regime/wf/run_folds.py`, using universes from `research/labelfree/folds.py`, with each control on the identical pair universe within each window, reporting universe size per window (SC-002)
- [ ] T032a [P] [US4] Implement `research/labelfree/baselines.py` — equal-weight buy-and-hold of the whitelist, and the PINNED 20/60-bar EMA trend rule with band ±0.1, frozen before any run and contributing no trial-log entries (it has no parameter learned from P&L). Brief 01 planned this module but never built it, because its Phase 6 was unreached (SC-003)
- [ ] T032b [P] [US4] Implement `research/labelfree/reproduce.py` — reload a saved arm configuration and seed in a fresh process and re-emit its trades. Brief 01 planned this module but never built it (SC-006)
- [ ] T033a [US4] RUN all four mandatory baselines via `research/labelfree/baselines.py` — the production control, equal-weight buy-and-hold of the whitelist, the pinned 20/60-bar EMA trend rule with band ±0.1, and the control at its cost-aware operating point — net of full costs on the identical universe (SC-003)
- [ ] T033b [US4] Compute the in-sample overfitting differential in `research/labelfree/stats.py` — the screening-fold advantage minus the held-out advantage over the control; if it exceeds 3.35pp/yr, record overfitting and stop the configuration (SC-008)
- [ ] T034 [US4] RUN cost stress at 1.5× fees on any surviving configuration via `research/labelfree/arms.py --stress 1.5` (SC-005)
- [ ] T035 [US4] Report BOTH P&L-control figures in `research/label-free-study/briefs/results/02-<date>.md` — re-run under this architecture and brief 01's original — so the architecture effect is visible separately from the label-prior effect (FR-004b, SC-010a)
- [ ] T036 [US4] Extend `research/labelfree/stats.py` to emit mean exposure, trades per year and mean holding time beside EVERY return, so a degenerate arm cannot rank on inaction (FR-019, SC-014)
- [ ] T037 [US4] Compute DSR and PBO over the FULL trial log in `research/labelfree/stats.py`, including every arm, blend weight and label; report the configuration count beside the figure, since CSCV is biased low at small counts (FR-004 of protocol §4, SC-004)

---

## Phase 7: Reproduction and close

**Purpose**: Per the brief, the final substantive tasks are the protocol §6 report and the
trial-log audit.

- [ ] T038 [US4] RUN `research/labelfree/reproduce.py` for any accepted configuration from its saved config and seed in a fresh process; assert identical trades (SC-006)
- [ ] T039 Re-run `/speckit-analyze` on the completed feature and append the report to `specs/010-labelfree-warmstart-finetune/analyze.md`; fix every CRITICAL and re-run until clean (FR-025, SC-016)
- [ ] T040 RUN the trial-log audit via `research/labelfree/trials.py::audit`; a failed audit invalidates the overfitting statistics and therefore the verdict (FR-015, SC-013)
- [ ] T041 Write `research/label-free-study/briefs/results/02-<date>.md` per protocol §6 — hypothesis, config and seeds, trials logged, pooled return with CI, folds positive, trades/yr, hold time, both control figures, cost stress, DSR/PBO, verdict, and **whether labels helped or hurt** relative to the P&L control. A delta below the floor reads "not resolved", never "no effect" (FR-021, SC-007). **Written even on a kill** — the directional finding is the deliverable (FR-022, FR-026, SC-009)
- [ ] T042 Stage ONLY the files this feature created or modified and commit to `master`; never `git add -A` (FR-024)

---

## Dependencies & Execution Order

```
Phase 1 (Setup)
   ↓
Phase 2 (LEAKAGE GATE)   ← 🚦 blocks every arm; each has a forward-looking term
   ↓
Phase 3 (US1 Warm-start) ← MVP: answers brief 01's open question
   ↓
Phase 4 (US2 Blended)    ← independent of US1's outcome; runs even if Arm A underperforms
   ↓
Phase 5 (US3 Auxiliary)
   ↓
Phase 6 (US4 Verdict)    ← needs at least one non-degenerate arm
   ↓
Phase 7 (Analyze → audit → report)
```

### User Story Dependencies

- **US1** depends only on Phase 2. It is the MVP and answers brief 01's ambiguity on its own.
- **US2** depends on Phase 2, not on US1 — by design, US1 scenario 4 routes an underperforming
  Arm A here rather than tuning it.
- **US3** depends on US2 only for its comparison, not for its execution.
- **US4** depends on at least one arm producing a configuration that trades.

### Parallel Opportunities

File-authoring only, never runs:

- Phase 2: T003, T004, T006 (T005 shares `labels.py` with T003/T004 and is therefore serial)
- Phase 3: T008, T009 (tests) and T010, T014
- Phase 4/5: T018 and T023 both edit `losses.py` — **not** parallel

## Implementation Strategy

### MVP (User Story 1 only)

Phases 1–3, ending at T017. This delivers the answer to brief 01's central ambiguity: whether the
failure was the objective or the optimisation. It is a genuine standalone result — if a
warm-started policy trades and profits where a randomly-initialised one did not, the objective is
exonerated and the whole programme's direction changes.

### Incremental Delivery

1. **Phases 1–2** → leakage control proven; nothing downstream can be silently invalid
2. **Phase 3** → the initialisation answer (MVP)
3. **Phase 4** → the blend answer, with collapse detectable
4. **Phase 5** → whether direction was necessary or any auxiliary target would serve
5. **Phases 6–7** → verdict against both controls, reproduction, audit, report

### Budget

The brief allows 1–2 sessions. Per-run training is not expected to exceed brief 01's ~200 s (the
windowed encoder measured 0.6× the streaming pass). The dominant costs are Arm B's 30 screening
runs and T029's 15 per-fold classifiers.

## Notes

- **Sequential runs are mandatory.** `[P]` marks file authoring only.
- **Build windows per chunk**, never panel-wide — 2.14 GB vs 0.04 GB.
- **A kill verdict still completes Phase 7.** The directional finding — did labels help or hurt —
  is the deliverable regardless of acceptance.
- **Never weaken a gate.** If a protocol gate proves hard, escalate as a finding.
- **Test values, not shapes.** Four of feature 009's defects were wrong-axis slices that MLX
  broadcast silently into plausible shapes and wrong numbers.
