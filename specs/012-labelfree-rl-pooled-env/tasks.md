# Tasks: labelfree-rl-pooled-env

**Feature**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md) · **Research**: [research.md](research.md)
**Data model**: [data-model.md](data-model.md) · **Contract**: [contracts/rl-cli.md](contracts/rl-cli.md)
**Validation**: [quickstart.md](quickstart.md)

## Format: `[ID] [P?] [Story] Description`

`[P]` = parallelisable (different files, no dependency on incomplete work). **File authoring
only — RUNS ARE NEVER PARALLEL.** Vectorised envs parallelise *inside* one run; two runs never
overlap.

## Path Conventions

Paths relative to `user_data/strategies/`. Commands from the **repo root** with
`PYTHONPATH=user_data/strategies`.

---

## Phase 1: Setup

- [X] T001 Create `research/labelfree/rl/__init__.py` and `research/labelfree/tests/test_rl.py`
- [X] T002 [P] Pin constants in `research/labelfree/rl/__init__.py`: `SCREEN_FOLD = "f07"`, `LIQUIDITY_FRACTION` imported from `sim` (never re-declared), `EXEC_LAG_BARS = 2`, `MIN_EXPOSURE`, `N_SEEDS = 5`, `STEPS_PER_FOLD`, with each value's source in the docstring
- [X] T003 [P] Assert the dependency surface in `test_rl.py` — gymnasium ≥1.3, stable-baselines3 ≥2.9, sb3-contrib ≥2.9, torch ≥2.13 — so a version drift fails loudly rather than changing PPO's behaviour silently

---

## Phase 2: Foundational — THE TWO GATES (Blocking)

**Nothing downstream is evidence until both pass.**

### Gate A — the env reproduces the simulator bit-for-bit (R1)

- [X] T004 Implement `research/labelfree/env.py` step economics in numpy, indexed by **`i`, the decision bar** (research R2 — one index symbol, feature-wide): `requested = |w_i − w_{i−1}|` with `w_{−1} = 0`, `executed = min(requested, LIQUIDITY_FRACTION × quote_volume_i)`, `reward = w_i × bar_return_i − cost_i × executed` (FR-002, FR-003)
- [X] T005 Implement the execution lag in `env.py`: `bar_return_t = log(open_{t+2}) − log(open_{t+1})`, matching `sim.bar_returns` (R2). An episode of *T* decisions consumes *T+2* bars
- [X] T006 Implement `research/labelfree/rl/equivalence.py` — step a path through `env.py` and compare accumulated rewards to `sim.net_returns` at **`rtol=0`** (FR-001, R1)
- [X] T007 [P] Cover the four cases a naive step gets wrong, in `test_rl.py`: the `w_{−1}=0` opening boundary, a path still in position at the end, the liquidity cap binding, and a zero-volume bar
- [X] T008 RUN `equivalence.py --folds f07 --n-paths 200`; require `max|diff| = 0.00e+00` (quickstart S1). **A mismatch halts the feature** — an unverified restatement is a different simulator wearing the same name

### Gate B — the environment can be reward-hacked (R5, SC-011)

- [X] T009 Implement `research/labelfree/rl/rewards.py` (FR-006) with two specs: `realised` (the only one that may produce a result) and `freqai_shaped` (+25/entry, −1/idle bar, `pnl × 100` on exit) — the positive control
- [X] T010 Make `freqai_shaped` structurally unable to produce a reported figure: runs using it are tagged `probe: true` and `evaluate.py` refuses them
- [X] T011 RUN the probe — train briefly on `freqai_shaped` and on `realised` at the same budget, and assert turnover under the shaped reward is **at least 2×** the realised run's, a multiple pinned before the probe runs. "Measurably higher" without a threshold passes on a 1% rise (analyze M1, quickstart S2)
- [X] T012 [P] Test in `test_rl.py` that the probe assertion can FAIL — feed it two identical turnover series and confirm it reports failure. **A control that cannot fail certifies nothing**; feature 010's liveness control severed nothing on its first version, and feature 011's planted leak needed a separation check before its threshold meant anything

**Checkpoint**: T008 and T011 both green. Until then no policy may be trained.

---

## Phase 3: User Story 1 - An environment that cannot be cheated (Priority: P1) 🎯 MVP

**Independent test**: the env pays realised economics, masks invalid actions, terminates on
drawdown — and demonstrably conveys reward shape.

- [X] T013 [US1] Implement action masking in `env.py` — "sell" invalid when flat, **masked not penalised** (FR-004)
- [X] T014 [US1] Implement drawdown termination (FR-005), and keep `terminated` (drawdown) distinct from `truncated` (window exhausted). Conflating them corrupts PPO's bootstrap at the boundary: a truncated episode's value must be bootstrapped, a terminated one's must not (R6)
- [X] T015 [P] [US1] Test that `terminated` and `truncated` are never both set, and that a truncated episode carries a bootstrappable value
- [X] T016 [P] [US1] Test that the reward contains no entry bonus, idle penalty or take-profit multiplier — assert a flat step's reward is exactly 0.0 and an entry's reward equals `−cost × executed` (FR-003)
- [X] T017 [US1] Fail the run on a non-finite reward or loss rather than letting it propagate (FR-007)
- [X] T018 [US1] Document the env's observation/action/reward contract in the module docstring, for briefs 5 and 6

**Checkpoint**: US1 is a standalone deliverable — an auditable pooled trading environment,
reusable whether or not PPO works.

---

## Phase 4: User Story 2 - One shared policy across all pairs (Priority: P2)

- [X] T019 [US2] Implement `research/labelfree/rl/episodes.py` — sample `(pair, start, length)` windows of 30–90 days from the fold's **training** range only (FR-009)
- [X] T020 [US2] Enforce the **two trailing bars** in episode validation: window + 2 bars must lie inside the training range. An episode borrowing `open[i+1]`/`open[i+2]` from the test range leaks by exactly two bars and nothing else would report it (R2)
- [X] T021 [P] [US2] Test with 5000 sampled episodes that no window, including its trailing bars, touches the test range (quickstart S3)
- [X] T022 [US2] Reject and **count** episodes containing a data gap; never interpolate
- [X] T023 [US2] Implement the shared-policy training loop in `research/labelfree/rl/train.py` — one parameter set for every pair, no per-pair adaptation (FR-008)
- [ ] T024 [P] [US2] Test that a pair absent from a fold's training range still receives an action at evaluation, and that the case is counted

---

## Phase 5: User Story 3 - Selection that never sees the test fold (Priority: P3)

- [X] T025 [US3] Select checkpoints on **training reward only** (SC-012), and write a `SelectionRecord` carrying the quantity used and the range it came from (FR-012)
- [X] T026 [US3] Make `train.py` refuse to record a selection whose range is the fold under evaluation
- [ ] T027 [P] [US3] Test that a selection record naming the evaluation fold is rejected (quickstart S5). Selecting on the test split is the specific defect the brief cites in FreqAI's `best_model.zip`
- [X] T028 [US3] Pre-register every configuration to `research/labelfree/trials.jsonl` **before** it runs (FR-013)

---

## Phase 6: User Story 4 - Seeds before variants (Priority: P4)

- [ ] T029 [US4] Implement `research/labelfree/rl/liveness.py` — one probe per swept parameter, each **in its active regime** (FR-019, SC-015)
- [ ] T030 [US4] Probe `band_width` specifically across seeds. R4 predicts a **fixed** band avoids brief 02's bimodality, where a *learned* band made turnover a coin flip between two attractors. If this shows bimodality the continuous arm inherits the defect and is **reported, not tuned**
- [ ] T031 [US4] Make `train.py` refuse to run until every swept parameter probes LIVE (quickstart S4)
- [ ] T032 [US4] Report the interquartile mean over ≥5 seeds with a bootstrap interval, never a best seed; seeds with identical trades count once (FR-014, FR-015)
- [ ] T033 [US4] Attach mean exposure, trades/yr and mean holding time to **every** return, so a policy that learned not to trade cannot rank on inaction (FR-016, SC-013)
- [ ] T034 [US4] Emit `degenerate: true` with the **return withheld** when mean exposure falls below the participation floor (FR-017, R8). Brief 1's six best-ranked configurations were all below it, two flat on 100% and 99.9% of bars — and a policy-gradient method finds that optimum faster
- [ ] T035 [P] [US4] Test that a degenerate arm's return is withheld and cannot be read from the artefact (quickstart S7)
- [ ] T035a [US4] Implement `research/labelfree/rl/evaluate.py` — turn a trained policy into an exposure path, apply the participation floor, and emit the arm record. The plan's structure names it and T037/T038/T042–T046 use it, but no task created it (analyze M3)
- [ ] T036 [US4] RUN the screening fold: `--fold f07`, **both** action parameterisations, 5 seeds each, sequentially — compute goes to seeds, never to algorithm comparison (FR-010, FR-011, quickstart S6)
- [ ] T036a [US4] Pre-register the operational definition of "distinguishable" in `research/labelfree/rl/__init__.py` **before the screen runs**: the RL and brief-1 exposure paths must differ on **more than 5% of decision bars** on the screening fold, at every seed. A bar defined after seeing the screen is not a bar (analyze H1)
- [ ] T037 [US4] Apply the SC-018 screen bar in `research/labelfree/rl/evaluate.py --screen`: **non-degenerate** (mean exposure above the participation floor) **and distinguishable** per T036a's pinned threshold. Deliberately weak — time is not the binding constraint, so a marginal screen is funded and only a degenerate or path-identical one is not. Print `FUND` or `STOP`
- [ ] T038 [US4] Make `--pooled` refuse to run without a `FUND` screen on record

---

## Phase 7: User Story 5 - The head-to-head (Priority: P5)

**Prerequisite discovered by analyze C1**: brief 1 persists **no trained policy**. `sweep.py`,
`train.py` and `arms.py` write JSON results only, never weights, so there is no artefact to
compare against and SC-009 cannot execute without rebuilding one.

- [ ] T038a [US5] Add policy persistence to `research/labelfree/rl/headtohead.py`'s dependency path: re-train brief 1's best configuration (`utility-penalty-g0.5`, its reported f08 optimum) on the screening fold at the same seeds, and **save the weights** to `research/labelfree/rl/brief1_policy.npz`. Budgeted as its own work item, not folded into T039 (analyze C1)
- [ ] T038b [P] [US5] Test that the persisted policy reloads and reproduces its own training-range exposure path bit-for-bit — a comparator that does not reload deterministically makes the head-to-head unrepeatable
- [ ] T038c [US5] Record in the trial log that brief 1's comparator was **re-trained for this feature**, not recovered from brief 1's own run. Its numbers are therefore a fresh draw, not brief 1's reported figures, and the report must say so

- [ ] T039 [US5] Implement `research/labelfree/rl/headtohead.py` — both policies evaluated **in the simulator**, identical folds, costs and liquidity caps (FR-018). Not through freqtrade; that is what keeps them comparable (R3)
- [ ] T040 [US5] Report the paired IQM difference with its interval against the floor, and a verdict of `RL BETTER BY THE FLOOR` · `MATCHED` · `WORSE`
- [ ] T041 [US5] Apply SC-010: `MATCHED` **closes the family** and records that the differentiable estimator is preferred for lower variance and backtestable state

---

## Phase 8: Pooled §5 evaluation — GATED on a FUND screen

**Do not start unless T037 printed `FUND`.**

- [ ] T042 [GATED] RUN the full 15-fold pooled walk-forward for the locked configuration, sequentially; report the annualised delta against the control with its bootstrap interval and whether it clears the floor (SC-001)
- [ ] T043 [GATED] Report f07's contribution to the pooled verdict **separately** (SC-019) — the screen selected on it, and a selection rule discovered after the fact is not a selection rule
- [ ] T044 [GATED] RUN W1/W2/W3 single-window and report against SC-002
- [ ] T044a [GATED] Add a span mode to `research/labelfree/baselines.py` — `--folds f00..f14` aggregating on the same annualisation the floor uses. It accepts `--fold` only today. **Feature 011 found this gap and wrote the fix as its T043a, which was then cancelled with the rest of that feature when SC-009 killed it** — so the fix never landed and this feature inherits the hole (analyze C2)
- [ ] T044b [P] [GATED] Test that a one-fold span reproduces `--fold` exactly, so the new path cannot silently diverge from the old
- [ ] T045 [GATED] RUN the four mandatory baselines over the evaluation span via `baselines.py --folds` (SC-003)
- [ ] T046 [GATED] RUN cost stress at 1.5× and 2× fees; report the fee level at which the edge crosses zero (SC-005)
- [ ] T047 [GATED] Write `NNNC/NNNC_RL.py` — consult the policy in `confirm_trade_entry` and `custom_exit`, **not** `populate_entry_trend` (R3, SC-017). Merge parameter dictionaries from the parent
- [ ] T048 [GATED] Report the hook-evaluation limitation explicitly: these hooks can only **veto** candidates the vectorised gate already proposed, so the policy cannot open a position the base strategy never considered. The §5 figure is bounded by that; the head-to-head is not
- [ ] T049 [GATED] Compute the in-sample overfitting differential (SC-008)
- [ ] T050 [GATED] Reproduce the accepted configuration in a fresh process via a reproduction path **owned by this package** (SC-006). `research/labelfree/reproduce.py` is `ArmConfig`/`train_arm`-specific and an SB3 policy shares neither; feature 011 hit the same wall and gave its package its own path (analyze M2)

---

## Phase 9: Close

- [ ] T051 Compute DSR and PBO over the **full** trial log (SC-004). **Expect `CANNOT BE EVALUATED`** — the log stores summary scalars, so seeds are the only axis. Fourth feature to hit this; report it as unevaluable, never passed on an artefact
- [ ] T052 Re-run `/speckit-analyze` and append to `analyze.md`; fix every CRITICAL and re-run until clean (FR-022, SC-016)
- [ ] T053 RUN the trial-log audit (`trials.py::audit`) (SC-014, quickstart S9)
- [ ] T054 Render a sub-floor delta as **"not resolved"**, never "no effect", and assert the phrase "no effect" cannot appear in a generated verdict line (FR-023, SC-007)
- [ ] T055 Write `research/label-free-study/briefs/results/04-<date>.md` per protocol §6, including the head-to-head with brief 1. **Written even on a kill** (FR-024)
- [ ] T056 If the feature stopped at the screen or the head-to-head, mark T042–T050 **cancelled with the reason inline** rather than leaving them open
- [ ] T057 Stage ONLY the files this feature created or modified and commit (FR-020, FR-021)

---

## Dependencies & Execution Order

```
Phase 1 (Setup)
   ↓
Phase 2 (TWO GATES)     ← 🚦 A: env == simulator, bit-for-bit
   ↓                       B: the reward CAN be hacked
Phase 3 (US1 env)       ← MVP, standalone deliverable
   ↓
Phase 4 (US2 shared policy)
   ↓
Phase 5 (US3 selection hygiene)
   ↓
Phase 6 (US4 seeds)     ← SC-018 screen gate: FUND or STOP
   ↓
Phase 7 (US5 head-to-head)  ← SC-009/SC-010, the brief's actual question
   ↓
Phase 8 (§5 pooled)     ← GATED on FUND
   ↓
Phase 9 (analyze → audit → report → commit)
```

### User Story Dependencies

- **US1** depends only on Phase 2, and is the MVP: an auditable pooled trading environment,
  reusable by briefs 5 and 6 whether or not PPO works.
- **US2** depends on US1's env.
- **US3** is orthogonal to US2's outcome — selection hygiene applies regardless.
- **US4** depends on US2 and US3.
- **US5** depends on US4's locked configuration.
- **Phase 8** depends on a FUND screen, not merely on US5 completing.

### Parallel Opportunities

File authoring only: T002/T003; T007; T012; T015/T016; T021; T024; T027; T035.

---

## Implementation Strategy

**MVP** = Phases 1–3, ending at T018.

**Two decision points precede the expensive work**: the SC-018 screen at T037, and the
head-to-head verdict at T041. Either can stop the feature before Phase 8's 15-fold run.

**Expected non-obvious outcomes**, stated now so they are not retrofitted:

1. **T051 will report `CANNOT BE EVALUATED`.** Fourth feature running into a trial log that
   stores summary scalars. That is the correct outcome, not a failure to fix.
2. **T034 is the likeliest place this feature stops.** "Learn not to trade" is the environment's
   degenerate optimum, and a policy-gradient method finds it faster than gradient descent on a
   smooth objective.
3. **T030 may show the continuous arm is bimodal after all.** R4 predicts otherwise; the probe
   decides, and a bimodal arm is reported rather than tuned.

## Notes

- **Runs are sequential. Always.**
- Both Phase 2 gates exist because the corresponding failure has already occurred in this study:
  a faithful-looking restatement that silently differs, and a control that passes vacuously.
- A negative result satisfying the gates is a successful outcome and is recorded with the same
  weight as a positive one.
