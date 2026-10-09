# Tasks: Label-Exit Coordination

**Feature**: `021-label-exit-coordination` | **Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

**Owner tags**: `@model-implementer` Python and tests · `@experiment-runner` building the
table, running the leaderboard · `@gate-reviewer` reviewing the leaderboard before any verdict
is accepted · `@main` the verdict narrative, and anything touching `Framework/BaseStrategy.py`
while a dry-run is live.

**Standing constraint, every phase**: the exit policy is held FIXED (FR-019). No task may
change an exit parameter. A task that appears to need one is a spec violation, not a licence.

---

## Phase 1: Setup

- [X] T001 [P] @model-implementer Create `research/outcomes/__init__.py` with a package
  version constant, the `.cache/` path, and the `sys.prefix` interpreter guard copied from
  `research/clfstudy/__init__.py` (the venv at `.venv`, never conda).
- [X] T002 [P] @model-implementer Create `research/outcomes/tests/__init__.py` and confirm
  pytest collects the new package.
- [X] T003 [P] @model-implementer Add `research/outcomes/.cache/` to the nested repo's
  `.gitignore`; cached feathers are derived artefacts and must not be committed.

---

## Phase 2: Foundational — the one exit model (BLOCKING)

Everything in US2–US4 calls this. Nothing else may start until T007 passes.

### The ordering that matters

- [X] T004 @main **Generate the SC-001 characterisation fixture (SC-001, FR-005, FR-021) FROM THE CURRENT,
  PRE-REFACTOR CODE** and see it pass, into
  `Framework/tests/fixtures/exit_policy_characterisation.json`. Record the stop decision for
  every bar of a real multi-pair, multi-year dataset, covering BOTH the grace and non-grace
  branches and BOTH `after_fill` values. **A fixture generated after the refactor proves
  nothing** — it would simply record whatever the new code does and assert the new code does
  it. This task must complete, and its test must be seen green against unmodified code,
  before T006 begins. Owner is `@main` because it touches the live-inherited
  `Framework/BaseStrategy.py` path while a 26-day dry-run is running.
- [X] T005 [P] @model-implementer Write the allowlist duplication test (SC-002, FR-003) in
  `Framework/tests/test_exit_policy_single_source.py`: walk every `.py` under
  `user_data/strategies/`, collect files referencing `atr_stoploss_multiplier`,
  `atr_stoploss_cap`, `atr_stoploss_floor`, `stoploss_grace_level`, `stoploss_grace_hours`,
  `use_volume_confirmation_stoploss`, and assert that set equals an explicit allowlist
  organised by role (canonical primitive / thin caller / strategy that *configures* a value /
  test). The failure message must say whether the flagged reference is configuration or
  implementation, and point at `utils.ExitPolicy`.
- [X] T006 @main Create `utils/ExitPolicy.py` and extract the arithmetic (FR-001, FR-002, FR-004) per
  `contracts/exit-policy.md`: `atr_stop_fraction`, `effective_stop_fraction`,
  `simulate_forward`. Pure functions — MUST NOT import freqtrade, take a `Trade`, or read a
  dataprovider. Then make all **three** duplicates thin callers:
  `BaseStrategy.custom_stoploss`, `BaseStrategy._resolve_normal_stop` (research R1 — this one
  is a duplicate *inside the same class* and its own docstring admits it), and
  `NNPredictStrategy._stop_aware_labels`, deleting their arithmetic. Keep freqtrade's return
  convention (`1.0` = no change, and the `(open_rate * (1+normal) / current_rate) - 1` rescale
  after grace) in `BaseStrategy` — it is glue, not arithmetic.
  **FOUND DURING IMPLEMENTATION**: the duplication is **four-way**, not three.
  `research/labelfree/es/exits.py::atr_stop_level` was a fourth copy, outside research
  R1's catalogue, surfaced by T005's own test on first run. Refactored into a caller per
  FR-002 ("no other code may re-implement any part of it") rather than allowlisted as an
  exception; its 233 labelfree tests passed unchanged, which is what licensed touching
  closed work. Its fidelity notes had independently rediscovered that the volume-tighten
  clamp is a literal −0.02 and not `atr_stoploss_cap` — two authors finding the same trap
  separately is the argument for one copy.
- [X] T007 @main Run T004's characterisation test against the refactored code (SC-001, FR-005, FR-021) and confirm
  100% of bars match (SC-001). If any bar differs, the extraction is wrong — fix it, do not
  regenerate the fixture.
- [X] T008 @model-implementer **Prove the duplication test can fail.** Plant a file naming
  `atr_stoploss_multiplier` and re-implementing the clamp, confirm T005's test fails and
  names it, then remove the file. A guard that cannot fail is worthless, and this repository
  has already lost a result to a wiring check that passed while the logic never ran.
- [X] T009 [P] @model-implementer Unit-test `simulate_forward`'s **same-bar rule precedence**:
  on a bar whose `low` breaches the stop AND whose `high` reaches the take-profit, the result
  MUST be `stop`. OHLC cannot order intra-bar events, and the other choice would let the
  table claim profits the live path would not collect (Principle V).
- [X] T010 [P] @model-implementer Unit-test the grace asymmetry: inside the grace window the
  non-`after_fill` call returns `1.0` and the `after_fill` call returns
  `stoploss_grace_level`. Losing that asymmetry would silently widen every live stop.

- [X] T011 [P] @model-implementer Test that the exit is actually held FIXED (FR-019): assert
  every `ExitConfig` field used by the table equals the live value on `BaseStrategy` /
  `BaseNNStrategy` (`atr_stoploss_multiplier` 2.5, `atr_stoploss_cap` -0.02,
  `atr_stoploss_floor` -0.04, and the grace / take-profit / hold-cap values from the Open
  operating point). FR-019 is otherwise a prose header that nothing checks, so a drift would
  pass silently and every outcome would describe an exit the strategy does not run.

**Checkpoint — REACHED**. `utils/ExitPolicy.py` is the only implementation of the exit
arithmetic; live behaviour is provably unchanged across 1,440 cases; and the guard has been
shown to fail on a planted duplicate (now a permanent self-test, not a one-off).

Two findings the plan did not anticipate:
* The duplication was **four-way**. `research/labelfree/es/exits.py::atr_stop_level` was a
  fourth copy, surfaced by T005's own test on first run and refactored into a caller per
  FR-002. Its 233 tests passed unchanged.
* `ExitParams`' grace fields now have **no default and raise if absent**, because three
  different grace configurations exist and a convenient default would have silently
  simulated an exit the strategy does not run. The guard immediately caught T009's own
  tests relying on that default, which is the behaviour one wants from it.

---

## Phase 3: US2 — the per-bar outcome table (P1)

**Goal**: evaluating a candidate label becomes a query instead of a 1.5-hour training run.

**Independent test**: build the table, then reproduce a known arm's trade outcomes from it
without running a backtest.

- [X] T012 [US2] @model-implementer Implement `ExitConfig` in `research/outcomes/table.py` per
  `data-model.md`: the eleven fields, the validation rules verbatim
  (`atr_stoploss_floor <= atr_stoploss_cap < 0`, `take_profit > 0`, `max_hold_days >= 1`,
  `fee_per_side >= 0`), and `config_hash` as the first 16 hex characters of SHA-256 over the
  canonical JSON of every field. **`fee_per_side` MUST be in the hash** — a positive outcome
  is net of fees (FR-006a), so a fee change changes every outcome, and omitting it is exactly
  the silent-staleness bug FR-007 exists to prevent.
  **HAZARD FOUND IN PHASE 2, read before implementing**: this repository holds THREE
  different grace configurations -- `BaseStrategy.sell_params` 2.6h/-0.17, and
  `NNNC_MLX_Open.json` (the production operating point) 6.5h/-0.27. `ExitParams` therefore
  gives the two grace fields NO default and `effective_stop_fraction` RAISES if they are
  absent, because a default of "no grace" would silently simulate a stop firing inside a
  window where production has effectively none (-27% for 6.5 hours). `ExitConfig` MUST
  source grace from the operating point being modelled, never from a default.
- [X] T013 [US2] @model-implementer Write the two-layer leakage guard and its tests FIRST (FR-009,
  before the builder that produces the columns): every outcome column named `__outcome_*`,
  and a **raising** assertion in the label builder that no column it passes onward carries the
  prefix. Test that feeding an outcome-bearing frame into the label path raises. A convention
  alone is not "structurally prevented".
- [X] T014 [US2] @model-implementer Implement the table builder in `research/outcomes/table.py`:
  one `OutcomeRecord` per bar per `data-model.md` — `__outcome_profit` net of
  `2 x fee_per_side`, `__outcome_hold_bars`, `__outcome_close_reason` in
  `{stop, take_profit, hold_cap, end_of_data}`, `__outcome_positive`, `__outcome_fillable`,
  `__outcome_fillable_fraction`, `__outcome_gapfill_share`. Calls `simulate_forward`; contains
  no exit arithmetic of its own.
- [X] T015 [US2] @model-implementer Implement fillability (FR-009a) using the same quote-volume cap
  the live path applies (`QUOTE_VOLUME_HEADROOM_MULT` in `BaseStrategy.custom_stake_amount`)
  — do not re-derive it.
- [X] T016 [P] [US2] @model-implementer Implement `__outcome_gapfill_share` (FR-020, SC-009): the
  share of each hold window that is gap-fill (volume 0, OHLC equal). 22–55% of 15m bars on six
  of eleven production pairs are gap-fill, so outcomes computed across them describe the
  absence of trading.
- [X] T017 [US2] @model-implementer Implement the feather cache and JSON sidecar per
  `contracts/outcome-table.md`. Requesting a table whose stored hash differs from the
  requested config's is an **error**, never a silent reuse (FR-007, SC-004).
- [X] T018 [P] [US2] @model-implementer Add the `--build`, `--status`, `--verify` CLI.
- [X] T019 [P] [US2] @model-implementer Test reproducibility (SC-003, FR-008): two builds from the same
  inputs are byte-identical.
- [X] T020 [US2] @experiment-runner Build the table for the 11 production pairs at 15m and report
  per-pair record counts, close-reason distribution and gap-fill share.
- [X] T021 [US2] @experiment-runner Validate against reality: reproduce a known arm's trade outcomes
  from the table and reconcile against `evaluation/pooled_NNNC_A_Raw_Open.json`. Report any
  systematic divergence — a table that disagrees with the harness is wrong, and better found
  here than in a verdict.

**Checkpoint**: the table exists, is reproducible, invalidates on exit change, cannot leak
into a model, and agrees with the harness.

---

## Phase 4: US3 — triggers as the only variable (P2)

**Goal**: labels become comparable, because only the trigger differs.

**Independent test**: express several existing labels as triggers, recombine with the shared
outcome, confirm the label sets differ only where the triggers do.

- [X] T022 [US3] @model-implementer Implement the trigger registry in (FR-010)
  `research/outcomes/triggers.py` per `contracts/trigger-cli.md`: `name`, `evaluate`,
  `required_columns`, `source`. Assert `required_columns` present before evaluation. Reject
  any trigger referencing a `__outcome_` column, and any pair-specific trigger
  (Constitution X).
- [X] T023 [P] [US3] @model-implementer Register the three required triggers (FR-013) (FR-014a): `gbb`
  (`guard_metric < 0.3 AND bb_width > 0.035`), `empty` (every bar — the forward_mae case and
  the verdict null), `geometry` (the DWT-smoothed zero-lag valley already in
  `labels_geometry`, never tested).
  **T023 FINDING, carried into T025's report**: `geometry` could NOT be ported. `labels_geometry`
  (a) raises with its own defaults (`price_mean * min_gain` with `min_gain=None`), (b) has its DWT
  smoothing commented out (`smoothed = close`), (c) is FORWARD-LOOKING — `find_peaks` over the whole
  array plus a 3-bar backfill, measured by perturbing future bars and watching the past change — and
  (d) derives `prominence` from whole-series `mean`/`std`, making it both lookahead and per-pair.
  The registered `geometry` is a CAUSAL confirmed-valley substitute (fires 4 bars after a ≥1% low),
  base rate 3.07% against the labeler's measured 54%. `gbb` IS a faithful port, bit-identical to the
  live backward-looking mask over 164,918 SOL bars.

- [X] T024 [US3] @model-implementer Implement label formation as `trigger AND __outcome_positive`
  (FR-011), with no label-specific outcome test anywhere.
- [X] T025 [US3] @model-implementer Audit the 28 `LabelMethod` members (FR-012): express as triggers
  those that reduce, and list those that do not with the reason. Per research R7, expect
  `mfe_mae_ratio`, `future_sharpe`, `future_sortino`, `future_expectancy` and
  `risk_adj_future` NOT to reduce — their conditions are purely forward-path properties, so
  they are competing *outcome* definitions that this design supersedes rather than ports.
  Report the actual finding; do not assume research R7 was right.
- [X] T026 [P] [US3] @model-implementer Test that two triggers combined with the shared outcome
  produce label sets differing only where the triggers differ.

**Checkpoint**: any trigger can be defined and turned into a label without writing an outcome
test.

---

## Phase 5: US4 — economic value vs learnability (P1)

**Goal**: settle whether a trigger earns its keep by selecting collectable moments, or by
handing the model an easy target. This is why the feature exists.

**Independent test**: a deliberately constructed trigger with known high selection value and
known low learnability lands in the expected quadrant.

- [X] T027 [US4] @model-implementer Implement economic value in `research/outcomes/leaderboard.py` (SC-010)
  from the table alone, NO model (FR-014): base rate, mean realised outcome, distribution —
  reported both over all triggered bars and over fillable triggered bars only (FR-009b).
- [X] T028 [US4] @model-implementer Implement the **block** bootstrap for the economic-value
  interval (research R9). Resample in contiguous blocks: adjacent bars' forward windows
  overlap heavily, so an i.i.d. bootstrap would understate the interval badly and could turn
  an indeterminate result into a false `selection_value`. Test that the block version gives a
  wider interval than i.i.d. on autocorrelated synthetic data.
- [X] T029 [US4] @model-implementer Implement the fixed probe in `research/outcomes/probe.py`
  (FR-015): `research/clfstudy`'s `RIDGE` type with its split set `v1`, its scaler discipline
  and its MCC computation. The SAME model, features and protocol for every trigger — varying
  it would make the column incomparable and the verdict unreliable. Report the seed/window
  spread as the standard error the verdict rule needs.
  **FR-015b AMENDMENT (2026-10-09, approved by the user after T029)**: the verdict rule consumes
  CONDITIONAL learnability — the probe restricted to triggered bars, predicting the outcome alone.
  Implemented as `probe.probe_conditional`; results in `research/outcomes/probe_conditional_results.json`.
  The unconditional column is retained as context and MUST NOT feed FR-014b.

- [X] T030 [US4] @model-implementer Implement the verdict rule (FR-014b) against the `empty` (SC-007)
  trigger, **total** over three outcomes: `selection_value` (economic value exceeds empty's by
  more than the CI on the difference), `decoration` (it does not, while learnability exceeds
  empty's by more than the SE on that difference), `indeterminate` (neither). Error if `empty`
  is absent from the run — without the null no verdict can be assigned.
- [X] T031 [P] [US4] @model-implementer Test the verdict rule with synthetic triggers constructed to
  land in each of the three quadrants, including the `empty`-vs-itself case which MUST come
  out `indeterminate`.
- [X] T032 [US4] @model-implementer Assemble the leaderboard per `TriggerScore` in `data-model.md` (SC-006),
  with economic value and learnability as SEPARATE columns that are never combined into a
  single score (FR-016).
- [X] T033 [US4] @experiment-runner Measure SC-005: time one trigger's FULL two-column
  evaluation (economic value with its block-bootstrap interval, plus the probe) and report the
  wall-clock. SC-005 claims seconds against the ~1.5 hours a trained comparison costs, and that
  claim is the whole justification for building the table -- yet no task measured it until this
  one. If it is not seconds, the design has not delivered its premise and that must be reported
  rather than explained away.
- [X] T034 [US4] @experiment-runner Run the leaderboard over every registered trigger (SC-007, SC-008) and produce
  the report.
- [X] T035 [US4] @gate-reviewer Review the leaderboard BEFORE any verdict is accepted: check base
  rates are plausible, that no trigger earns its value from unfillable or gap-fill-heavy bars,
  that intervals are block-bootstrapped, and that the `empty`-vs-itself control came out
  `indeterminate`. Produce a short memo; do not edit results.

**Checkpoint**: every trigger carries two independent columns and one of three verdicts.

---

## Phase 6: Polish

- [X] T036 @main Write the report narrative: the verdict for each trigger, and what it implies.
  If `gbb` is `selection_value`, trigger discovery becomes the next line of work; if
  `decoration`, the apparent edge is the entry rule and the model is ornamental, which
  contradicts nothing measured so far but changes everything downstream.
- [X] T037 @main **State the probe's bias in the report** (research R8) (FR-015a): ridge cannot express
  a logical AND — it scored 0.533 against the gbb trigger rule's 0.673 — so it UNDERSTATES
  learnability for trigger-defined labels. The verdict rule is unaffected because both sides
  of every comparison use the same probe, but the absolute numbers are not ceilings and a
  reader could otherwise take them as such.
- [X] T038 [P] @main Record in the report that the table's economic value is a LOOKAHEAD
  BOUND, never a realisable return, and that any P&L claim must come from `evaluation/` and
  its ~3–4pp/yr floor (FR-017). Classifier metrics are context, never an acceptance input
  (FR-018).
- [X] T039 [P] @model-implementer Run the full suite; report the count and confirm no
  pre-existing failures were introduced.
- [X] T040 [P] @main Update `research/clfstudy/report/methods.md` with the three-way
  duplication finding (research R1), so the next person does not rediscover it.

---

## Dependencies

```
Phase 1 (setup)
    └─> Phase 2 (exit model) ── BLOCKING, and strictly ordered:
            T004 (fixture from PRE-refactor code)
              └─> T006 (extract)  ──> T007 (verify unchanged)
            T005 (duplication test) ──> T008 (prove it fails)
    └─> Phase 3 (US2 table)   ── needs simulate_forward from T006
            └─> Phase 4 (US3 triggers) ── needs the table's __outcome_positive
                    └─> Phase 5 (US4 leaderboard) ── needs both
                            └─> Phase 6 (polish)
```

**US2 → US3 → US4 is a genuine chain**, not a convenience: a trigger cannot be scored without
outcomes, and a verdict cannot be reached without both columns.

## Parallel opportunities

- T001–T003 together
- T005 alongside T004 (different files; T005 does not touch the live path)
- T009, T010 together once T006 lands
- T016, T018, T019 together once T014 lands
- T023, T026 together once T022 lands
- T031 alongside T032
- T038, T039, T040 together

## MVP

**Phase 2 alone is a shippable increment of real value**: it removes a three-way duplication
of the live stop arithmetic and proves live behaviour unchanged, which is worth doing whether
or not the rest of the feature ever runs.

The analytical MVP is Phase 2 + Phase 3 + the `empty` and `gbb` triggers only — enough to
answer the one question the feature exists for, before any broader trigger search.
