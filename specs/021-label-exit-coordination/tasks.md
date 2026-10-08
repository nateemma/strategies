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

- [ ] T001 [P] @model-implementer Create `research/outcomes/__init__.py` with a package
  version constant, the `.cache/` path, and the `sys.prefix` interpreter guard copied from
  `research/clfstudy/__init__.py` (the venv at `.venv`, never conda).
- [ ] T002 [P] @model-implementer Create `research/outcomes/tests/__init__.py` and confirm
  pytest collects the new package.
- [ ] T003 [P] @model-implementer Add `research/outcomes/.cache/` to the nested repo's
  `.gitignore`; cached feathers are derived artefacts and must not be committed.

---

## Phase 2: Foundational — the one exit model (BLOCKING)

Everything in US2–US4 calls this. Nothing else may start until T007 passes.

### The ordering that matters

- [ ] T004 @main **Generate the SC-001 characterisation fixture FROM THE CURRENT,
  PRE-REFACTOR CODE** and see it pass, into
  `Framework/tests/fixtures/exit_policy_characterisation.json`. Record the stop decision for
  every bar of a real multi-pair, multi-year dataset, covering BOTH the grace and non-grace
  branches and BOTH `after_fill` values. **A fixture generated after the refactor proves
  nothing** — it would simply record whatever the new code does and assert the new code does
  it. This task must complete, and its test must be seen green against unmodified code,
  before T006 begins. Owner is `@main` because it touches the live-inherited
  `Framework/BaseStrategy.py` path while a 26-day dry-run is running.
- [ ] T005 [P] @model-implementer Write the allowlist duplication test (SC-002, FR-003) in
  `Framework/tests/test_exit_policy_single_source.py`: walk every `.py` under
  `user_data/strategies/`, collect files referencing `atr_stoploss_multiplier`,
  `atr_stoploss_cap`, `atr_stoploss_floor`, `stoploss_grace_level`, `stoploss_grace_hours`,
  `use_volume_confirmation_stoploss`, and assert that set equals an explicit allowlist
  organised by role (canonical primitive / thin caller / strategy that *configures* a value /
  test). The failure message must say whether the flagged reference is configuration or
  implementation, and point at `utils.ExitPolicy`.
- [ ] T006 @main Create `utils/ExitPolicy.py` and extract the arithmetic per
  `contracts/exit-policy.md`: `atr_stop_fraction`, `effective_stop_fraction`,
  `simulate_forward`. Pure functions — MUST NOT import freqtrade, take a `Trade`, or read a
  dataprovider. Then make all **three** duplicates thin callers:
  `BaseStrategy.custom_stoploss`, `BaseStrategy._resolve_normal_stop` (research R1 — this one
  is a duplicate *inside the same class* and its own docstring admits it), and
  `NNPredictStrategy._stop_aware_labels`, deleting their arithmetic. Keep freqtrade's return
  convention (`1.0` = no change, and the `(open_rate * (1+normal) / current_rate) - 1` rescale
  after grace) in `BaseStrategy` — it is glue, not arithmetic.
- [ ] T007 @main Run T004's characterisation test against the refactored code and confirm
  100% of bars match (SC-001). If any bar differs, the extraction is wrong — fix it, do not
  regenerate the fixture.
- [ ] T008 @model-implementer **Prove the duplication test can fail.** Plant a file naming
  `atr_stoploss_multiplier` and re-implementing the clamp, confirm T005's test fails and
  names it, then remove the file. A guard that cannot fail is worthless, and this repository
  has already lost a result to a wiring check that passed while the logic never ran.
- [ ] T009 [P] @model-implementer Unit-test `simulate_forward`'s **same-bar rule precedence**:
  on a bar whose `low` breaches the stop AND whose `high` reaches the take-profit, the result
  MUST be `stop`. OHLC cannot order intra-bar events, and the other choice would let the
  table claim profits the live path would not collect (Principle V).
- [ ] T010 [P] @model-implementer Unit-test the grace asymmetry: inside the grace window the
  non-`after_fill` call returns `1.0` and the `after_fill` call returns
  `stoploss_grace_level`. Losing that asymmetry would silently widen every live stop.

**Checkpoint**: `utils/ExitPolicy.py` is the only implementation of the exit arithmetic, live
behaviour is provably unchanged, and the guard protecting both facts has been shown to fail
when violated.

---

## Phase 3: US2 — the per-bar outcome table (P1)

**Goal**: evaluating a candidate label becomes a query instead of a 1.5-hour training run.

**Independent test**: build the table, then reproduce a known arm's trade outcomes from it
without running a backtest.

- [ ] T011 [US2] @model-implementer Implement `ExitConfig` in `research/outcomes/table.py` per
  `data-model.md`: the eleven fields, the validation rules verbatim
  (`atr_stoploss_floor <= atr_stoploss_cap < 0`, `take_profit > 0`, `max_hold_days >= 1`,
  `fee_per_side >= 0`), and `config_hash` as the first 16 hex characters of SHA-256 over the
  canonical JSON of every field. **`fee_per_side` MUST be in the hash** — a positive outcome
  is net of fees (FR-006a), so a fee change changes every outcome, and omitting it is exactly
  the silent-staleness bug FR-007 exists to prevent.
- [ ] T012 [US2] @model-implementer Write the two-layer leakage guard and its tests FIRST (FR-009,
  before the builder that produces the columns): every outcome column named `__outcome_*`,
  and a **raising** assertion in the label builder that no column it passes onward carries the
  prefix. Test that feeding an outcome-bearing frame into the label path raises. A convention
  alone is not "structurally prevented".
- [ ] T013 [US2] @model-implementer Implement the table builder in `research/outcomes/table.py`:
  one `OutcomeRecord` per bar per `data-model.md` — `__outcome_profit` net of
  `2 x fee_per_side`, `__outcome_hold_bars`, `__outcome_close_reason` in
  `{stop, take_profit, hold_cap, end_of_data}`, `__outcome_positive`, `__outcome_fillable`,
  `__outcome_fillable_fraction`, `__outcome_gapfill_share`. Calls `simulate_forward`; contains
  no exit arithmetic of its own.
- [ ] T014 [US2] @model-implementer Implement fillability (FR-009a) using the same quote-volume cap
  the live path applies (`QUOTE_VOLUME_HEADROOM_MULT` in `BaseStrategy.custom_stake_amount`)
  — do not re-derive it.
- [ ] T015 [P] [US2] @model-implementer Implement `__outcome_gapfill_share` (FR-020, SC-009): the
  share of each hold window that is gap-fill (volume 0, OHLC equal). 22–55% of 15m bars on six
  of eleven production pairs are gap-fill, so outcomes computed across them describe the
  absence of trading.
- [ ] T016 [US2] @model-implementer Implement the feather cache and JSON sidecar per
  `contracts/outcome-table.md`. Requesting a table whose stored hash differs from the
  requested config's is an **error**, never a silent reuse (FR-007, SC-004).
- [ ] T017 [P] [US2] @model-implementer Add the `--build`, `--status`, `--verify` CLI.
- [ ] T018 [P] [US2] @model-implementer Test reproducibility (SC-003): two builds from the same
  inputs are byte-identical.
- [ ] T019 [US2] @experiment-runner Build the table for the 11 production pairs at 15m and report
  per-pair record counts, close-reason distribution and gap-fill share.
- [ ] T020 [US2] @experiment-runner Validate against reality: reproduce a known arm's trade outcomes
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

- [ ] T021 [US3] @model-implementer Implement the trigger registry in
  `research/outcomes/triggers.py` per `contracts/trigger-cli.md`: `name`, `evaluate`,
  `required_columns`, `source`. Assert `required_columns` present before evaluation. Reject
  any trigger referencing a `__outcome_` column, and any pair-specific trigger
  (Constitution X).
- [ ] T022 [P] [US3] @model-implementer Register the three required triggers (FR-013): `gbb`
  (`guard_metric < 0.3 AND bb_width > 0.035`), `empty` (every bar — the forward_mae case and
  the verdict null), `geometry` (the DWT-smoothed zero-lag valley already in
  `labels_geometry`, never tested).
- [ ] T023 [US3] @model-implementer Implement label formation as `trigger AND __outcome_positive`
  (FR-011), with no label-specific outcome test anywhere.
- [ ] T024 [US3] @model-implementer Audit the 28 `LabelMethod` members (FR-012): express as triggers
  those that reduce, and list those that do not with the reason. Per research R7, expect
  `mfe_mae_ratio`, `future_sharpe`, `future_sortino`, `future_expectancy` and
  `risk_adj_future` NOT to reduce — their conditions are purely forward-path properties, so
  they are competing *outcome* definitions that this design supersedes rather than ports.
  Report the actual finding; do not assume research R7 was right.
- [ ] T025 [P] [US3] @model-implementer Test that two triggers combined with the shared outcome
  produce label sets differing only where the triggers differ.

**Checkpoint**: any trigger can be defined and turned into a label without writing an outcome
test.

---

## Phase 5: US4 — economic value vs learnability (P1)

**Goal**: settle whether a trigger earns its keep by selecting collectable moments, or by
handing the model an easy target. This is why the feature exists.

**Independent test**: a deliberately constructed trigger with known high selection value and
known low learnability lands in the expected quadrant.

- [ ] T026 [US4] @model-implementer Implement economic value in `research/outcomes/leaderboard.py`
  from the table alone, NO model (FR-014): base rate, mean realised outcome, distribution —
  reported both over all triggered bars and over fillable triggered bars only (FR-009b).
- [ ] T027 [US4] @model-implementer Implement the **block** bootstrap for the economic-value
  interval (research R9). Resample in contiguous blocks: adjacent bars' forward windows
  overlap heavily, so an i.i.d. bootstrap would understate the interval badly and could turn
  an indeterminate result into a false `selection_value`. Test that the block version gives a
  wider interval than i.i.d. on autocorrelated synthetic data.
- [ ] T028 [US4] @model-implementer Implement the fixed probe in `research/outcomes/probe.py`
  (FR-015): `research/clfstudy`'s `RIDGE` type with its split set `v1`, its scaler discipline
  and its MCC computation. The SAME model, features and protocol for every trigger — varying
  it would make the column incomparable and the verdict unreliable. Report the seed/window
  spread as the standard error the verdict rule needs.
- [ ] T029 [US4] @model-implementer Implement the verdict rule (FR-014b) against the `empty`
  trigger, **total** over three outcomes: `selection_value` (economic value exceeds empty's by
  more than the CI on the difference), `decoration` (it does not, while learnability exceeds
  empty's by more than the SE on that difference), `indeterminate` (neither). Error if `empty`
  is absent from the run — without the null no verdict can be assigned.
- [ ] T030 [P] [US4] @model-implementer Test the verdict rule with synthetic triggers constructed to
  land in each of the three quadrants, including the `empty`-vs-itself case which MUST come
  out `indeterminate`.
- [ ] T031 [US4] @model-implementer Assemble the leaderboard per `TriggerScore` in `data-model.md`,
  with economic value and learnability as SEPARATE columns that are never combined into a
  single score (FR-016).
- [ ] T032 [US4] @experiment-runner Run the leaderboard over every registered trigger and produce
  the report.
- [ ] T033 [US4] @gate-reviewer Review the leaderboard BEFORE any verdict is accepted: check base
  rates are plausible, that no trigger earns its value from unfillable or gap-fill-heavy bars,
  that intervals are block-bootstrapped, and that the `empty`-vs-itself control came out
  `indeterminate`. Produce a short memo; do not edit results.

**Checkpoint**: every trigger carries two independent columns and one of three verdicts.

---

## Phase 6: Polish

- [ ] T034 @main Write the report narrative: the verdict for each trigger, and what it implies.
  If `gbb` is `selection_value`, trigger discovery becomes the next line of work; if
  `decoration`, the apparent edge is the entry rule and the model is ornamental, which
  contradicts nothing measured so far but changes everything downstream.
- [ ] T035 @main **State the probe's bias in the report** (research R8): ridge cannot express
  a logical AND — it scored 0.533 against the gbb trigger rule's 0.673 — so it UNDERSTATES
  learnability for trigger-defined labels. The verdict rule is unaffected because both sides
  of every comparison use the same probe, but the absolute numbers are not ceilings and a
  reader could otherwise take them as such.
- [ ] T036 [P] @main Record in the report that the table's economic value is a LOOKAHEAD
  BOUND, never a realisable return, and that any P&L claim must come from `evaluation/` and
  its ~3–4pp/yr floor (FR-017). Classifier metrics are context, never an acceptance input
  (FR-018).
- [ ] T037 [P] @model-implementer Run the full suite; report the count and confirm no
  pre-existing failures were introduced.
- [ ] T038 [P] @main Update `research/clfstudy/report/methods.md` with the three-way
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
- T015, T017, T018 together once T013 lands
- T022, T025 together once T021 lands
- T030 alongside T031
- T036, T037, T038 together

## MVP

**Phase 2 alone is a shippable increment of real value**: it removes a three-way duplication
of the live stop arithmetic and proves live behaviour unchanged, which is worth doing whether
or not the rest of the feature ever runs.

The analytical MVP is Phase 2 + Phase 3 + the `empty` and `gbb` triggers only — enough to
answer the one question the feature exists for, before any broader trigger search.
