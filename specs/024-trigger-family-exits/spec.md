# Feature Specification: Trigger-Family Exit Presets

**Feature Branch**: `024-trigger-family-exits` (spec directory; work is committed to master)

**Created**: 2026-10-11

**Status**: Draft

**Input**: User description: "Trigger-family exit presets for SimpleTrigger (and later NNNC). Feature 023 showed that the exit components that matter interact with the entry trigger: under arm D's single shared exit stack, the take-profit helps the reversion trigger gbb (+12.9pp) but hurts the band-position momentum triggers breakout_bb and breakout_gbb (removing it gives +19.9 / +24.4pp), and deferring the 12h unclog exit to 24h moves breakout_gbb's late half from negative to positive (+25.19pp, positive in both halves). The ATR stop band also splits by trigger group. The grace stop helps every trigger and stays shared. Goal: classify each registered trigger by family (reversion or momentum) from what its condition tests, not its name; give each family its own exit parameter set on top of the ARM_D_OPEN preset (take-profit, unclog/hold window, ATR band, volume-confirm; grace and ROI shared); carry the family through freqtrade's enter_tag so a trade's exits are chosen at entry and never re-evaluated; and support a combined arm that runs a reversion trigger (gbb) and a momentum trigger (breakout_gbb) together, with gbb priority on overlap bars (overlap is ~2.3% of trades and was measured exit-indifferent), reporting entries lost to slot/one-position-per-pair contention. The unclog times are currently hardcoded in BaseStrategy.custom_exit; the mechanism must make them per-family without breaking the live dry-run strategy that imports BaseStrategy. Every arm's framework fingerprint must cover both parameter sets and the tag-to-set mapping. Per-family parameters are fixed on early folds and judged on late folds, one parameter at a time, under the existing walk-forward harness and quality gates (gate 2: no distant era, so results are provisional). Success = the combined arm reproduces st_gbb's trades on its reversion side when the momentum side is disabled, reproduces st_bgo_mx on its momentum side when the reversion side is disabled, and is reported against both single-family arms by half."

## Background *(evidence from feature 023, all PROVISIONAL under quality gate 2)*

| finding | evidence (`specs/023-simple-trigger-family/report.md`) |
|---|---|
| The grace stop helps every trigger | +6 to +81pp, 15/15 triggers, 14/15 same sign in both halves (Addendum 2) |
| The take-profit helps reversion and hurts band-position momentum | `gbb` +12.9pp with it; `breakout_bb` / `breakout_gbb` +19.9 / +24.4pp without it (Addendum 2) |
| A longer hold helps band-position momentum | `breakout_gbb` with no TP and a 24h hold: +25.19pp, positive in both halves (Addendum 3) |
| The ATR band splits by trigger group | +14..+20pp for breakout-type and `geometry`; −10..−31pp for `adx` / `votes_*` (Addendum 2) |
| Overlap between `gbb` and `breakout_gbb` is small and exit-indifferent | 2.3% of each trigger's trades; matched diff p = 0.86 / 0.49 (Addendum 5) |
| The two triggers' entries are nearly disjoint, not orthogonal in return | ~190 shared trades; per-fold correlation 0.39–0.54 (Addendum 3) |
| True price breakouts are not rescued by exits, guards, or timing filters | Addenda 3, 4, 7. They are out of scope here. |

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a trigger with its own family's exits (Priority: P1)

The researcher declares a SimpleTrigger arm by naming a trigger. The arm automatically uses the
exit settings of that trigger's family: reversion triggers get the reversion exit set, momentum
triggers get the momentum exit set. The researcher no longer hand-writes per-arm exit overrides
or mixins, as was done for the 023 `_MomExit` arms.

**Why this priority**: It is the mechanism everything else depends on. On its own it already
replaces 023's ad hoc `_MomExit` classes with one declared, fingerprinted mechanism.

**Independent Test**: Re-run `gbb` and `breakout_gbb` through the family mechanism and compare
them with 023's recorded arms. `gbb` on the reversion set must reproduce `st_gbb`, and
`breakout_gbb` on the momentum set must reproduce `st_bgo_mx`, trade-for-trade in 15/15 folds.

**Acceptance Scenarios**:

1. **Given** a trigger classified as reversion, **When** an arm is declared with it and no exit
   overrides, **Then** it runs the reversion exit set, its trades equal `st_gbb`'s for `gbb`, and
   its framework fingerprint names the reversion set.
2. **Given** a trigger classified as momentum, **When** an arm is declared with it, **Then** it
   runs the momentum exit set, and for `breakout_gbb` its trades equal `st_bgo_mx`'s in every fold.
3. **Given** two arms that differ only in one family's exit set, **When** both are fingerprinted,
   **Then** their fingerprints differ.
4. **Given** an arm that overrides one family's exit value (merged into that family's set),
   **When** it runs, **Then** the override takes precedence over the family default, and the
   fingerprint records it. An arm that sets the same exit key both as a general override and in
   a family set is refused at declaration, because the precedence would be ambiguous.

---

### User Story 2 - Run a reversion trigger and a momentum trigger together (Priority: P2)

The researcher declares one combined arm with `gbb` (reversion) and `breakout_gbb` (momentum).
Each trade is tagged with its family when it opens and keeps that family's exits for life. When
both triggers fire on the same bar, the trade is tagged reversion (`gbb` priority). The run
reports how many entries each family lost to the other through shared position slots and the
one-position-per-pair rule.

**Why this priority**: This is the payoff test of the family idea: two nearly disjoint entry
sets with exits suited to each. It depends on Story 1.

**Independent Test**: Disable one side, then the other. Each must reproduce its single-family
arm exactly. With both enabled, compare the result with both single-family arms by half.

**Acceptance Scenarios**:

1. **Given** the combined arm with the momentum side disabled, **When** it runs, **Then** its
   trades equal `st_gbb`'s in 15/15 folds.
2. **Given** the combined arm with the reversion side disabled, **When** it runs, **Then** its
   trades equal `st_bgo_mx`'s in 15/15 folds.
3. **Given** a bar where both triggers fire, **When** an entry is taken, **Then** it is tagged
   reversion and exits on the reversion set.
4. **Given** a completed combined run, **When** it is reported, **Then** the report gives
   per-family trade counts, per-family P&L, and entries lost to contention, early and late.
   Contention means a signal that would have entered in its single-family arm but did not here.
5. **Given** a completed combined run, **When** it is compared, **Then** it is paired against
   each single-family arm and against the sum of the two, with the halves split and the
   per-fold return correlation between the families.

---

### User Story 3 - Settle each family's exit parameters without overfitting (Priority: P3) — DEFERRED to a follow-up feature (Q1 → A)

The researcher chooses each family's exit settings by changing one parameter at a time, fixing it
on the early folds and judging it on the late folds. The order of the momentum set is the
take-profit, then the hold window, then the ATR band, then volume-confirm. The reversion set
starts from arm D's values. Every result is labelled provisional.

**Why this priority**: The mechanism and the combined arm deliver value with the values 023
already measured. Tuning refines them, and it carries the most overfitting risk.

**Independent Test**: For each parameter step, produce an early-fold choice and a late-fold
verdict against the previous step. Every arm passes a liveness check.

**Acceptance Scenarios**:

1. **Given** a parameter step, **When** candidate values are compared, **Then** the choice uses
   only early folds (f00–f07), and the late folds (f08–f14) judge it.
2. **Given** a candidate value, **When** its arm runs, **Then** the exit reason it governs moves
   (proven live), or the step is reported as INERT.
3. **Given** a step whose late-fold verdict is not IMPROVED, **When** the step closes, **Then**
   the previous value is kept.

---

### Edge Cases

- **A trigger that is neither clearly reversion nor clearly momentum** (`adx`, `votes_*`, the
  broad filters that fire on 10–14 trades/day): it is classified as **unassigned** and runs the
  shared ARM_D_OPEN exits. It never silently picks up a family's set.
- **A trade whose tag is missing or unrecognised** (manual entries, trades opened before the
  feature, a renamed family): it uses the shared exits, and the run logs and counts these cases.
  An unknown tag is never an error that leaves a trade without exits.
- **A trade open across a family parameter change**: tags are fixed at entry, and a backtest run
  uses one configuration, so this cannot occur within a run.
- **The live dry-run strategy**, which imports the same base framework: its exit behaviour is
  byte-identical before and after this feature, as 023's SC-008 proved for 023.
- **The hold window conflicting with a hard time cap** (a momentum hold longer than the 1-day
  unclog or the maximum hold): the order of time-based exits is defined and documented. A hold
  window that cannot take effect is refused when it exceeds the maximum hold, and flagged when it
  exceeds the 1-day unclog. It is never assumed live.
- **Overlap priority**: with `gbb` priority the momentum side never takes a bar where both fire.
  The momentum side's single-family reproduction (Story 2, scenario 2) is therefore run with the
  reversion side fully disabled, not merely deprioritised.
- **A family with no triggers in an arm**: the arm behaves exactly as a single-family arm.

## Clarifications

### Session 2026-10-11

- Q: Is per-family tuning (User Story 3) part of 024? → A: Deferred. 024 = mechanism + combined
  arm with 023's measured values. Tuning is a follow-up, scoped by the combined arm's results.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Every registered trigger MUST carry a family: `reversion`, `momentum`, or
  `unassigned`. The family is decided from what the trigger's condition tests (buying weakness vs
  buying strength), not from its name, and is recorded with a one-line justification.
  Classification at the start: reversion = `gbb`, `gbb_tight`, `gbb_stale`, `geometry`,
  `votes_tech`, `failed_breakout`; momentum = `breakout_bb`, `breakout_gbb`,
  `breakout_gbb_nofresh`, `breakout`, `breakout_vol`, `breakout_squeeze` and their `_gm`
  variants; unassigned = `adx`, `votes_ind2`, `votes_ind3`, `votes_ind4`, `votes_consensus`,
  `empty`, the unscoreable `trends`/`bands`, and the controls, except `gbb_stale`. `gbb_stale`
  is a lagged `gbb`, so its condition tests weakness and it is reversion. The classification MUST be checked by a test, so a newly registered
  trigger cannot be left without a family.
- **FR-002**: Each family MUST have one named exit parameter set layered on top of the
  ARM_D_OPEN preset. A set may override the take-profit, the hold window before the 12h unclog,
  the 1-day unclog timing, the ATR stop band and the volume-confirmation stop. The grace stop,
  ROI and every entry-side setting MUST stay shared. They are not per-family.
- **FR-003**: The starting values MUST be 023's measured values. Reversion = ARM_D_OPEN exactly.
  Momentum = no take-profit and the 12h unclog deferred to 24h, as in `st_bgo_mx`. Unassigned =
  ARM_D_OPEN.
- **FR-004**: A trade's family MUST be fixed when it opens, recorded on the trade, and used by
  every exit and stop decision for that trade's whole life. It MUST never be re-evaluated.
- **FR-005**: A trade with no recognised family MUST use the shared ARM_D_OPEN exits, and the run
  MUST report how many such trades occurred. In a SimpleTrigger run that count is expected to
  be zero.
- **FR-006**: The 12h and 1-day unclog times MUST become configurable per family without changing
  the exit behaviour of any existing strategy that does not opt in. The live dry-run strategy's
  dumped exit behaviour MUST be identical before and after (the 023 SC-008 method).
- **FR-007**: The framework fingerprint MUST cover each family's resolved exit set, the
  trigger-to-family mapping in use, and the overlap priority rule. Two arms that differ in any of
  these MUST have different fingerprints. Arms that do not use families MUST keep their existing
  recorded fingerprints (arm D `303456c1fa322b4c`).
- **FR-008**: A combined arm MUST accept two or more triggers from different families. It MUST
  resolve same-bar overlap with a declared priority (default: reversion first) and tag each entry
  with the family whose trigger won.
- **FR-009**: Each side of a combined arm MUST be able to be disabled without changing anything
  else, so the single-family reproduction tests are exact.
- **FR-010**: A combined arm's report MUST give per-family trade counts and P&L. It MUST give
  entries lost to contention per family, measured against that family's single-family arm, and
  overlap bars resolved by priority. All of these MUST be split by early and late halves.
- **FR-011**: The combined arm MUST be compared with each single-family arm (paired bootstrap and
  halves, as in 023). It MUST also report the per-fold return correlation between the two
  families (constitution IX). All verdicts MUST be labelled PROVISIONAL (gate 2).
- **FR-012**: Per-family parameter tuning MUST change one parameter at a time. It MUST choose on
  folds f00–f07, MUST judge on folds f08–f14, and MUST keep the previous value unless the late
  verdict is IMPROVED (constitution VIII).
- **FR-013**: Every tuning arm MUST prove the parameter it changes is live, by a change in the
  exit-reason counts it governs. An unchanged arm MUST be reported INERT (constitution IV).
- **FR-014**: The mechanism MUST be strategy-agnostic. No family-specific or trigger-specific
  logic may live in the shared framework, beyond reading a family's parameter set by tag
  (constitution: framework stays strategy-agnostic).
- **FR-015**: Applying families to NNNC is OUT of scope. The mechanism MUST NOT depend on
  anything SimpleTrigger-specific, so that a later NNNC feature can adopt it.
- **FR-016**: Per-family parameter tuning (User Story 3) is DEFERRED to a follow-up feature
  (clarification Q1 → A). 024 delivers the mechanism and the combined arm with 023's measured
  values (FR-003). FR-012 and FR-013 are the binding rules for that follow-up. The combined arm's
  contention and per-family results decide which parameters it tunes.

### Key Entities

- **Trigger family**: a label on a registered trigger (`reversion` / `momentum` / `unassigned`)
  with a one-line justification of what the condition tests.
- **Family exit set**: a named set of exit-parameter overrides on top of ARM_D_OPEN, one per
  family. Shared components (grace, ROI, entry settings) are excluded by definition.
- **Trade family tag**: the family recorded on a trade at entry, read by every exit decision for
  that trade.
- **Combined arm**: an arm with more than one trigger, an overlap priority rule, and per-side
  enable switches. Its fingerprint covers all of them.
- **Contention record**: for each family, the entries its single-family arm took that the
  combined arm did not, split by cause where determinable (pair already occupied by the other
  family / no free slot).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `gbb` run through the family mechanism (reversion set) reproduces `st_gbb`
  trade-for-trade in 15/15 folds. `breakout_gbb` (momentum set) reproduces `st_bgo_mx` in 15/15.
- **SC-002**: The combined arm with one side disabled reproduces the other side's single-family
  arm in 15/15 folds, for both sides.
- **SC-003**: Changing any one family's exit value, the mapping, or the priority changes the
  fingerprint. Arm D's recorded fingerprint is unchanged.
- **SC-004**: The live dry-run strategy's dumped exit and stop behaviour is identical before and
  after the feature.
- **SC-005**: The combined arm reports per-family counts, P&L and contention losses by half. It
  reports paired verdicts against both single-family arms and the per-fold correlation between
  families. Every figure is labelled PROVISIONAL.
- **SC-006**: Every registered trigger has a family and a justification, and a test fails if a
  new trigger is registered without one.
- **SC-007**: Zero trades in any SimpleTrigger run fall back to the shared exits because of a
  missing or unknown tag.

## Assumptions

- Data, folds, harness, causality gate and quality gates are 023's, unchanged. No distant era
  exists, so every result is PROVISIONAL (gate 2) and persistence is tested by halves only.
- Position slots and the one-position-per-pair rule stay shared across families. Splitting slots
  per family is a possible follow-up once contention has been measured.
- ROI stays global because freqtrade applies it to every trade alike. 023 found it helps 14/15
  triggers by a few pp, so sharing it is an accepted compromise.
- `gbb` priority on overlap is the default. 023 measured the overlap as 2.3% of trades and
  exit-indifferent, and this priority keeps the reversion side exactly reproducible.
- The true price breakouts stay out of scope. 023 Addenda 3, 4 and 7 found no exit, guard or
  timing fix for them.
- Offline screens and per-bucket trade splits are not used to choose parameters. Only
  walk-forward arms decide (023 Addendum 7; memory `feedback_bucket_filters_get_substituted`).
- Regime-specific parameters remain out of scope, as in 023.
- **Correction found in planning (research R9):** the live dry-run strategy
  (`MomentumRegimeBasket15mFast`) derives from freqtrade's `IStrategy` and imports nothing from
  `Framework/`, so `BaseStrategy` edits cannot reach it. FR-006 and SC-004 stand as cheap
  safeguards. The binding neutrality evidence is the hash-identical reproduction of recorded arms
  that do use `BaseStrategy`.
