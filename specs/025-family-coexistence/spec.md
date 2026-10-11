# Feature Specification: Family Coexistence and Reversion Late-Half Fix

**Feature Branch**: `025-family-coexistence` (spec directory; work is committed to master)

**Created**: 2026-10-11

**Status**: Draft

**Input**: User description: "Family coexistence and reversion late-half fix (follow-up to feature 024). Feature 024 built per-family exits (reversion = gbb on ARM_D exits, momentum = breakout_gbb with no take-profit and a 24h hold) and a combined arm. The combined arm made +15.39pp (8/15 folds), between gbb alone (+11.04) and breakout_gbb alone (+25.19) and far below the sum of the two (+36.23); late half -20.37pp. Contention analysis showed the 10-slot cap never binds (slots_full = 0) — all contention is one-position-per-pair: about 3,250 gbb entries and 1,250 breakout_gbb entries were blocked because the other family held the pair, and inside the combined arm the momentum side still earns in both halves (+12.31 / +4.33pp) while the reversion side carries gbb's late collapse (+23.45 / -24.70pp). Goal: (1) make the two families coexist without pre-empting each other's pairs, using a structural, strategy-agnostic rule (e.g. a per-family cap on how many pairs a family may hold, or an entry rule that keeps reversion off a pair while momentum conditions hold) — no pair-specific code; (2) fix the reversion side's late-half loss by tuning the reversion exit set one parameter at a time, choosing on early folds f00-f07 and judging on late folds f08-f14, with every changed parameter proven live by exit-reason counts; this is 024's deferred FR-012/FR-013, scoped to the reversion family. Success = the combined arm's momentum side recovers its standalone late-half result (it is not crowded out), contention losses drop, and the combined arm is reported against both single-family arms and against momentum alone by half. All results PROVISIONAL under quality gate 2 (no distant era). Reuse 024's mechanism (FamilyExitMixin, ARM_D_FAMILY, TRIGGERS, contention.py) and 023's harness."

## Background *(024 evidence, PROVISIONAL under quality gate 2)*

| arm (024) | per-fold sum | positive folds | early | late |
|---|---|---|---|---|
| combined `st_fam_combo` (gbb + breakout_gbb) | +15.39 | 8/15 | +35.76 | −20.37 |
|   … its momentum trades | +16.64 | — | +12.31 | **+4.33** |
|   … its reversion trades | −1.25 | — | +23.45 | **−24.70** |
| reversion alone `st_gbb` | +11.04 | 7/15 | +30.33 | −19.28 |
| momentum alone `st_bgo_mx` | **+25.19** | 9/15 | +22.68 | +2.50 |

**Contention** (`evaluation/contention_st_fam_combo.md`): the slot cap never bound (`slots_full` = 0
in every cell). The losses came from one position per pair, mostly because the other family held
the pair:

| family | lost to the other family | total lost |
|---|---|---|
| reversion | 3,258 | 3,896 |
| momentum | 1,252 | 1,853 |

The overlap priority decides only 239 same-bar entries. It is not a coexistence lever.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Families coexist without pre-empting each other's pairs (Priority: P1)

The researcher runs the combined arm with a coexistence rule. The rule is structural: it depends
on family and position state, never on which pair. Under it, one family's open trades can no
longer crowd the other out of its entries. The run reports, per family and half, the entries lost
to the other family, against 024's combined arm and against each single-family arm.

**Why this priority**: 024 measured pair-level pre-emption as the reason the combined arm falls far
short of the sum of its parts. Until the families coexist, tuning either one is measured on a
distorted entry set.

**Independent Test**: Run the combined arm with each candidate rule. Each candidate must, with its
rule disabled (neutral setting), reproduce 024's `st_fam_combo` trade-for-trade. Then compare the
candidates' contention and P&L by half.

**Acceptance Scenarios**:

1. **Given** a coexistence rule at its neutral setting, **When** the combined arm runs, **Then** its
   trades equal `st_fam_combo`'s in 15/15 folds.
2. **Given** a coexistence rule at an active setting, **When** the combined arm runs, **Then** the
   report shows the change in entries each family lost to the other, per half. The rule is proven
   live: at least one entry was refused or allowed by it, and the run counts these.
3. **Given** several candidate rules or settings, **When** one is chosen, **Then** it is chosen on
   folds f00–f07 only and judged on f08–f14. The choice and its late-half verdict are reported.

---

### User Story 2 - Fix the reversion side's late half by exit tuning (Priority: P2)

With coexistence settled, the researcher tunes the reversion exit set one parameter at a time. The
fixed order is take-profit, then the 12h unclog timing, then the ATR stop band, then
volume-confirmation. Each step chooses its value on f00–f07, is judged on f08–f14, and keeps the
previous value unless the late verdict is IMPROVED. Every changed value must be proven live by a
change in the exit-reason counts it governs.

**Why this priority**: The reversion side's late-half collapse (−24.70pp inside the combined arm) is
the other half of the gap. This is 024's deferred FR-012/FR-013, scoped to the reversion family.

**Independent Test**: For each step, a table of candidate values with early-fold results, the
chosen value, its late-fold paired verdict against the previous step, and the exit-reason counts
proving it live.

**Acceptance Scenarios**:

1. **Given** a tuning step, **When** candidates are compared, **Then** only f00–f07 results
   influence the choice.
2. **Given** a chosen value, **When** it is judged, **Then** it is paired against the previous
   step on f08–f14. A verdict other than IMPROVED keeps the previous value.
3. **Given** a candidate whose governed exit-reason counts do not move, **When** it runs, **Then**
   it is reported INERT, not as a null result.

---

### User Story 3 - Report the combined arm against every reference by half (Priority: P3)

The researcher gets one report comparing the final combined arm (coexistence rule plus the tuned
reversion set) with momentum alone, reversion alone, 024's combined arm, and the sum of the two
singles. It covers the early and late halves, per-family contributions, contention, and the
per-fold correlation between the families. Every figure is labelled PROVISIONAL.

**Why this priority**: The decision this work feeds is whether a two-family arm beats momentum
alone. That needs one place where all references sit side by side.

**Independent Test**: The report exists and every row has early and late figures, a paired verdict
and the PROVISIONAL label.

**Acceptance Scenarios**:

1. **Given** the final combined arm, **When** the report is produced, **Then** it states whether
   the combined arm's momentum side reached momentum-alone's late-half result (+2.50pp), and
   whether the whole arm beat momentum alone in each half.
2. **Given** a negative outcome (the combination still does not beat momentum alone), **When**
   it is reported, **Then** it is recorded as a result, not suppressed (constitution: a negative
   result that passes the gates is a successful outcome).

---

### Edge Cases

- **A rule that removes all of one family's entries.** For example, a pair cap of 0. It is reported
  as degenerate, not as a winning setting, because it reduces the arm to a single-family one.
- **Open trades at a cap boundary.** The rule only refuses NEW entries. It never closes or evicts an
  open trade.
- **Both families signal on the same bar for the same pair.** The 024 overlap priority still
  applies first. The coexistence rule acts on entries to pairs held, or about to be held, by the
  other family.
- **The reversion tuning step changes which pairs reversion holds**, and so the coexistence
  behaviour. Each tuning step is run inside the combined arm with the chosen rule, so the
  interaction is included in what is measured.
- **A tuning candidate that violates the exit-set validation** (take-profit ≤ 0, an ATR floor above
  its cap, a hold window beyond the maximum hold): it is refused at declaration, as in 024.
- **Momentum's late half is small** (+2.50pp standalone on about 2,700 trades). "Recovered" is
  reported with its paired interval, never as a point comparison alone.

## Clarifications

### Session 2026-10-11

- Q: Where is reversion exit tuning measured? → A: Inside the combined arm with the chosen
  coexistence rule, because 024 showed the families interact through pair holding.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The combined arm MUST support a coexistence rule that limits how one family's open
  positions block the other family's entries. The rule MUST depend only on family and position
  state, never on pair identity (constitution X), and MUST be strategy-agnostic.
- **FR-002**: At least two candidate rules MUST be evaluated, each with a neutral setting that
  reproduces 024's `st_fam_combo` exactly:
  - (a) a per-family cap on how many pairs a family may hold at once;
  - (b) an entry rule that refuses a reversion entry on a pair where the momentum trigger fired
    within a recent window. While it is active, a bar where both families fire goes to momentum,
    because the rule exists to keep momentum's pairs free (analyze G2).
- **FR-003**: Each candidate rule MUST be proven live. The run counts entries the rule refused or
  allowed, per family, and a rule whose counts are zero is reported INERT.
- **FR-004**: The coexistence rule and its setting MUST be chosen on folds f00–f07 only and judged on
  f08–f14 (constitution VIII). Candidate settings MUST be a small pre-declared set, not a search.
- **FR-005**: The contention report (024 C6) MUST be produced for every coexistence candidate, so
  the entries lost to the other family are compared with 024's numbers by half.
- **FR-006**: Reversion exit tuning MUST change one parameter at a time, in the fixed order
  take-profit → 12h unclog timing → ATR stop band → volume-confirmation. Grace, ROI and entry
  settings stay shared and are not tuned.
- **FR-007**: Each tuning step MUST choose on f00–f07, MUST be judged on f08–f14 against the previous
  step, and MUST keep the previous value unless the late verdict is IMPROVED.
- **FR-008**: Each tuning arm MUST prove its parameter live by a change in the exit-reason counts it
  governs. An unchanged arm MUST be reported INERT (constitution IV).
- **FR-009**: Reversion tuning MUST be measured inside the combined arm with the chosen
  coexistence rule (clarification Q1 → A). This captures how reversion's exits change which pairs
  it holds, and so what momentum gets. The paired comparison for each step is the combined arm at
  the previous step's value.
- **FR-010**: The momentum exit set MUST stay at 024's values throughout. This feature tunes only the
  reversion family.
- **FR-011**: The final report MUST compare the final combined arm with momentum alone, reversion
  alone, 024's combined arm and the sum of singles, by half. It MUST include paired verdicts,
  per-family contributions, contention and the per-fold correlation. Every figure MUST be labelled
  PROVISIONAL (gate 2).
- **FR-012**: Every coexistence and tuning configuration MUST be distinguishable by its framework
  fingerprint. Arms that use neither MUST keep their recorded fingerprints.

### Key Entities

- **Coexistence rule**: a named, structural entry restriction between families, with a neutral
  setting and a small declared set of active settings. Each setting has a refusal/allow counter.
- **Tuning step**: one reversion exit parameter, its candidate values, the early-fold choice, the
  late-fold verdict and the liveness evidence.
- **Final combined arm**: the chosen coexistence rule plus the tuned reversion set plus 024's
  momentum set.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every coexistence candidate at its neutral setting reproduces `st_fam_combo` in 15/15
  folds.
- **SC-002**: Every active coexistence setting and every tuning arm has non-zero liveness evidence,
  or is reported INERT.
- **SC-003**: Every choice (coexistence rule and setting, and each tuning step) is documented as made
  on f00–f07 and judged on f08–f14, with the late verdict stated.
- **SC-004**: The final report states, with paired intervals:
  - whether the combined arm's momentum side reaches momentum-alone's late half (+2.50pp);
  - whether entries lost to the other family fell against 024 (3,258 reversion / 1,252 momentum);
  - whether the final combined arm beats momentum alone in each half.
- **SC-005**: Arm D (`303456c1fa322b4c`) and every 023/024 recorded arm keep their fingerprints and
  stay reproducible.

## Assumptions

- Data, folds, harness, causality gate and quality gates are unchanged from 023/024. No distant era
  exists, so every result is PROVISIONAL (gate 2) and persistence is tested by halves only.
- The triggers stay `gbb` (reversion) and `breakout_gbb` (momentum). Swapping triggers is out of
  scope.
- `max_open_trades` stays 10. 024 showed it never binds with ≤10 pairs, so slot budgeting is not a
  lever here.
- A rule may only refuse new entries. Evicting open positions is out of scope (freqtrade cannot do
  it cleanly, and it would change the exit model).
- The candidate settings for each rule and each tuning step are small and pre-declared, set in the
  plan, to keep the early-fold choice from becoming a search (constitution VIII).
- Offline screens and per-bucket trade splits are not used to choose settings; only walk-forward arms
  decide (023 Addendum 7).
- Regime-specific parameters remain out of scope.
