# Feature Specification: NNNC / SimpleTrigger Merge

**Feature Branch**: `029-nnnc-simpletrigger-merge` (spec directory; work is committed to master)

**Created**: 2026-10-12

**Status**: Draft

**Input**: User description: "NNNC merge (023 FR-007a): restructure NNNC so it shares ONE framework definition with SimpleTrigger instead of being a sibling held together only by the 023 equivalence test. Decision A: a pure restructure with NO behaviour change -- the model becomes one entry source inside the SimpleTrigger framework, and arm D must reproduce its recorded trades byte-for-byte, including the 56.5% of its entries outside the gbb trigger. Requiring a trigger AND the model is added only as a default-off switch. NNNC gains by inheritance, all default-off or behaviour-neutral: stop invariants, per-family exits incl. the 028 ATR take-profit switch, the regime gate, merging presets/overrides, the fingerprint. Constraints: no behaviour change; the ~300 NNNC variants keep working; the live bot is untouched; Framework/ stays strategy-agnostic; parameter dicts merge; no pair-specific code; one backtest chain at a time. Open question: which family NNNC's model entries are tagged with. Out of scope: switching C2 or the trigger gate on for NNNC, retraining, portfolio combination. Success = one framework hierarchy, arm D and reference NNNC arms byte-identical, stop invariants on every NNNC arm, any SimpleTrigger framework feature switchable on an NNNC arm by declaration alone."

## Background

**Today the two families are siblings.**
- *SimpleTrigger* is the framework with no model. Since 023 it has gained:
  - stop invariants (a wrong-direction or too-loose stop halts the run);
  - per-family exits (024), including 028's ATR take-profit switch;
  - the reversion regime gate (026);
  - family coexistence rules (025);
  - presets whose overrides merge from parent to child.
- *NNNC* is the model family: about 300 strategy variants, including arm D (`NNNC_A_Raw_Open`, the
  research operating point `NNNC_MLX_Open`). It enters when the model's prediction clears its
  threshold and has none of the above.
- *What links them:* only 023's equivalence test. An NNNC arm whose model accepts every trigger
  bar reproduces SimpleTrigger's trades (15/15 folds). Anything added to one family since then has
  to be re-added to the other by hand, and nothing stops the two drifting apart.

**What this feature is.** The restructure 023 recorded as its follow-up (FR-007a): NNNC and
SimpleTrigger derive from one framework definition, and the model is one entry source inside it.
- **The researcher chose a pure restructure** (2026-10-12, decision A). Nothing any existing
  strategy does may change.
- **Arm D enters on the model alone.** 56.5% of its trades fall outside the gbb trigger, and they
  must stay.
- **Trigger AND model** (enter only on trigger bars the model accepts) becomes a switch, off by
  default. Evaluating it is a later feature.

**What it is not.** It doesn't switch any new behaviour on for NNNC: no ATR take-profit, regime
gate or trigger gate. Those become possible by declaration and are judged in later features.

**Production is not affected.** The live dry-run bot runs a separate basket strategy that does not
share this framework.

## Clarifications

### Session 2026-10-12

- Q: Which family do NNNC's model entries belong to? → A: Label-driven (A′). Each label declares
  the family it implies, and a model's entries take it. Labels that imply no entry state, such as
  arm D's forward-peak label, leave entries untagged, on the shared exit.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One framework, no behaviour change (Priority: P1)

The researcher restructures NNNC onto the shared framework. Before the change, arm D and its
equivalence twin are re-run on the current code to fix a byte-level baseline, and every NNNC
strategy's framework fingerprint is recorded. After the change, the same runs and fingerprints
are compared.

**Why this priority**: The merge is only worth having if it is invisible. A restructure that
moves arm D's trades would contaminate every model comparison made after it.

**Independent Test**: arm D and the twin re-run after the change; per-fold trade sets compared
with the pre-change baseline; fingerprints of every loadable NNNC class compared.

**Acceptance Scenarios**:

1. **Given** the pre-change baseline, **When** arm D runs after the change with the same borrowed
   per-fold models, **Then** its trades are identical in 15 of 15 folds.
2. **Given** the equivalence twin (model accepts every trigger bar), **When** it runs after the
   change, **Then** its trades equal both its baseline and SimpleTrigger's gbb arm in 15 of 15
   folds.
3. **Given** every NNNC strategy class that loads, **When** its fingerprint is computed after the
   change, **Then** it equals the pre-change fingerprint. Any class that changes is listed and
   explained, or the change is not accepted.
4. **Given** any NNNC strategy, **When** it is loaded or run, **Then** no model is retrained.

---

### User Story 2 - Stop invariants on every NNNC arm (Priority: P2)

Every NNNC run checks its stops the way SimpleTrigger runs do: the right sign, never looser than
the documented bound, and the grace rescale never moving the stop down for a trade still above
it. A violation halts the run.

**Why this priority**: 023 found stop defects that silently changed results. Today only
SimpleTrigger and the twin are protected.

**Independent Test**: arm D's baseline run reports its violation count. After the change, every
NNNC run carries the check and arm D still reproduces.

**Acceptance Scenarios**:

1. **Given** arm D's pre-change baseline, **When** the stop check is applied in report-only mode,
   **Then** the number of violations is recorded before anything is merged.
2. **Given** zero violations in the baseline, **When** arm D runs after the change with the check
   halting, **Then** it completes and reproduces its trades.
3. **Given** a planted violation, **When** an NNNC arm runs, **Then** the run halts and names the
   trade.

---

### User Story 3 - SimpleTrigger features available to NNNC by declaration (Priority: P3)

A later NNNC experiment can switch on any framework feature by declaring it, with no new code:
- per-family exits, including the ATR take-profit;
- the reversion regime gate;
- trigger-AND-model entries.

**Why this priority**: This is what the merge is for. The next features (C2 for NNNC, the trigger
gate, the model's out-of-trigger entries) need it.

**Independent Test**: for each feature, a minimal declared NNNC test class shows the feature
changing a decision (liveness). With the declaration removed, the behaviour is the pre-change
behaviour.

**Acceptance Scenarios**:

1. **Given** an NNNC class declaring the trigger-AND-model switch with trigger gbb, **When** it is
   evaluated on a fold, **Then** every entry is on a bar where both the trigger fired and the model
   accepted, and the count of entries dropped is reported.
2. **Given** an NNNC class declaring a family exit (e.g. the ATR take-profit for its model
   entries' family), **When** it runs, **Then** that family's trades use the declared exit.
3. **Given** an NNNC class declaring the regime gate, **When** it runs, **Then** entries in the
   gate's "off" state are dropped as in SimpleTrigger.
4. **Given** no declaration, **When** any of these runs, **Then** it behaves exactly as before.

---

### User Story 4 - Report (Priority: P4)

A short report records the baseline, the fingerprint comparison, the reproduction result, the
stop-check result, the liveness of each switch, and how a later feature declares each one.

**Why this priority**: This is the hand-over to the NNNC experiments that follow.

**Independent Test**: the report exists and every success criterion has a recorded value.

**Acceptance Scenarios**:

1. **Given** the feature finished, **When** the report is read, **Then** it lists every NNNC class
   whose fingerprint was compared and the result, plus arm D's and the twin's per-fold
   reproduction.

### Edge Cases

- **The baseline differs from older records.** Arm D's stored records may predate later fixes
  (such as the post-grace stop sign fix). The comparison is against a fresh pre-change baseline,
  not old records. Differences between the baseline and old records are reported, not hidden.
- **The baseline finds stop-invariant violations.** If arm D violates an invariant today, halting
  on it would change arm D's results. The feature then stops before the merge, and the violation
  is reported as a defect to be fixed in its own change. "No behaviour change" and "halt on
  violation" cannot both hold for a defective arm.
- **NNNC classes that don't load today** (missing files, retired experiments) are listed as
  pre-existing failures and excluded from the comparison; they are not counted as merge
  regressions.
- **Model entries have no family tag today.** Per-family exits apply only to tagged trades; an
  untagged trade uses the strategy's shared exit. Under FR-007, arm D's entries stay untagged (its
  label implies no family), and no NNNC class declares a family exit today. The tags change no
  existing exit either way.
- **A trigger and the model fire on the same bar.** Under the default (model-only), the entry is
  the model's, exactly as today. Its tag is the label's family (FR-007), not the trigger's.
- **A model trained on a label without a declared family** (most legacy NNNC variants) is
  untagged, the same as arm D.
- **NNNC's own entry guards and threshold** (the per-fold prediction thresholds, the guard chain)
  must keep their current order and effect. The shared framework must not apply a second guard
  pass.
- **Coexistence and pre-emption rules (025/026)** are per-family and default off. They remain off
  for NNNC.
- **Startup length.** The regime gate needs a long warm-up. Declaring it on an NNNC arm must
  enforce that warm-up, as SimpleTrigger does, rather than silently running on a cold
  instrument.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: NNNC strategies and SimpleTrigger strategies MUST derive from one shared framework
  definition, which provides:
  - entry sources;
  - stop invariants;
  - per-family exits;
  - the regime gate;
  - merging overrides.

  NNNC MUST NOT carry its own copy of any of these.
- **FR-002**: The model MUST be an entry source of that framework. By default an NNNC strategy
  enters exactly when it does today: the model's prediction against its threshold, through its
  existing guard chain.
- **FR-003**: A trigger-AND-model entry mode MUST exist as a declared switch, off by default. When
  it is on, an entry requires both the declared trigger(s) and the model's acceptance.
- **FR-004**: Before any code changes, a baseline MUST be recorded on the current code:
  - arm D's and the twin's per-fold trade sets on f00–f14, re-run under fresh tags with their
    borrowed per-fold models;
  - the framework fingerprint of every loadable NNNC class;
  - arm D's stop-invariant violation count, in report-only mode.
- **FR-005**: After the change:
  - arm D and the twin MUST reproduce their baseline trade sets in 15 of 15 folds;
  - the twin MUST still equal SimpleTrigger's gbb arm;
  - every loadable NNNC class MUST keep its baseline fingerprint, unless the fingerprint gains
    only the default-off switch fields, which are omitted while off as in 028.
- **FR-006**: Every NNNC run MUST carry the stop invariants of 023 FR-017, and a violation MUST
  halt the run. If the baseline (FR-004) finds violations in arm D, the feature MUST stop before
  merging and report them.
- **FR-007**: Model entries MUST carry a family tag that leaves every existing exit unchanged.
  The tag MUST be label-driven (clarified 2026-10-12, A′):
  - each label declares the family it implies, with a one-line reason, as triggers already do
    (for example, gbb → reversion: its positives are weak band positions; breakout_gbb →
    momentum: strong band positions);
  - a model's entries take its training label's family;
  - a label that implies no entry state leaves its entries untagged, so they keep the shared exit.
    This includes the forward-peak label arm D is trained on (≥ 2% within 48 bars: a bounce and a
    breakout both qualify).

  Which family arm D's entries belong to, for example per bar by market state, is left to a later
  feature.
- **FR-008**: A later NNNC feature MUST be able to switch on, by declaration alone:
  - each per-family exit, including the ATR take-profit;
  - the regime gate;
  - the trigger-AND-model mode.

  Each switch MUST be shown live (it changes a decision) on at least one fold, and inert when not
  declared.
- **FR-009**: No model MUST be trained or retrained by any step of this feature. A missing
  borrowed model MUST fail loudly.
- **FR-010**: The shared framework definition MUST stay strategy-agnostic: no NNNC- or
  SimpleTrigger-specific names in the framework layer. Parameter dictionaries MUST merge from the
  parent. No pair-specific code.
- **FR-011**: The existing test suites MUST pass, apart from failures that already fail before the
  change, which are listed with their cause.
- **FR-012**: The report MUST record the baseline, the fingerprint comparison per class, arm D's
  and the twin's per-fold reproduction, the stop-check result, and each switch's liveness and
  declaration.

### Key Entities

- **Entry source**: something that proposes entry bars: a trigger, or the model with its threshold.
  It carries the family its entries are tagged with: a trigger's own family, or the model's
  training-label family.
- **Label family**: the entry state a label's positives imply (reversion, momentum), declared with
  a reason, or none.
- **Framework definition**: the one shared base that turns entry proposals into tagged entries
  and applies exits, stops, gates and invariants.
- **Baseline**: the pre-change per-fold trade sets, fingerprints and violation count that the
  change is judged against.
- **Declared switch**: a default-off framework feature an NNNC strategy turns on by declaration.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Arm D reproduces its pre-change trades in 15 of 15 folds. The twin does too, and it
  still equals the no-model gbb arm in 15 of 15 folds.
- **SC-002**: 100% of loadable NNNC classes keep their pre-change fingerprint. Classes that fail to
  load are listed, and fail identically before the change.
- **SC-003**: Every NNNC run carries the stop check. Arm D's baseline violation count is recorded,
  and is 0 for the merge to proceed.
- **SC-004**: Each of the declared switches (per-family exit, ATR take-profit, regime gate,
  trigger-AND-model) changes at least one decision when declared on an NNNC test arm, and none when
  not declared.
- **SC-005**: No model is trained during the feature (the count of training events is 0).
- **SC-006**: The test suites pass, except for failures shown to fail identically before the
  change.

## Assumptions

- Arm D is `NNNC_A_Raw_Open` run per fold as `NNNC_ARO_f00–f14` with its saved models and per-fold
  thresholds (plan `plan_NNNC_A_Raw_Open.json`, records `wf_aro_fNN`). The twin is
  `NNNC_TriggerGate` (`NNNC_TGT_fNN`, borrowing the same models).
- The baseline re-run uses fresh tags, because the harness skips existing records. Model-free
  determinism has been shown repeatedly (028 re-check). Model runs load fixed weights, so the
  baseline is expected to be deterministic; the baseline itself checks this by running one fold
  twice.
- Model backtests are slower than SimpleTrigger's (prediction per bar). The full baseline plus
  the after-change runs are two chains of 30 folds each, run sequentially (one chain at a time).
- Feature scalers are not regenerated: the feature include list does not change.
- The live dry-run bot (`BasketStrategy`) doesn't import this framework, and is not touched or
  restarted.
- NNNC's per-fold prediction thresholds and guard chain are part of "today's entry", and are
  reproduced, not redesigned.
- Every result in this feature is an equivalence check, not a performance claim. No quality-gate-2
  label is needed for "identical".
