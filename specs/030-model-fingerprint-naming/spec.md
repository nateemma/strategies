# Feature Specification: Model Naming by Model Fingerprint

**Feature Branch**: `030-model-fingerprint-naming` (spec directory; work is committed to master)

**Created**: 2026-10-12

**Status**: Draft

**Input**: User description: "Model naming by resolved model fingerprint (queued after 029). Saved
weights are keyed off the strategy CLASS NAME. Consequences: the self-training hazard (a renamed,
derived or per-fold class finds no weights and silently trains inside its backtest); reusing
identical weights needs class-name redirects (BorrowedModelMixin, hand-written NNMT overrides); a
directory name doesn't say what model it holds. Goal: name and locate models by a fingerprint of
everything that shapes the weights (classifier type and architecture, label method and parameters,
features and scaler, augmentation, training window, seed), with a readable prefix + short hash and
a manifest beside the weights. Migrate existing saved_data by aliasing, never retraining.
Strategy-level settings that don't shape weights stay out of the key. A key collision is worse
than a retrain; a missing model is a loud failure, never a silent retrain. No behaviour change:
arm D and the reference NNNC arms reproduce 15/15, 029 fingerprints unchanged, zero training.
Open scope question: which NN families adopt it. Out of scope: retraining, changing any model, the
NNNC framework study (queued next)."

## Background

**Weights are found by class name today.** An NN strategy saves and loads its model in a directory
named after its own class, and trains a new model whenever that directory is empty. That training
happens inside whatever run first loads the class, including a backtest.

This causes three problems, all met in recent features:
- **The self-training hazard.** A renamed class, a subclass or a per-fold class finds no weights
  and quietly trains a new model in the middle of an evaluation. The walk-forward harness carries
  guards against this, including renaming classes and model directories per seed.
- **Reusing weights takes redirects.** Testing an exit or entry switch on an existing model means a
  new class, which by default would train a new model. Reuse works only through a hand-written
  redirect:
  - `BorrowedModelMixin` serves the 023 twin, the 029 probe, the 029 liveness arms and the 12
    switch-experiment arms;
  - about 11 NNMT classes carry their own path overrides.
- **Names don't describe models.** `NNNC_ARO_f00` and `NNNC_GBR_f00` differ only by an
  abbreviation. Which label, targets, features or window produced them isn't recorded with the
  weights.

**Shared scalers are an unkeyed input.** Models also depend on scaler files kept at the top of the
model store (e.g. `main_scaler.pkl`), shared by name across models. If one were regenerated, every
model trained on the old version would silently be served the new one.

**What this feature is.** Weights are located by a **model fingerprint**, a key resolved from
every input that shapes them. Two strategy classes that would train the same model therefore find
the same weights, and a class whose model inputs differ in any way can never load another's.
Settings that act after the model (entry guards, prediction threshold, exits, stops, family exits,
regime gate, trigger) are not part of the key. Every exit or entry variant of a model therefore
uses its weights directly.

**Why now.** The next feature studies entry, exit and stop-loss features directly on NNNC model
entries. That study declares many variants of a few models. It needs the variants to share
weights safely, without redirects or training.

## Clarifications

*(Open questions are marked inline; see Q1–Q3.)*

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Variants of a model share its weights automatically (Priority: P1)

The researcher declares a new strategy that differs from an existing model's strategy only after
the model: a different exit, stop, guard, threshold or regime gate. Run on the same walk-forward
folds, it loads the existing per-fold weights with no redirect and trains nothing.

**Why this priority**: it is what the next feature (the NNNC framework study) needs. It also
retires the most-used workaround, `BorrowedModelMixin`.

**Independent Test**:
1. Declare a subclass of arm D that changes only an exit parameter, with no mixin.
2. Run it on f00–f14.
3. Confirm that no training occurred, that it loaded arm D's per-fold weights, and that with the
   exit change removed it reproduces arm D's per-fold trades.

**Acceptance Scenarios**:

1. **Given** arm D's per-fold models exist, **When** a subclass that changes only exits is run on
   any fold, **Then** it resolves to the same model key as arm D on that fold and trains nothing.
2. **Given** a former `BorrowedModelMixin` user (the twin, the probe, the liveness arms, the switch
   arms) without the mixin, **When** it is run, **Then** it loads the same weights it loaded with
   the mixin, and its per-fold trades are unchanged.
3. **Given** two classes with identical model inputs but different names, **When** either is
   loaded, **Then** both resolve to one key.

---

### User Story 2 - A different model can never load another's weights (Priority: P1)

The researcher changes any input that shapes the weights: label method or targets, horizon,
features, scaler, classifier or architecture, augmentation, training window or seed. The strategy
gets a different key, so it can never silently load weights trained for something else.

**Why this priority**: a key collision serves wrong weights and produces a plausible-looking but
meaningless result. That is worse than a retrain, so the key must be complete.

**Independent Test**: for each weight-affecting input, derive a class that changes only that input
and confirm its key differs from the parent's. For each post-model setting, confirm the key is
unchanged.

**Acceptance Scenarios**:

1. **Given** arm D, **When** a subclass changes the label gain target (or any other
   weight-affecting input), **Then** its key differs, and with no weights under the new key the run
   fails loudly rather than training (subject to Q2).
2. **Given** a model trained on a shared scaler, **When** that scaler's contents change, **Then**
   the model's key changes. The old weights are never paired with the new scaler.
3. **Given** two walk-forward folds of one strategy, **When** their keys are resolved, **Then** they
   differ by training window only.

---

### User Story 3 - Existing models keep working with no retraining (Priority: P1)

Every existing trained model is indexed under its fingerprint without retraining and without
moving or altering its weights. Every recorded arm reproduces its trades.

**Why this priority**: per-fold model sets are expensive and are the baseline for every result
since 021. Losing or silently swapping them would invalidate the record.

**Independent Test**:
1. Run the migration.
2. Re-run arm D and the reference NNNC arms on all folds.
3. Compare per-fold trades with today's baseline. Confirm that no weight file changed and that no
   training occurred.

**Acceptance Scenarios**:

1. **Given** the existing model store, **When** migration runs, **Then** each walk-forward model
   gets a key and a manifest, its weight files are byte-identical, and nothing is retrained.
2. **Given** two existing directories that resolve to the same key but hold different weights,
   **When** migration runs, **Then** it refuses to merge them and reports the conflict. Neither is
   served under the shared key until the conflict is resolved.
3. **Given** a model whose training inputs can't be fully reconstructed (e.g. an old single model
   with an unrecorded training window), **When** migration runs, **Then** it is handled under Q3.
   It is never given a key that a different model could also resolve to.

---

### User Story 4 - A model directory says what it holds (Priority: P2)

Browsing the model store, the researcher reads each model's type, label and targets from its
directory name. The full resolved inputs are in a manifest beside the weights.

**Why this priority**: useful and cheap once the key exists, but the safety stories come first.

**Independent Test**: for arm D's f00 model, the directory name shows classifier type, label
method and gain target. The manifest lists every key input and the classes known to use the model.

**Acceptance Scenarios**:

1. **Given** any keyed model, **When** its manifest is read, **Then** it lists:
   - every input that went into the key;
   - the training window;
   - the seed;
   - the scaler identity;
   - the date the manifest was written.
2. **Given** a strategy class, **When** the researcher asks which model it uses, **Then** a command
   prints the resolved key and whether weights exist under it.

### Edge Cases

- **A class overrides model loading by hand** (the NNMT path overrides): it is either migrated to
  the fingerprint or kept working unchanged, with the choice recorded per class. It must never
  half-apply.
- **Seed:** the key carries the seed. The walk-forward harness no longer needs to rename classes
  per seed to keep seeds apart, but existing seed-suffixed directories still resolve.
- **Training window:** a per-fold class must know its training window to resolve its key. A class
  run outside the harness has a window too (the run's), and the key must not silently fall back to
  a different model when it is unknown.
- **Auxiliary artefacts:** Markov matrices, GAN discriminators and PCA files live beside or near
  the weights. Each either travels with the key it belongs to or is itself keyed.
- **Code changes:** a change to the classifier's code that changes what training produces is not
  visible from declared inputs. The key carries an explicit model-code version that is bumped by
  hand when training behaviour changes, so old weights are not served to new code silently.
- **Concurrent writes:** two runs resolving the same key must not both train into it (one training
  chain at a time remains the rule).

## Requirements *(mandatory)*

### Functional Requirements

**The key**

- **FR-001**: Every NN strategy in scope (Q1) MUST locate its weights by a model key resolved from
  all weight-affecting inputs:
  - classifier type and architecture/hyperparameters;
  - label method and every label parameter (gain and loss targets, horizon, stop/MAE limits, label
    thresholds);
  - the feature set and its preprocessing;
  - the identity of every scaler or fitted transform the model is trained on (by content, not by
    file name);
  - augmentation settings;
  - the training window;
  - the seed;
  - a model-code version.
- **FR-002**: Settings that act only after the model MUST NOT enter the key. These are entry
  guards, prediction threshold, exits, stops, family exits, regime gate, trigger, entry mode and
  coexistence rules. Changing any of them MUST leave the key unchanged.
- **FR-003**: The key MUST be deterministic: the same inputs give the same key in every process,
  regardless of class name, module path or dictionary order.
- **FR-004**: Inputs MUST be resolved as the strategy actually uses them, after class inheritance
  and parameter-file merging, not as declared on one class.
- **FR-005**: The location MUST combine a human-readable prefix (classifier type, label method, main
  targets) with a short hash of the full key. Only the hash decides identity; the prefix is for
  reading.

**Manifest and lookup**

- **FR-006**: Each keyed model MUST have a manifest beside its weights. It records the resolved key
  inputs, the training window, the seed, scaler identities, the model-code version, when it was
  written, and the classes known to use it.
- **FR-007**: A command MUST report, for a given strategy class (and fold), its resolved key, its
  location and whether weights exist there.
- **FR-008**: When a strategy loads weights, the manifest's recorded inputs MUST be checked against
  the strategy's resolved inputs. A mismatch MUST halt the run with a message naming the differing
  inputs.

**Missing weights**

- **FR-009**: In walk-forward evaluation, a missing model MUST be a loud failure that names the key
  and its inputs, never a training run.
- **FR-010**: Training into a key MUST follow the policy chosen in
  [NEEDS CLARIFICATION: Q2 — may a run outside walk-forward evaluation still train on a missing
  key automatically, or only an explicitly requested training run?]

**Migration**

- **FR-011**: Existing trained models MUST be indexed under their keys without retraining and
  without altering any weight file. Indexing by reference is acceptable, so weights need not be
  physically moved.
- **FR-012**: Migration MUST refuse to assign one key to two directories whose weights differ, and
  MUST report every such conflict.
- **FR-013**: Models whose training inputs can't be fully reconstructed MUST be handled per
  [NEEDS CLARIFICATION: Q3 — legacy models with unrecorded inputs].
- **FR-014**: Seed-suffixed and per-fold directories created under the old naming MUST continue to
  resolve for the classes that used them.

**Scope and retirement**

- **FR-015**: The families that adopt the key in this feature are
  [NEEDS CLARIFICATION: Q1 — NNNC only, all BaseNNStrategy families, or also TSPredict].
- **FR-016**: `BorrowedModelMixin` MUST become unnecessary: each of its current users MUST load the
  same weights without it. The mixin is then removed, or left as a documented no-op if removing it
  would change a recorded fingerprint.
- **FR-017**: The walk-forward harness's checks for an existing model MUST use the resolved key
  rather than a class or directory name.

**No behaviour change**

- **FR-018**: Arm D and the reference NNNC arms MUST reproduce their per-fold trades 15/15 after the
  change. The reference arms are GbbRaw, GbbFwdMae, ForwardMAE_Raw and NNNC_MLX, together with the
  twin and a switch-experiment arm.
- **FR-019**: Every NNNC strategy's 029 framework fingerprint MUST be unchanged, except where a
  change is listed and justified.
- **FR-020**: No training MUST occur during migration or verification, and no weight file's bytes
  may change.
- **FR-021**: The resolver MUST be strategy-agnostic if placed under `Framework/`. Strategy-family
  knowledge (which attributes shape a family's weights) MUST be declared by the family, not
  hard-coded in Framework.
- **FR-022**: The live dry-run bot's strategy is not an NN strategy and MUST be untouched.

### Key Entities

- **Model key**: the resolved set of weight-affecting inputs, plus its hash. It identifies one
  trained model.
- **Model location**: the readable prefix plus the hash, where the weights and manifest live (or
  are indexed).
- **Manifest**: a record beside the weights holding the key inputs, window, seed, scaler identities,
  code version, write time and known user classes.
- **Model index**: the mapping from keys to existing weight locations, built by migration. It
  covers legacy class-named directories.
- **Weight-affecting input declaration**: per family, the list of attributes and parameters that
  shape its weights. This is what keeps the resolver generic.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Arm D and every reference NNNC arm reproduce their baseline per-fold trades in 15/15
  folds, with zero training runs and zero changed weight bytes.
- **SC-002**: Every former `BorrowedModelMixin` user, run without the mixin, reproduces its
  recorded trades 15/15 (or on its recorded folds).
- **SC-003**: For every declared weight-affecting input, a class differing only in that input gets
  a different key. For every post-model setting, the key is unchanged. Both are checked by tests,
  with 100% of declared inputs covered.
- **SC-004**: Migration indexes 100% of the existing walk-forward model sets (ARO, GBR, GFM, FMR, WF
  and their seed variants). It reports every conflict and every legacy model it couldn't key.
- **SC-005**: A class renamed with no other change finds its weights and trains nothing, on the
  first attempt.
- **SC-006**: Every NNNC strategy fingerprint is unchanged (approximately 300 strategies), or the
  change is listed and justified.

## Assumptions

- **Training stays inside a run.** Training is still triggered from within a strategy run today: the
  walk-forward harness trains a fold by running the class over its training window. That mechanism
  is kept; what changes is where the weights go and when training is allowed (Q2).
- **The per-fold training window is known.** It is in the walk-forward plan, so per-fold classes can
  be told their window (e.g. by the generated fold class). For the five walk-forward sets the
  window is recorded in their plans.
- **Seeds are recorded.** For walk-forward sets, the seed is the plan's seed (`_sN` suffix, or 1 by
  default).
- **Weight-identical directories.** Two existing directories that resolve to the same key are
  expected to be rare. Any found is reported, not merged (FR-012).
- **Out of scope:**
  - retraining or changing any model;
  - the NNNC framework-feature study, which is queued next;
  - TSPredict's per-pair model files, unless Q1 brings them in.
- **Model-code version.** It starts at one value for all existing models. Bumping it is a manual
  step when training behaviour changes, documented in the agent guide.
