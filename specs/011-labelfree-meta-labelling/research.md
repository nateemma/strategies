# Research: labelfree-meta-labelling (brief 03)

Phase 0. Every NEEDS CLARIFICATION from Technical Context resolved here, plus the measurements
the plan depends on. Citations to `research/label-free-study/01-literature-review.md`.

---

## R1 — Store the primary's probabilities as debug columns in the framework

**Decision**: `get_predictions()` writes three `%`-prefixed debug columns —
`%predict_prob_sell`, `%predict_prob_hold`, `%predict_prob_buy` — carrying the probabilities
that actually drive the decision. The meta-label feature extractor reads those columns.

**Write position**: *after* the Markov-smoothing branch, so the columns always hold whatever
`argmax_with_threshold` consumed. **That branch is currently inert** —
`use_markov_smoothing = False` (`BaseNNStrategy:167`), no `NNNC_WF_f*` fold class overrides it,
and no `*_markov.npy` matrix exists for any fold model, so the smoothing cannot engage even if
the flag were flipped without also producing the matrices. For this feature the columns
therefore carry the raw `classifier.predict()` output. Writing after the branch rather than
before costs nothing today and keeps the columns correct if smoothing is ever enabled — the
alternative silently starts recording a different quantity than the one being traded on.

**Superseded**: an earlier revision of this decision had `reinfer.py` call the classifier
directly to avoid touching `Framework/`. That was the wrong trade. Calling the classifier
directly **duplicates the prediction path**, and a duplicated path drifts from the code that
actually runs — the exact failure this study has already recorded (`feedback_verify_training_
steps_by_measurement.md`, and the loss-logic duplication that let a term silently not run).
Storing what the framework already computes is strictly safer.

**Why the constitution permits it**: `Framework/` must stay *strategy-agnostic*, not frozen.
Probabilities the base class already computes for every NN strategy are agnostic by definition;
nothing strategy-specific is added.

**Finding it addresses**: `get_predictions()` computes `pred_probs`, optionally smooths it
(lines 1160–1162, inert here), prints its per-class statistics (1183–1189) — and then returns
only the thresholded class array. `process_predictions()` stores just:

```python
dataframe["predict_buy"]  = np.where(predictions == TradingAction.BUY, 1, 0)
dataframe["predict_sell"] = np.where(predictions == TradingAction.SELL, 1, 0)
```

A trade opened at confidence 0.81 and one at 0.99 are indistinguishable afterwards, and 0.8 is
the production `prediction_threshold` — so nearly every trade sits near the boundary where the
discarded digits matter most.

**The `%` prefix is load-bearing, not cosmetic.** `DataframeUtils.remove_debug_columns()` drops
every column matching `^%`, and it is applied before scaling (`FeatureNormalizer:334, 512`) and
in scaler creation (`CreateScalers:186`). That guarantees the model's own output **cannot** be
picked up as a training feature. Without the prefix this change would create a
predict-your-own-prediction leak. The convention already exists in the framework
(`%train_buy`, `%train_sell`).

**Alignment — the part that is easy to get wrong.** `predictions` is front-padded when the model
emits fewer rows than the dataframe:

```python
offset = original_length - predicted_length
preds = np.zeros(original_length, dtype=int); preds[offset:] = predictions
```

`pred_probs` receives no such treatment, so a raw assignment either raises on length mismatch or
silently misaligns every row. The three columns MUST therefore be:

1. **initialised to NaN at full dataframe length** — following the `%train_buy` init-then-set
   precedent (`BaseNNStrategy:954`); and
2. written at the **same `offset`** used for `predictions`.

**NaN, never 0.0.** Zero is a legitimate probability meaning "confidently not this class". Using
it as padding would mark every warm-up bar as a confident non-buy and quietly corrupt the
feature rows that fall there. With NaN, a trade whose entry bar lands in the warm-up region
yields an unbuildable row, which FR-006 already requires be **excluded and counted** rather than
imputed — the two rules compose correctly.

**Alternatives considered**: stashing `pred_probs` on `self` for `process_predictions` to write
(rejected: hidden cross-call state, and `get_predictions` already holds the dataframe); adding a
non-prefixed column (rejected: it would enter the feature list and leak); recomputing confidence
from the thresholded flag (rejected: it is one bit, which is what we need more of).

## R2 — Recomputed floor for the evaluation span: **4.681 pp/yr**

**Decision**: SC-001 is measured against **4.681 pp/yr**, not 3.350.

**Method**: the same statistic the harness uses — 20,000-resample bootstrap over per-trade P&L,
seeded 0, on the pooled trades restricted to folds 5–14, annualised by the span.

| quantity | value |
|---|---|
| trades / span | 287 / 2.45 yr |
| realised return over span | +19.087 pp |
| bootstrap 95% CI | [+7.489, +30.447] pp |
| half-width | 11.479 pp over span |
| **floor** | **4.681 pp/yr** |
| full-archive floor, for contrast | 3.350 pp/yr (378 trades / 3.68 yr) |

**Why it matters**: the floor is **40% higher** than the study-wide figure. Inheriting 3.350
would have accepted deltas the data cannot resolve. The brief anticipated this ("quote the floor
recomputed for that span"); this is that number.

---

## R3 — Discrimination metric against a 73.2% base rate

**Decision**: report **AUC** as the screening metric (SC-009's 0.55 gate), and report
**precision at the veto's operating point** beside it. Accuracy is excluded.

**Rationale**: 210 of 287 evaluation trades are winners. A model that accepts everything scores
73.2% accuracy, so accuracy cannot distinguish skill from the majority class. AUC is
threshold-free and base-rate-insensitive, which is what SC-009 needs. But AUC alone does not say
whether the *vetoed* trades were losers — precision at the operating point does, and it is the
quantity the P&L consequence actually follows from.

**Alternatives considered**: F1 (rejected: conflates the two error types this feature must keep
separate — vetoing a winner costs money, admitting a loser costs money, and they are not
symmetric); balanced accuracy (rejected: no advantage over AUC here and less familiar).

---

## R4 — Model family and capacity at n≈91→287

**Decision**: gradient-boosted trees via **LightGBM** (installed, 4.7.0), heavily capacity-
constrained, with `xgboost` (3.2.0) available as a cross-check if a result survives.

**Rationale**: the brief specifies "LightGBM / sklearn gradient boosting. No neural net — n ≈ 400".
The nested design is thinner than that headline: the fold-5 model trains on **91** trades, and
only the last folds see anything near 287. Capacity must be set for the *first* fold, not the
last. Concretely: shallow trees, a high minimum-samples-per-leaf, and few estimators, all pinned
before any run and recorded in the trial log.

**Risk accepted**: with 91 rows and ~9 features the fold-5 model may be too weak to be
informative. That is a real outcome, not a defect — it is reported per fold with its trade
count (FR-014), never averaged into a single number that hides it.

---

## R5 — Proving a feature cannot see the future

**Decision**: recompute each feature with all bars strictly after `open_date` **deleted from the
input frame**, and assert the value is bit-identical (FR-003).

**Rationale**: this is the only check that cannot be argued with. Inspection misses indicator
warm-up and centred windows; a shuffle test detects *informativeness*, not *causality*. Deleting
the future and recomputing is a direct test of the actual claim.

**The audit must be able to fail (FR-004)**: a planted-leak control — a feature set to the
trade's own realised `profit_abs` — is injected and the audit is required to flag it. This
mirrors the negative control in `research/labelfree/liveness.py`, which was added after a first
version of that control passed vacuously by not actually severing anything. An audit that cannot
detect a known leak certifies nothing.

**Alternatives considered**: permutation-only auditing (rejected: the brief's own shuffle test
measures whether a feature carries information, which is orthogonal to whether it is causal —
both are run, but only deletion tests causality).

---

## R6 — Nested assignment and the W3 probe

**Decision**: the model evaluated on fold *k* trains on trades whose `close_date` precedes fold
*k*'s `test_start`; the W3 probe inverts this deliberately.

**Rationale**: keying on `close_date` rather than `open_date` is stricter — a trade that opened
before the boundary but closed after it would otherwise contribute an outcome not yet knowable.
With 90-day folds and ~3h median holds the cost is a handful of rows, and the guarantee is
clean.

**W3 (2023) is not evaluable out-of-sample.** It lies entirely inside the warm-up folds, so no
nested model can precede it. Per clarification, it is run as an **anti-causal persistence
probe**: trained on later folds, tested on 2023, and reportable only as evidence about whether
the entry-feature/outcome relationship holds in a distant era. Its figure may never be quoted as
an out-of-sample return and never counts toward SC-001. This keeps Constitution Principle II
(NON-NEGOTIABLE; W1 and W3 are ~2.5 years apart) satisfiable without laundering a probe into a
return claim.

---

## R7 — Why pair features are excluded

**Decision**: no pair identity, pair group, or pair-derived history in the feature vector.

**Rationale**: the brief lists "the pair's recent hit rate (causal)". It *is* causal and would
pass R5's audit. But it hands the model a per-pair quality score, and the cheapest route to
discrimination then becomes ranking pairs — which is SC-010's kill condition and a finding this
repo has already recorded and closed (`project_universe_selection_no_lever.md`). Excluding pair
features means the model cannot rank pairs, so any measured gain is necessarily within-pair and
SC-010 becomes a genuine test rather than a formality.

**Cost accepted**: lower discrimination. The weaker-but-unambiguous test is the useful one.

---

## R8 — Re-inference reproduction gate

**Decision**: the re-inferred entry set must equal the archived `(pair, open_date)` set exactly;
any divergence halts the feature for diagnosis.

**Rationale**: the archived trades were produced 2026-09-09; features come from re-running those
models today. If the sets differ, the features describe a different model than the one whose
outcomes are the training target, and every downstream number is void. Feature 010's
`reproduce.py` achieved |d| = 0.00e+00 on a comparable path, so exact reproduction is a
realistic bar in this codebase rather than an aspiration.

**Failure is informative**: a mismatch means either the fold models, their pinned params, or the
feather data have changed since 2026-09-09 — each of which is a finding the harness should
surface anyway.

---

## R10 — What "in-sample" means for SC-008 under a nested design

**Decision**: in-sample = each fold-*k* model's advantage **on its own training trades**;
held-out = that same model's advantage on fold *k*. The differential is aggregated across
evaluation folds 5–14 and compared to the recomputed 4.681 pp/yr floor.

**Why this needs deciding now**: `stats.py::overfitting_differential` was written for a design
with a *screening* fold and a *held-out* fold. This design has neither — every evaluation fold
is held out by construction, and there is no screening pass. So SC-008 has no natural mapping,
and whoever implements it would have to invent a split. Inventing it **after** results exist is
exactly the post-hoc choice the gate exists to prevent, so it is fixed here, before any model
is trained.

**Why this split and not another**: it is the only pairing where both sides come from the same
model, which is what makes the difference attributable to overfitting rather than to the model
changing between folds. Comparing fold *k* against fold *k+1*, or warm-up against evaluation,
would confound overfitting with regime drift.

**Known limitation**: at fold 5 the in-sample side is 91 trades, so its advantage is measured on
a small sample and will be noisy. The differential is therefore reported **per fold with its
training count**, never as a single aggregate that hides which folds drove it — the same rule
FR-014 applies to returns.

---

## R9 — Prior evidence grade

Hudson & Thames' E-mini meta-labelling study reports OOS Sharpe 0.18 → 0.40 and 0.26 → 0.32,
with validation gains far larger than OOS — the gap itself being the caution. Their stated
boundary condition is the one that matters here: *"if the primary model is bad, meta-labelling
only reduces the downside."* The primary here is positive but concentrated, which is the regime
where the mechanism is plausible. Grade: **medium-weak**, with trivial cost — which is why this
brief is worth running and why its kill gates are strict.
