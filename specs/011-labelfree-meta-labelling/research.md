# Research: labelfree-meta-labelling (brief 03)

Phase 0. Every NEEDS CLARIFICATION from Technical Context resolved here, plus the measurements
the plan depends on. Citations to `research/label-free-study/01-literature-review.md`.

---

## R1 — The primary model's entry confidence is computed and then thrown away

**Decision**: capture `pred_probs[:, TradingAction.BUY]` inside the prediction path rather than
reading a dataframe column, because no column carries it.

**Finding**: `Framework/BaseNNStrategy.get_predictions()` computes a full probability array
`pred_probs` and even prints its statistics, but `process_predictions()` stores only the
**thresholded** result:

```python
dataframe["predict_buy"]  = np.where(predictions == TradingAction.BUY, 1, 0)
dataframe["predict_sell"] = np.where(predictions == TradingAction.SELL, 1, 0)
```

The raw probability is discarded. A trade that opened at confidence 0.81 and one at 0.99 are
indistinguishable after this point, and 0.8 is the production `prediction_threshold`.

**Consequence for the plan**: the re-inference module must call the classifier directly and keep
the probability vector. It MUST NOT modify `BaseNNStrategy` — the framework is
strategy-agnostic by constitution, and a feature-local capture keeps it that way.

**Alternatives considered**: adding a `predict_confidence` column to the framework (rejected:
changes the shared prediction path for every strategy to serve one research feature);
re-deriving confidence from the thresholded flag (rejected: it is one bit, which is precisely
the information we need more of).

---

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

## R9 — Prior evidence grade

Hudson & Thames' E-mini meta-labelling study reports OOS Sharpe 0.18 → 0.40 and 0.26 → 0.32,
with validation gains far larger than OOS — the gap itself being the caution. Their stated
boundary condition is the one that matters here: *"if the primary model is bad, meta-labelling
only reduces the downside."* The primary here is positive but concentrated, which is the regime
where the mechanism is plausible. Grade: **medium-weak**, with trivial cost — which is why this
brief is worth running and why its kill gates are strict.
