# Research: Label-Exit Coordination (Phase 0)

## R1 — The duplication is three-way, not two-way

**Finding**: the ATR stop arithmetic — `-multiplier * atr_pct`, clamped to
`[floor, cap]`, then optionally divided by `sqrt(rvol)` and re-floored at −0.02 — appears
in **three** places, not the two named in the spec:

1. `Framework/BaseStrategy.custom_stoploss`, in the `use_atr_adaptive_stoploss and after_fill`
   branch.
2. `Framework/BaseStrategy._resolve_normal_stop`, whose own docstring says it *"Mirrors the
   after_fill ATR logic in `custom_stoploss`"* — a documented duplicate inside the same class.
3. `NNPredict/NNPredictStrategy._stop_aware_labels`, whose docstring says it uses *"the SAME
   stop arithmetic and parameter names as `Framework.BaseStrategy.custom_stoploss` /
   `_resolve_normal_stop`"*.

**Decision**: the extraction target is the arithmetic only, and all three call sites become
callers. FR-004 names only the NNPredict copy; this finding extends the work to the
intra-class duplicate as well, which is strictly within FR-001/FR-002's "exactly one place".

**Why it matters**: the spec's test (FR-003) would have failed immediately on the
`_resolve_normal_stop` copy. Discovering that at implementation time rather than now would
have looked like a test bug rather than a real finding.

## R2 — What is arithmetic and what is glue

**Decision**: `utils/ExitPolicy.py` holds pure functions over numbers and arrays. It does not
import freqtrade, does not take a `Trade`, and does not read a dataprovider.

Specifically, the shared primitive owns:

- the stop *level* given volatility and volume: `atr_stop_fraction(atr_pct, rvol, params)`
- the grace decision given trade age: `effective_stop_fraction(age_hours, atr_pct, rvol, params)`
- the forward simulation: given a price path and the parameters, when and why the trade closes

The live path keeps, in `BaseStrategy`:

- sourcing `atr_pct_roll` and volume from `self.dp.get_analyzed_dataframe`
- freqtrade's return convention — `1.0` meaning "no change", and the
  `(open_rate * (1 + normal) / current_rate) - 1` rescaling after the grace window expires

**Rationale**: the convention is freqtrade-specific glue whose meaning depends on how
freqtrade interprets the return value; moving it offline would be meaningless and moving it
into a shared primitive would make the primitive depend on a `Trade`. The arithmetic is what
was duplicated and the arithmetic is what moves.

**Alternative considered**: move the whole of `custom_stoploss`. Rejected — it would force
the offline simulator to construct fake `Trade` objects and a fake dataprovider, which is
more coupling than the duplication it removes.

## R3 — How the duplication test detects a second copy without being brittle

**Decision**: an allowlist test. The test walks every `.py` file under
`user_data/strategies/`, collects those that reference any of the exit-arithmetic parameter
names (`atr_stoploss_multiplier`, `atr_stoploss_cap`, `atr_stoploss_floor`,
`stoploss_grace_level`, `stoploss_grace_hours`, `use_volume_confirmation_stoploss`), and
asserts that set equals an explicit allowlist: the canonical primitive, the two thin callers
that must name the parameters to pass them through, the strategy classes that legitimately
*configure* them, and the tests. Any new file naming them fails the test, and the failure
message says to call `utils.ExitPolicy` or to extend the allowlist with a reason.

**Rationale**: the constraint we actually care about is "nobody re-implements this", and
re-implementing it requires naming the parameters. An allowlist expresses that directly.

**Alternatives considered**:

- *AST pattern matching* for "a unary-negated multiply followed by a `max(min(...))` clamp".
  Rejected: the matcher is itself a heuristic. It would miss a copy written with
  `np.clip`, and would fire on unrelated clamps. A test that can be fooled by reformatting is
  worse than no test, because it creates false confidence in exactly the invariant we are
  trying to protect.
- *String-grep for the expression text*. Rejected: brittle to whitespace and variable naming.
- *Coverage-based* — assert the shared function is called during a backtest. Rejected: proves
  the shared path runs, not that a second copy is absent. This is the failure mode the repo
  already hit, where wiring checks passed while a duplicated loss term never ran.

**Note**: the allowlist distinguishes *configuring* a parameter (a strategy setting
`atr_stoploss_multiplier = 2.5`) from *implementing* the arithmetic. Only the latter is
forbidden, so the allowlist is organised by role and the test message explains which kind of
reference is being flagged.

## R4 — Proving live behaviour unchanged (SC-001)

**Decision**: a characterisation test. Before the refactor, record the stop decision for
every bar of a real multi-pair, multi-year dataset into a fixture; after it, assert the new
path reproduces that fixture exactly. Both the grace and non-grace branches, and both
`after_fill` values, must be covered, because the grace branch returns `1.0` on the
non-`after_fill` call and that asymmetry is easy to lose.

**Rationale**: FR-005 demands proof, not inspection. The repo's own precedent is
`Framework/tests/test_training_pipeline_characterization.py`, so the pattern is established.

**Risk accepted**: the fixture must be generated from the pre-refactor code, so the ordering
is: write the test against current behaviour, see it pass, refactor, see it still pass. Doing
it in the other order would prove nothing.

## R5 — Outcome table storage and cache key

**Decision**: one feather file per `(pair, exit-config hash)` under
`research/outcomes/.cache/`, with a sidecar JSON recording the exit configuration in full.
The hash covers every parameter the simulation reads — the multiplier, cap, floor, grace
hours and level, the volume-confirmation flag, the take-profit, the hold cap, and the fee —
canonicalised and SHA-256'd, first 16 hex characters.

**Rationale**: mirrors `research/clfstudy`'s label-keyed feather cache, which worked. The
sidecar means a stale cache can be diagnosed by reading it rather than by guessing.

**Why the fee is in the hash**: a positive outcome is net of fees (FR-006a), so changing the
fee changes every outcome. Omitting it would be a silent-staleness bug of exactly the kind
FR-007 exists to prevent.

## R6 — Preventing outcome columns from reaching a model (FR-009)

**Decision**: structural, two layers.

1. **Naming**: every outcome column carries a reserved prefix (`__outcome_`). The existing
   feature-selection path builds its input from `FeatureNormalizer.include_list`, so a
   prefixed column cannot be selected by accident.
2. **An assertion at the boundary**: the trigger/label builder asserts that no column it
   hands onward carries the prefix. A test feeds a frame containing outcome columns into the
   label path and asserts it raises.

**Rationale**: FR-009 says "structurally prevented", and a convention alone is not structural.
Naming plus a raising assertion means the mistake fails loudly at the one place it could be
made.

**Alternative considered**: keep outcomes in a separate frame never joined to the feature
frame. Rejected as the primary mechanism — the join is needed to form labels, so the two must
meet somewhere; better to make that meeting point explicit and guarded than to rely on them
never touching.

## R7 — Expressing existing LabelMethods as triggers

**Finding**: of the 28 `LabelMethod` members, the ones that reduce cleanly to a trigger are
those whose BUY condition is a conjunction of a present-time condition and a forward-return
test. `labels_gbb` is the clearest: `guard_metric < 0.3 AND bb_width > 0.035` is the trigger,
the forward-gain test is the outcome. `labels_geometry` is a trigger (a DWT-smoothed valley)
plus guard filters. `forward_mae` has **no** trigger — it is the empty trigger, every bar.

Labels that do **not** reduce are those whose condition is itself a property of the forward
path with no present-time component — `mfe_mae_ratio`, `future_sharpe`, `future_sortino`,
`future_expectancy`, `risk_adj_future`. These are alternative *outcome* definitions, not
triggers, and under this design they are superseded by the shared outcome rather than ported.

**Decision**: the registry holds triggers only. The report lists the non-reducible labels with
that reason, satisfying FR-012. This is a genuine finding rather than bookkeeping: it means
roughly a third of the label family was never a labelling idea at all, but a competing
opinion about what "good" means — and this feature replaces all of them with one answer.

## R8 — The fixed learnability probe

**Decision**: ridge-regularised logistic regression on the flattened window, reusing
`research/clfstudy`'s existing `RIDGE` classifier type, its split set `v1`, its scaler
fitting discipline and its MCC computation. Same features, same protocol, every trigger.

**Rationale**: FR-015 requires one fixed probe so the column is comparable. clfstudy
measured a 14× parameter range spanning 0.0027 MCC on these labels, so a small probe loses
almost nothing — and RIDGE is already a first-class classifier type there with a frozen
config id, so reusing it means the number is comparable to ~2,000 existing runs.

**Why not the study's best architecture**: an LSTM costs ~1.5 hours for 15 folds, which would
reinstate the cost this feature exists to remove, and would make the learnability column
depend on an architecture choice that the verdict rule would then inherit.

**Caveat to carry**: RIDGE cannot express a logical AND, which is why it scored 0.533 against
the trigger rule's 0.673 on gbb. For a trigger-defined label the probe will therefore
*understate* learnability relative to a nonlinear model. That is acceptable because the
verdict rule compares each trigger's learnability against the **empty trigger's** measured
with the same probe, so the bias is common to both sides of the comparison — but it must be
stated in the report, because a reader could otherwise read the absolute numbers as ceilings.

## R9 — Uncertainty estimates for the FR-014b verdict rule

**Decision**: both differences get an uncertainty from the same source as the quantity.

- *Economic value*: bootstrap over triggered bars, resampled in contiguous blocks to respect
  autocorrelation, reporting a 95% interval on the difference from the empty trigger. Block
  resampling matters because adjacent bars' outcomes overlap heavily — the forward windows
  intersect — and an i.i.d. bootstrap would understate the interval badly.
- *Learnability*: the seed spread the probe already produces across clfstudy's seeds and
  windows, reported as a standard error on the difference.

**Rationale**: FR-014b needs "more than the measurement uncertainty on that difference", so
each side needs an interval, and the two are not the same kind of quantity.

**Explicit non-goal**: these intervals are *not* the pooled harness's ±3–4pp/yr. That floor
governs P&L claims (FR-017). The table's economic value is a different quantity measured a
different way, and conflating the two would be the error of treating a lookahead bound as a
realisable return.
