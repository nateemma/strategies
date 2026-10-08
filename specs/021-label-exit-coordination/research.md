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

## R10 — T021 FOUND THE SPEC'S EXIT POLICY IS INCOMPLETE (2026-10-08)

The reconciliation did its job. Matching all 1,410 control-arm trades to table entries by
`(pair, open_date)` — 100% matched — gives:

| metric | arm (realised) | table | divergence |
|---|---|---|---|
| win rate | 0.6702 | 0.6972 | **+2.70pp** |
| mean profit/trade | 0.004679 | 0.005182 | **+10.7% relative** |

The table is **optimistic**, which is the dangerous direction (Principle V).

**Root cause: FR-001 enumerates the exit policy incompletely.** It lists the ATR-adaptive
stop with cap/floor, the static fallback, the grace window, the take-profit and the hold cap.
The live Open operating point ALSO has:

* `roi = {"0": 0.05}` — a 5% ROI exit. 37 of the arm's 1,410 trades exited this way.
* `unclog_12h` — hardcoded in `BaseStrategy.custom_exit`, deferred by `NNNC/_hold_window.py`.
  86 trades.
* `unclog_1d` — 1 trade.
* `force_exit` — 2 trades.

`simulate_forward` models stop / take-profit / hold-cap only, so it is faithful to the
contract and the contract was wrong. This is a specification defect, not an implementation
bug, and the fix is to amend FR-001 and the exit-policy contract before changing code.

**One misread in the original report, corrected.** It treated the arm's 462
`trailing_stop_loss` exits as a fifth missing mechanism. They are not: `trailing_stop` is
`false` at this operating point, and `trailing_stop_loss` is simply how freqtrade LABELS an
exit taken via `custom_stoploss`. Reconciling the mix accordingly:

* table `stop` 427  vs  arm `trailing_stop_loss` 462 + `stop_loss` 3 = **465** (table under
  by 38)
* table `take_profit` 983  vs  arm `take_profit` 819 + `roi` 37 + `unclog_12h` 86 +
  `unclog_1d` 1 + `force_exit` 2 = **945** (table over by 38)

The two offsets match, which is consistent with the single cause: the missing exits fire
EARLIER than the stop would have, so the table credits a later take-profit where the arm
had already left, and misses stops the arm avoided by leaving first.

**What this does NOT justify.** The report's verdict of "TABLE IS INVALID — DO NOT USE" is
stronger than the evidence. Mean profit per trade diverges 10.7% relative, not
catastrophically, and the win-rate error is largely offsetting (false winners and false
losers of similar magnitude). The table is *biased*, not garbage.

**But it cannot be used for verdicts as-is**, for a reason the magnitudes do not show: the
bias only cancels out of a trigger COMPARISON if it is uniform across triggers. Triggers
selecting longer-held or choppier trades would be mismodelled more, and the verdict rule
(FR-014b) compares triggers against each other. An unquantified, possibly trigger-dependent
10% bias in the ranking quantity is not acceptable, so this blocks US3/US4 until fixed.

## R11 — Reconciliation converged on the aggregate; two subset biases remain (2026-10-08)

Four reconciliation runs against the control arm's 1,410 trades (100% matched on
`(pair, open_date)` every time):

| run | all trades | stop subset | profit subset | win rate |
|---|---|---|---|---|
| 1 incomplete exit model | +10.7% | — | — | +2.70pp |
| 2 exit model completed | +72.1% | +0.0176 | — | +0.64pp |
| 3 stop fill price fixed | −35% | +0.0116 | −0.0082 | +0.64pp |
| **4 fee netting fixed** | **−4.7%** | **+0.0107** | **−0.0056** | **−0.71pp** |

Run 4: table +0.004459 against the arm's +0.004679, delta −0.00022, inside ±0.0022.
Win rate −0.71pp, inside ±2.45pp.

**The aggregate agreement is arithmetic cancellation, not fidelity.** The control's exit
mix is 33% stop / 67% profit, and `0.33 × 0.0107 − 0.67 × 0.0056 = −0.0002` — which is the
aggregate delta exactly. A trigger whose exit mix differs carries a bias of up to ±0.01 per
trade against a mean of 0.0047, so a stop-heavy trigger is flattered and a
profit-exit-heavy one penalised.

### The one-bar-offset hypothesis is REFUTED

It was attractive because a single cause would explain both residuals' opposite signs: if
the table exited a bar early, it would be too shallow on stops in a falling move and too
early on take-profits in a rising one. Measuring the arm's hold duration against the
table's kills it:

- all trades: **median difference 0** bars; 500 of 1,410 differ by exactly 0
- **profit exits: mean +0.197, median 0** — timing essentially matches, so there is no
  offset to explain this residual
- **stop exits: mean +5.465, median +1** — the arm holds substantially LONGER, with a long
  right tail

So the two residuals have DIFFERENT causes, and only the stop one is a timing problem. The
table exits stops too early (which is why it is optimistic there: an earlier exit in a
falling move is a smaller loss), but not by a fixed offset — the median is one bar while the
mean is five and a half, so it is a tail effect on a minority of trades rather than a
systematic shift.

**Why this is worth recording rather than fixing blind**: a plausible single-cause theory
was disposed of by one measurement that needed no code change. The profit-exit residual is
NOT a timing artefact and still has no diagnosis.

## R12 — the entry convention was the dominant cause; my own estimate of its size was wrong

Run 5, after entering at `open[start_index]` instead of `close[start_index]`:

| | run 4 | run 5 |
|---|---|---|
| entry-bar slope | −1.072 | **−0.028** |
| correlation | −0.495 | **−0.022** |
| all trades | −0.000220 | **+0.000187** (inside ±0.002199) |
| stop-exit subset | +0.010740 | **+0.004304** (−60%) |
| profit-exit subset | −0.005610 | **−0.001838** (−67%) |
| win rate | −0.71pp | −0.50pp (inside ±2.45pp) |

The slope and correlation collapsed to ~0, confirming the mechanism was correctly
identified and fully removed.

**I predicted this would close only 25–30% of the subset bias. It closed 60–67%, and the
reason I was wrong is worth recording.** The 25–30% figure came from linearly de-trending
the divergence against the entry-bar move — which removes only the *additive* component of
the entry-price effect. The real effect is path-dependent: the entry price sets the stop
level itself (`entry × (1 + stop_frac)`) and the take-profit level, so it changes *which*
exits fire and *when*, not merely the arithmetic of a fixed outcome. A linear adjustment
cannot capture that, so it understated the true contribution by a factor of two.

The general lesson: de-trending estimates the share of variance a cause explains, not the
share of BIAS removing it will fix, whenever the cause feeds back into the mechanism.

**Where this leaves the table.** The aggregate is inside its interval and the entry effect
is gone. Both subsets remain outside, but at 1.4× and 1.8× their intervals rather than 3.6×
and 5.6×. Residual bias is +0.0043 on stop exits and −0.0018 on profit exits, so a trigger
with an extreme exit mix is exposed to at most ~0.004 per trade against a population mean
of 0.0047 — down from ~0.011. The control's own 33/67 mix nets to +0.0002.
