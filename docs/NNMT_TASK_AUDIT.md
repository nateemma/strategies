# NNMT multi-task audit — pre-screens before a task-combination search

Question: is there a cheap way to find the best combination of tasks without
training all 2^n? `trading` is always present. Findings below are from code
reading + static model inspection — **zero training runs**.

> **CORRECTION (2026-09-01).** A first version of this audit claimed the
> inference task-filter gates entries using untrained heads. **That was wrong.**
> `apply_task_filters` resolves to **False**, so the inference filter does not
> run. The tasks act as MTL regularisers, as designed. The corrected analysis is
> below; what survives is smaller but still relevant to task reduction.
>
> Why the error: `Framework/BaseNNStrategy.buy_params` sets
> `"apply_task_filters": True`, and I took that as the effective value.
> But `NNMTStrategy.buy_params` **REPLACES** that dict rather than merging it
> (the `{**BaseNNStrategy.buy_params, ...}` form is commented out at
> NNMTStrategy.py:68), so the key never reaches NNMT and the
> `BooleanParameter(default=False)` applies. Resolution order matters here and
> I asserted it instead of tracing it.

## Where each task actually acts

`_CLASSIFIER_TASK_WEIGHTS = {trading 4.0, regime 0.0, risk 1.0, momentum 1.0,
flow 1.0, profit 0.0}`

| task | shapes the trading LABEL? | trained as an aux head? |
|---|---|---|
| trading | it *is* the label | yes (4.0) |
| profit | **YES** (ground truth) | **NO** (0.0) |
| regime | **YES** (ground truth) | **NO** (0.0) |
| momentum | YES (ground truth) | yes (1.0) |
| flow | YES (ground truth) | yes (1.0) |
| **risk** | **NO** | yes (1.0) |

Two mechanisms, and they are NOT the same set:

1. **Label definition (always on).** `get_trading_classes` calls
   `_filter_trading_by_tasks` unconditionally with GROUND-TRUTH task values:
   `buy = raw_buy & (>=3 of [profit!=LOSS, regime!=BEAR, momentum!=NEG,
   flow!=DECREASE])`; the sell side is unfiltered. So profit/regime/momentum/flow
   shape the target. `risk` does not — it is never referenced in the live body
   (verified by AST; it appears only inside a 32-line dead string literal).
2. **Auxiliary MTL heads (the intended mechanism).** Only tasks with non-zero
   loss weight get gradient: `MLXClassifierMultiTask:450` multiplies each task
   loss by its normalised weight. So the actual regularisers are
   **risk, momentum, flow**.

## What survives, and why it matters for task reduction

- **`regime` and `profit` are dead heads.** Weight 0.0 means no gradient, and
  with the inference filter off their predictions are never consumed either. They
  contribute nothing as regularisers. **The effective config is already 4 tasks
  (trading + risk + momentum + flow), not 6** — two are removed in all but name.
  Note this does NOT mean they are irrelevant: their GROUND-TRUTH labels still
  shape the target. Not predicting something is compatible with it defining your
  label.
- **`risk` is the only pure regulariser** — trained, but it neither shapes the
  label nor gates anything. It is the cleanest single ablation candidate: drop it
  and the target is unchanged, so the run isolates the MTL effect exactly.
- **The dead 32-line string literal** in `_filter_trading_by_tasks` should go; it
  documents a rule that is not the live one and it is what made `risk` look used.

## The cheap search (still not 2^n)

The filter-side exhaustive sweep proposed earlier is **moot** — the inference
filter is off, so there is nothing to sweep. What remains is the MTL question,
and it is the cheap one:

1. **Leave-one-out, n+1 runs.** Only 3 tasks are actually trained, so this is
   ~4 runs: all-tasks, minus-risk, minus-momentum, minus-flow (plus the existing
   `NNMT_MLX_tronly` as the zero-aux end). Changing a task's WEIGHT does not
   change the target, so these runs are clean comparisons.
2. **One run, gradient affinity.** Log the cosine between each aux head's
   gradient and the trading head's gradient on the shared trunk. Negative cosine
   identifies a task that actively fights trading. Gives the full picture from a
   single training run.
3. **Free pre-screen: label correlation.** Correlate the six task label vectors.
   Highly correlated auxiliaries are redundant regularisers.

Separately, and more expensive: "which tasks should DEFINE the label" (the
`>=3 of 4` rule in `get_trading_classes`) does require retraining per variant,
because changing it changes the target. Worth keeping distinct from the MTL
question — they are different experiments that happen to touch the same names.


## FIXED (2026-09-01) — the latent trap is closed

`apply_task_filters` is `BooleanParameter(optimize=True)`, so a hyperopt search
can switch it on. If it did, the filter would have gated entries on `profit` and
`regime` — heads with loss weight 0, i.e. never trained. That is the trap: the
code is currently harmless only because a default happens to be False.

**Fix, in `_filter_trading_by_tasks`:**

- New `active_tasks` argument. `None` (the default) uses all four tasks, which is
  what LABEL construction wants — `get_trading_classes` passes GROUND-TRUTH task
  values, and those are valid whether or not the head is trained.
- The inference path now passes `self._active_filter_tasks()`, which returns only
  tasks with non-zero `_CLASSIFIER_TASK_WEIGHTS`. Under the current weights that
  is `('momentum', 'flow')`; `profit` and `regime` are excluded automatically,
  and it self-corrects if the weights change later.
- Threshold generalised from a hardcoded `>= 3` to `len(conds) - 1` — the same
  "tolerate one dissenting task" rule, which reproduces 3-of-4 exactly when all
  four are active. A hardcoded 3 with only 2 active tasks would have been
  unsatisfiable and blocked every buy.
- Degenerate case (no trained filter tasks) passes trading through unchanged
  rather than silently blocking all entries.
- Removed the unused `risk_predictions` parameter and the 32-line dead string
  literal. Between them they were what made `risk` look load-bearing.
- Logs which tasks it is ignoring and why, when the filter is enabled.

**Verified on 20k random rows:**

| property | result |
|---|---|
| label path identical to previous implementation | **True** |
| inference path with trained-only heads differs | True (3989 -> 5945 buys) |
| no usable heads -> passes through, does not block all | True |
| `_active_filter_tasks()` under current weights | `('momentum', 'flow')` |

The label-path equivalence is the important one: `get_trading_classes` runs
regardless of `apply_task_filters`, so any change there would have altered every
NNMT target. It does not.


## Ownership fix (2026-09-01) — MT-only params removed from the base classes

`apply_task_filters` is multi-task-only: it is declared as a `Parameter` solely
in `NNMTStrategy` and read only at `NNMTStrategy:1083`. It had no business in the
shared base classes, and its presence there is what produced the wrong reading in
the first version of this audit.

It was in TWO base dicts:

- `BaseStrategy.buy_params` → `False`
- `BaseNNStrategy.buy_params` → `True` (overriding the above)

Neither ever reached NNMT — `NNMTStrategy.buy_params` is a fresh dict, not a
merge — but both propagated a dead key into every strategy that DOES merge:
`SimpleStrategy`, `TSPredict`, `NNAnomalyStrategy`, `DebugNNStrategy`. Those
carry a buy_params entry with no `Parameter` to receive it.

The same applied to four neighbours in the same dict — `bias_profit_high`,
`bias_profit_low`, `bias_trading_buy`, `bias_trading_sell` — all declared as
Parameters only in `NNMTStrategy`. All five arrived as hyperopt-dump leftovers
(hence the `# value loaded from strategy` markers on every line).

**Removed all five from `BaseStrategy.buy_params`, and `apply_task_filters` from
`BaseNNStrategy.buy_params`.** Verified inert:

| class | buy_params keys | MT-only leaked |
|---|---|---|
| BaseStrategy | 11 | none |
| BaseNNStrategy | 11 | none |
| NNNCStrategy | 8 | none |
| NNMTStrategy | 9 | none |

`NNMTStrategy.apply_task_filters` still exists as the `BooleanParameter` and
still defaults to `False` — behaviour is unchanged everywhere, because the
removed entries were already shadowed by the fresh dicts in NNNC and NNMT.

**Deliberately NOT re-added to `NNMTStrategy.buy_params`.** The `BooleanParameter`
default is now the single source of truth; adding a redundant buy_params entry
would recreate the same two-places-define-it problem that caused the original
misreading.


## Label-space analysis (2026-09-01) — zero training runs

107,129 rows, 6 pairs, 15m, real feathers. Labels produced by the actual
strategy functions (`DataframePopulator.add_minimal_indicators` ->
`add_additional_indicators` -> the `get_*_target` functions), so these are the
labels the model is really trained against.

### Class balance

| task | 0 | 1 | 2 |
|---|---|---|---|
| **trading** | 0.071 | 0.869 | 0.059 |
| profit | 0.218 | 0.550 | 0.232 |
| regime | 0.228 | 0.554 | 0.218 |
| momentum | 0.168 | 0.749 | 0.083 |
| risk | 0.420 | 0.455 | 0.125 |
| flow | 0.437 | 0.171 | 0.392 |

`trading` is heavily imbalanced (5.9% buy, 7.1% sell); the auxiliaries are all
far better balanced, which is part of why they can act as regularisers at all.

### Normalised mutual information with `trading`

| task | NMI | trained today? |
|---|---|---|
| **profit** | **0.130** | **NO (weight 0)** |
| risk | 0.035 | yes |
| momentum | 0.028 | yes |
| regime | 0.017 | NO (weight 0) |
| flow | 0.011 | yes |

**The current configuration trains the three tasks with the LOWEST
trading-relevance and excludes the one with by far the highest.** `profit`
carries 4x the mutual information of the next-best task and 12x that of `flow`.

That is not an artefact of the metric: `profit` is the triple-barrier label, and
`trading` is constructed from buy/sell signals gated on it (among others). It is
the most trading-aligned auxiliary available, and it is switched off.

### Pairwise NMI among the auxiliaries — all near zero

|  | profit | regime | momentum | risk | flow |
|---|---|---|---|---|---|
| profit | 1.000 | 0.009 | 0.018 | 0.047 | 0.037 |
| regime | 0.009 | 1.000 | 0.039 | 0.002 | 0.006 |
| momentum | 0.018 | 0.039 | 1.000 | 0.007 | 0.088 |
| risk | 0.047 | 0.002 | 0.007 | 1.000 | 0.002 |
| flow | 0.037 | 0.006 | 0.088 | 0.002 | 1.000 |

Maximum off-diagonal is 0.088. The auxiliaries are **essentially mutually
independent**, so there is no redundancy to exploit — you cannot drop one on the
grounds that another covers it. Each carries distinct information, or distinct
noise.

### What this does and does not establish

NMI measures ALIGNMENT WITH THE TARGET, not MTL benefit. High alignment can mean
positive transfer or it can mean "no new signal"; low alignment can mean useless
noise or valuable orthogonal regularisation. **This ranks candidates; it does not
decide.** The decision needs the leave-one-out runs below.

## Queued confirmatory runs (after the rebuild)

Only 3 tasks are trained today, so the space is small and leave-one-out is cheap.
Ordered by predicted value:

1. **Re-enable `profit`** (weight 0 -> 1.0). Highest NMI by 4x, independent of
   the others, currently contributing nothing. The single highest-value
   experiment here.
2. **Drop `flow`** (weight 1.0 -> 0). Weakest current auxiliary (NMI 0.011).
3. **Drop `risk`** (weight 1.0 -> 0). The cleanest PURE ablation: `risk` does not
   appear in the label filter, so removing it changes the MTL effect ONLY and
   leaves the target identical. Best isolation of the multi-task effect itself.
4. **Drop `momentum`** (weight 1.0 -> 0), to complete leave-one-out.
5. Existing `NNMT_MLX_tronly` provides the zero-auxiliary end of the range.

Changing a task WEIGHT does not change the target (the label filter uses
ground-truth values regardless), so these are clean comparisons against the same
labels — no confound from a shifting target.

**Still expensive and kept separate:** changing which tasks DEFINE the label (the
`>= n-1` rule in `get_trading_classes`) does alter the target and needs its own
retrain per variant. Do not mix it with the MTL question.
