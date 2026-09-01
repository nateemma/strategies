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
