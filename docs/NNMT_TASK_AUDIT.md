# NNMT multi-task audit — pre-screens before any task-combination search

Question: is there a cheap way to find the best combination of tasks, without
training all 2^n? Answer: the pre-screens found the configuration is
**internally inconsistent**, so a combination search would currently be
measuring noise. Fix the inconsistency first.

`trading` is always present by requirement. All findings below are from code
reading and a static model inspection — **zero training runs**.

## The current configuration

`NNMT_MLX._CLASSIFIER_TASK_WEIGHTS = {trading 4.0, regime 0.0, risk 1.0,
momentum 1.0, flow 1.0, profit 0.0}`

| task | loss weight | trained? | used in the filter? | verdict |
|---|---|---|---|---|
| trading | 4.0 | yes | — (it is what gets filtered) | primary |
| **profit** | **0.0** | **NO** | **YES** | **untrained head gates entries** |
| **regime** | **0.0** | **NO** | **YES** | **untrained head gates entries** |
| momentum | 1.0 | yes | YES | working gate |
| flow | 1.0 | yes | YES | working gate |
| **risk** | 1.0 | **yes** | **NO** | **trained but never consumed** |

### 1. Weight 0 means the head is never trained

`MLXClassifierMultiTask` line 450: `total = total + task_weight_arr[i] *
loss_fns[task](...)`. Weights are renormalised, so a 0.0 entry contributes
nothing to the loss and its head receives **no gradient** — it stays at
initialisation. `regime` and `profit` are both 0.0.

### 2. ...but both are used to gate entries

`_filter_trading_by_tasks` (the LIVE code — statement [1] of that function is a
32-line dead string literal holding an older 5-condition version):

    buy_conditions = [profit != LOSS, regime != BEAR,
                      momentum != NEGATIVE, flow != DECREASE]
    buy_mask  = (trading == BUY) & (buy_conditions.sum() >= 3)
    sell_mask = (trading == SELL)          # sell side is NOT filtered

So the buy gate is 3-of-4, and **two of those four inputs are untrained heads.**

**Why this is worse than it looks.** An untrained head emits an arbitrary but
roughly constant class. If the untrained `regime` head happens to settle on
anything other than BEAR, `regime != BEAR` is permanently True — a free pass, and
the gate degenerates to 2-of-3 on the real signals. If it settles on BEAR, that
condition is permanently False and the gate hardens to 3-of-3. **Which of those
you get is random per training run**, so the effective entry strictness varies
seed to seed for reasons unrelated to the data. That is a plausible contributor
to NNMT's seed instability.

### 3. `risk` is trained but never consumed

Verified by AST: `risk_predictions` is a parameter of
`_filter_trading_by_tasks` and is **never referenced in the live body** — only in
the dead string literal. It costs a head, gradient and capacity, and its only
possible remaining value is as an MTL regulariser on the shared trunk.

### 4. The trading LABEL is built from the other tasks

`get_trading_classes` calls the SAME `_filter_trading_by_tasks`, using
**ground-truth** task labels. So the filter is applied twice:

- **training**: labels = raw buy/sell signals filtered by TRUE task values
- **inference**: predictions = trading head filtered by PREDICTED task values

The auxiliary tasks are therefore not independent regularisers — they are the
components that *define* the target. And there is a train/inference mismatch: the
label was defined with a real profit/regime filter, while inference gates on
untrained profit/regime heads.

## What this means for the original question

A 2^n combination search is premature. Two of the four gates are noise and one
trained task is unused, so any ranking would mostly measure which random
initialisation each untrained head landed on.

**Cheapest fixes, in order — none needs a combination search:**

1. **Resolve the weight-0 / in-filter contradiction.** Either train `profit` and
   `regime` (non-zero weight) or drop them from the filter. This is a one-line
   change either way and is very likely the single highest-value edit here.
2. **Decide `risk`'s purpose** — drop it, or use it in the filter. Currently it
   is paying training cost for at most an MTL side-effect.
3. **Delete the 32-line dead literal** in `_filter_trading_by_tasks` so the live
   rule is unambiguous.

## Only then, the cheap search (still not 2^n)

- **Filter side (inference):** dump per-task predictions + trading predictions +
  forward returns from ONE backtest, then enumerate all subsets and every k-of-n
  threshold offline by boolean masking. Seconds, exhaustive, no retraining.
  Valid because the inference filter is a SECOND gate on top of an
  already-filtered target.
- **MTL side (training):** leave-one-out is n+1 runs, not 2^n. Cheaper still,
  log per-task gradient cosine against the trading head on the shared trunk in a
  single run — negative cosine identifies tasks that actively fight trading.
- Note the label-definition question ("which tasks should DEFINE the target")
  does require retraining, because changing it changes the target.
