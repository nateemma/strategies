# T013 — is the profit head trained? (research R2's open question)

**Answer: NO. The profit head carries ZERO loss weight, so R2's label/profit-head
coupling is COSMETIC and T014's pinning is NOT load-bearing.**

Established by reading the code, no training required.

1. `NNMT/NNMT_MLX.py` sets
   `_CLASSIFIER_TASK_WEIGHTS = {'trading': 4.0, 'regime': 0.0, 'risk': 1.0,
   'momentum': 1.0, 'flow': 1.0, 'profit': 0.0}` — `profit` is **0.0**.
2. `NNMTStrategy` (~1488) pushes that dict onto the classifier as
   `clf.task_weights_override`.
3. `Predictors/MLXClassifierMultiTask._task_weights()` renormalises it, and the
   training step multiplies each per-task loss by `active_task_weights[t]`.
   A weight of 0.0 removes the head's loss from the objective entirely.
4. `NNMTStrategy` (~1365) independently excludes zero-weight tasks from the
   active task list: `tuple(t for t in self._FILTER_TASKS if
   float(weights.get(t, 1.0)) > 0.0)`.

**Consequence for T014.** `MIN_BUY_GAIN_THRESHOLD` is read at NNMTStrategy ~438
and ~510, where it sets the profit head's class boundaries at `PROFIT_HORIZON`.
Those labels are still COMPUTED, but nothing trains against them, so changing
the label threshold does not move the model through that path. The label arm is
effectively single-variable without the pin.

The pin is retained anyway: it costs nothing, it keeps the arm single-variable
by construction rather than by an argument about loss weights, and it stays
correct if the profit head is ever re-enabled. Constitution VIII is satisfied
either way.

Corroborates the existing project memory note "NNMT task audit: filter OFF,
2 heads untrained, `profit` disabled".
