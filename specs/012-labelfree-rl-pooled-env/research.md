# Research: labelfree-rl-pooled-env (brief 04)

Phase 0. Decisions, each with the measurement or code reading behind it.

---

## R1 — The env step must be numpy, and proven equivalent to the MLX simulator

**Decision**: implement per-step economics in numpy; assert that stepping a whole path through
the env reproduces `sim.net_returns`'s vectorised MLX output **bit-for-bit**. That equivalence
test is this feature's calibration gate.

**Why it is needed at all**: FR-001 says reuse brief 1's simulator as the step function. But
`sim.net_returns(w, open_px, cost_per_unit, ...)` is **vectorised over a complete exposure
path** and built on MLX. An RL environment needs a function called once per bar, of order 10^8
times, inside SB3's CPU rollout loop. Calling MLX per step would pay Metal dispatch overhead on
a scalar, and SB3's vectorised envs run in worker processes where a Metal context is the wrong
tool entirely.

**Why equivalence rather than reuse**: the economics are cleanly separable per bar —

```
requested[i] = |w[i] - w[i-1]|                      (w[-1] = 0)
executed[i]  = min(requested[i], 0.10 * quote_volume[i])     hard cap
reward[i]    = w[i] * r[i] - cost[i] * executed[i]
```

so a numpy step is a faithful restatement, not an approximation. But *"faithful"* is exactly
the claim this study has been burned by: a duplicated loss term that silently never ran, a
purge that landed in an already-discarded tail, four parameters that reached nothing. The
equivalence test is what converts the claim into a measurement.

**Alternatives considered**: calling `net_returns` on a 2-element window each step (rejected:
MLX overhead per scalar, and the `w[-1] = 0` boundary would be re-applied every step, silently
zeroing the carried position); refactoring `sim.py` to expose a step function (rejected: brief
1's simulator is calibrated to the arbiter at 0.00000000 bps and is a shared dependency of
briefs 2, 5 and 6 — changing its internals to serve brief 4 risks all of them).

---

## R1a — AMENDMENT (measured 2026-09-12): "bit-for-bit against MLX" was over-specified

**Decision**: the equivalence gate has **two parts** — a LOGIC part that must be exact, and a
PRECISION part that is bounded rather than zero.

**What R1 got wrong**: it required the numpy step to reproduce `sim.net_returns` bit-for-bit.
Measured, that is unachievable and tests the wrong thing. **MLX's `log()` and numpy's `log()`
disagree by ~5 ulps at float32**, and the discrepancy is already present in `bar_returns`
before any of the env's arithmetic runs:

```
bar_return, numpy f64 vs MLX f32 : max|d| 6.36e-07
bar_return, numpy f32 vs MLX f32 : max|d| 6.47e-07      float32 eps = 1.19e-07
```

Neither implementation is wrong. A gate that fails on this fails on something no change to
this feature can fix, and — worse — it would have been "resolved" by loosening the tolerance,
which would also have absorbed a real boundary error.

**The fix isolates the two**. Feeding the simulator's **own** bar returns into the env's step
leaves only this feature's economics under test:

| case | max\|diff\| |
|---|---|
| random | **0.00e+00** |
| opens_long (the `w_{-1}=0` boundary) | **0.00e+00** |
| ends_in_position | **0.00e+00** |
| cap_binds | **0.00e+00** |
| zero_volume | **0.00e+00** |

So the economics — boundary, cap, accumulation, lag — are **identical**, and the residual is
purely the library's `log`. Part 2 reports that residual against a ceiling of 5e-06 (a few
float32 ulps); a value *above* the ceiling is a real defect, a value at it is two correct
implementations disagreeing in the last bits.

**Why this matters beyond the gate**: the original formulation would have produced a failing
gate on correct code, and the tempting fix — widen the tolerance until it passes — destroys
the gate's ability to catch the boundary errors it exists for. Splitting logic from precision
keeps rtol=0 exactly where it is meaningful.

---

## R2 — Execution lag: decide at *i*, fill at *i+1*, measure to *i+2*

**Decision**: the env's reward at step *i* uses `log(open[i+2]) − log(open[i+1])`, matching
`sim.bar_returns`:

```python
return mx.log(o[..., 2:]) - mx.log(o[..., 1:-1])
```

**THE INVARIANT, stated once and used everywhere** — every artefact uses the index *i* for the
decision bar, and no artefact introduces a second symbol:

> The action taken on the observation **ending at bar *i*** earns the return between
> **`open[i+1]` and `open[i+2]`**.

**Consequence for the env**: an episode of *T* decisions needs *T+2* bars of price. The last
two bars of any window are consumable as outcome but not as decision points, and an off-by-one
here would hand the agent a return it could not have earned. The observation window ends at bar
*i*; the reward is the realised consequence of acting on it, which is not lookahead.

**Why it is called out**: this exact axis confusion produced five wrong-axis slice bugs in
feature 009, each of which "produced plausible shapes and wrong numbers with no error".

---

## R3 — A position-aware policy reaches freqtrade through per-candidate hooks

**Decision** (clarification Q2): keep position, bars-in-trade and unrealised P&L in the
observation. For the §5 evaluation, the policy is consulted in `confirm_trade_entry` (which
receives the candidate) and `custom_exit` (which receives the open `Trade`), not in
`populate_entry_trend`.

**Why the obvious path fails**: `populate_entry_trend` is vectorised over a whole dataframe and
has no notion of the position held at each bar. A policy whose action depends on current
position cannot be expressed as a position-blind boolean column. Brief 04 criticises FreqAI's
`add_state_info` for precisely this and then specifies the same state for its own agent; the
hooks are the resolution.

**What this costs**: `confirm_trade_entry` can only *veto* a candidate the vectorised layer
already proposed, so the entry universe is bounded by whatever gate populates it. The RL policy
therefore cannot open a position the base strategy never considered. This is a real restriction
on the §5 evaluation and must be stated in the report rather than discovered in it.

**Why the head-to-head is unaffected**: SC-009 runs in the simulator, where both the RL policy
and brief 1's differentiable policy see an identical environment and neither is routed through
freqtrade.

---

## R4 — The continuous band does NOT inherit brief 02's bimodality

**Decision**: implement the no-trade band as a **fixed** transform of the policy's target
exposure. Do not reuse `policy.ClassifierEncoderPolicy`'s band head.

**Rationale, from reading the code**: brief 02's `band_hold` bimodality — at a fixed band width,
seeds land either in a churn attractor (~600 turnover) or a frozen one (~0.1) — arises because
the band is **learned**: `self.band = nn.Linear(cfg.hidden, 2)`, scaled by `band_width`. The
optimiser finds one of two fixed points of the band/exposure interaction. A band that is a fixed
hyperparameter has no such interaction and no attractor to fall into.

**This is a prediction, not an established fact**, and SC-015's liveness probe over the band
width is what will test it: if band width proves inert, or bimodal across seeds, the continuous
arm inherits the defect after all and must be reported as such rather than tuned.

---

## R5 — The reward-hacking probe is a positive control, and it must be able to fail

**Decision**: implement FreqAI's shipped reward shape as a selectable env reward — +25 per
entry, −1 per idle candle, `pnl × 100` on exit — train briefly against it, and assert turnover
rises sharply relative to the specified reward.

**Why it is a gate and not an experiment** (SC-011): this session has twice produced a control
that passed *vacuously*. Feature 010's liveness negative control did not actually sever the
parameter it claimed to sever. Feature 011's first planted-leak formulation had to be checked
for separation from honest features before its detection threshold meant anything. An
environment whose reward cannot be *shown* to be hackable has not been shown to be unhackable.

**Expected direction**: the FreqAI reward pays for entries and punishes idling, so the agent
should churn. If it does not, either the env is not conveying the reward or the training budget
is too small to express a preference — both of which invalidate everything downstream.

---

## R6 — SB3 on CPU alongside MLX

**Decision**: torch/SB3 for the policy on CPU; MLX confined to the equivalence test and to
brief 1's policy in the head-to-head. Vectorised envs via SB3's `SubprocVecEnv`, worker count
set from the 10 performance cores, not the 14 physical.

**Measured environment**: gymnasium 1.3.0, stable-baselines3 2.9.0, sb3-contrib 2.9.0 (for
`MaskablePPO`), torch 2.13.0. All installed; no new dependency.

**Constraint inherited from AGENT_GUIDE**: runs are sequential. Vectorised envs parallelise
*within* one run; two runs never overlap. Measured precedent: two concurrent jobs put load 60 on
14 cores and finished slower than one.

**Gymnasium 1.3 API**: `step` returns `(obs, reward, terminated, truncated, info)`. Conflating
`terminated` (drawdown stop) with `truncated` (window ran out) corrupts PPO's bootstrap value
at the episode boundary — a truncated episode's value must be bootstrapped, a terminated one's
must not.

---

## R7 — The screening fold contaminates the pooled verdict, and is bounded rather than avoided

**Decision** (SC-019): screen on f07 as the brief directs; pre-register that the screen selects
**only** between the two action parameterisations; report f07's contribution to the pooled
verdict separately.

**Why not just exclude f07 from the pooled set**: it would cost a fold from an already thin
15, and the selection pressure from a single binary choice is small. But "small" must be stated
and bounded before the run, not argued afterwards — feature 009 voided 60 runs by ranking on
the wrong range, and the lesson recorded from it is that a selection rule discovered after the
fact is not a selection rule.

---

## R8 — "Learn not to trade" is this environment's degenerate optimum

**Decision**: the participation floor is applied in **reporting**, not as a reward penalty. A
policy whose mean exposure falls below the floor is INCONCLUSIVE and its return is withheld
(FR-017, SC-013).

**Why not penalise inaction in the reward**: an idle penalty is exactly what makes FreqAI's
shipped reward hackable, and FR-003 forbids it. Zero exposure earning zero reward is the honest
economics; the problem is not that the agent finds it, it is that a zero return would *rank
well* against arms that lose money. That is a reporting defect, and it is fixed where it occurs.

**Precedent**: brief 1's six best-ranked configurations were all below the participation floor,
two of them flat on 100% and 99.9% of bars. The same failure is available here and is more
likely, because a policy-gradient method will find it faster than gradient descent on a smooth
objective.
