# Data Model: labelfree-rl-pooled-env

Phase 1. Entities described by meaning, not storage.

---

## Episode

A training window. **Drawn only from a fold's training range** (FR-009).

| field | meaning | validation |
|---|---|---|
| `pair` | instrument | in the fold's universe |
| `start` | first decision bar | window end + 2 bars ≤ fold `train_end` |
| `length` | decisions in the episode | 30–90 days of bars |
| `fold_id` | owning fold | `f00`–`f14` |

**Rules**

- The window plus **two trailing bars** must lie inside the training range — the reward at the
  last decision needs `open[i+1]` and `open[i+2]` (R2). An episode that borrows those from the
  test range leaks by exactly two bars, and nothing would report it.
- An episode containing a data gap is **rejected and counted**, never interpolated.
- Episodes are sampled i.i.d. across pairs so one shared policy sees every instrument.

---

## Observation

What the agent sees at a decision bar.

| component | meaning |
|---|---|
| feature window | the normalised entry-time features, ending at **bar *i***, the decision bar |
| `position` | current exposure in [0, 1] |
| `bars_in_trade` | how long the position has been held |
| `unrealised_pnl` | mark-to-market on the open position |

**Rules**

- Nothing after bar *i* may enter the observation. The reward uses `open[i+1]`/`open[i+2]`;
  the observation does not. This is the R2 invariant and it is the only indexing convention
  used in this feature.
- The last three components are the state the brief says FreqAI cannot backtest. They are why
  the §5 evaluation must route through per-candidate hooks (R3).

---

## Action

| parameterisation | shape | masking |
|---|---|---|
| discrete | {flat, long} | "sell" invalid when flat; masked, **not penalised** (FR-004) |
| continuous | target exposure in [0, 1] | passed through a **fixed** no-trade band (R4) |

Both are screened; one is chosen on the screening fold and locked (FR-010, SC-019).

---

## StepResult

One bar of environment economics. **Must reproduce `sim.net_returns` bit-for-bit** (R1).

Indexed by **`i`, the decision bar**, matching research R2. No second index symbol is used
anywhere in this feature — feature 009 lost five bugs to axis ambiguity, each producing
plausible shapes and wrong numbers.

| field | definition |
|---|---|
| `requested` | `abs(w_i − w_{i−1})`, with `w_{−1} = 0` |
| `executed` | `min(requested, 0.10 × quote_volume_i)` — the hard liquidity cap |
| `bar_return` | `log(open_{i+2}) − log(open_{i+1})` |
| `reward` | `w_i × bar_return − cost_i × executed` |
| `terminated` | drawdown exceeded the configured maximum |
| `truncated` | the episode window ran out |

**Rules**

- `terminated` and `truncated` are **distinct**. Conflating them corrupts PPO's bootstrap at the
  boundary: a truncated episode's value must be bootstrapped, a terminated one's must not.
- Costs are charged on **executed**, not requested — the cap is not a free brake.
- No entry bonus, no idle penalty, no take-profit multiplier (FR-003).

---

## RewardSpec

Selects which economics the env pays.

| variant | purpose |
|---|---|
| `realised` | the specified reward; the only one used for any result |
| `freqai_shaped` | +25/entry, −1/idle bar, `pnl × 100` on exit — the **positive control** (R5) |

**Rules**

- `freqai_shaped` exists to be *hacked*. If an agent trained on it does not churn measurably
  more than one trained on `realised`, the environment has not been shown to convey reward
  shape at all, and no downstream number is admissible (SC-011).
- It may never produce a reported result.

---

## PolicyRun

One trained policy.

| field | meaning | validation |
|---|---|---|
| `action_space` | discrete-masked or continuous-band | pre-registered |
| `seed`, `fold_id`, `total_steps` | configuration | in the trial log **before** the run |
| `selection_quantity` | what chose the checkpoint | **must be a training-range quantity** (FR-012) |
| `mean_exposure`, `trades_per_year`, `mean_hold_hours` | activity | mandatory beside every return (FR-016) |
| `degenerate` | mean exposure below the participation floor | if true, **return withheld** (FR-017) |

**Rules**

- A degenerate policy is INCONCLUSIVE, never a return. Zero exposure earns zero, which would
  otherwise rank above arms that lose money — brief 1's six best-ranked configurations were all
  below the floor (R8).
- Seeds producing identical trades are one measurement (FR-015).

---

## HeadToHead

SC-009: the brief's actual question.

| field | meaning |
|---|---|
| `rl_returns`, `brief1_returns` | per-seed, same folds, same costs, same caps |
| `delta_iqm`, `ci` | interquartile mean of the paired difference, with interval |
| `verdict` | RL better by the floor · matched · worse |

**Rules**

- Both policies run **in the simulator**, not through freqtrade — that is what keeps them
  comparable (R3).
- "Matched" closes the family and records that the differentiable estimator is preferred for
  lower variance and backtestable state (SC-010).

---

## SelectionRecord

| field | meaning | validation |
|---|---|---|
| `decision` | checkpoint, variant or hyperparameter chosen | |
| `criterion` | the quantity used | |
| `range` | where it was computed | **must not be the fold under evaluation** (SC-012) |

Recorded so the audit can *verify* the claim rather than accept it — selecting on the test split
is the specific defect the brief cites in FreqAI's `best_model.zip`.

---

## Relationships

```
Fold ──> Episode ──> StepResult ──> PolicyRun ──┬──> HeadToHead   (SC-009, in the simulator)
           │            │                        └──> Pooled §5    (SC-001, via freqtrade hooks)
           │            │
           │        RewardSpec ──(freqai_shaped gates everything)──> SC-011
           │
      training range only (FR-009)      SelectionRecord ──(gates)──> SC-012
```

`RewardSpec`'s control and the equivalence test both **gate** every downstream figure: until the
env provably reproduces the simulator and provably conveys reward shape, no PolicyRun is
evidence.
