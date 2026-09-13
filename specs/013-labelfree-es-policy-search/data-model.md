# Data Model: labelfree-es-policy-search

Phase 1. Entities, their fields, and the invariants that make the gates checkable.

---

## 1. `Space` — the search space

`research/labelfree/es/space.py`. Nine dimensions, each with bounds and the **incumbent** value
read from `NNNC/NNNC_MLX.json` and `Framework/BaseStrategy.py`.

| # | parameter | incumbent | bounds | why it is in the space |
|---|---|---|---|---|
| 1 | `atr_stoploss_multiplier` | 2.5 | [1.0, 5.0] | scales the ATR stop — 25.9% of exits |
| 2 | `atr_stoploss_floor` | −0.04 | [−0.10, −0.02] | loosest stop allowed |
| 3 | `atr_stoploss_cap` | −0.02 | [−0.04, −0.005] | tightest stop allowed |
| 4 | `stoploss_grace_hours` | 6.5 | [0.0, 12.0] | how long the stop is held wide |
| 5 | `stoploss_grace_level` | −0.27 | [−0.40, −0.05] | how wide it is held |
| 6 | `cexit_take_profit` | 0.024 | [0.010, 0.060] | **51.6% of exits, +179.7% of P&L — added per R2** |
| 7 | `cexit_max_days` | 10 | [2, 28] | the holding-time bound |
| 8 | `entry_rvol_threshold` | 2.0 | [0.25, 4.0] | the dominant entry guard |
| 9 | `entry_guard_threshold` | 0.3 | [0.0, 1.0] | the second entry guard |

**Invariants** (enforced at decode, not merely documented):

- `atr_stoploss_floor < atr_stoploss_cap < 0` — a candidate violating this is **inadmissible**,
  not clipped, because clipping silently maps distinct candidates to one point and corrupts the
  trial count.
- `cexit_max_days` is an integer; CMA-ES searches it continuously and rounds at decode. Two
  candidates rounding to the same integer are **one evaluation for scoring, two for the trial
  count** (spec edge case), because the deflation counts attempts.
- A candidate whose decoded value equals a bound is flagged `on_bound`, which FR-019 requires be
  reported rather than silently accepted.

**Encoding**: CMA-ES operates in an unbounded z-space; `encode`/`decode` map through the bounds
so the optimiser never sees a constrained problem. `decode(encode(x)) == x` is a unit test.

---

## 2. `Candidate`

```
candidate_id : str    # deterministic hash of the decoded parameter vector
params       : dict   # the 9 decoded values
source       : str    # "cmaes" | "random" | "incumbent"
generation   : int    # 0 for random search and the incumbent
on_bound     : list   # parameter names sitting on a bound
```

`candidate_id` is a hash of the **decoded** values, so identical candidates from the two methods
collide by construction and the duplicate rule above is detectable.

---

## 3. `Evaluation` — the unit of trial accounting

```
eval_id       : str        # monotonic, assigned at WRITE time
candidate_id  : str
fold_id       : str        # "f00".."f14"
range         : str        # "train" | "test"   <- the leakage axis
method        : str        # "cmaes" | "random"
written_at    : timestamp  # BEFORE the score exists
score         : float|null # null until scored
admissible    : bool|null
reason        : str|null   # why inadmissible
n_trades      : int|null
trades_per_yr : float|null
mean_hold_h   : float|null
exposure      : float|null
block_returns : list|null  # 16 blocks, the time axis DSR/PBO needs
```

**The write-before-score invariant (FR-013)** is what makes the record auditable: an evaluation
is appended with `score: null`, then updated. A crash therefore leaves an evidence of an attempt,
which is exactly what the deflation must count. An evaluation that exists only once it succeeded
would undercount attempts and inflate every downstream statistic.

**The leakage invariant (FR-010/011)**: for each `fold_id`, `count(range == "test") == 1`, and
its `written_at` is later than every `range == "train"` evaluation for that fold. Both are
checkable from the file alone, by an auditor who does not trust the code.

---

## 4. `History` — the sidecar

`research/labelfree/es/history.jsonl`, one `Evaluation` per line.

Every evaluation from both methods, including abandoned and discarded candidates (FR-014). This
is the input to PBO/CSCV and to the deflation's total count (FR-015).

The shared `research/labelfree/trials.jsonl` receives **one summary row per search run** with the
sidecar's path, its line count and its sha256, so the shared log points at the full record rather
than duplicating it. **The trial-log audit reads both files** (SC-018) — an audit that read only
the summary would certify a record it never examined.

---

## 5. `CalibrationSample` — the blocking gate's evidence

```
candidate_id    : str
surrogate_score : float
backtest_score  : float   # a real freqtrade backtest, ~175 s each
fold_id         : str
```

40 candidates (≈2 hours of backtests), spanning the space rather than clustered near the
incumbent. Produces:

- `spearman_rho` — the rank agreement, reported as a number (FR-003)
- `win_disagreements` — pairs the two orderings rank oppositely
- **the gate**: if surrogate and backtest disagree on which candidate *wins*, the search is
  optimising the wrong thing and **no result is admissible** (SC-017a)

Level agreement is deliberately not required. The search only ever uses the surrogate to order
candidates; a constant offset is harmless, a reordering is fatal. Demanding level agreement
failed in feature 012 on an MLX-vs-numpy `log()` difference that no change could fix.

---

## 6. `LivenessReport` — Principle IV applied to the space

For each of the 9 dimensions: sweep it alone across its bounds with the other 8 at the
incumbent, and record the **span of the objective**.

```
parameter : str
span_pp   : float   # max - min of the objective over the dimension
verdict   : str     # "live" | "inert"
```

`inert` means the span is below a threshold the report states — the dimension cannot move the
objective and is reported as such rather than searched. This is cheap (9 × ~30 evaluations) and
runs before the search, because this study has four parameters that were configured, logged,
believed active and reaching nothing.

---

## 7. `Verdict`

```
pbo              : float   # the headline number
n_evaluations    : int     # reported BESIDE the PBO (FR-016)
dsr              : float   # deflated by n_evaluations, not by survivors (FR-015)
best_candidate   : Candidate
delta_vs_incumbent_pp : float
within_sweep_noise    : bool   # |delta| < 4.93 pp -> "not resolved" (FR-018)
beats_random_control  : bool   # else "the optimiser fit noise" (FR-008)
admissible            : bool   # from admissibility.compare
status           : str     # "ACCEPT" | "KILL" | "NOT RESOLVED"
```

`status` is derived, never typed by hand:

- `pbo > 0.2` → **KILL** (clarification Q3: kills the candidate; the measured PBO is the finding)
- `within_sweep_noise` → **NOT RESOLVED**
- `not beats_random_control` → **KILL**, recorded as the optimiser fitting noise
- `not admissible` → **KILL** (no trades, or losing to doing nothing)
- otherwise gates SC-001…SC-006 decide

The report is written in every case (FR-020).
