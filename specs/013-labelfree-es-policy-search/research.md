# Research: labelfree-es-policy-search (brief 05)

Phase 0. Every decision carries the measurement or code reading behind it.

---

## R1 — The evaluator is a NEW discrete-event position simulator, and its calibration blocks everything

**Decision**: build an exit surrogate that tracks an open position bar by bar and applies the
strategy's exit stack; calibrate it against real freqtrade backtests **before any search runs**.

**Why it cannot be an extension of `net_returns`**. Brief 05 says "using the vectorised
simulator from brief 1", but that simulator has no concept of a position's *history*:

- `sim.replay()` is documented as *"Replay a **FIXED** entry/exit series"* — the trade series is
  an **input**
- `sim.net_returns(w, open_px, cost, volume)` is exposure-path economics: `w·r − c·|Δw|`, with no
  stop, no grace window, no holding limit

Every parameter being searched — `stoploss_grace_hours`, the ATR multiplier, `cexit_max_days`,
`cexit_take_profit`, the guard thresholds — **determines** the trade series. The simulator
consumes what the search is trying to change.

**Why not just run backtests**: measured this session, one fold backtest is ~175 s. The brief's
24,000 evaluations would cost ~1,167 hours.

**Why calibration is a gate, not a validation step**: the surrogate re-implements freqtrade's
exit stack. That is duplication, and duplication drifts — this study has paid for it repeatedly
(a loss term that silently never ran; a purge landing in an already-discarded tail; four inert
parameters). An uncalibrated surrogate does not merely risk a wrong answer; it means the entire
search budget optimises a fiction. Feature 009's simulator was calibrated to the arbiter at
0.00000000 bps; feature 012's environment was proven bit-exact against `net_returns`. This is
the same bar.

---

## R2 — MEASURED: the exit stack the brief targets governs the ENTIRE loss side

Exit reasons across the 378 archived walk-forward trades:

| exit reason | trades | share | share of P&L | mechanism |
|---|---|---|---|---|
| `take_profit` | 195 | **51.6%** | +179.7% | `cexit_take_profit` |
| `trailing_stop_loss` | 98 | **25.9%** | **−120.2%** | **the ATR-adaptive stop + grace stack** |
| `model_exit` | 45 | 11.9% | +21.9% | the model's own signal |
| `unclog_12h` | 27 | 7.1% | +7.6% | portfolio-level rule |
| `roi` | 10 | 2.6% | +14.8% | `minimal_roi` |
| `stop_loss` | 1 | 0.3% | −4.7% | the static −0.05 initial stop |
| `metric_overbought`, `force_exit` | 2 | 0.5% | +1.0% | |

**Reading the label `trailing_stop_loss` correctly is the whole point.** It does **not** mean
freqtrade's native trailing stop, which is off:

- `BaseStrategy.trailing_stop = False` (`Framework/BaseStrategy.py:371`), and its own comment
  records that `trailing_stop_positive`/`_offset` are *unreachable* while it is False
- `interface.py:1654-1658`: a stop exit is labelled `STOP_LOSS`, and **upgraded to
  `TRAILING_STOP_LOSS` whenever `trade.is_stop_loss_trailing`** — i.e. whenever the stop that
  fired is not the initial stop
- the fold strategies are `NNNC_WF_f*_s7 → NNNC_MLX → NNNC → BaseNNStrategy`, and
  `BaseNNStrategy.py:145` sets **`use_atr_adaptive_stoploss = True`**; the incumbent params set
  `stoploss_grace_hours = 6.5`, `stoploss_grace_level = -0.27`

So `custom_stoploss` moves the stop on essentially every trade, and every stop exit downstream of
that move is reported as `trailing_stop_loss`. **Those 98 trades are the ATR + grace stack
firing.** The single `stop_loss` trade is the one case where the static −0.05 was hit before the
custom stack moved anything.

**Corrected consequence**: the brief's headline levers — `atr_stoploss_multiplier`,
`atr_stoploss_floor`, `atr_stoploss_cap`, `stoploss_grace_hours`, `stoploss_grace_level` — are
not inert. They govern **99 of 378 exits (26.2%)** and **the entire loss side of the ledger**
(−124.9% of the −124.9% total loss). The search space is aimed at the right mechanism.

**What this changes about the design**:

1. The surrogate **must** model the ATR stop faithfully — it is not a rarely-taken branch that
   could be approximated. It is one of the two mechanisms that decide the P&L.
2. The two dominant mechanisms **oppose** each other: `take_profit` at +179.7% and the ATR stop
   at −120.2%. A joint search can trade one against the other, which one-at-a-time sweeps
   cannot — this is the brief's hypothesis, and the census supports rather than undercuts it.
3. `cexit_take_profit` must be **in** the search space. It is 51.6% of exits and the larger of
   the two P&L terms, and the brief's ≈8 dimensions as listed do not name it.

**Principle IV still applies to the search space**, but as a probe rather than a prediction: each
dimension is measured for its ability to move the objective *before* the search, and any
dimension that cannot is reported inert. Four parameters in this study have been configured,
logged, believed active and reaching nothing; the probe is cheap and the study has paid for
skipping it before.

## R3 — What the surrogate models, and what it explicitly does not

**Decision**: model the exit stack that actually fires; declare the rest out of scope and
measure the cost of excluding it.

| mechanism | modelled | why |
|---|---|---|
| `cexit_take_profit` | **yes — and ADDED to the search space** | 51.6% of exits, +179.7% of P&L; the brief's list omits it (R2) |
| trailing stop | **yes** | 25.9% of exits, −120.2% of P&L |
| `cexit_max_days` | **yes** | bounds holding time |
| ATR-adaptive stop + floor/cap | **yes — highest fidelity required** | 25.9% of exits, −120.2% of P&L (R2) |
| `stoploss_grace_hours` / `_level` | **yes — highest fidelity required** | gates when the ATR stop becomes reachable; wide early, anchored to entry, does not trail |
| volume-confirmation tightening | **yes** | `1/sqrt(rvol)` when rvol > 1, clamped at −0.02 |
| entry guards | **yes** | they decide which trades exist at all |
| `model_exit` | **no — replayed** | depends on the model's per-bar prediction, which is not a searched parameter |
| `unclog_12h` / `unclog_1d` | **yes — recomputed** | `BaseStrategy.py:1211-1215` shows both are purely per-trade (`num_hours >= 12 & profit > 0.005`; `num_days >= 1 & profit >= 0`) with no portfolio state. **Correction**: an earlier draft of this decision called `unclog_12h` portfolio-level. It is not, and that error would have made minimum holding time unsearchable. |
| partial fills, funding, slippage beyond the cost model | **no** | inherited from brief 1's cost model unchanged |

**The replayed mechanism is the honest risk**: `model_exit` alone is **11.9%** of trades (45 of
378). The surrogate takes it from the archive rather than recomputing it, because it depends on
the model's per-bar prediction, which is not a searched parameter. So a parameter change that
*would* have altered when the model exits is invisible to the surrogate. That limitation is
stated in the report, not discovered in it.

Because `unclog_12h` is recomputed rather than replayed, **minimum holding time is searchable**
and enters the space as dimension 10 (`hold_window_hours`, data-model.md §1). The spec names it
as a lever at spec.md:14; under the earlier mistaken reading it would have been silently dropped.

---

## R4 — Calibration means RANK agreement, measured on candidates scored both ways

**Decision**: sample candidates, score each with the surrogate **and** with a full freqtrade
backtest, and report (a) the rank correlation and (b) every pair the two orderings disagree on.

**Why rank and not level**: the search only ever uses the surrogate to *order* candidates. A
constant offset in level is harmless; a reordering is fatal. Demanding level agreement would
fail on cost-model details that do not affect the search, exactly as feature 012's "bit-for-bit"
gate failed on an MLX-vs-numpy `log()` difference that no change could fix.

**The gate**: if the two orderings disagree on which candidate *wins*, the search is optimising
the wrong thing and no result is admissible (SC-017a).

**Sample size**: bounded by backtest cost at ~175 s each. A sample of 40 candidates is ~2 hours
and gives a usable rank correlation; that is the pre-registered size.

---

## R5 — The random control must match on EVALUATIONS, not on wall-clock

**Decision**: both methods consume the same **count** of surrogate evaluations, recorded per
method and asserted equal.

**Why it matters**: the brief makes this a kill gate — if CMA-ES does not beat random search
out-of-sample, the optimiser is fitting noise (Chen & Navet 2007). That comparison is only
meaningful if the budgets are identical, and "identical budget" is ambiguous between wall-clock
and evaluation count. Evaluations is the right unit: it is what the deflation counts, and it is
what the optimiser is being credited with using well.

**Both methods draw from the same bounded space**, so the control is a genuine null rather than
a differently-shaped search.

---

## R6 — The evaluation history lives in a sidecar; the audit must read both

**Decision** (clarification Q2): the shared `trials.jsonl` receives **one summary row per search
run**; the full history goes to `research/labelfree/es/history.jsonl`. The overfitting statistic
reads the sidecar. **The audit verifies both files.**

**Why**: the shared log holds 1,158 lines for the 505 trials of four completed features and is
committed to git. This brief implies up to 48,000 evaluations — roughly 20 MB, two orders of
magnitude more than the study's entire history, and every future feature's audit would pay for
it.

**The failure this avoids being introduced**: an audit that reads only the summary would certify
a record it never examined. That is worse than the storage problem, so SC-018 makes the
two-file audit explicit rather than leaving it to be remembered.

---

## R7 — PBO over the full history, with the count beside it

**Decision**: CSCV over the complete evaluation matrix — every candidate, every generation, both
methods — with the evaluation count reported alongside, and the deflated figure using the
**total** count rather than the surviving count.

**Why the count must appear**: `stats.pbo_cscv`'s own docstring records that CSCV is biased LOW
below ~50 configurations. Here the opposite regime applies — tens of thousands of evaluations —
so the figure is reliable, but the count is what makes it interpretable either way.

**And this feature can finally satisfy SC-004**: the trial log gained a time axis
(`trials.block_returns`, 16 blocks per trial) after four consecutive features reported DSR/PBO
as unevaluable. This is the first feature able to use it, and the first where the statistic is
the headline rather than a footnote.

---

## R8 — The expected outcome, stated before the work

Two measured numbers and one literature record point the same way:

- an entire rvol sweep spans **4.93 pp** inside a **7.633 pp** noise floor — the search space's
  total range of effect is smaller than the uncertainty on measuring it
- Allen & Karjalainen 1999, NEAT 2025, *Red Queen's Trap* 2025, MadEvolve 2026 — uniformly
  negative, with the gap widening as the search space grows

The census (R2) is the one piece of evidence pointing the *other* way: the search space reaches
the two mechanisms that decide the P&L, and they oppose each other, which is exactly the shape a
joint search can exploit and a one-at-a-time sweep cannot. That does not overturn the noise-floor
arithmetic — it is why the feature is worth running rather than assumed.

**A quantified negative is the expected deliverable**, per clarification Q3: PBO > 0.2 kills the
candidate, the measured PBO is the finding. What would change the conclusion is a search result
that beats the random control **and** sits more than 4.93 pp from the incumbent **and** clears
the floor — three conditions, none of which the prior evidence suggests.
