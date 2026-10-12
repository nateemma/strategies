# Research: Target-Aware Exits (027)

Every decision below is fixed before any measurement. PROVISIONAL under quality gate 2.

## R1 — What the exit does today (verified in code, 2026-10-12)

| element | where | reference arms' value |
|---|---|---|
| grace stop | `BaseStrategy.custom_stoploss`, first branch | −27% for 6.5 h (shared; `FamilyExits` forbids per-family grace) |
| stop after grace | `_resolve_normal_stop` → `ExitPolicy.atr_stop_fraction` | `-4 × atr_pct_roll`, clamped to [−6%, −4%], tightened by 1/√rvol when rvol > 1 (to no tighter than −2%), entry-anchored |
| post-grace ratchet | freqtrade only tightens a stop | each call recomputes from the LATEST bar's ATR, so the stop in force is the tightest seen since grace expired |
| take-profit | `ExitPolicy.custom_exit_decision` | reversion 2.2% net; momentum none (`cexit_take_profit` 10.0) |
| time exits | same | unclog_12h (≥ 12 h and > 0.5% net), unclog_1d (≥ 24 h and ≥ 0), max_hold 10 days; momentum defers unclog_12h to 24 h (`HoldWindowMixin`) |
| ROI | freqtrade native | `ignore_roi_if_entry_signal` |

**The declared targets.** NNNC's labels use `TrainingConfig.MIN_BUY_GAIN_THRESHOLD = 0.02` and
`HORIZON = 48` (`NNNC/LabelParamsMixin`, `BaseNNStrategy`). Both labels (`labels_gbb`,
`labels_breakout_gbb`) test the forward max CLOSE against close: a gross gain ≥ `min_gain` within
`horizon` bars, with no loss barrier. SimpleTrigger triggers are the causal part of those labels,
with no target attached. So: **gain target 0.02 gross, horizon 48 bars (12 h)** for both families,
DECLARED on the strategy (FR-007), not read from a labeler default (the labelers' own defaults are
0.01 / 36, which is exactly why reading them implicitly would be wrong).

## R2 — Feasibility measurements (US1), definitions

Inputs: the 15-fold trades of `st_bgo_mx` (momentum) and `st_book_rev2` (gated reversion), read
from the backtest archives (open_rate, close_rate, exit_reason, dates), and 15m OHLCV from
`research.clfstudy.data.load_pair`, with `atr_pct_roll` from the same populator the strategy uses.

Per trade, the path is the bars from the entry bar (entry at its open) to the earlier of the
horizon end or the max-hold end (10 days). Price levels are relative to `open_rate`:

- **target touch:** first bar whose HIGH ≥ open × 1.02. If that bar's LOW also breaches the stop
  then in force, the order is unknown: the trade is counted as **ambiguous** and excluded from
  (a)/(b), with the count reported (spec edge case).
- **MAE before target:** the lowest LOW from entry up to the bar before the touch, as a fraction.
- **post-grace stop in force at bar t:** the tightest entry-anchored `atr_stop_fraction` over bars
  from grace expiry to t, using `ExitPolicy` itself (one implementation).

Measurements, per family × half:
- **(a) stopped too early:** of stop-loss exits (`stop_loss` / `trailing_stop_loss`), the share whose
  price touched the target after the exit, within 48 bars of ENTRY, and within the max hold.
- **(b) killed winners:** of entries whose path touches the target within 48 bars ("label-positive
  entries"), the share whose MAE before the touch breaches the post-grace stop in force at the
  breach bar after grace expiry. Plus the MAE distribution (p50/p80/p90/p95) vs the stop level.
- **(c) time to target:** the distribution of bars to the touch for label-positive entries; and, for
  trades open at bar 48 without a touch, their eventual P&L and the P&L of closing at bar 49's
  open.
- **(d) after the target:** for trades that touched the target but did not exit at or above it, the
  realised P&L vs the target, and the give-back.

**Counterfactual replays** (C3, C5) use the same path machinery: re-exit each trade under the
changed rule and compare its pp with the recorded pp. They ignore re-sequencing: an earlier exit
frees the pair for a later entry the archive does not hold (023 Addendum 7). That is why they only
choose; walk-forward arms judge.

## R3 — Candidate couplings: the mechanism, the early-fixed value, and the support rule

Each rule is evaluated per family and must hold in BOTH halves (FR-003). The value is chosen on
f00–f07 only, from the grid shown (FR-004).

| | mechanism (when switched on) | early-fixed value (grid) | SUPPORTED iff, in both halves |
|---|---|---|---|
| **C1 stop depth** | post-grace stop no tighter than `-q`, where `q` = the early p-th percentile of label-positive entries' MAE before target; hard bound −10% | p ∈ {80, 90} — the one maximising early replay pp | (b) killed-winner share ≥ 15% **and** the C1 replay gain at the chosen p is > 0 |
| **C2 volatility units** (reversion only — momentum has no take-profit) | take-profit = `k × atr_pct_roll` at entry, clamped to [1%, 5%] net | `k` = median of `0.022 / atr_pct` over early reversion entries (scale-neutral on average) | target hit-rate within 48 bars in the lowest entry-ATR tercile < 0.5 × that of the highest tercile |
| **C3 horizon rule** | at bar 48 without a target touch: close at the next open, or move the stop to entry (break-even) | action ∈ {close, breakeven} — the one maximising early replay pp | the replay gain of the chosen action is > 0 |
| **C4 grace to horizon** | grace window length = horizon (12 h) instead of 6.5 h, same −27% depth | none (one setting) | share of killed winners (b) whose breach bar falls between grace expiry and bar 48 ≥ 50% **and** ≥ 10% of all label-positive entries |
| **C5 post-target lock** | after a target touch on a CLOSED candle, stop no lower than entry × (1 + λ × 0.02) | λ ∈ {0, 0.5} — the one maximising early replay pp | the replay gain of the chosen λ is > 0 |

Replays for C1 and C4 re-run the path with the changed stop. Liveness (C1–C5) is counted per arm
(FR-010).

Rationale for thresholds: 15% / 10% / 50% / 0.5× are round numbers chosen to require a material
effect, not a significance test. US1 only decides what gets a walk-forward arm; the arm's paired
verdict is the test.

## R4 — Where each coupling lives (one exit definition, FR-005)

`ExitParams` (pure, frozen) gains OPTIONAL fields, all `None` = off:
`target_gain`, `target_horizon_bars`, `stop_min_depth` (C1), `take_profit_atr_mult` (C2, with its
clamp), `horizon_action` ("close" | "breakeven") (C3), `grace_hours_override` (C4),
`lock_after_target` (λ, C5).

- **C1:** `_resolve_normal_stop` returns `min(atr_stop, stop_min_depth)`, so it is never tighter
  than `stop_min_depth`.
- **C2:** `custom_exit_decision` takes an optional `take_profit_override`. `BaseStrategy` computes
  it from the entry candle's `atr_pct_roll`, read from the analysed dataframe at
  `trade.open_date − 1 bar` (the signal bar, causal).
- **C3:** `custom_exit_decision` takes `bars_open` and `target_touched`; it returns
  "horizon_close", or the stop path applies break-even.
- **C4:** `custom_stoploss` reads the grace length from `ExitParams` when set. Grace is shared
  (forbidden per family), so C4 is arm-wide.
- **C5:** in `custom_stoploss`, a stop at entry × (1 + λ·target), returned as a ratio relative to
  the current rate, only when a PREVIOUS closed candle's high touched the target.

**Target touch is read from closed candles only.** freqtrade's `should_exit` updates
`trade.max_rate` with the CURRENT candle's high before calling `custom_stoploss` and `custom_exit`
(`interface.py:1448`). A lock keyed on `max_rate` would place a stop that the same candle's low can
trigger, after a high whose order within the candle is unknown. That is an optimistic artefact. The
touch is therefore computed from the analysed dataframe (rows after entry, up to the last closed
candle). It is one helper, shared by C3 and C5.

`simulate_forward` (021 outcome tables) does NOT model the new fields. It raises if any is set,
rather than silently simulating an exit the strategy does not run (the 021 lesson).

## R5 — Fingerprint and reproduction (FR-006)

`fingerprint._family_members` hashes `dataclasses.asdict(ExitParams)` for family-mode arms
(`st_book_rev2`, combined arms). Adding fields would change every such sha. Fix: new fields are
dropped from the hashed dict while `None` (the `OPTIONAL_ATTRIBUTE_MEMBERS` idiom). For non-family
arms (`st_bgo_mx`, arm D), the coupling values are hashed through new optional class attributes
(`EXIT_TARGETS`, `EXIT_COUPLING`), only when set.

Checks:
- every recorded arm's sha is unchanged (a test over the 023–026 classes);
- with every switch off, the two reference arms reproduce f00 trade-for-trade (`trades_hash`);
- arm D (`NNNC_MLX_Open`) sha is unchanged.

## R6 — Stop invariants (FR-009)

`StopInvariantMixin._stop_bound_inputs` adds `stop_min_depth` to the floor:
`floor = min(atr_floor, static, stop_min_depth)` when set. A C5 lock and a C3 break-even stop are
TIGHTER, so the bound is unaffected. The direction check (first post-grace stop vs the last
in-grace stop) stays as is, and C4 only moves when grace ends. New tests:
- C1 at −10% passes, and −11% under a −10% bound raises;
- a C5 lock on a trade below the lock price returns `STOP_AT_MARKET_RATIO` (fire now), not a
  positive ratio.

## R7 — Run cost and order

The feasibility tool has no backtests (minutes). Walk-forward arms: one per SUPPORTED coupling per
family that it applies to (≤ 9), plus one combination per family if ≥ 2 are adopted, plus 2
switch-off reproductions (f00 only). Arms run one at a time (memory). Each 15-fold SimpleTrigger
arm takes ~6–8 min, so ≤ ~1.5 h total.

## R8 — Judging (FR-011/012)

Each arm is judged on f08–f14 against its own reference arm:
- `halves.split` gives the paired P&L verdict (adoption) and the risk verdict (reported);
- adoption requires IMPROVED;
- per family; a combination is run and judged before it is recommended.

No stage runner is needed: values are fixed by the US1 early replays, so each coupling is one arm.
