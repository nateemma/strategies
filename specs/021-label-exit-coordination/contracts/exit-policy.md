# Contract: utils/ExitPolicy.py — the single exit primitive

**FR-001 through FR-005.** Pure functions over numbers and arrays. MUST NOT import
freqtrade, MUST NOT take a `Trade`, MUST NOT read a dataprovider.

## Surface

```
atr_stop_fraction(atr_pct, rvol, cfg) -> float
    The stop level relative to entry, negative. The arithmetic that is currently
    duplicated three ways:
        stop = -cfg.atr_stoploss_multiplier * atr_pct
        stop = max(min(stop, cfg.atr_stoploss_cap), cfg.atr_stoploss_floor)
        if cfg.use_volume_confirmation_stoploss and rvol > 1:
            stop = min(stop / sqrt(rvol), -0.02)
    Returns cfg.static_stoploss when atr_pct <= 0 or ATR adaptation is disabled.

effective_stop_fraction(age_hours, atr_pct, rvol, cfg) -> float
    atr_stop_fraction, except inside the grace window where it is
    cfg.stoploss_grace_level.

custom_exit_decision(current_profit, age_hours, guard_metric, cfg) -> str | None
    AMENDED 2026-10-08 (T021 follow-up, FR-001's amendment). The single source of
    truth for BaseStrategy.custom_exit's decision, minus the two things that stay in
    BaseStrategy as glue (the inert rsi_* branches, and the strategy_custom_exit
    hook, not overridden anywhere in the NNNC chain). `current_profit`/`age_hours`
    MUST be computed from the bar's OPEN price (see "Precedence" below for why).
    Returns metric_overbought / take_profit / unclog_12h / unclog_1d / max_hold, or
    None to defer to the native stop / ROI.

simulate_forward(high, low, close, atr_pct, rvol, start_index, cfg, *, open_=None,
                  guard_metric=None) -> OutcomeRecord
    Walks forward from start_index applying the full policy bar by bar until a
    closing rule fires. Returns profit (net of 2x cfg.fee_per_side), hold_bars and
    close_reason. `open_`/`guard_metric` are optional (None = "not modelled", same
    convention as `rvol`); omitting them reproduces this function's pre-amendment
    behaviour exactly (stop/take_profit/hold_cap/end_of_data only).
```

## The complete exit surface (FR-001, AMENDED 2026-10-08 after T021)

T021's reconciliation found the first outcome table optimistic by 2.70pp win-rate / 10.7%
mean-profit-per-trade because this contract originally enumerated only stop / take-profit /
hold-cap / static fallback. The live Open operating point's `custom_exit` ALSO fires on:

- `metric_overbought` — `guard_metric > 0.98` while in any profit (rare: 0.3-2.6% of bars,
  but not inert)
- `unclog_12h` — age >= 12h AND profit > 0.5%
- `unclog_1d` — age >= 1d AND profit >= 0%
- `max_hold` — age >= `cexit_max_days` (this is the SAME rule `simulate_forward`'s loop bound
  already enforced; the amendment gives it an explicit decision-function branch for the LIVE
  caller, which has no loop)

and freqtrade itself, OUTSIDE `custom_exit`, also applies the native ROI exit (`minimal_roi`,
5% at the Open operating point via `NNNCStrategy`, not a freqtrade `Parameter`).

Two exits remain inert at this operating point and are documented, not modelled: the three
`rsi_*` branches (no `rsi` column from the MINIMAL populator) and `strategy_custom_exit` (not
overridden in the NNNC chain). `enable_exit_signal` is also false.

## Precedence, verified against freqtrade source (not assumed)

`freqtrade/strategy/interface.py::should_exit` appends the custom-exit outcome to its `exits`
list BEFORE the native stoploss, and the native ROI after that ("# Sequence: Exit-signal /
Stoploss / ROI / Trailing stoploss"). `freqtrade/optimize/backtesting.py::_check_trade_exit`
returns on the FIRST entry in that list that fills. `should_exit` calls `custom_exit` with
`rate = row[OPEN_IDX]`, so every decision INSIDE `custom_exit` — including `take_profit` — is
a function of the bar's OPEN, not its high or low; the native stop triggers on the LOW; the
native ROI is evaluated against the HIGH. (`research/labelfree/es/exits.py`'s independent
fidelity notes, written for a different feature, corroborate all three facts against the same
source lines.)

Per bar, in order:

1. `custom_exit_decision` on the OPEN — `metric_overbought`, then `take_profit`, then
   `unclog_12h`, then `unclog_1d` (ONLY reached if `open_` is supplied to `simulate_forward`;
   see "What changed and what did not" below)
2. the native stop, on the LOW (unchanged)
3. the legacy HIGH-based `take_profit` check — ONLY reached when `open_` is NOT supplied
4. the native ROI exit, on the HIGH (only if `cfg.roi` is set)
5. `hold_cap` — the loop's own bound reaching `max_hold_days` (unchanged)
6. `end_of_data`

## What changed and what did not

**UNCHANGED, still "fixed and non-negotiable" when a caller does not supply `open_`**: if a
bar's low breaches the stop AND its high reaches the take-profit in the same bar, the result
is `stop`. This is `simulate_forward`'s PRE-amendment fallback path, pinned by
`test_exit_policy_simulate_forward.py` (T009) exactly as before.

**CHANGED, deliberately, when `open_` IS supplied** (which `research/outcomes/table.py`
always does): `take_profit` is decided correctly from the OPEN price, before the stop is even
checked, per the precedence above — and the legacy high-based check is skipped, because
running it alongside the correct one would re-admit exactly the optimistic bias T021 found
(a bar whose high alone reaches the take-profit level, with an open that does not, would not
have triggered a live exit at all). This is the one place this amendment changes decision
MECHANICS rather than purely adding exit types. See the feature's handback report for the
residual uncertainty this still carries (the SAME-bar interaction between the new open-based
checks and `max_hold`/ROI has not been independently reconciled against real trades the way
T021 reconciled the original two-exit model).

## Callers after the refactor

| caller | keeps | delegates |
|---|---|---|
| `BaseStrategy.custom_stoploss` | freqtrade's return convention (`1.0` = no change; the `(open_rate * (1+normal) / current_rate) - 1` rescale after grace); sourcing `atr_pct_roll` and volume from the dataprovider | `atr_stop_fraction`, `effective_stop_fraction` |
| `BaseStrategy._resolve_normal_stop` | nothing — becomes a one-line delegation | `atr_stop_fraction` |
| `BaseStrategy.custom_exit` | the inert `rsi_*` branches, the `strategy_custom_exit` hook call, reading `guard_metric`/age off the dataframe/trade | `custom_exit_decision` |
| `NNPredictStrategy._stop_aware_labels` | its target-construction logic | `effective_stop_fraction` |
| `research/outcomes/table.py` | nothing | `simulate_forward` (which itself calls `custom_exit_decision`) |

## Guarantees

- **SC-001**: live stop decisions identical on 100% of bars, proven by a characterisation
  fixture generated BEFORE the refactor (research R4 — the ordering matters; a fixture
  generated afterwards proves nothing).
- **SC-001 (custom_exit)**: live `custom_exit` decisions identical on 100% of a 960-case grid,
  proven the same way (`test_custom_exit_characterisation.py`, generated before T021's
  refactor).
- **SC-002**: the arithmetic exists here and nowhere else, proven by the allowlist test
  (research R3).
- Grace and non-grace branches, and both `after_fill` values, are covered. The grace branch
  returns `1.0` on the non-`after_fill` call, and losing that asymmetry would widen every
  live stop silently.

## AMENDED 2026-10-08 — entry convention, and a consequence for the same-bar tie rule

`simulate_forward` now enters at **`open_[start_index]`**, not `close[start_index]`, and the
replay includes the entry bar (its own high and low come after the fill). The live strategy
enters at the open of the bar stamped `trade.open_date`; the table had been entering at the
same bar's close.

Evidence, from the control arm's 1,410 matched trades: divergence against the entry bar's
move correlated −0.495 with a perfectly monotonic quintile pattern, and regressed with slope
**−1.072** against the −1.0 a pure entry-price mismatch predicts, intercept −0.00001.

**Consequence for the same-bar precedence rule above.** That rule — "the stop wins a
stop/take-profit tie" — only has meaning on the legacy HIGH-based take-profit path. With
`open_` supplied (which the real caller always does), take-profit is decided by the
OPEN-based `custom_exit_decision`, which resolves before the stop is checked, so no
same-bar ambiguity arises. The rule is therefore **inert for the production path** and
survives only for callers that omit `open_`. It is retained rather than deleted because
those callers still exist in tests, but it should not be read as governing the outcome
table.
