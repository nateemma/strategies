# Research: Regime Gate on the Reversion Family (026)

Phase 0. Code and data facts were read at `7e9a13e` on 2026-10-11.

## R1 — Pre-declared instruments (FR-001), one fixed construction each

All four are computed on 15m bars, from the 10 walk-forward pairs plus BTC as context. Lengths are
fixed here and never tuned:

| id | instrument | construction (value at bar t uses bars ≤ t only) |
|---|---|---|
| `breadth` | market breadth | share of pairs with `close > EMA(close, 768)` (768 bars = 8 days); NaN if fewer than 6 pairs have data |
| `volreg` | cross-pair volatility regime | per pair, `std(log-returns, 96)` (1 day); take the cross-pair median `m_t`, then `m_t / median(m, trailing 2880)` (30 days) |
| `persist` | trend persistence | per pair, the variance ratio `VR(16) = var(16-bar returns, 2880) / (16 · var(1-bar returns, 2880))`; cross-pair median. VR > 1 means trending, < 1 mean-reverting |
| `btctrend` | BTC trend | `close_BTC / EMA(close_BTC, 768) − 1` |

**Rationale.**
- `breadth` and `btctrend` are trend-state measures. Their prior use was as a "require uptrend"
  filter; here they are scored on whether they separate reversion's LOSING trades, which is the
  reverse framing (spec background).
- `persist` is the direct statistical test of "reverting vs trending". It is absent from the
  populator (memory `project_regime_switching_entry`).
- `volreg` is the regime axis 022 used, kept for comparability.
- One construction each keeps the multiple-comparison count small: 4 instruments × 6 threshold
  options.

**Alternatives rejected.**
- Per-pair states. These are pair-specific, which constitution X forbids, and "market regime" is a
  cross-pair notion anyway.
- ADX. It is already a registered momentum-flavoured trigger (`adx`), and a per-pair measure.
- Hurst. It is noisy at these window lengths and the variance ratio covers the same question.

## R2 — Two-state rule, fixed on early folds (FR-003)

For each instrument:
1. Collect its value at every early-fold (f00–f07) reversion entry of `st_fam_combo`. The entry's
   signal bar is `open_date − 15 min`.
2. Candidate thresholds are the early-entry **33rd, 50th and 67th percentiles**. Candidate
   directions are "off" when the value is ≥ the threshold, or when it is < the threshold. That
   gives 6 options.
3. Keep only options whose "on" state retains ≥ 50% of early reversion trades (FR-004's third
   condition, applied at choice time).
4. Choose the option that minimises the "off" state's early reversion P&L. In other words, "off"
   should capture the early losers. Ties go to the higher retained share.

The choice reads only f00–f07 trades. The late half is untouched until scoring.

**Why choose on early losers, when the early half is profitable overall?** The gate has to identify
the state where reversion loses. If the early "off" state is not already worse than "on", the
instrument carries no information. FR-004 then requires the "off" state to be negative in BOTH
halves.

## R3 — Feasibility scoring (FR-004, US1)

- **P&L unit:** per-trade pp contribution (`profit_abs × scale_pp_per_abs`), the same unit as the
  024/025 family columns.
- **Scored on:**
  - (i) the reversion trades of `st_fam_combo`, which is primary and the deployment context;
  - (ii) `st_gbb`'s trades, as a cross-check.
- **PASS** requires all three conditions on (i): "off" negative in both halves, "on" positive in the
  early half, and on-share ≥ 50% early. (ii) is reported; disagreement between (i) and (ii) is
  flagged, not used as a veto.
- **Also reported:**
  - the share of BARS in the "off" state per half (calendar-switch detector, spec edge case);
  - warm-up and undefined bars (counted as "on");
  - the number of "off"-state runs per half (one long run = a calendar switch in disguise).

- **Calendar-switch rule (analyze M1, pre-declared).** An instrument whose late-half "off" state is
  a single contiguous run covering > 80% of late bars is a STOP for that instrument, whatever its
  P&L split. It would be "reversion off late" in disguise.

**Overall STOP** if no instrument passes on (i). There are no backtests in US1.

**Caveat carried into the report:** a bucket split ignores substitution. Pairs freed by a refused
reversion entry may be taken by later entries (023 Addendum 7). That is why US2 exists.

## R4 — Live path for a market-wide state (US2 only)

**Facts.**
- `BaseStrategy` has no `informative_pairs`. A market-wide state needs every whitelist pair plus BTC
  at each bar.
- freqtrade supports `informative_pairs()` plus `self.dp.get_pair_dataframe(pair, timeframe)` in
  backtests and live trading. Data for the whitelist pairs is already loaded; BTC/USDT 15m exists
  locally.

**Decision.**
- **`utils/RegimeState.py`:** pure, strategy-agnostic functions `breadth(frames)`,
  `volreg(frames)`, `persist(frames)` and `btctrend(btc)`. Each maps {pair: OHLCV frame} to a
  date-indexed series. One implementation serves both the feasibility script and the strategy.
- **SimpleTrigger:**
  - adds `REVERSION_REGIME_GATE = None | {"instrument", "threshold", "off_when": "ge"|"lt"}`;
  - `informative_pairs()` returns BTC/USDT 15m when the gate uses `btctrend`;
  - the whitelist pairs come from `self.dp.current_whitelist()`;
  - in `populate_entry_trend`, it computes the series once per call (cached by last date) and aligns
    it to the pair's dates;
  - it drops reversion-only signal rows in the "off" state, and tags overlap rows momentum (the same
    pattern as 025 rule B);
  - the sink records `regime_gate_refused` per pair (025 C3 format) and the per-pair count of
    "off" bars.
- **Neutral:** `REVERSION_REGIME_GATE = None` reproduces `st_fam_combo` exactly (FR-006).

**Risk:** in backtests freqtrade calls `populate_*` per pair with the full history, so other pairs'
frames from `dp.get_pair_dataframe` are full-history too. Causality must come from the instrument
functions themselves, which use only rolling and backward windows. This is verified by the
truncation test (R5), not assumed.

**Warm-up (analyze H1, confirmed).**
- *The problem.* A fold backtests its 90-day window plus `startup_candle_count` bars, which is 64
  in ARM_D, and other pairs' frames span the same range. The instruments need up to ~2,900 bars:
  - `volreg` and `persist` use 2,880-bar trailing windows, plus 96/16;
  - `breadth` and `btctrend` use 768-bar EMAs, which need several times their span to settle.

  At 64 bars of startup, `volreg` and `persist` would be NaN ("on") for roughly the first 30 days
  of each 90-day fold, and the EMAs would be start-biased. US2 would then test a different, partly
  inert gate from the one US1 scored.
- *The decision.* Gated arms set `startup_candle_count = 3600` (37.5 days; it covers 2,880 + 96
  with margin, and lets the 768-bar EMAs settle beyond 4.5 spans). That changes data loading and
  the warm-up of every other indicator, so it may change trades by itself.
- *Control.* `st_reg_on3600`: gate None, startup 3600. If it reproduces `st_fam_combo` 15/15, the
  startup change is neutral and `st_fam_combo` stays the always-on comparison. Otherwise
  `st_reg_on3600` replaces it as the always-on comparison and the stage's `--previous`, and the
  difference is reported.
- *US1 parity.* Feasibility values come from full history, which is at least as warm as 3600
  bars. A unit check compares one instrument at 3600-bar truncation against full history at the
  fold starts. It requires agreement within 1e-3 relative for the EMA instruments (the residual
  start-up weight after 3600 bars of a 768-bar EMA is about 9e-5 of the initial deviation), and
  exact agreement for the rolling-window ones.
- *Alternative rejected:* a precomputed instrument file read by date. It is simpler, but has no
  live path and is a second implementation to keep in sync.

## R5 — Causality (FR-002)

Each instrument is checked with the 023 truncation method: compute on full data, then on data
truncated at 16 cut points; values at or before each cut must be identical. This is a unit test in
`utils/tests/test_regime_state.py` on real data slices. A planted non-causal control (a centred
rolling mean) must FAIL.

## R6 — Fingerprint and budget

- `REVERSION_REGIME_GATE` joins the optional family members (non-None only). Existing shas are
  unchanged.
- **Budget:** US1 is minutes with no backtests. US2 is ≤ 4 gated arms + 1 neutral reproduction +
  the startup control (H1), about 35 minutes, only if US1 passes.

## R7 — Degeneracy rule for the regime stage (analyze H2, confirmed)

**Problem.** `evaluation/stage.py` excludes a candidate as degenerate if EITHER family keeps < 25%
of its single-family trades in EITHER half. A gate that works is meant to cut reversion's
late-half trades, possibly below 25%, so the rule would exclude the intended winner silently.

**Decision.**
- For `--stage regime`, degeneracy is checked only on the momentum family (both halves) and on
  reversion's EARLY half. Reversion's late count is unconstrained.
- The early on-share ≥ 50% rule (R2) already protects reversion's early half.
- Other stages keep the 025 rule unchanged. A test covers both.
