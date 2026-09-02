# Phase 0 Research: Market-Phase Framework and Phase-Adaptive Entry Gate

**Feature**: 001-market-phase-entry-gate | **Date**: 2026-09-02

Every unknown in Technical Context is resolved below. Findings marked **VERIFIED** were checked against the repository or the data in this session; findings marked **DECISION** are choices made here with rationale.

---

## R1. Estimators for the direct trend-versus-reversion measures

**Decision**

| Measure | Estimator | Parameters |
|---|---|---|
| Variance ratio | Lo–MacKinlay `VR(q) = Var(r_q) / (q · Var(r_1))` on log returns | `q ∈ {2, 4, 8, 24}` bars (30m, 1h, 2h, 6h at 15m); rolling window 480 bars (5 days) |
| Hurst | Log-regression of `VR(q)` against `q` across the same `q` grid: `H = 1 + slope/2` | Reuses the VR computation; same 480-bar window |
| Return autocorrelation | Rolling Pearson correlation of `r_t` against `r_{t-1}` | 480-bar window |

**Rationale**: `VR > 1` means trending, `VR < 1` means reverting — this is the property the switch actually needs, measured directly rather than proxied by ADX. Deriving Hurst from the VR curve rather than R/S rescaling avoids a second estimator entirely: R/S is materially biased on windows this short, and the VR grid is already computed. The 480-bar window is chosen so that window ≫ largest `q` (480/24 = 20 non-overlapping blocks at the longest horizon), which is the minimum for a stable variance estimate.

**Alternatives considered**: DFA for Hurst — more robust on non-stationary series but requires a new implementation and is materially slower over 78 pairs × 198K bars, for a measure that only needs to rank-order states. R/S rescaling — rejected as biased at these window lengths. Wavelet-based Hurst — the repo already has wavelet machinery, but it estimates a different quantity and would not be comparable to the VR-derived value.

**Open risk**: a 480-bar (5-day) window is itself a lookback choice. Principle VII forbids making it adaptive without a trackability test, so it stays constant. It is recorded as a fixed parameter, not a tuned one.

---

## R2. Where the new indicator code lives

**Decision**: `utils/custom_indicators.py`.

**VERIFIED**: that module is a flat collection of 12 dataframe-to-series functions (`zema`, `RMI`, `mastreak`, `pcc`, `SSLChannels`, `WaveTrend`, `T3`, `SROC`, …), 207 lines, no strategy coupling. It is the established home for exactly this kind of function.

**Rationale**: satisfies Constitution "Framework must stay strategy-agnostic" without needing to touch `Framework/` at all. `utils/DataframePopulator.py` is the consumer layer and is only modified if mechanism M4 is reached, since M4 is the only path that needs these as *model input features* rather than as *study inputs*.

**Alternatives considered**: adding directly to `DataframePopulator` — rejected, it would compute the measures for every strategy run whether or not they are used, and the study (Phases A–C) does not run inside a strategy at all.

---

## R3. The three evaluation windows

**VERIFIED data coverage**: `user_data/data/binanceus/` holds 78 pairs at 15m. `BTC_USDT-15m.feather` spans 2021-01-01 to 2026-08-31 (198,486 bars). Pairs with history before a cutoff: 24 before 2022-01, 44 before 2023-01, 54 before 2024-01, 58 before 2025-01. Median pair start is 2022-07.

**Decision**:

| Window | Range | Role | Pairs available |
|---|---|---|---|
| **W1** | 2025-09-01 → 2026-08-31 | **Deployment regime — carries the headline verdict** (FR-027a) | 58+ |
| **W2** | 2024-06-01 → 2025-05-31 | Near-era confirmation | 54+ |
| **W3** | 2023-01-01 → 2023-12-31 | **Temporally distant era** (Principle II) | 44 |

**Rationale**: W1 is the regime the strategy would actually deploy into and is therefore the headline per Principle I. W3 sits in a structurally different era (post-FTX bear and recovery) separated from W1 by roughly two years, satisfying the non-negotiable distant-era requirement. W2 fills the gap so the three windows can show whether a result degrades monotonically or flips.

W1 and W2 are separated by only three months and MUST NOT be treated as independent evidence of persistence — W3 is the window that can establish or destroy a persistence claim. This is recorded because the recorded day-of-week failure came from exactly this mistake.

All three are pinned as literal date ranges. See R8.

---

## R4. Minimum-coverage rule for the market-breadth level

**Decision**: at each bar, breadth measures require at least **20 pairs** each having at least **200 bars** of prior history. Below that threshold the market-level label is `UNKNOWN`, and bars labelled `UNKNOWN` are excluded from every per-state statistic rather than folded into a neighbouring state.

**Rationale**: the universe ramps from 24 pairs in early 2021 to 78 by 2026. Cross-sectional dispersion and breadth computed over a handful of pairs are dominated by whichever few coins happened to be listed, which would silently make early history look like a distinct "phase" that is really a listing artifact. The 200-bar history requirement prevents a newly listed pair from entering the breadth calculation with an undefined trend state.

**Consequence**: the rule is satisfied throughout all three evaluation windows (W3 begins 2023-01 with 44 qualifying pairs), so `UNKNOWN` should appear only in pre-2022 history. If `UNKNOWN` appears inside any evaluation window, that is a defect to investigate, not a state to interpret.

---

## R5. The aggressive and conservative configurations (FR-018)

**VERIFIED**: `prediction_threshold` is declared in `Framework/BaseStrategy.py` as `DecimalParameter(0.2, 0.9, default=0.5, space="buy")` and consumed in `Framework/BaseNNStrategy.py` at the top of `get_predictions`. `NNNC/NNNC_MLX.json` sets it to **0.69**, alongside `entry_adx_threshold` 38.0, `entry_guard_threshold` −0.4, `entry_rvol_threshold` 3.1, `entry_close_norm_threshold` −0.1.

**Decision**: the two configurations differ **only** in `prediction_threshold`.

| Configuration | `prediction_threshold` |
|---|---|
| Aggressive | 0.55 |
| Constant baseline | 0.69 (the tuned operating point) |
| Conservative | 0.80 |

**Rationale**: this is the dominant entry filter, so it is the lever with enough dynamic range for a switch to matter. The bracket straddles the tuned operating point rather than sitting to one side of it, so neither arm is measured at an unpowered extreme. All remaining parameters — exits included — are held identical, so any measured difference is attributable to entry aggressiveness alone.

**CRITICAL GOTCHA — VERIFIED**: `NNNC/NNNC_MLX.json` is **untracked** in git (it is the user's working state) and, per recorded prior art, a `<Strategy>.json` **overrides** the `buy_params` declared in the strategy class. Setting `prediction_threshold` in Python will therefore silently do nothing while this file exists. Every arm MUST vary the value through the JSON (or a per-arm copy of it) and MUST confirm the value took effect via the R7 liveness measurement. This is precisely the inert-parameter shape that Principle IV exists to catch.

---

## R6. Trade duration for the dwell-time kill criterion

**Decision**: measure `NNNC_MLX`'s own median and interquartile trade duration on W1 as the first task of Phase B, and derive the dwell threshold from that measured number. Do not use a borrowed figure.

**Correction recorded**: the 1.4–31 hour range quoted during spec drafting came from the live `MomentumRegimeBasket15mFast` dry run, which is a **different strategy family** with a different holding profile. The NN family's own recorded figure is different again — measured trailing-stop exits averaging 2:44 against a 2.6-hour grace on NNMT — and `NNNC_MLX.json` sets `stoploss_grace_hours` to 5.6, different from both. Carrying any of these across would set the kill criterion from the wrong distribution.

**Threshold rule**: an instrument is ELIMINATED when its median dwell time is shorter than the **25th percentile** of host trade duration. Rationale: if a quarter or more of trades would span a phase change, the phase in force at entry does not describe the conditions the trade actually experiences, so the gate cannot act on it.

---

## R7. Extracting realized trades, and proving a parameter is live

**VERIFIED**: `scripts/test_strat.sh` builds `freqtrade backtesting --cache none --breakdown month --timerange=... -c ... --strategy-path ... --strategy-list ...`. It does **not** pass `--export`. Backtest result archives are nonetheless present under `user_data/backtest_results/` as dated `.zip` plus `.meta.json` pairs.

**Decision**: the study supplies its own thin runner that adds `--export trades --export-filename` with a per-arm path, rather than modifying `test_strat.sh`. Modifying the shared script would touch code outside this change (Constitution: surgical changes) and would alter behaviour for every other experiment in the repo.

**Liveness proof (Principle IV, FR-031)**: every arm reports three counts from the same run — bars where the model condition passed, bars where the full entry chain passed, and resulting trade count. An arm whose full-chain pass count is identical to baseline is reported as **INERT** and its trading numbers are discarded, not recorded as a null result.

---

## R8. Pinning the time range

**VERIFIED**: `test_strat.sh` accepts `-t | --timeframe` taking an explicit `YYYYMMDD-YYYYMMDD` range (line 93), while `-n | --ndays` and `-o | --offset` compute the range **relative to today's date** (lines 90–91) via `set_start_date`.

**Decision**: every A/B arm uses `-t` with a literal date range from R3. `-n`/`-o` MUST NOT be used for any comparison arm.

**Rationale**: `-n 720 -o 0` run on two different days silently produces two different windows, making arms non-comparable. This is the mechanical cause behind the recorded "pin the timerange for multi-day A/B" rule.

---

## R9. Lookahead validation method

**Decision**: for each instrument, compute the label series over full history, then recompute over history truncated at a cut point `T`, and assert the two label series are **identical** over `[start, T]`. Repeat at three cut points spread across the data.

**Rationale**: this is the end-to-end measurement Principle V requires, and it catches the failure modes that per-line reasoning misses — centred rolling windows, `ffill` reaching backwards, and cross-sectional operations that implicitly use the full panel. The cross-sectional breadth measures are the highest-risk instruments here, because a naive dispersion calculation over a DataFrame with future rows present will leak.

**Explicitly not sufficient**: freqtrade's own `lookahead-analysis`. Recorded prior art established that an end-to-end recipe beats it, and it does not inspect study code that runs outside a strategy at all.

---

## R10. Reuse of existing prior-art code

**VERIFIED and reused**:

| Asset | What it provides | How it is used |
|---|---|---|
| `btc_corr/exp4_regime.py` | The five-state BTC taxonomy: `EMA(close,50)` with `pct_change(24)` for direction, `ADX(high,low,close,14)` for strength, cut at ADX<20 Sideways / 20–30 Weak / ≥30 Strong | Reproduced verbatim as the baseline instrument every new instrument is measured against (FR-004) |
| `btc_corr/btcdata.py` | `load_ohlcv`, `log_returns`, absolute DATA path, `__file__`-relative layout | The pattern the new `regime/` study directory follows (FR-006) |
| `utils/BtcRegimeGate.py` | The correct cross-pair merge: declare BTC in `informative_pairs()`, pull via `dp.get_pair_dataframe()` at the strategy timeframe, merge on the tz-aware `date` column, `ffill`, fail open when BTC is unavailable | Extended or wrapped for the Phase D strategy-side path; the merge is not rewritten |
| `Basket/tools/rolling_optimal_lookback.py` | Harness shape for trackability: rolling windows, "does this window's best choice predict the next window's", adaptive error versus constant error | The structural model for the Phase C harness (FR-014) |

**Rationale**: the Phase C question is the same question `rolling_optimal_lookback.py` already answered for a different parameter, and it answered NO. Reusing its shape means the two results are directly comparable, and it means a NO here will be recognisable as the same shape of negative result rather than being re-litigated.

---

## R11. Environment

**VERIFIED**: `pandas` is not installed on the system Python. All study code runs under `.venv/bin/python`, invoked from the freqtrade root with `PYTHONPATH=.` (the pattern already used in `rolling_optimal_lookback.py`'s docstring).

**Host strategy**: `NNNC/NNNC_MLX.py` — `class NNNC_MLX(MLXClassifierMixin, NNNCStrategy)`, `classifier_type = ClassifierTypeMLX.LSTM`, `entry_trend_filter_enable = False`, `entry_trend_ema_period = 100`. The `NNNC_DDPM_` prefixed variants activate a GAN and MUST NOT be used (FR-033).

---

## R12. The forecast metric, the baseline, and the horizon grid

Added 2026-09-02 after cross-artifact analysis found the error metric undefined, the horizon grid unpopulated, and the baseline ambiguous. These are one decision and are resolved together.

### Decision — decision blocks

Each evaluation window is split into non-overlapping blocks of length `H`. For each block `b`, compute the risk-adjusted return of each arm from **its realized trades falling in that block** — `R_agg(b)` and `R_con(b)`. The block's realized winner is the argmax.

### Decision — regret, not MAE

For any rule `C` that picks a configuration per block:

```
regret(b) = R_winner(b) − R_C(b)          (≥ 0 by construction)
error(C)  = mean regret over qualifying blocks
```

**Rationale**: the precedent harness used MAE because it was forecasting a *numeric parameter* (a lookback length). Here the forecast is a *binary choice*, so MAE has no meaning. Regret is in return units, so the margin reads directly as "what the switch cost us" rather than as an abstract error.

### Decision — the baseline is the BETTER FIXED CHOICE

```
constant_error     = min( error(always-aggressive), error(always-conservative) )
phase_choice_error = error(phase rule)
beats_constant     = phase_choice_error < constant_error
```

The tuned status-quo configuration (`prediction_threshold` 0.69) is reported as a **reference column**, not as the bar.

**Rationale — this is the correction that matters most.** Scoring the phase rule against 0.69, which is neither available choice, would let a pure threshold-tuning result masquerade as trackability: if 0.55 or 0.80 is simply better than 0.69 across the board, a phase rule picking between them beats 0.69 while carrying no information about phase at all. That is the same shape of artifact as the original BTC gate's proportional de-risking — a number that moves for a reason unrelated to the hypothesis. Requiring the rule to beat the better fixed choice makes a PASS strictly harder and is the only version of the test that isolates the phase signal.

### Decision — horizon grid `H ∈ {14d, 30d, 60d}`

Long enough to accumulate a stable risk-adjusted number per block, short enough to give several decision points per window. W3 (365 days) yields 26 / 12 / 6 blocks respectively.

### Decision — block qualification

A block in which **either** arm produced fewer than `N` trades is excluded and reported, never scored as a tie (FR-018b, FR-032). **`N` is derived at task T013** from the host's measured trade rate, not fixed here — the same reasoning as R6: a threshold guessed ahead of the measurement is a borrowed figure.

### Decision — margin floor

A PASS requires the margin between `phase_choice_error` and `constant_error` to exceed the paired-seed noise band (FR-030). Three horizons constitute a small multiple comparison; the existing requirement for at least two temporally distant windows is what controls it. **A PASS at exactly one horizon in exactly two windows is the weakest admissible result and MUST be reported as such**, not as a clean positive.

---

## Summary of resolved unknowns

| # | Unknown | Resolution |
|---|---|---|
| R1 | Estimator definitions | Lo–MacKinlay VR; Hurst from the VR curve; rolling autocorr(1); 480-bar window |
| R2 | Where new indicators live | `utils/custom_indicators.py` |
| R3 | Evaluation windows | W1 2025-09→2026-08 (headline), W2 2024-06→2025-05, W3 2023 (distant) |
| R4 | Breadth coverage rule | ≥20 pairs with ≥200 bars, else `UNKNOWN` |
| R5 | Aggressive vs conservative | `prediction_threshold` 0.55 / 0.69 / 0.80, all else identical |
| R6 | Dwell-time threshold | Measured from the host's own W1 duration distribution; eliminate below the 25th percentile |
| R7 | Trade extraction and liveness | Study-local runner adding `--export trades`; three pass-rate counts per arm |
| R8 | Pinning | `test_strat.sh -t YYYYMMDD-YYYYMMDD`; never `-n`/`-o` |
| R9 | Lookahead validation | Truncated-history versus full-history label identity at three cut points |
| R10 | Prior-art reuse | exp4 taxonomy, btcdata pattern, BtcRegimeGate merge, rolling_optimal_lookback harness |
| R11 | Environment | `.venv/bin/python`, `PYTHONPATH=.`; host `NNNC_MLX` |
| R12 | Forecast metric, baseline, horizon | Mean regret over decision blocks; baseline = **better fixed choice** (0.69 as reference only); `H ∈ {14d, 30d, 60d}`; min trades per block `N` derived at T013 |

No `NEEDS CLARIFICATION` items remain.
