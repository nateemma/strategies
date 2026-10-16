# Research: 037 Indicator Screen

Each decision resolves an open technical question from the plan's Technical Context.

## R1 — How the harvest calls each strategy's own code (FR-002, FR-003)

**Decision.**
- **One base frame per pair.** Build it once: `DataframePopulator.add_indicators(MINIMAL)` +
  `SimpleStrategy.add_additional_indicators`. This is the same base every SimpleStrategy sees, and
  `SimpleStrategy` is the only class in the family that overrides `add_additional_indicators`.
- **Then, per strategy class:**
  1. copy the base frame;
  2. set `atr_ok = True` (the only filter column read inside a signal function, by EhlersBase);
  3. call the class's own `get_entry_signals` / `get_exit_signals` with its current parameters,
     loaded through freqtrade's `StrategyResolver` as `pick_threshold._load_fold_context` does;
  4. harvest the new columns (the column-set difference) plus the returned raw signals.
- **Columns already in the base frame** (the 24 model inputs, guard_metric, bb_width, ...) are
  harvested once, under source `base`.

**Rationale.**
- Calling `get_*_signals` directly skips `get_entry_conditions`, where SimpleStrategy applies the
  atr / market-state / regime / trend-mode / slope filters. So the signal is taken before every
  family-level filter, without copying any formula (FR-002).

**Strategy-internal filters, found by an AST scan of every `get_entry_signals` /
`get_exit_signals`.** Three places filter inside the signal code itself. Each gets a declared
override that still calls the strategy's own code:

| Strategy | Internal filter | Override (raw signals emitted) |
|---|---|---|
| EhlersBase family (11 classes) | `_get_regime_masks` (cycle/trend vote), strict regions, `atr_ok` | call the class's own `get_cycle_entry_trigger`, `get_trend_entry_trigger`, `get_cycle_exit_trigger`, `get_trend_exit_trigger` → four raw signals. The as-built `ehler_buy` is not a raw signal. |
| BBBreakout | `trend_mode` selects which band crossing fires | call `get_entry_signals` twice, with the `trend_mode` column forced to 1 and then to 0 → two raw signals (`…:tm1`, `…:tm0`); same for exits |
| Ichimoku | `entry_use_cloud_thickness`, `entry_use_htf_filter` | both parameters set to False on the instance. The higher-timeframe columns are never built, since `populate_indicators` is bypassed. |

Combo's use of guard_metric is one of its four indicator votes, not a filter, so it is kept as built.

**Liveness (Principle IV).** Each override has a test showing that it changes the signal
(override ≠ as-built on real data), and that the forced-column trick reaches the strategy's own
code path.

**Alternatives rejected.**
- Re-implementing the formulas: forbidden by FR-002, and it drifts from the strategies.
- Running `populate_entry_trend`: it applies the filters.
- A freqtrade backtest per strategy: far slower, and it yields entries, not raw signals.

## R2 — Which classes are in scope (FR-001)

**Decision.** Every concrete class in `SimpleStrategies/` that:
- subclasses `SimpleStrategy` (directly, or via `EhlersBase` / `Ehlers_Indicator`);
- is not a subclass of `TriggerOnlyStrategy`;
- is not one of the trigger arms (`Breakout*Open`, `Gbb*Open`, `GeometryOpen`).

Discovery is by import and `issubclass`, not by file list. The resolved class list and the
exclusions are written to the harvest manifest.

**Rationale.** Trigger arms wrap registered triggers already scored in 021. The gbb trigger itself
enters as the candidate flag (FR-006, FR-010).

## R3 — Data range, pairs and BTC

**Decision.**
- **Pairs:** the `config_wf.json` whitelist (10 pairs, the same set as the NNNC controls).
- **BTC/USDT 15m** is loaded alongside (2021-01-01 → present on disk) for the BTC-valued indicators
  and the B/C BTC targets. BTC is not a scored pair.
- **Range:** 2021-01-01 → 2026-09-07 (the plan `wf_aro_g24` `data_end`), plus the startup warm-up.
- **Warm-up:** the first 1,000 bars per pair are excluded from every score.
- **Windows:** fold test windows f00–f14 (2023-01-01 → 2026-09-07; early f00–f07, late f08–f14),
  plus an extra descriptive era **E0** = 2021-04-01 → 2022-12-31.

**Rationale for E0.** Constitution II asks for a genuinely distant era. The early and late halves
are adjacent. E0 (2021–22, including the 2022 bear market) is scored for the single-indicator and
pair screens and reported alongside. It needs no training, so it is available. Multivariate models
cannot be scored on E0, because no training data precedes it. Their persistence rests on early/late.

## R4 — Label targets (FR-008)

**Decision.**
- **BUY and SELL labels:** from the ARO control class (`NNNC_A_Raw_Open` with the
  `wf_aro_g24` params snapshot), via `strat.get_training_labels` on each pair's full-range frame.
  These are the labels the ARO model trains on (`HORIZON` = 48 bars).
- **The last 48 bars of data** are not computable and are dropped.
- **Check:** a test shows that the full-range labels equal the per-fold labels built the
  label-screen way (`research/label_screen/audit._frames`) on one fold, for every pair. Labels are
  a deterministic function of the price path, so this must hold exactly.

## R5 — Candidate population (FR-010)

**Decision.** `research.outcomes.triggers.evaluate("gbb", frame)`: the registered trigger, with its
own conditions and its asymmetric NaN convention. It is stored as the harvest column
`base:gbb_candidate`.

## R6 — Scores (FR-012, FR-013)

**Decision.**
- **Strength and direction:** AUC by the rank (Mann–Whitney) formula, with ties averaged.
  - direction = sign(AUC − 0.5);
  - strength = |AUC − 0.5|.
  - Computed per window on rows where both the column and the target are defined.
- **Non-monotone association:** mutual information, in bits, normalised by the target's entropy.
  - The column is cut into 20 quantile bins, with edges taken from that fold's TRAINING window
    (Principle XI).
  - For E0, the edges come from E0 itself (descriptive; flagged in the output).
  - Binned MI is deterministic (SC-007); the k-NN estimator is not.
- **Raw signals:** fire rate, plus lift = P(target | signal fires) ÷ P(target).
  - Lift is not computable when the signal fires on fewer than `min_fires` bars, or on more than
    95% of the window's bars.
- **Aggregation over windows:** the median, and the share of computable windows whose direction
  matches the median's. Windows with fewer than `min_rows` rows are reported but excluded from
  aggregation.
- **Instrument check (SC-002):** the rank of guard_metric and bb_width by median strength for BUY on
  all bars, over f00–f14.

## R7 — Residual strength beyond bb_width + guard_metric (FR-015)

**Decision. Stratified AUC.**
- Candidate bars are binned on a 5 × 5 grid of bb_width × guard_metric quintiles, with edges from
  the fold's training-window candidate bars.
- Within each cell, the column's AUC is computed.
- Cells are combined by weighting each cell's AUC by n_pos × n_neg (the van Elteren-style weighted
  average).
- The result is the column's ability to rank WITHIN cells where bb_width and guard_metric are
  roughly fixed. bb_width itself scores about 0.5 here by construction.
- **Width-sort baseline:** bb_width's unstratified AUC on candidate bars, shown next to every
  column's unstratified AUC.

**Alternatives rejected.**
- Logistic-regression residualisation: it requires a fitted model per fold. It adds tuning without
  changing the question.
- Partial correlation: it assumes linearity, and guard_metric's effect is thresholded.

## R8 — Redundancy groups (FR-014)

**Decision.**
- Spearman correlation between every pair of columns, on a fixed-seed sample of 200,000 candidate
  bars drawn from the early windows only.
- Average-linkage clustering on 1 − |ρ|, cut at |ρ| ≥ 0.8.
- Each column's ρ with guard_metric and with bb_width is reported (on all bars and on candidate
  bars).

## R9 — Pairwise combinations (FR-016)

**Decision.**
- Every unordered pair of raw BUY signals, combined with AND. With ~90 raw buy signals after
  overrides, that is about 4,000 pairs.
- Each pair is scored by lift for the BUY label on candidate bars, per early window.
- **Selection rule (pre-registered):**
  - computable in ≥ 6 of 8 early windows;
  - median early lift ≥ max(lift of either component) × 1.10;
  - median early lift ≥ 1.20;
  - from those, the top 20 by median early lift.
- Survivors are scored once on f08–f14. A survivor is CONFIRMED if its late median lift is also
  ≥ 1.20 and above both components' late lifts.
- E0 lift is reported for survivors, descriptively.

## R10 — Multivariate models (FR-017, FR-018)

**Decision.**
- **Models (scikit-learn 1.9.0, already in the venv):**
  - a **rule model**: `DecisionTreeClassifier(max_depth=3, min_samples_leaf=500,
    class_weight="balanced", random_state=0)`;
  - a **flexible model**: `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
    max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
    random_state=0)`.
  - All settings are fixed in the pre-registration; no per-fold tuning.
- **Inputs:** every harvested value column, plus raw signals as 0/1. Missing values are passed
  through, since both models handle NaN natively.
- **Training rows:** candidate bars in [2021-01-01 + warm-up, test_start − purge).
  - purge = 48 bars for label targets, N bars for regime targets.
  - Built per fold (Principle XI).
- **Scored:** test-window AUC.
- **Comparisons, on the same rows:**
  - the restricted model, the same class trained on [bb_width, guard_metric] only;
  - the width sort (raw bb_width AUC);
  - the ARO model's p_buy, where the fold's signal table exists
    (`research.nnnc_framework.signals.fold_table`, BUY target only).
- **Candidate rule:** a multivariate result is a candidate only if the full model's median AUC
  exceeds the restricted model's in both the early and the late halves.
- **Rules output:** for each fold, the tree's paths are exported as text. The most frequent split
  columns across folds are counted.
- **One fold per process.** CPU only; no Metal/MLX.

## R11 — Regime targets (FR-011, FR-020, FR-021)

**Decision.** Horizons N ∈ {48, 288, 672} bars (12 h, 3 d, 7 d). Each forward quantity is computed
per bar, from bars (t, t+N] only.

| Target | Forward quantity | Scope |
|---|---|---|
| A1 | mean `profit` of outcome-table rows (Open exit, `research/outcomes/table.load_table`) whose bar is a gbb candidate and whose open time is in (t, t+N], pooled across all 10 pairs | market-wide (same value for every pair at t) |
| A2 | mean `profit_ratio` of `w_aro` backtest trades (plan `plan_w_aro.json`, folds f00–f14) opened in (t, t+N], pooled | market-wide; only exists inside fold test windows |
| B-btc / B-pair | log return close(t+N)/close(t), for BTC / for the pair | market-wide / per pair |
| C-btc / C-pair | std of 15m log returns over (t, t+N], for BTC / for the pair | market-wide / per pair |

**Classes.**
- Bars in the top tercile = 1, bottom tercile = 0; the middle tercile is dropped.
- Cut points come from the target's values in the fold's training window. For A2, which has no
  training-window trades, the cut points come from the early test windows (f00–f07). This is
  pre-registered and flagged.
- A bar where no trigger or trade falls in (t, t+N] is "not computable" for A1/A2.

**Persistence baselines (FR-021).** The same quantity over the trailing window [t−N, t):
- B: the trailing return;
- C: the trailing volatility;
- A1/A2: the mean outcome of candidates and trades that CLOSED before t (realised, so causal).

Each is scored with the same AUC as the indicators. A regime candidate must beat its baseline in
both halves.

**BTC-valued indicators.** The harvest is also run on BTC/USDT. Its columns are joined to each pair
by timestamp, under the prefix `btc:`.

**Autocorrelation.** Regime targets at 3 d / 7 d are heavily autocorrelated: one 90-day window
holds only about 13–30 independent 7-day periods. Window-level medians and the same-direction
share are therefore the unit of evidence; row counts overstate the sample. This is stated in the
report.

## R12 — Lookahead test for columns (FR-004)

**Decision.** Reuse `research.outcomes.causality`:
- its era slices (`_load_slice`, three pairs × three eras: SOL/LTC/AAVE at 2021-10, 2023-10,
  2025-07);
- its cut scheme (16 cuts, warm-up 200 bars).

Through a small adapter, the populate step becomes "harvest one pair" (R1). Every column (values
and raw signals) is compared bar by bar up to each cut. NaN == NaN counts as equal, and float
values must match within 1e-9 relative. Any mismatch → EXCLUDED, with the first mismatch recorded.
The result is written to `research/indicator_screen/causality.json`.

## R13 — Storage, size and cost

**Decision.**
- **Harvest cache:** one feather per pair at `user_data/data/indicator_screen/<PAIR>.feather`,
  outside the nested repo and never committed. A sidecar JSON holds the code sha, the class list,
  the columns and the parameters.
- **Size estimate:** 11 frames (10 pairs + BTC) × ~200k bars × ~350 float32 columns ≈ 3 GB on disk.
- **Screens read only the columns they need,** per pair (feather column selection). Peak memory is
  one column across all pairs, or one fold's candidate rows for the models (~300k × 350 ≈ 0.4 GB).
- **Cost:** harvest time is measured on one pair before the full run (task-level gate). Expected to
  be minutes per pair. The Ehlers super-smoother and the pandas-ta calls dominate.
- **Committed outputs:** small JSON plus `report.md` under `research/indicator_screen/`.

## R14 — Plotting strategy (FR-007)

**Decision.**
- `SimpleStrategies/AllIndicators.py`: a `SimpleStrategy` subclass whose `populate_indicators`
  runs the R1 harvest for the pair and adds every column. Raw signals are prefixed `raw_buy:` /
  `raw_sell:`.
- It never enters: `populate_entry_trend` / `populate_exit_trend` set no signals.
- It exists for `freqtrade plot-dataframe`. The screens use the cache, not this class.

## R15 — Pre-registration and late-window lock (FR-023, SC-005)

**Decision.**
- `research/indicator_screen/preregistration.json` holds:
  - `min_rows`, `min_fires`;
  - the pair-selection rule (R9) and the model settings (R10);
  - the tercile rule (R11), the horizons, and the A2 cut-point exception;
  - the instrument-check rule.
- Any screen asked to score a late window (f08–f14) first checks that `preregistration.json` is
  committed and unchanged since that commit (git). It refuses otherwise.
- Early and E0 scoring may run before the commit, so the rules can be calibrated on early data.
  This is stated as such.
