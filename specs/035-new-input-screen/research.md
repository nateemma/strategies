# Research: New-Input Screen for NNNC (035)

Decisions taken in planning, each with its rationale and the alternatives rejected. Code
references are relative to `user_data/strategies`.

## R1. Normalisation: causal, fitted on nothing, pre-normalised (spec FR-013 amended)

**Decision:** each input is normalised inside its builder by a causal rolling transform, then
declared in `pre_normalized_columns`. The trailing 2880-bar (30-day) z-score is clipped to ±5 and
divided by 5. Bounded shares are centred instead (for example 2s − 1).
- The shared `main_scaler.pkl` is a RobustScaler fitted on exactly 6 named columns (bb_width,
  vwap_pos, vwap_neg, di_diff_scaled, sar_ratio, spread_ma). It is therefore untouched and is read
  by the survivor exactly as by the control.

**Rationale:**
- `FeatureNormalizer.rolling_dataframe_normalise` scales only the columns not in
  `pre_normalized_columns` (`Framework/FeatureNormalizer.py:355-357`).
- A new column outside that list would need a new scaler file, and through
  `ModelKey._file_inputs` (`Framework/ModelKey.py:215-229`) a different scaler for the original 6
  columns. That is a second variable against the control.
- A rolling transform has no fitted artefact, so no artefact can see test bars (constitution XI).
  Its causality is proven by the truncation test (R12).

**Spec amendment:** FR-013 said that scalers for the new input set are fitted per fold and that
existing scalers are not read. It now reads: the new inputs carry no fitted artefact (causal
rolling normalisation), and the original 6 columns use the control's own scaler unchanged.

**Pre-existing exposure (recorded, not fixed here):** every fold of the control already reads one
shared `main_scaler.pkl` (written 2026-09-10, fit range not recorded). Control and survivors are
exposed identically, so the comparison is unaffected. The report notes it.

**Alternatives:**
- A per-fold scaler for the survivor only: two variables.
- Per-fold scalers for control and survivor alike: this retrains the control and voids
  `cs_aro`/`w_aro`/`wf_aro_g24` as comparators, so it is out of scope.

## R2. Inputs are precomputed into a per-pair cache, then merged by the strategy

**Decision:** `research/new_inputs/build.py` writes one parquet per pair to
`user_data/data/new_inputs/<PAIR>.parquet`, outside the strategies repository and uncommitted. It
has a `date` column and one column per input (plus the positioning missing-flags). A mixin's
`add_additional_indicators` calls `super()` and then left-merges the declared columns on `date`.
- The same cache feeds S1 (no strategy involved) and S2 training, so both read identical values.

**Rationale:**
- Market-state inputs need every pair's history. Computing them inside the strategy needs
  `startup_candle_count` 4096 (026, `utils/EntryFramework.py:61`), and startup is a model-key
  input that changed 4/15 folds by itself.
- A cache keeps startup and every other run input equal to the control's.
- `add_additional_indicators` is the established hook (`Framework/BaseNNStrategy.py:1053`;
  pattern in `NNNC/NNNC_ForwardMAE_Raw.py:73-91`).

**Alternatives:**
- In-strategy computation through `dp.get_pair_dataframe`: changes startup.
- Freqtrade informative pairs: they cannot carry archive data.

## R3. Time alignment

Freqtrade stamps a candle at its open: candle `date` closes at `date + 15m`. A signal at bar t can
use what has closed by `t + 15m`.
- **15-minute kline sources** (global spot, Binance.US, premium index): the row with the same open
  time covers the same interval and closes at the same moment. It is aligned to the same `date`.
- **5-minute metrics** (`create_time`): treated conservatively as available only 5 minutes after
  their stamp. Bar t uses the last snapshot with `create_time + 5m ≤ t + 15m`, as a backward
  as-of merge.
- **Windows:** every rolling window is trailing and closed on the right.

The truncation test (R12) proves the pipeline is causal given these rules. The rules themselves are
declared in the pre-registration.

## R4. Archive formats

- **Spot klines:** 12 columns, no header (open_time, OHLCV, close_time, quote_volume, trades,
  taker_buy_base, taker_buy_quote, ignore).
  - Spot files from 2025-01 stamp time in microseconds, and older files in milliseconds. The
    parser detects the unit per file by magnitude (> 1e14 means µs).
- **Metrics:** CSV with a header (`create_time, symbol, sum_open_interest,
  sum_open_interest_value, count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
  count_long_short_ratio, sum_taker_long_short_vol_ratio`), verified on ZEC 2023-06-01.
- **Futures premium index klines:** kline layout; whether the files have a header is checked per
  file.
- **Checksums:** each archive has a `.CHECKSUM` (sha256). It is verified before parsing.

## R5. Missing values

- **Absent source rows** (pair not on the source, day missing, pre-history) give NaN in the raw
  series. After normalisation NaN becomes the neutral 0 (the framework's own fill: `FeatureNormalizer.py:359-361`).
- **Positioning inputs** also carry `<name>_missing`, per clarification Q3. The flag is 1 wherever
  the normalised value is undefined: the raw source value is absent (before 2021-12-01, or a
  missing day) OR it falls in the z-score warm-up (fewer than 960 trailing values).
- **Binance.US-native order flow** on gap-filled bars (volume 0) is NaN, never a computed 0
  share. A 0 traded volume makes the share undefined, not 0.
- **S1 coverage** counts a trade as covered only if its raw value is present (not filled). An
  input under 90% is INCONCLUSIVE.

## R6. Input definitions (declared once; windows never tuned)

A window of 4 bars is 1 h, 16 bars 4 h, 96 bars 24 h, 2880 bars 30 d. Every value is computed at
the bar's close.
- `z(x)` = (x − trailing 2880 mean) / trailing 2880 std, with min_periods 960, clipped to ±5 and
  divided by 5.
- `c(s)` = 2s − 1.

| tag | family | raw definition | normalised |
|---|---|---|---|
| OF_TBG | order flow, global | taker_buy_base / volume over the last 4 bars | c |
| OF_TRG | order flow, global | log(trades over 4 bars / trailing 96-bar median of the 4-bar trade count) | z |
| OF_SZG | order flow, global | log(avg trade size over 4 bars / trailing 96-bar median) | z |
| OF_GUS | order flow, cross-venue | log((global quote vol + 1) / (US quote vol + 1)) over 4 bars | z |
| OF_TBU | order flow, Binance.US | taker_buy_base / volume over the last 4 bars (NaN if volume 0) | c |
| OF_TRU | order flow, Binance.US | as OF_TRG on Binance.US trades | z |
| PO_OI | positioning | log(OI now / OI 16 bars ago) | z, + flag |
| PO_FL | positioning | flush: max(0, −PO_OI raw) × max(0, −16-bar log return) | z, + flag |
| PO_TT | positioning | Δ over 16 bars of log top-trader long/short (sum) ratio | z, + flag |
| PO_TK | positioning | log mean taker long/short vol ratio over the last 12 snapshots (1 h) | z, + flag |
| MS_VR | market state | 026 volreg: cross-pair median of 96-bar log-return std / its trailing 2880 median | z |
| MS_BR | market state | 026 breadth: share of pairs with close > EMA768 | c |
| MS_BV | market state | BTC 96-bar log-return std | z |
| MS_BA | market state | BTC ADX(14) / 100 | c |
| MS_RR | market state | pair 96-bar log return − cross-pair median 96-bar log return | z |
| BA_LV | basis | premium index close, mean of the last 4 bars | z |
| BA_CH | basis | BA_LV − its value 16 bars ago | z |

- **Market-state pool:** the 10 `config_wf` pairs from `user_data/data/binanceus`. A pair with a
  gap-filled bar at t is excluded from that bar's cross-sectional statistic. The statistic needs at
  least 6 pairs (026's MIN_PAIRS).
- **026 reuse:** MS_VR and MS_BR apply `utils/RegimeState.py`'s `volreg`/`breadth` functions,
  unchanged, to frames from which gap-filled bars have been removed. They are 026's definitions on
  cleaned data, so they are not numerically equal to 026's gate series.

## R7. Data acquisition

- **Global sources:** data.binance.vision:
  - `spot/monthly/klines/<SYM>USDT/15m` for the 10 pairs and BTC, 2021-01..2026-08, plus `daily`
    for 2026-09-01..07;
  - `futures/um/daily/metrics/<SYM>USDT`, 2021-12-01..2026-09-07: about 17,400 files of about
    14 KB, with no monthly variant (verified 404);
  - `futures/um/monthly/premiumIndexKlines/<SYM>USDT/15m`.
- **Binance.US order flow:** `api.binance.us/api/v3/klines` paged at 1000 bars per request from
  each pair's listing (ZEC 2022-01-13, LINK 2022-01-14, NEAR 2022-01-20, AAVE 2021-07-15,
  AVAX 2021-11-18, DOT 2021-12-02; XRP, LTC and BCH from 2020; SOL 2020-09). The fetch is
  throttled.
- **Storage:**
  - raw archives in `user_data/data/external/<source>/`, with a `manifest.json` (url, sha256,
    bytes, fetched_at);
  - archives are read with `zipfile` into pandas and never extracted to run;
  - the downloader runs once, resumable, and skips files already present with a verified checksum.

## R8. S1 instrument

Reuse 034's S1 path: `research/label_screen/separation.py`, `research/nnnc_framework/entries.py`
`Model.simulate`, and the width twin cuts.
- **Trades:** the 412 early width-book trades are simulated once from the control's ARO tables.
- **Per trade:** the signal row is located as in `research/model_calibration/width_sizing.py:47-58`
  (signal date = `bars_for(pair)[1][e-1]`). bb_width, p_buy and every input are attached by
  `(pair, date)`.
- **Reproduction gate:** 412 trades and the control-at-share gap 0.1004 must reproduce before any
  input is read.
- **Partial Spearman:** rank-transform the input, pp, bb_width and p_buy. Regress rank(input) and
  rank(pp) on [1, rank(bb_width), rank(p_buy)], then take the Pearson correlation of the
  residuals. This is computed pooled, and per fold with ranks taken within the fold.
- **Split gap (reported):** within each fold, split at the input's median, giving the
  high − low pp per trade and its sign.

## R9. S2 classes and drivers

- **Survivor class:** `NNNC_NI_<TAG>(NewInputMixin, NNNC_A_Raw_Open)` (`NNNC/_new_inputs.py`).
  - The mixin declares `NEW_INPUTS = ()` (default: no effect) and `MODEL_KEY_ATTRIBUTES =
    ("NEW_INPUTS",)`.
  - The generated class sets `include_list` and `pre_normalized_columns` as new lists (parent +
    cols). They are never mutated in place (`process_one_hot_columns` mutates `self.include_list`).
  - It overrides `add_additional_indicators`.
  - The new include_list changes the model key automatically. LSTM_RAW's hidden size is unchanged
    (nearest power of 2 is 32 for 24..31 features); only the input width grows.
- **Training:** the 032 driver (`research/candidate_training/train.py`), registered at runtime
  (as 034).
- **Tables and picks:**
  - signal tables via `research.nnnc_framework.signals` with runtime `NI_<TAG>` registration;
  - sizer picks with `research/control_sizer/pick.py`, generalised to take `--plan` (the default
    stays `plan_cs_aro.json`).
- **S2 gap:** 034's separation scoring on the `NI_<TAG>` tables, against 0.1004.
- **Sizer twin:** `NNNC_NI_<TAG>_SZ` = `NNNC_CS_ARO`'s bases with the mixin. Same model key as
  the entry class (sizer attributes are in `MODEL_KEY_IGNORED`).

## R10. S3

- The winner trains f08–f14, then picks on all 15 folds (one per process) and `run_folds` from
  the repo root with `PYTHONPATH=user_data/strategies`.
- The instrument is the control sizer's, with an ARO_→NI_ table switch.
- `judge_one` runs against `cs_aro` and `w_aro` (pass needs both), with `wf_aro_g24` reported.
- Small margin under 1 pp: seed-2, the winner retrained at seed 2. `cs_aro` at seed 2 needs ARO
  seed-2 weights on 15 folds, which do not exist, so they are trained only if this rule fires.

## R11. Live feasibility (record only, clarification Q2)

| family | live source from this location |
|---|---|
| OF_*G (global) | none: api.binance.com returns 451; the archive lags by a day |
| OF_*U, OF_GUS (US leg) | Binance.US REST klines carry the same fields (verified) |
| PO_* | none on Binance; the nearest equivalent is OKX (another venue) |
| MS_* | yes: derived from the venue's own candles |
| BA_* | none on Binance from here; OKX mark/index is the nearest equivalent |

## R12. Truncation test (FR-005)

- **Shared cuts:** 200 cut times, spread evenly over 2021-03 (after the 2880-bar warm-up)
  to 2026-09-07. At each cut, every source of every pair is truncated at the cut's close and all
  pairs' inputs are rebuilt once (market state needs every pair). The value of every input, for
  every pair, at the cut bar must equal the full-history value (atol 1e-12, NaN = NaN). Pairs not
  yet listed at a cut are checked as NaN = NaN.
- **Cost:** 200 all-pair rebuilds, each truncated to the cut, so earlier cuts are cheaper.
- **Planted control:** a centred-window variant of OF_TBG must fail.
- **Tests:** a unit test runs the harness on synthetic data. The full run writes
  `research/new_inputs/causality.json`.
- **Pattern:** `utils/tests/test_regime_state.py` and `research/outcomes/causality.py`.

## R13. Multiple testing and budget

- 17 inputs at about 2 standard errors: roughly 0.8 expected false S1 passes, so S1 is a filter.
- S2 training (at most 3 × 8 ARO folds, about 3 min each) is the test that matters.
- S3 is about 7 folds of training plus 15 picks and 15 backtests.
- The downloads (about 30–45 min) and the truncation test (200 all-pair rebuilds, R12) are the
  largest fixed costs.
