# Research: Width-Relative Label for NNNC Models (033)

## R1. Where the pooled relabel step lives

**Decision:** a generic, default-off hook at the start of `TrainingEngine._train_keyed`, before
`train_model`. It is declared as data on the strategy (`POOLED_LABEL`, read with `getattr`) and
implemented in a new strategy-agnostic module `Framework/PooledLabel.py`. The NNNC values (outcome
parameters, reference column, bucket count, share) live in a new mixin `NNNC/_width_relative.py`.

**Rationale:**
- `_train_keyed` is the first point where every pair's frame and labels are together: the
  aggregate path buffers all whitelist pairs, then calls it once.
- Per-pair label functions (`TrainingSignals.METHODS`) see one pair, so they cannot pool.
- The frames there are the training run's frames: the training timerange plus startup candles, all
  before the test window. No test bar can reach the buckets.
- The same pattern as 032's `TRAIN_ROW_FILTER`: generic code in `Framework/`, values on the
  strategy, absent means unchanged.

**Alternatives considered:**
- A new `LabelMethod` id in `TrainingSignals`, as the description proposed. Rejected as the main
  mechanism because a per-pair function cannot pool. The generic path-outcome function does go in
  `TrainingSignals` (R2), next to `labels_gbb_path`, so it shares that function's helpers.
- Relabelling inside `prepare_training_data`. Rejected: that function is about windowing and
  scaling, and 032 already pinned its byte-identical-off fixture.

## R2. The outcome (Clarification Q1: A)

**Decision:** a new generic function `TrainingSignals.path_outcome(df, target, stop, horizon)`. It
returns, per bar, `(outcome, bars_to_exit)`, close-based exactly like `labels_gbb_path`:
- walk forward up to `horizon` closes;
- the first close with return ≥ `target` gives `outcome = target`;
- a close at or below `stop` before that gives `outcome = stop`;
- otherwise the close return at `horizon`;
- NaN when the horizon runs past the frame end.

NNNC passes its own label parameters: `MIN_BUY_GAIN_THRESHOLD` 0.02, `HORIZON` 48 (ARO and GBR),
and `stop_frac` −0.05, `labels_gbb_path`'s default.

**Rationale:** these are the gbb_path label's own path rules, so the only change against the control
label is "relative to width". A unit test checks that `outcome == target` exactly when
`labels_gbb_path` (with its rule gates off) is positive.

**Alternatives considered:** the close return at the horizon (Q1 B) and the strategy's real exit
(Q1 C) were both declined in clarification.

## R3. Relative to width: median split, not mean (spec FR-003/FR-004 refined)

**Finding (diagnose before design):** on the ARO early folds' eligible bars, the gbb_path hit rate
rises steadily with band width:

| width decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| hit rate | 0.45 | 0.47 | 0.47 | 0.48 | 0.50 | 0.52 | 0.51 | 0.54 | 0.58 | 0.59 |

The outcome is capped at +2%, so every target hit scores the same 0.02. With "outcome > bucket
mean", every hit is positive whenever the mean is under 0.02. The positive rate per bucket would
then track the hit rate, 0.45 → 0.59, and the label would still correlate with width. That fails
SC-003 by construction.

**Decision:** within each width bucket, a bar is positive if it ranks in the top `share` (default
0.5) of that bucket's outcomes. Ties are broken deterministically:
1. outcome, higher is better;
2. then bars to exit: fewer is better for target hits, more is better for stops; closes at the
   horizon all have the same value;
3. then the bar's position in the pooled frame (stable).

This is the bucket median split with a defined tie-break. Every bucket is about 50% positive, so
width carries almost no label information, while the meaning stays "did better than bars of similar
width".

The margin (FR-004) becomes the share (default 0.5). The spec's FR-003/FR-004 and US1 scenario 3
are updated to say "the bucket's median outcome, with ties broken by speed to target".

**Alternatives considered:**
- Mean plus margin: fails SC-003, as shown above.
- More buckets with the mean: does not remove the cap problem.
- Uncapped first-touch close return: changes Q1's outcome.

## R2b. The exit side (Clarification 2026-10-14, after planning)

**Decision:** `path_outcome` takes a `side`. For `"sell"` it mirrors `labels_gbb_path_sell`,
close-based:
- the first close at or below a −`min_loss` fall (NNNC 0.02) gives `outcome = −min_loss`;
- a close at or above +`take_profit` (0.022, the sell labeller's default) before that gives
  `outcome = +take_profit`;
- otherwise the close return at the horizon (48);
- NaN past the frame end.

Exit-eligible bars are guard_metric > 0.3 AND bb_width > 0.035, disjoint from entry-eligible bars.
They get their own 10 width buckets. The bottom half by outcome is SELL. The tie-break is a fall
reached sooner ranking lower, then a take-profit reached later ranking higher, then position.

A unit test checks that `outcome == −min_loss` exactly when `labels_gbb_path_sell` (rule gates off)
labels the bar 1.

**Prior evidence on model exits (memory):**

| setting | effect of model exits |
|---|---|
| NNMT, open gate | +10.4 pp |
| breakout model | `model_exit` 416 times at +1.79% mean |
| NNNC, tight gate | +0.03 pp |
| NNNC, exit gate also opened (COCONUT) | −5.15 pp |
| NNNC_MLX | runs with exits on |

The gate study's note says loosened gates require a re-test. Hence an exit-on arm, co-primary.

**Exit gate:** unchanged from the controls: rvol > `exit_rvol_threshold` 2.0 AND guard_metric >
`exit_guard_threshold` −0.1. It is active because `entry_enable_guards` is True. MLX already trades
on it.

## R4. Bars without a valid outcome

**Decision:** the relabel step writes `%wr_keep` into each raw frame: 1.0 for an entry- or
exit-eligible bar with a valid outcome, NaN otherwise. The arm mixin's `TRAIN_ROW_FILTER` is
`(("%wr_keep", "gt", 0.5),)`. That one column carries the union, because 032's filter spec is a
conjunction.
032's filter reads the raw frame and fails NaN, so these bars are excluded, not labelled HOLD.

**Rationale:** 032's filter path is reused unchanged. The `%` prefix keeps the column out of the
features, because `remove_debug_columns` drops `^%` columns before scaling.

## R5. Label classes

**Decision:**
- entry-eligible valid bars in the top share → BUY; the rest → HOLD;
- exit-eligible valid bars in the bottom share → SELL; the rest → HOLD;
- all other bars keep their class, because the row filter removes them anyway.

The class weights then come from about 25% BUY, 25% SELL and 50% HOLD over the kept rows. Labels keep the existing one-hot / class format that
`prepare_training_data` receives.

The relabel report (`width_relative.json`, written beside the weights like 032's `row_filter.json`)
records per bucket:
- the edges;
- the count;
- the positive share;
- the median outcome;
- the Spearman of the label with width;
- the number of invalid bars.

## R6. The sizer arm (Clarification Q2: B)

**Decision:** a new mixin `NNNC/_width_sizer.py` (`WidthSizerMixin`), used on top of the existing
`WidthBookMixin`:
- **Entries:** exactly the width book's: rule AND `bb_width >= WIDTH_CUTS[fold]`, then the base gates.
- **Agree:** the model's own `predict_buy` at the signal bar (argmax BUY and `p_buy` > threshold),
  carried in the entry tag (`:agree` / `:disagree`). freqtrade passes `entry_tag` to
  `custom_stake_amount`, so no dataframe lookup is needed.
- **Exits:** they do not depend on the model. Both controls' snapshots set
  `enable_exit_signal: false`, and no exit path reads predictions. So a flat-stake twin must
  reproduce the width book's `trades_hash` exactly (instrument check).
- **Stake:** `custom_stake_amount` returns
  `min(BaseStrategy's fillable cap, max_stake, proposed × w)`, with:
  - `w = 1.0 / c` on agree and `0.5 / c` otherwise;
  - `c = s + 0.5 × (1 − s)`, where `s` is the pre-registered agree share (0.5, so `c` = 0.75 and
    the weights are 1.333 and 0.667).

**Rationale:**
- The width book already runs the model (its exit side needs it), so the predictions exist.
- Scaling by `c` keeps the expected mean stake equal to the width book's 900, so total pp is
  comparable (spec FR-008).
- BaseStrategy's liquidity cap is kept: the shortfall is measured, never bypassed.

**Stake limits:** with stake 900, 10 open trades and a 10,000 wallet, all-agree books could need
up to 12,000. The instrument reads each trade's realised stake from the result file and reports,
per fold, the trades whose stake is below the intended one. More than 5% flags the fold (spec edge
case).

## R6b. The exit-on variant

**Decision:** the same class with `enable_exit_signal: True` in its `.json`. Everything else
(weights, thresholds, stakes, exit gate) is identical.
- The `model_exit ` tag (it has a trailing space, a known artefact) is counted per fold.
- Exit-gate-passing bars during open trades are counted from the trade windows. A fold with such
  bars but zero model exits is a liveness failure.
- It is judged against the exit-off arm (`judge_one`) and the width book (pp).

The trades-hash identity applies only to the exit-off arm: earlier exits change pair availability.

## R7. Threshold and agree share (Clarification Q3: B)

**Decision:**
- The arm classes redefine `prediction_threshold` as `DecimalParameter(0.0, 1.0, decimals=4,
  space="buy", load=True)`, on the new mixin only.
- The per-fold threshold is the quantile `1 − s` of the model's effective BUY probability (argmax
  BUY ? p_buy : 0) over the width-book candidate bars of the fold's validation tail (the last 20% of
  each pair's training frame; `run_match.py`'s approach).
- Scoring calls `ft_bot_start()` (032 amendment 2), runs one fold per process, and verifies the
  loaded threshold equals the picked one to 4 decimals.

**Rationale:**
- The validation tail is training data the weights never fit; 032's run-match showed it is the
  cleaner reference.
- Width-book candidates are the bars the sizer actually prices.
- The redefinition changes nothing for existing classes; their parameter stays the class-level
  2-decimal one.

**Alternatives considered:** the whole training window, which is in-sample for the weights.

## R8. Model key and fingerprints

**Decision:**
- `POOLED_LABEL` joins `MODEL_KEY_ATTRIBUTES` on the width-relative mixin, alongside 032's
  `TRAIN_ROW_FILTER`, which is now `%wr_keep`.
- `WIDTH_CUTS`, the sizer weights and `s` go in `MODEL_KEY_IGNORED`: they do not shape the weights.
- The model-key audit test classifies every new name.
- Existing keys are pinned by the 032 fixture (`model_keys_pre032.json`). The 033 build captures a
  `model_keys_pre033.json` before any Framework edit.

## R9. Evaluation stages (pre-registered before training)

- **S0, instrument and label:**
  - per fold and side, every bucket's labelled share is in [0.40, 0.60], and |Spearman| ≤ 0.05 per
    side;
  - |Spearman(label, bb_width)| ≤ 0.05 on eligible valid bars;
  - kept rows > 0, BUY > 0 and SELL > 0;
  - simulator fidelity ≥ 0.95 on the sizer arm's recorded entries; entries equal the width book's
    (same pairs and candles);
  - thresholds load to 4 decimals;
  - the stake shortfall is reported;
  - the sizer arm's per-fold `trades_hash` (pair, dates, profit ratio; not stake or tag) equals
    `w_<m>`'s. A fold that differs had a trade blocked or changed by the wallet; it is reported and
    flagged, not hidden.
- **Per arm:** S0 for the exit-on arm is entry-rule violations 0, exit liveness, and parameters
  loaded. Its S2/S3 comparators are the exit-off arm (judge_one, late) and the width book (pp, both
  halves).
- **S1, mechanism (exit-off arm only):**
  - the agree − disagree pp per trade gap of the width-relative model, on the IDENTICAL width-book
    trades (simulator, as in `co_sizing.py`), must exceed the current model's in both halves (ARO
    0.072 / 0.085, GBR 0.063 / 0.072);
  - `agree` is recomputed from the signal tables' `buy_argmax` and `p_buy` at the fold threshold.
- **S2:** early f00–f07; the exit-off arm's pp must beat the width book's AND the control's; the
  exit-on arm's pp must beat the exit-off arm's AND the width book's.
- **S3:** late f08–f14. The exit-off arm needs `judge_one(arm, control)` PASS AND late pp above the
  width book's. The exit-on arm needs `judge_one(exit-on, exit-off)` PASS AND late pp above the width
  book's. `judge_one(arm, width)` is reported for both.
- **Small margin:** a stage passing by < 1 pp is INCONCLUSIVE until a paired seed-2 repeat.
- **Outcome:** PASS (PROVISIONAL), FAIL or INCONCLUSIVE. Confirmation: the folds after 2026-09-07,
  or dry-run.
- **Order:** the judge refuses records older than the pre-registration commit, and later stages are
  unread when an earlier one fails (032's `judge.py` pattern, reused).

## R10. Cost and running

- **Training:** ARO about 12 min per fold. GBR trained 15 folds in 17 min in 032.
- **Then:** one threshold pick process per fold (memory) and one backtest chain per model.
- **No parallel chains.** The live bot stays untouched.
