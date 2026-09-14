# 00 — Shared evaluation protocol

Every brief in this study is evaluated the same way. The protocol exists because the repo has
already measured that (a) a 12-month window at ~100 trades has a ±7.6pp noise floor, (b) the
pooled walk-forward harness resolves ~3.35pp/yr, and (c) parameter-scale effects (1–2pp) will
never be resolvable on this data (`regime/WALKFORWARD_POWER.md`, `AGENT_GUIDE.md` "Evaluating
a change"). Label-free methods are *more* prone to overfitting than the supervised pipeline
(they optimise the evaluation metric directly), so the protocol is stricter, not looser.

## 1. The simulator (shared by briefs 1, 2, 4, 5, 6)

One vectorised, differentiable simulator, `research/labelfree/sim.py` (MLX first; a PyTorch
mirror is acceptable if MLX blocks something). Contract:

- **Inputs:** per-pair feature tensors produced by the existing pipeline
  (`DataframePopulator` → `FeatureNormalizer` scalers from `saved_data/`), so features are
  identical to what NNNC/NNMT see. Pair-agnostic features only (AGENT_GUIDE "Feature
  Engineering Rules").
- **Decision timing:** the policy sees bars `≤ t` and emits a target exposure `w_t ∈ [0, 1]`
  per pair (long-only spot). Fills happen at bar `t+1` open (or a VWAP proxy). No other
  lookahead. This mirrors freqtrade's next-candle execution.
- **Costs inside the objective:** taker fee per side (Binance.US: use the real tier, ≈10 bp),
  half-spread, and a volatility-scaled slippage term. Training cost scale `γ` is a
  hyperparameter (DeePM sweeps {0.5, 1.0, 1.5}); **evaluation is always at full or
  pessimistic cost**.
- **Liquidity cap:** executed `Δw` is limited to what ≤10% of the candle's quote volume can
  fill given the stake; the unfilled remainder carries to the next bar. Use a soft-min in
  training so the gradient does not vanish where the cap binds; the hard cap applies in
  evaluation. This is the "wall 2" correction from `us_spot_market_study.md`.
- **Portfolio layer:** `max_open_trades`, stake sizing and `tradable_balance_ratio` from
  `config/config_binanceus.json` are replicated so the simulator's P&L is comparable to
  freqtrade's. The simulator is validated by feeding it a *fixed* NNNC_MLX signal and
  matching freqtrade's backtest P&L for the same window to within fee rounding — this
  calibration step is a deliverable of brief 1 and blocks everything else.
- **Outputs:** per-bar net log return, turnover, position, and a trade list in the same shape
  as the backtest archive (`pair, open_date, close_date, profit_abs`) so `evaluation/pool.py`
  and `compare_pooled.py` work unchanged.

The freqtrade backtest is the **arbiter**; the simulator is for training. Any candidate is
finally re-run through `evaluation/run_folds.py` as a real strategy (a thin
`BaseNNStrategy` subclass whose `get_predictions` loads the policy).

## 2. Windows and folds

- Training range and windows are those in `regime/regimedata.py`. W1 (`20250901-20260831`)
  is the sole single-window acceptance window; W2/W3 are in-sample and reported only.
- **Primary evidence is the pooled walk-forward** (`evaluation/`, feature 007): 15 anchored
  folds, 730-day minimum training, 90-day test folds, trades pooled before any statistic.
  SUI/USDT excluded per `config/config_wf.json`.
- Purge and embargo: when a method uses any forward-looking quantity in training (labels for
  brief 2 and 3's trade outcomes), purge the maximum horizon and embargo one horizon after
  each fold boundary (López de Prado, *AFML* ch. 7).

## 3. Baselines (all four are mandatory in every report)

| Baseline | Why |
|---|---|
| `NNNC_MLX` control, production config | The thing to beat. Pooled-WF record exists (`evaluation/pooled_nnnc.json`; fold plan via `evaluation/foldplan.py`). |
| Equal-weight buy-and-hold of the whitelist | The "did you just buy beta" check; the market fell in W1 so this also exposes short-side-free luck. |
| TSMOM rule with a no-trade band (e.g. 20/60-bar EMA sign, band ±0.1) | A label-free rule with zero parameters learned from P&L; if the model cannot beat it, the objective is doing nothing. |
| `NNNC_MLX` + cost-aware threshold (0.80 operating point from `CLOSING_THE_LEARNABILITY_GAP.md`) | Isolates "new objective" from "better operating point". A label-free model that only reproduces this is not new. |

## 4. Statistics

- **Effect size vs floor.** Report the pooled annualised return delta against the control with
  its bootstrap 95% CI (`evaluation/compare_pooled.py`), and state whether it clears 3.35pp/yr.
  Below that: "not resolved", not "no effect".
- **Seeds.** ≥5 seeds per configuration. Report the interquartile mean and stratified-bootstrap
  CI across seeds (Agarwal et al. 2021, `rliable`), never the best seed. Two seeds giving
  identical trades is one measurement.
- **Trial accounting.** Every configuration evaluated on any fold — seed, loss variant, γ,
  architecture, band width, threshold — is appended to `research/labelfree/trials.jsonl`
  *before* its result is known. The final report computes the Deflated Sharpe Ratio (Bailey &
  López de Prado 2014) over the full trial count and the Probability of Backtest Overfitting
  via CSCV (Bailey et al. 2015; `pypbo`). **PBO > 0.2 fails the brief regardless of the point
  estimate.**
- **Cost stress.** Re-evaluate the accepted configuration at 1.5× and 2× fees and with the
  liquidity cap at 5%. Report the fee level at which the edge crosses zero.
- **Turnover.** Report trades/yr and mean holding time next to every return figure. A method
  whose edge appears only with turnover far above the control's is flagged.
- **Reproduction.** A separate process must reproduce the accepted result from the saved
  config and seed (byte-identical trades), per AGENT_GUIDE's reproducibility note.
- **Spec-kit provenance.** Every result must come from a `specs/0NN-<slug>/` feature with
  `spec.md`, `plan.md`, `tasks.md` and a clean `/speckit-analyze` report produced *before*
  implementation and again *after* (README "How to run a brief"). The spec's success criteria
  must be the gates in §5, verbatim or stricter — never looser. A result without this trail is
  inadmissible regardless of its numbers.

## 5. Acceptance and kill gates (used verbatim by the briefs)

**Accept** (promotes to a `specs/` feature and dry-run):
1. Pooled-WF annualised return ≥ control + 3.35pp/yr, CI excluding zero.
2. Positive in ≥ 10 of 15 folds and in each of W1/W2/W3 when run single-window.
3. Beats all four baselines net of full costs.
4. PBO ≤ 0.2; DSR > 0 over the full trial log.
5. Survives 1.5× fees with the delta still positive.
6. Reproduced from config by a fresh process.

**Kill** (write it up in the brief's "Result" section and stop):
- After the budgeted sessions, no configuration clears gate 1 → "not resolved".
- Any configuration whose in-sample (W2) advantage exceeds its W1 advantage by more than the
  floor → overfitting; stop and record.
- Simulator vs freqtrade P&L disagree by more than fee rounding on the calibration signal → the
  simulator is wrong; nothing downstream is evidence until fixed.

## 6. Reporting template (one file per brief run: `briefs/results/<brief>-<date>.md`)

```
Hypothesis · Config (path, seed list) · Trials logged (n) ·
Pooled WF: return/yr, CI, folds positive, trades/yr, hold time ·
Baselines table · Cost stress · PBO / DSR · Verdict (accept / not resolved / kill) ·
What was learned that changes the next brief
```
