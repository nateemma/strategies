# Draft input for /speckit-specify (feature 037): indicator screen against labels and regimes

Indicator screen: which SimpleStrategies indicators, alone or combined, track the gbb_path labels and
market regimes (feature 037).

**Background:**
- **What SimpleStrategies tests today.** It has 67 indicator strategies: 56 on `SimpleStrategy` and
  11 on `EhlersBase`. Each tests one indicator by P&L, behind its own guards and regime filters.
  None of them says whether the indicator carries information about the outcomes NNNC is trained on.
- **What is known.**
  - gbb_path BUY labels are built partly from guard_metric and bb_width.
  - Within gbb candidate bars, the NNNC model is about a band-width sort (AUC 0.53–0.62; memory
    project_model_check_width_sort).
  - Labels (034) and new non-OHLCV inputs (035) found nothing that beats it.
  - The model sees 24 indicators; most SimpleStrategies indicators have never been scored against
    the label.
- **Regime.** No dependable regime indicator exists (001 BTC gate FAIL; 026 volreg gate fails on
  NNNC entries, because NNNC's trades in the "off" state are profitable).
- **Prior art to reuse, not rebuild:**
  - `Debug/DebugAnalyseIndicators.py`: Pearson, Spearman and MI against gbb labels, but one pair,
    the DataframePopulator columns only, and no walk-forward;
  - `research/label_screen/audit.py`: labels per pair per fold window;
  - `research/outcomes/causality.py`: a truncation test for lookahead;
  - `research/outcomes/table.py`: per-bar trigger outcomes under the Open exit.

**Goal:** a descriptive screen of every SimpleStrategies indicator, alone and in combination, against
(1) the gbb_path BUY and SELL training labels and (2) three regime definitions. It answers:
- which indicators carry information beyond guard_metric and bb_width;
- whether any combination beats the band-width sort on gbb candidate bars;
- whether any indicator, or combination of indicators, identifies a regime.

This is a screen. It makes no P&L claim; candidates go on to backtests in later features.

**Stories (confirm in spec):**
1. **(P1) An all-indicators harvest.**
   - An `AllIndicators` strategy calls each SimpleStrategies strategy's indicator code on the same
     data and keeps:
     - every indicator column it adds;
     - its raw buy and sell signals at its current parameters, taken from
       `get_entry_signals`/`get_exit_signals`, i.e. BEFORE `get_entry_conditions` applies the
       atr/regime/trend-mode/slope filters.
   - No entry or exit guards anywhere in the harvest.
   - Each strategy's formula is called, not copied by hand. A strategy that fails is recorded and
     excluded, never silently dropped.
   - Every column must pass the causality truncation test; a failure is reported and excluded
     (fractals, Ichimoku chikou and any centred filter are the suspects).
   - It is a real freqtrade strategy, so the indicators can be plotted.
   - Columns are cached per pair, so the screens never recompute them.
2. **(P1) Single-indicator screen against gbb_path.**
   - Targets: gbb_path BUY and SELL labels (the ARO control's label parameters; labels are targets
     only).
   - Scored on each of the 15 walk-forward fold TEST windows (non-overlapping), pooled over the
     whitelist. Per indicator:
     - AUC (direction-free strength |AUC − 0.5|);
     - mutual information (catches non-monotone shapes);
     - for each strategy's raw signal, lift = label rate when the signal fires ÷ base rate.
   - Reported as the median over folds plus the share of folds with the same sign.
   - Two populations:
     - (a) all bars: the INSTRUMENT CHECK. guard_metric and bb_width must rank at or near the top
       for BUY, or the tool is wrong and nothing else is read;
     - (b) gbb candidate bars only: the real question.
   - Redundancy: cluster indicators by rank correlation, so that "guard_metric in disguise" is
     visible.
   - Residual score: each indicator's AUC/MI after removing what bb_width and guard_metric explain
     (within population b).
3. **(P2) Combinations.**
   - (i) Pairwise: AND of two raw buy signals, about 2,200 pairs, scored by lift on population (b).
   - (ii) Multivariate: a shallow tree model (depth 2–3, readable as rules) and a gradient-boosted
     model on all harvested columns.
     - Fitted on each fold's TRAINING window and scored on its test window.
     - Compared with the same models given only bb_width + guard_metric, and with the band-width
       sort.
     - Only the comparison counts as evidence; the absolute score does not.
   - Multiple-comparison control:
     - selection uses the EARLY folds (f00–f07) only;
     - survivors are checked once on the LATE folds (f08–f14);
     - an indicator or combination counts only if it holds in both halves.
4. **(P2) Regime screen.** The same singles and combinations against three regime targets, each
   defined forward-looking over N days (default N = 3 and 7; confirm), as top vs bottom tercile:
   - **A — strategy outcome:** the mean realised outcome of gbb-trigger bars under the Open exit
     (the outcome table), opened in the next N days, pooled across pairs. Answers "when do our
     trades work".
   - **B — market direction:** the forward N-day return of BTC, and of the pair itself.
   - **C — volatility:** the forward N-day realised volatility of BTC, and of the pair.

   Indicators are scored both on the pair's own values and on BTC's values (the market-wide view).
   Same early-select, late-confirm rule.

**Constraints:**
- Read-only with respect to strategies and the live bots. No training queue is needed, except for
  the small CPU tree models, which run one fold per process. This must not disturb the live NNNC
  recording (pid in `user_data/logs/live_nnnc.pid`) or the momentum bot (pid 2661).
- Framework/ untouched. No pair-specific code.
- The labels and regime targets are forward-looking; they are used only as targets, never as inputs.
  Every input column is causal (story 1).
- Outputs go to `research/indicator_screen/`: JSON per screen plus `report.md`, and a findings
  section in docs.

**Out of scope:** backtests of any candidate, changes to NNNC inputs or gates, and SimpleStrategies'
own regime filters (they are bypassed, not studied).
