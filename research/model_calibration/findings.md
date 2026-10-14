# NNNC model check: what does the model select? (2026-10-13) — DIAGNOSTIC, PROVISIONAL

Inputs: [data.md](data.md) (calibration data); [within_candidates.py](within_candidates.py) →
`within_candidates.json`; [width_twin.py](width_twin.py) → `width_twin.json`;
[width_plus_model.py](width_plus_model.py) → `width_plus_model.json`. No training, no backtests:
the books are rebuilt with the NNNC framework study's simulator, which reproduces the recorded model
books at 99.7–100%. It has not been fidelity-checked on a non-model entry mask.

## 1. The high AUC is mostly the label's rule

- The gbb label is first a rule: `guard_metric < 0.3` and `bb_width > 0.035`. About 20% of bars
  pass it, and about half of those are labelled positive.
- Overall AUC is 0.93–0.98. On rule-eligible bars only, it falls to **0.60–0.62** (ARO, MLX),
  **0.58–0.59** (GBR) and **0.53–0.54** (GFM).
- The probability compression in data.md follows from this: label rates of 0.35–0.9 are squeezed
  into p 0.65–0.85.

## 2. Within the candidates, the model behaves like a band-width sort

The test holds selectivity fixed: the same number of rule-eligible bars as the model selects, chosen
by depth (`guard_metric`, declared), by width (`bb_width`, post-hoc) and at random. Mean 48-bar
forward return:

| model | half | model | depth | width (post-hoc) | random |
|---|---|---|---|---|---|
| ARO | early / late | +2.11 / +1.25 | +0.79 / +0.76 | +2.15 / +1.26 | +0.60 / +0.20 |
| GBR | early / late | +2.12 / +1.13 | +0.78 / +0.78 | +2.29 / +1.28 | +0.61 / +0.21 |
| GFM | early / late | +0.69 / +0.85 | +0.72 / +0.71 | +1.83 / +1.20 | +0.63 / +0.21 |
| MLX | early / late | +0.72 / +0.25 | +0.67 / +0.21 | +0.69 / +0.24 | +0.61 / +0.21 |

- **The declared rule passes, 3 of 4.** The model beats depth in both halves on ARO, GBR and MLX.
- **MLX's pass means little.** Its 0.69 threshold takes 80–93% of all eligible bars, so its
  selection is mostly the rule itself.
- **Against width (post-hoc), the model loses on ARO, GBR and GFM.** Its 48-bar +2.2% hit rate is
  a little higher (0.71–0.75 vs 0.67–0.71).

## 3. Book level: a model-free width book matches the model books

Built with the same gates and the gbb rule, then `bb_width ≥` a per-fold cut. The tradeable
variant uses the previous fold's cut. Simulator, 24 h grace, no held-pair add, pp early / late:

| | model book | width book (tradeable cut) | width AND model | width AND p ≥ 0.6 |
|---|---|---|---|---|
| ARO | 35.2 / 14.7 (607 / 569 tr) | 35.7 / 15.6 (412 / 519) | 19.4 / 13.5 (128 / 184) | 36.2 / 16.5 |
| GBR | 37.6 / 15.7 | 36.5 / 17.1 | 30.8 / 21.6 | 33.0 / 18.6 |
| GFM | 32.4 / 7.2 | **47.1 / 17.0** | 0.5 / 2.4 | 45.6 / 17.9 |
| MLX | 22.2 / 14.5 | 20.8 / 14.8 | 20.6 / 14.6 | 20.8 / 14.8 |

- **The width book alone matches ARO, GBR and MLX, and clearly beats GFM.**
- **Where the model agrees with width, it raises profit per trade:** ARO 0.15 vs 0.09 pp per trade,
  early. The cost is volume, and the 10-slot book is rarely full, so volume matters.

## Reading

This agrees with the 2026-09 guard ablation ("the model does not pick the trades"), now on the
guards-open, 24 h stack. It is also consistent with 023's finding that the model beats the plain
gbb trigger twin: the trigger twin did not sort by band width.
- **The model's measurable selection is close to one rule feature.** Its distinctive value is
  higher precision per trade, not more profit per book.
- **This does not say the model is useless.** It says the current label teaches mostly the rule,
  and what the model learns beyond the rule (within-candidate AUC about 0.6) is roughly what band
  width already supplies.

## Options (fix-oriented; the model stays)

1. **Make the width book the baseline that every model change must beat.** It costs nothing at run
   time, it is model-free, and it already exists in the simulator. A model variant that does not
   beat it adds nothing over one sorted feature.
2. **Re-target training at what the rule and width do not know.**
   - Train on rule-eligible bars only, so capacity goes to ranking candidates instead of learning
     the rule.
   - Optionally use a label defined relative to width: for example, forward outcome above the
     width-rank expectation.
   - The within-candidate AUC and the width-book comparison are the success measures, followed by
     walk-forward P&L.
3. **Use the model as a priority or sizing signal on the width book.** Agreement trades earn about
   1.7× per trade on ARO. For example, the width book's entries with a full stake when the model
   agrees and a smaller one when it doesn't. This is a cheap simulator screen.

**Before any of these:** confirm the width book with a real backtest. The simulator has not been
fidelity-checked on a non-model entry mask.

## 4. Confirmed by real backtest (`width_backtest.py` → `width_backtest.json`)

Arms `w_aro`, `w_gbr`, `w_gfm`, `w_mlx` (`NNNC_W_<M>` = `WidthBookMixin` + the control's base; the
model loads but does not select; held-pair add off; 24 h grace) against each model's own 24 h
control, f00–f14:

| | width early / late | model early / late | width − model (15 folds) | trades width / model | Sharpe width / model | maxDD width / model | paired |
|---|---|---|---|---|---|---|---|
| ARO | 35.85 / 15.59 | 35.28 / 14.68 | **+1.48** | 930 / 1,176 | 2.41 / 2.54 | 4.9 / 3.7 | UNCHANGED p 0.90 |
| GBR | 36.72 / 16.96 | 37.94 / 15.51 | **+0.22** | 928 / 2,021 | 2.41 / 1.60 | 5.4 / 9.7 | UNCHANGED p 0.99 |
| GFM | 47.29 / 16.78 | 32.56 / 7.20 | **+24.31** | 1,146 / 1,879 | 2.70 / 1.72 | 5.0 / 6.6 | UNCHANGED p 0.08 |
| MLX | 19.90 / 14.20 | 21.29 / 13.90 | **−1.09** | 351 / 360 | 2.17 / 2.24 | 2.5 / 2.3 | UNCHANGED p 0.76 |

- **The simulator is faithful on this non-model entry mask too.** Trade recall is 99.7–99.9%,
  precision 99.7–99.8%, and simulated pp is within 0.6 per half.
- **The finding holds at backtest level.** A model-free book of the gbb rule plus a band-width cut
  matches ARO, GBR and MLX, which are all UNCHANGED, with gaps of +1.5, +0.2 and −1.1 pp over 15
  folds. It beats GFM by 24 pp.
- **GBR matches with less than half the trades, a higher Sharpe and half the drawdown.**
- **Labels.** This is a diagnostic, not a pre-registered test: both halves were seen in the
  simulator first, and f00's cut is not causal.

**Consequence.** The width book is the honest baseline for NNNC model work. A model variant that
does not beat it adds nothing over one sorted rule feature.

## 5. The model as a sizer on the width book (`width_sizing.py` → `width_sizing.json`)

Simulator screen, EXPLORATORY (the agree side was already seen in `width_plus_model.py`). The width
book picks the trades; each trade keeps its simulated exit and is re-weighted by the model at its
signal bar. "pp per capital" is the sized book's pp divided by its mean weight, i.e. its return on
the capital it actually used.

| | agree n / pp per trade, early | disagree, early | agree, late | disagree, late |
|---|---|---|---|---|
| ARO | 73 / 0.146 | 339 / 0.074 | 118 / 0.096 | 401 / **0.011** |
| GBR | 178 / 0.114 | 316 / 0.051 | 175 / 0.082 | 261 / **0.010** |
| GFM | 11 / 0.035 | 563 / 0.083 | 31 / 0.047 | 542 / 0.029 |
| MLX | 180 / 0.113 | 4 / 0.107 | 166 / 0.088 | 1 / 0.218 |

pp per unit capital, early / late:

| | unsized | agree 1.0 / 0.5 | agree 1.0 / 0.25 | rank (0.5–1.0 by p_buy) |
|---|---|---|---|---|
| ARO | 35.7 / 15.6 | 39.4 / 21.9 | 44.2 / 29.4 | 39.0 / 19.6 |
| GBR | 36.5 / 17.1 | 41.8 / 22.5 | 46.8 / 27.3 | 39.2 / 19.3 |
| GFM | 47.1 / 17.0 | 46.5 / 17.5 | 45.6 / 18.4 | 48.9 / 17.1 |
| MLX | 20.8 / 14.8 | 20.8 / 14.8 | 20.8 / 14.7 | 22.4 / 17.2 |

- **Declared rule: NOT a useful sizer (2 of 4 models; needed 3).** ARO and GBR pass both parts; GFM
  and MLX fail.
- **The failures are structural, not a reversal.** MLX's model book already IS the width book (5 of
  351 trades disagree), so there is nothing to size. GFM's model agrees on only 42 of 1,147 width
  trades, and those are no better.
- **Where the model and width disagree, the model is right on ARO and GBR.** Late-half trades the
  model would not take earn about 0.01 pp each, against 0.08–0.10 for trades it would. So within the
  width book the model does carry information, but only on the two models whose threshold leaves
  room for disagreement.
- **The continuous rank weight beats unsized per unit capital on all four models in both halves**
  (listed before the run, but not the declared test).
- **Caveat.** Return per unit capital assumes the freed capital is worth nothing; scaling agree
  trades up instead would meet the liquidity cap.

## 6. Width-book variants (`width_variants.py` → `width_variants.json`)

Simulator screen; early half selects within each family, late half checks (declared before the
run). pp early / late (trades over 15 folds):

| variant | ARO | GBR | GFM | MLX |
|---|---|---|---|---|
| width (count-matched cut) | 35.7 / 15.6 (931) | 36.5 / 17.1 (930) | 47.1 / 17.0 (1,147) | 20.8 / 14.8 (351) |
| q50 | 53.3 / 10.8 (4,341) | same | same | 20.4 / 13.1 (216) |
| q80 (family pick) | 56.9 / 11.3 (2,265) | same | same | 14.1 / 8.8 (128) |
| q90 | 45.7 / 10.8 (1,330) | same | same | 8.2 / 9.1 (85) |
| **w+rvol (family pick)** | 37.5 / 15.3 (884) | 39.4 / **22.3** (891) | 46.2 / **19.3** (1,082) | 13.6 / **16.7** (238) |
| w+depth | 23.4 / 10.0 | 26.1 / 17.0 | 28.5 / 9.7 | 15.8 / 16.3 |
| w+low_close_norm | 26.0 / 12.3 | 29.2 / 20.9 | 31.7 / 10.2 | 17.5 / 16.8 |
| w+atr_pct | 28.6 / 7.9 | 28.0 / 8.6 | 35.4 / 8.2 | 16.1 / 10.0 |
| w+adx | 19.9 / 2.4 | 19.5 / 7.1 | 21.3 / 1.2 | 8.7 / 8.4 |

(ARO, GBR and GFM share gates, so a model-free quantile cut gives them the same book.)

- **Model-free quantile cuts: NOT adopted (late beats width on 0 of 4).** More trades, more early pp,
  much less late pp. The count-matched cut sits above q90 and the declared grid stopped at q90.
- **width + rvol: passes the declared rule (late beats width on GBR, GFM, MLX; ARO −0.3).** The margins
  are small (+5.2, +2.3, +1.9 late), and MLX gives back 7.2 early, so over 15 folds it is +1.5 / +8.1
  / +1.4 / −5.3. This is within simulator noise for everything except GBR.
- **Every other second feature makes the book worse.** Width is doing nearly all of the work.
