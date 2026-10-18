# ATR-barrier study: S1 screen (2026-10-17)

Pre-registration `preregistration.json` (69d250ae). Arms: cs_tba (ARO on 037 tbatr, volume > 0, 24
inputs), cs_tbb (+ 7 composite inputs). S1 pass = candidate-bar tbatr AUC above ARO's in both halves.

| Median over folds | composite | ARO | cs_tba | cs_tbb |
|---|---|---|---|---|
| candidate AUC early | 0.554 | 0.541 | 0.503 | 0.492 |
| candidate AUC late | 0.552 | 0.539 | 0.523 | 0.525 |
| candidate MCC early | — | 0.007 | −0.003 | 0.006 |
| candidate MCC late | — | 0.004 | 0.057 | 0.064 |
| all-bar AUC early / late | — | 0.518 / 0.497 | 0.497 / 0.505 | 0.493 / 0.510 |

**S1: both arms FAIL; no backtests run.** B > A in both halves: no.

Diagnostic (not part of the gate): both models output near-constant p_buy (mean ≈ 0.36, sd ≈ 0.035,
max ≈ 0.44, e.g. f02 and f13), so they sit near the base rate rather than learning the label. cs_tbb
has the composite's inputs yet scores below the composite's fixed equal-weight readout (0.49 vs 0.55
early), which points at training, not at the inputs. Late folds f12/f13 do reach 0.55–0.61.

## Diagnostics after S1 (2026-10-17; not gates)

**1. What the trained model learned.** Training is 3-class (sell 43% / hold 20% / buy 37% on f02);
val MCC sits at ≈ 0.14 from epoch 1, then early-stops at epoch 26 after two LR cuts. The test windows
show where that MCC comes from:

| CS_TBA, median over folds | early | late |
|---|---|---|
| p_buy, barrier hit vs timeout (all bars) | ≈ 0.70 | ≈ 0.71 |
| p_buy, up vs down (candidates; the S1 score) | 0.503 | 0.523 |
| **p_buy − p_sell**, up vs down (candidates) | **0.542** | **0.563** |
| p_buy − p_sell, up vs down (all bars) | 0.527 | 0.527 |
| corr(p_buy, p_sell) on candidates | 0.84 | 0.84 |

p_buy and p_sell both rise with "a barrier will be hit" (volatility); the direction is in their
difference. S1 scored the wrong readout for a 3-class model. (Full softmax regenerated outside the
repo; `research/nnnc_framework/signals.py` now also writes p_sell / p_hold.)

**2. What is learnable from the same inputs** (`reference.py`, `reference.json`; last bar only,
hourly training rows to 48 bars before each test window):

| Median AUC, candidates | early | late |
|---|---|---|
| direction, A inputs (24): logistic / HGB | 0.552 / 0.538 | 0.560 / 0.555 |
| direction, B inputs (31): logistic / HGB | 0.545 / 0.537 | 0.570 / 0.558 |
| composite (fixed, no fit) | 0.554 | 0.552 |
| trained CS_TBA, p_buy − p_sell | 0.542 | 0.563 |
| hit vs timeout, A / B HGB | 0.679 / 0.692 | 0.670 / 0.684 |

Direction tops out near 0.55–0.57 for every model on these inputs; the trained LSTM, read as
p_buy − p_sell, is at that level. The composite's members add ≤ 0.01 (B ≈ A). Barrier-hit (volatility)
is far more learnable (≈ 0.68 on candidates, 0.73 on all bars).

## Direction readout arms (preregistration_dir.json, 32c4d02a)

S1 (`screen_dir.json`), candidate AUC of (1 + p_buy − p_sell)/2 vs ARO's p_buy 0.541 / 0.539:
cs_tba_d 0.542 / 0.563 **PASS**; cs_tbb_d 0.537 / 0.539 FAIL.

S2 (`verdicts_dir.json`), cs_tba_d; instrument PASS (trades = w_aro's in every fold):

| vs | arm | comparator | early / late gain | paired | Decision |
|---|---|---|---|---|---|
| w_aro | 48.06 | 51.44 | −3.29 / −0.10 | UNCHANGED (p 0.18) | FAIL |
| wf_aro_g24 | 48.06 | 49.97 | −2.72 / +0.81 | UNCHANGED | FAIL |
| cs_aro (question) | 48.06 | 59.95 | −8.14 / −3.75 | **WORSENED** (p 0.003) | FAIL |

**FAIL.** A direction score at ARO's AUC level on tbatr sizes the width book worse than equal stakes:
the trades are paid by the fixed take-profit / stop exits (gbb_path's outcome, which ARO's label is),
not by which 4×ATR barrier is hit first.

## Exit replay: do label-matched exits let the direction score pay? (`replay.py`, `replay.json`)

Descriptive (replays choose, backtests judge; no re-sequencing or wallet effects). The width book's
930 trades re-exited from their recorded entries. Fidelity: the null replay reproduces 99.8% of
recorded exits (reason and bar) and the recorded P&L (35.66 / 15.60 vs 35.85 / 15.59). (First run
read the class-default exit params, 62% fidelity; fixed to the folds' own params JSON before reading.)

pp of wallet, early / late, by stake source:

| Exit | equal (w_aro) | cs_aro | cs_tba_d |
|---|---|---|---|
| today's (null) | 35.66 / 15.60 | 40.63 / 19.25 | 32.43 / 15.50 |
| C2: TP 2.897×atr_pct_roll, today's stop | 35.52 / 21.69 | 41.07 / 25.59 | 32.11 / 21.37 |
| label barriers 4×ATR14 on closes, 48-bar timeout | 19.64 / 15.60 | 25.09 / 18.22 | 17.10 / 13.71 |
| same, intrabar TP/stop orders | 18.50 / 18.79 | 22.66 / 22.33 | 14.58 / 16.16 |
| 3×ATR14 on closes | 17.28 / 20.77 | 22.62 / 24.19 | 14.95 / 17.14 |
| TP 2×ATR14, today's stop + grace | 39.17 / 19.11 | 44.64 / 23.38 | 35.13 / 18.99 |
| TP 4×ATR14, today's stop + grace | 38.66 / 12.55 | 44.24 / 15.78 | 34.43 / 13.88 |

- Exits matching the label cost the width book ~16 pp early (the 24 h grace's recoveries are cut;
  at 4×ATR 40–54% of trades time out at 12 h); late is flat to +3.
- **Under every exit, cs_tba_d's stakes do worse than equal stakes, and cs_aro's do better.** The
  direction score's failure is not an exit mismatch: on the width book's own trades it does not
  rank outcomes, even outcomes defined by its own barriers.
- Side result: TP 2×ATR14 with today's stop (+3.5 / +3.5 vs null) and C2 (−0.1 / +6.1) are the
  only exits better than today's; both are picked after seeing 9 variants.

## Direction-only training: cs_tbd (preregistration_dironly.json, 7cd06148)

Arm A retrained on windows whose last bar hit a barrier (HOLD dropped; kept share 0.60 on f00 vs
0.83, 0.50 on f14). S1 (`screen_dironly.json`) PASS: candidate AUC 0.560 / 0.554 vs ARO 0.541 /
0.539 (cs_tba_d's readout 0.542 / 0.563). S2 (`verdicts_dironly.json`), instrument PASS:

| vs | cs_tbd | comparator | early / late gain | paired | |
|---|---|---|---|---|---|
| w_aro | 54.34 | 51.44 | −0.78 / +3.68 | UNCHANGED (p 0.26) | FAIL (early) |
| wf_aro_g24 | 54.34 | 49.97 | −0.22 / +4.59 | UNCHANGED | FAIL (early) |
| cs_aro (question) | 54.34 | 59.95 | −5.63 / +0.03 | UNCHANGED (p 0.11) | FAIL |
| cs_tba_d | 54.34 | 48.06 | +2.51 / +3.78 | | PASS |

**FAIL** (early half). Direction-only training is the best tbatr sizer (+6.3 pp over the 3-class
readout, +2.9 over equal stakes) but does not beat equal stakes early, nor cs_aro.

## TP 2×ATR(14) on the width book: w_aro_tp2 (preregistration_tp2.json, 40f88e46)

Instrument PASS: replaying the arm's own 873 trades under its exit reproduces 99.9% of exits. TP
levels: median 2.57% (5–95%: 1.28–5.93%); 37% below the fixed 2.2%.

| vs | w_aro_tp2 | comparator | early / late gain | paired | |
|---|---|---|---|---|---|
| w_aro | 50.99 | 51.44 | −1.96 / +1.50 | UNCHANGED (p 0.94) | **FAIL** (early) |
| wf_aro_g24 | 50.99 | 49.97 | −1.39 / +2.41 | UNCHANGED | FAIL (early) |

The replay's +3.53 / +3.51 did not survive: the exit runs faithfully, but the arm takes 57 fewer
trades (873 vs 930). Higher targets hold trades longer (ROI 5% exits 18 → 81, take-profit 557 →
426), and a held pair cannot re-enter -- the one-trade-per-pair blocking also seen in the
SimpleTrigger grace study. Replays fix the trade list, so they cannot see it.

## Held-pair re-entry (preregistration_reentry.json, 50dd2754)

Instrument PASS both arms: adds in both halves (w_aro_hp 88 / 113, w_aro_tp2_hp 105 / 127), 100% on
trades in profit, no exit on an add candle; w_aro_tp2_hp's 619 single-fill trades replay to 99.8%.

| comparison | arm | comparator | early / late gain | paired | trades | |
|---|---|---|---|---|---|---|
| **primary** w_aro_tp2_hp vs w_aro_hp | 52.11 | 62.06 | +4.77 / **−14.71** | UNCHANGED (p 0.44) | 851 vs 915 | **FAIL** |
| w_aro_hp vs w_aro (reported) | 62.06 | 51.44 | +3.54 / +7.07 | UNCHANGED (p 0.053) | 915 vs 930 | PASS |
| w_aro_tp2_hp vs w_aro (reported) | 52.11 | 51.44 | +8.31 / −7.64 | UNCHANGED | 851 vs 930 | FAIL |

- Re-entry does not rescue the 2×ATR take-profit: it still takes fewer trades (851 vs 915) and loses
  the late half badly (late Sharpe and maxDD WORSENED vs w_aro_hp).
- **Side result (reported, not the question):** 031's in-profit add on the width book gives 62.06
  (+3.5 / +7.1 vs w_aro; maxDD 4.9 → 4.2, Sharpe 2.41 → 2.50) -- above cs_aro's 59.95, which runs
  with the add OFF. cs_aro and all its comparators have the add off, so the like-for-like test is
  cs_aro + add vs w_aro_hp; no leverage control was run here.

## Sizer x add stack (preregistration_stack.json, 49713c6b) -- PASS (PROVISIONAL)

cs_aro_hp = cs_aro + 031's in-profit add (cs_aro's thresholds unchanged; add stake 1x, sizer weights
on the first fill only). Instrument PASS: adds 89 / 113, 100% in profit, 0 exits on an add candle,
agree share 0.489 vs cs_aro 0.489.

| comparison | arm | ctl | early | late | paired p | trades |
|---|---|---|---|---|---|---|
| vs w_aro_hp (co-primary) | 69.93 | 62.06 | +4.45 | +3.42 | 0.002 IMPROVED | 915 / 915 |
| vs cs_aro (co-primary) | 69.93 | 59.95 | +3.14 | +6.84 | 0.046 IMPROVED | 915 / 930 |
| vs w_aro (reported) | 69.93 | 51.44 | +7.99 | +10.49 | 0.002 IMPROVED | 915 / 930 |

Sharpe 2.67 (cs_aro 2.59, w_aro_hp 2.50, w_aro 2.41; all UNCHANGED); maxDD 4.03 (4.49 / 4.18 / 4.88).
The sizer and the add stack. Capital deployed (stake x hours, descriptive): cs_aro_hp 24.3k / 30.1k
vs cs_aro 17.8k / 22.2k (+36% / +35%), so part of the add's gain may be more capital at work; daily
SD 0.37 vs 0.33. No leverage control was run. One era, seed 1. Nothing adopted.

### Leverage control (031's rule, research/held_pair/exposure.py; leverage_stack.json)

The adds' extra stake-hours x the comparator's pp per stake-hour, per fold. A gain no larger than this is
"explained by extra capital" (031 pre-registration cb158ba, fixed before this arm). Computed, not backtested.

| add on | half | gain pp | leverage pp | beats | avg stake arm / ctl |
|---|---|---|---|---|---|
| cs_aro_hp vs cs_aro | early | +3.14 | 11.98 | no | 137 / 103 |
| cs_aro_hp vs cs_aro | late | +6.84 | 5.02 | yes | 195 / 149 |
| w_aro_hp vs w_aro | early | +3.54 | 9.93 | no | 138 / 107 |
| w_aro_hp vs w_aro | late | +7.07 | 3.92 | yes | 202 / 156 |

The add's gain is explained by extra capital early and beats it late, on both books. The sizer's part
(cs_aro_hp vs w_aro_hp, +4.45 / +3.42) is not leverage: average committed stake 137 / 138 and 195 / 202.
The 031 control assumes the extra capital would earn the base book's rate linearly (no wallet or liquidity cap).

## Backtested leverage control (preregistration_lev.json, 56e298cd) -- INCONCLUSIVE (instrument failed)

cs_aro_lev = cs_aro with stakes x1.3187. Instrument FAILED, no P&L read: median stake ratio vs cs_aro
1.00 (needed 1.3187 +- 0.02); average committed stake 0.865 / 0.862 of cs_aro_hp's (needed 0.90-1.10).
Cause: 68% of cs_aro's entries are already held below their intended stake by the fillable cap
(<= 1/QUOTE_VOLUME_HEADROOM_MULT of the entry candle's quote volume; median stake 0.46 of intended), so
a bigger stake cannot lever them; only the ~27% uncapped entries scale. The in-profit add reaches more
capital because it fills on a LATER candle with its own liquidity headroom -- it buys capacity a bigger
first stake cannot.

## Controls v2 (preregistration_lev2.json, 177385f3) -- both primaries FAIL

Instruments PASS. cs_aro_lev2 (stakes x1.7432): avg committed stake 1.001 / 0.977 of cs_aro_hp's, 99.7%
of stakes on the cap model, 0 adds. cs_aro_hpany (any-signal add): adds 202 / 262, 335 not in profit
(in-profit share 0.28), 0 exits on an add candle, agree share 0.497.

| comparison | arm | ctl | early | late | paired | Sharpe | maxDD | verdict |
|---|---|---|---|---|---|---|---|---|
| cs_aro_hp vs cs_aro_lev2 (primary) | 69.93 | 82.66 | -13.72 | +1.00 | p 0.14 | 2.67 / 2.55 | 4.03 / 6.60 | FAIL |
| cs_aro_hp vs cs_aro_hpany (primary) | 69.93 | 82.96 | -17.34 | +4.31 | p 0.15 | 2.67 / 2.66 | 4.03 / 5.71 | FAIL |
| cs_aro_lev2 vs cs_aro (reported) | 82.66 | 59.95 | +16.87 | +5.84 | IMPROVED p 0.001 | 2.55 / 2.59 | 6.60 / 4.49 WORSENED | FAIL (risk) |
| cs_aro_hpany vs cs_aro (reported) | 82.96 | 59.95 | +20.48 | +2.53 | IMPROVED p 0.014 | 2.66 / 2.59 | 5.71 / 4.49 | PASS |

Avg committed stake early / late: cs_aro 103 / 149, cs_aro_hp 137 / 195, cs_aro_lev2 137 / 191,
cs_aro_hpany 159 / 243. At matched capital, staking more beats the in-profit add early (-13.7) and
loses to it narrowly late (+1.0); the any-signal add (more capital still) beats it early, loses late.
The in-profit add's gain over cs_aro is not shown to exceed leverage. Descriptive, not pre-registered:
pp per maxDD point cs_aro_hp 17.4, cs_aro_hpany 14.5, cs_aro 13.4, cs_aro_lev2 12.5 -- the in-profit
add has the lowest drawdown of the four.

## Capital through the liquidity cap (preregistration_cap.json, 441ca595)

- Q1 stack (cs_aro_hpany_lev2 = any-signal add + first fills x1.7432): **PASS (PROVISIONAL), 97.56** --
  vs cs_aro_hpany 82.96 +9.33 / +5.27 (IMPROVED p 0.046), vs cs_aro_lev2 82.66 +12.94 / +1.96 (p 0.15),
  vs cs_aro +29.81 / +7.80; Sharpe 2.51, maxDD 8.48 (UNCHANGED); avg stake 197 / 284. Returns-only: PASS.
- Q2 / Q3 (max_adds 2 / 3): **INCONCLUSIVE, instrument failed, no P&L read.** Trades did reach 2 / 3 adds
  in both halves (2 adds: 87 / 128; 3 adds: 28 / 64), but 2 exits fell on an add candle in each arm: LTC
  and AAVE on 2025-10-10 21:15 (the crash) -- the 2nd add filled at the open, then the stop hit inside
  the same 15 m candle. The rule targets the exit-due-at-open bug; intrabar stops are genuine. Amending
  it is the user's call.

## All-available entry (preregistration_all.json, e40e4e31) -- INCONCLUSIVE (instrument mis-specified)

cs_aro_all2k / all3k: stakes match min(L x w, fillable cap) for 99.9% / 99.0% of shared trades,
wallet-limited 0.1% / 1.0%, 0 adds, agree share 0.484 / 0.476; trades 921 / 911 vs 930. FAILED only on
"median stake > cs_aro's": the median entry (367.7) is liquidity-capped, so the condition could never
pass -- my pre-registration error. Avg stake 155 / 212 (2k), 185 / 246 (3k) vs cs_aro 103 / 149. No P&L read.

### Q2 / Q3 after the instrument amendment (preregistration_cap_amend.json, e5309bba) -- both FAIL

Amendment (user, before any P&L read): stops hit inside an add candle no longer fail the instrument.
Both arms then PASS the instrument (2 stop exits on an add candle each, reported).

| comparison | arm | ctl | early | late | Sharpe | maxDD | verdict / returns-only |
|---|---|---|---|---|---|---|---|
| m2 vs cs_aro_hpany (Q2) | 73.02 | 82.96 | -3.29 | -6.64 | 1.88 / 2.66 WORSENED | 12.49 / 5.71 WORSENED | FAIL / FAIL |
| m3 vs m2 (Q3) | 64.27 | 73.02 | +1.92 | -10.67 | 1.37 / 1.88 WORSENED | 16.92 / 12.49 WORSENED | FAIL / FAIL |

Descriptive: 2025-10-10 trades -4.33 pp (1 add), -13.27 (2 adds), -17.21 (3 adds); the crash day is more
than m2's whole late shortfall (-8.94 of -6.64), but m2 also loses early (-3.29). More adds per trade =
averaging down into moves the any-signal add cannot tell from dips (72-78% of adds not in profit).

### All-available after the instrument amendment (preregistration_all_amend.json, 2603dea6)

Amendment (user, before any P&L read): the stakes-up test applies to entries whose fillable cap exceeds
cs_aro's intended stake (281 / 270 trades): median stake ratio vs cs_aro 2.22 (2k) / 2.86 (3k). Both PASS.

| comparison | arm | ctl | early | late | paired | maxDD | verdict / returns-only |
|---|---|---|---|---|---|---|---|
| all2k vs cs_aro (Q4) | 95.50 | 59.95 | +25.76 | +9.79 | IMPROVED p 0.0003 | 7.70 / 4.49 WORSENED | FAIL (risk) / PASS |
| all3k vs all2k (Q5) | 114.17 | 95.50 | +13.90 | +4.77 | IMPROVED p 0.028 | 8.22 / 7.70 | PASS / PASS |
| all3k vs cs_aro_lev2 | 114.17 | 82.66 | +22.80 | +8.73 | IMPROVED p 0.004 | 8.22 / 6.60 | PASS |
| all3k vs cs_aro_hpany | 114.17 | 82.96 | +19.18 | +12.04 | p 0.084 | 8.22 / 5.71 | PASS |

Sharpe 2.48 (2k) / 2.51 (3k) vs cs_aro 2.59, all UNCHANGED. Backtest vs replay upper bound: 2k 95.50 vs
106.07, 3k 114.17 vs 134.24 (wallet and re-sequencing cost ~10-20 pp). The gain comes from the ~30% of
entries with spare liquidity. Not modelled: slippage at up to 10% of a 15 m candle's volume on a 3,000
USDT order -- live depth (036) is the calibration. all3k (no add) > the stacked arm 97.56, not judged head to head.
