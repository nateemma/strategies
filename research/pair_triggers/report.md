# Pair triggers: report

**Sources:** `preregistration.json` (commit 09013ee2, before any backtest); `verdicts.json`. Run 2026-10-17.
**Label: PROVISIONAL**; the pairs were chosen with the late half in view. Nothing is adopted.

## Decision: no candidate trigger (two arms FAIL, two INCONCLUSIVE)

| arm | trades | alone pp (gbb 11.04) | vs gbb early / late | paired | beside width book: early / late vs w_aro (35.85 / 15.59) | paired (whole) |
|---|---|---|---|---|---|---|
| `st_pt_kst_z` | 708 | 18.48 | −13.35 / +20.78 → FAIL | UNCHANGED (p 0.69) | 49.99 / 10.82 → FAIL | UNCHANGED (p 0.28) |
| `st_pt_cci_copp` | 459 | 17.72 | −13.28 / +19.95 → FAIL | UNCHANGED (p 0.72) | 48.57 / 8.29 → FAIL | UNCHANGED (p 0.49) |
| `st_pt_cmo_macd` | 5 | – | INCONCLUSIVE (instrument) | | | |
| `st_pt_cmo_dcb` | 1 | – | INCONCLUSIVE (instrument) | | | |

- Both tradeable pairs beat gbb in total and in the late half, but lose to it in the early half;
  beside the width book they add in the early half and cost in the late half. The halves disagree
  every time, so nothing is shown. Max drawdown vs gbb: IMPROVED for both.
- Beside the width book, ~9–10 pp of the arms' P&L was refused by the shared wallet and 223–339
  trades held a pair the width book also held (two sub-accounts needed).

## Instrument finding: the CMO pairs fire on zero-volume bars

The two CMO arms made 5 and 1 trades although 037's cache shows ~1,000–1,600 co-fires in the test
windows. Traced on f05: the co-fires are almost all on **zero-volume (gap-filled) bars**, where the
strategy, correctly, never enters (ZEC 136 co-fires, 7 with volume). 32% of 037's `data_ok` bars
have zero volume; `data_ok` only excludes mostly-filled stretches. Share of each confirmed pair's
fires on zero-volume bars:

- tbatr pairs: CMO∧DCBreakout 1.00, DCBreakout∧TSI 0.99, CMO∧PVO 0.98, CMO∧TEMACross 0.89, the PVO
  pairs on candidate bars 0.87–0.98; KST∧ZScore 0.04, CCI∧KST 0.05.
- tb pairs: 0.00–0.05 except KST∧VBM 0.32, COPP∧VBM 0.25, COPP∧MI 0.26.
- comparators: width book 0.19, gbb 0.25.

So the volatility-neutral (tbatr) pair results are mostly a data artifact, and every 037 score
includes ~32% untradeable bars. Fix: add `volume > 0` to the screens' row filter and re-run.

## Also found

037's harvest ran every SimpleStrategies class at its Parameter **defaults**, not the class's
`buy_params` (e.g. CMO period 9 / −60, not 16 / −66). The triggers port the defaults (0 mismatches
vs the cache on ~2.0M bars per leg).
