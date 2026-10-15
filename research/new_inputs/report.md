# 035 new-input screen: report

**Sources:**
- Pre-registered in `preregistration.json` / `.md` (commit 1f79efa5), before any S1 score. Four
  amendments were made before that commit and are listed in it.
- Outputs: `causality.json`, `coverage.json`, `screen.json`, `survivors.json`, and `fetch_log.md`.
- Run: 2026-10-15.

**Label: PROVISIONAL.** There is one era (2023–26). Nothing is adopted or promoted.

## Decision: no survivors; all 17 inputs stop at S1

None of the 17 inputs ranks the 412 early width-book trades beyond bb_width and the control's
p_buy by the pre-registered rule:
- |partial Spearman| ≥ 0.10;
- the same sign in ≥ 6 of 8 folds;
- |ρ with bb_width| ≤ 0.5;
- coverage ≥ 0.90.

So S2 (training) and S3 (backtest) did not run. **No model was trained, no arm record exists, and
no late-half data was read.**

| tag | family | partial | sign folds | ρ bb_width | ρ p_buy | split gap | S1 |
|---|---|---|---|---|---|---|---|
| OF_TRU | order flow, Binance.US trade count | **+0.159** | 5/8 | +0.15 | −0.07 | +0.075 | FAIL (sign) |
| OF_GUS | order flow, global ÷ US volume | **−0.142** | 4/8 | −0.23 | +0.08 | −0.054 | FAIL (sign) |
| MS_BV | BTC 24 h volatility | +0.096 | **6/8** | +0.12 | −0.01 | +0.048 | FAIL (magnitude) |
| PO_TK | taker long/short ratio | −0.084 | 5/8 | −0.11 | −0.06 | −0.042 | FAIL |
| MS_RR | return vs basket | −0.077 | 5/8 | +0.02 | +0.05 | −0.033 | FAIL |
| OF_TRG | order flow, global trade count | +0.072 | 3/8 | +0.16 | −0.11 | +0.044 | FAIL |
| MS_VR | market volatility regime | +0.067 | 4/8 | +0.24 | −0.18 | +0.060 | FAIL |
| PO_FL | liquidation-flush proxy | −0.064 | 6/8 | +0.18 | −0.19 | −0.017 | FAIL |
| OF_SZG | global average trade size | −0.054 | 4/8 | −0.00 | +0.13 | −0.002 | FAIL |
| PO_OI | open-interest change | +0.049 | 4/8 | −0.24 | +0.17 | −0.009 | FAIL |
| PO_TT | top-trader long/short change | +0.048 | 6/8 | −0.16 | +0.16 | −0.002 | FAIL |
| OF_TBU | Binance.US taker-buy share | +0.047 | 5/8 | +0.06 | −0.14 | +0.041 | FAIL |
| OF_TBG | global taker-buy share | −0.046 | 3/8 | +0.07 | −0.13 | +0.005 | FAIL |
| MS_BA | BTC ADX | −0.046 | 5/8 | +0.12 | −0.21 | −0.017 | FAIL |
| BA_LV | spot–perp premium | +0.024 | 3/8 | −0.05 | +0.05 | −0.033 | FAIL |
| MS_BR | market breadth | +0.007 | 4/8 | −0.11 | +0.05 | −0.006 | FAIL |
| BA_CH | premium change | +0.001 | 4/8 | −0.11 | +0.16 | −0.005 | FAIL |

- **Instrument:** the screen reproduced 412 trades and a control-at-share gap of 0.1003 (0.1004 ±
  0.002) before any input was read.
- **Orthogonality:** every input passes the cap. The largest |ρ with bb_width| is 0.24 (MS_VR,
  PO_OI).
- **The two near misses swing widely by fold:**
  - OF_TRU runs from −0.68 (f02) to +0.36 (f04);
  - OF_GUS from −0.27 to +0.22.

  With about 50 trades per fold, a fold partial has a standard error near 0.14. The pooled values
  are about 3 standard errors from zero, but the sign is not stable.
- **Chance level:** 17 inputs were tested at about 2 standard errors, so roughly 0.8 passes were
  expected by chance. Two inputs reached the magnitude cutoff and neither held its sign. This is
  consistent with no stable incremental information on this trade set.

## S0 (before the pre-registration commit)

- **Data:** every archive was checksum-verified (`fetch_log.md`). 2 of 17,420 metrics days are
  absent.
- **Causality:** 200 shared cuts and 1,749 pair-cut checks. Every input was prefix-stable, and the
  planted centred control failed (135 mismatches).
- **Coverage on the 412 trades:** 100%, except OF_TRU at 95.9%.
- **Live sources:**
  - Binance.US order flow and market state are live from the venue;
  - global order flow, positioning and basis have none reachable from here.

## SC-003: existing classes unchanged

`wf_aro_g24` f00 and f07 were re-run with the held-pair add declared off, as the records were made
(see 034). The trade hashes are 6e32009f6dd2bc9c and b3fe1bf90ca9841d, **byte-identical**
(`regime/arms/sc003_035_bytecheck_aro_off_*`, local). The pre-034 model-key fixture is unchanged
(`tests/test_mixin.py`).

## Harness lessons

1. **Gap-filled bars break cross-pair pooling.** Removing them left fewer than 6 real pairs on
   22–54% of bars, and 026's `volreg` (a 2880-bar base median with full min_periods) was never
   defined. The fix was to carry each pair's real-bar value forward at most 24 h. This was amended
   before the pre-registration commit.
2. **Mixins and model keys:** under feature 030 method hashing, any class that overrides a key
   method gets its own key. A mixin can be default-off for other classes, but never key-neutral
   for its own.
3. **New inputs can skip the scaler.** A causal rolling transform inside the builder, plus a
   pre-normalised column, leaves the control's 6-column RobustScaler untouched. A survivor would
   have been a true one-variable change.
4. **Data sources:**
   - Data Vision spot files from 2025 use microsecond stamps;
   - UM metrics come only as daily files (about 17,400 for 10 pairs);
   - Binance.US REST omits bars with no trades, which count as zero, not unknown.

## Fix-oriented follow-ups (not run)

1. **OF_TRU with a stability-first design.** Binance.US trade count was the strongest pooled
   signal (+0.159) and is live-feasible. Its fold-sign failure could be noise at about 50 trades
   per fold. Two options:
   - re-test it on a larger trade set, for example the full candidate set (all gbb-eligible bars,
     not just width-book trades), pre-registered with the same partial statistic;
   - or train it into ARO directly as a one-input S2 arm, labelled exploratory.
2. **A larger S1 sample.** The width book's 412 early trades give per-fold standard errors near
   0.14, so sign stability over 8 folds is a strict test. The same screen on the gbb candidate
   bars' forward outcomes (thousands of bars) would have more power, at the cost of measuring
   something one step further from traded P&L.
3. **The control-sizer follow-ups stand** as the open lead: the GBR repeat, restoring the intended
   stake ratio, and forward confirmation.

## Provenance

- **Code:** `research/new_inputs/` (inputs, sources, build, causality, coverage, trades, screen,
  survivors, fetch), `NNNC/_new_inputs.py` (unused by any class), the `--plan` option on
  `research/control_sizer/pick.py`, and the `NI_` registration in
  `research/nnnc_framework/signals.py`.
- **Data (not committed):** `user_data/data/external/` (manifest) and `user_data/data/new_inputs/`
  (cache, table ee1da6f6878d9184, code 09a12a3a523962a5).
