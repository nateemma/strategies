# 035 pre-registration: new-input screen for NNNC

This is committed BEFORE any S1 score is computed, before any survivor model is trained, and
before any `ni_*` arm record exists. The judge refuses any arm record older than this file's
commit, and the screen refuses to run while this file is uncommitted or dirty.

**Machine-readable rules:** `preregistration.json`. The input table is
`research/new_inputs/inputs.py`, whose hash `table_sha` is recorded in the JSON.

**Question:** does information the model has never seen rank the early width-book trades beyond
bb_width and the current ARO model's p_buy? And if trained into the model, does it improve the
current sizer `cs_aro`?

**Inputs (17).** Each is computed at the bar's close from trailing data only and normalised with
no fitted artefact. Every value is in [−1, 1], or NaN, which the framework fills with 0.
Positioning inputs also carry a `_missing` flag: 1 when the value is undefined, meaning the raw
value is absent or still in warm-up.

| tag | family | definition (windows in 15 m bars) | norm |
|---|---|---|---|
| OF_TBG | order flow, Binance global | taker-buy share over 4 | 2s − 1 |
| OF_TRG | order flow, Binance global | log((4-bar trades + 1) / (96-bar median + 1)) | z |
| OF_SZG | order flow, Binance global | log(4-bar avg trade size / its 96-bar median) | z |
| OF_GUS | order flow, cross-venue | log((global 4-bar quote vol + 1) / (US 4-bar quote vol + 1)) | z |
| OF_TBU | order flow, Binance.US | taker-buy share over 4; NaN on gap-filled bars | 2s − 1 |
| OF_TRU | order flow, Binance.US | as OF_TRG on Binance.US trades; NaN on gap-filled bars | z |
| PO_OI | perp positioning | log(OI / OI 16 bars earlier) | z + flag |
| PO_FL | perp positioning | max(0, −PO_OI raw) × max(0, −16-bar log return) | z + flag |
| PO_TT | perp positioning | 16-bar change of log top-trader long/short ratio | z + flag |
| PO_TK | perp positioning | log mean taker long/short vol ratio over the last 12 snapshots | z + flag |
| MS_VR | market state | real-bar 96-bar vol, carried ≤ 24 h, cross median (≥ 6 pairs) / its 2880-bar median | z |
| MS_BR | market state | 026 breadth: share of pairs with close > EMA768 | 2s − 1 |
| MS_BV | market state | BTC 96-bar log-return std | z |
| MS_BA | market state | BTC ADX(14) / 100 | 2s − 1 |
| MS_RR | market state | the pair's 96-real-bar log return − cross median (carried ≤ 24 h) | z |
| BA_LV | spot–perp basis | premium index, 4-bar mean | z |
| BA_CH | spot–perp basis | BA_LV − BA_LV 16 bars earlier | z |

z = (x − trailing 2880 mean) / trailing 2880 std, with min_periods 960, clipped to ±5, divided by 5.

**Alignment:**
- Freqtrade stamps bars at their open. A kline row with the same open stamp closes with the bar.
- A 5-minute metrics snapshot is usable 5 minutes after its `create_time`, and is absent if more
  than 60 minutes old at the bar close.
- On Binance.US, no row means no trades in that bar, while a gap-filled venue bar is undefined.

**Stages:**
1. **S0** (done before this commit; outcomes below): fetch with verified checksums, build, then
   the truncation test (200 shared cut times; every input of every pair at the cut must equal the
   full-history value). A planted centred control must fail. Coverage is recorded, and the
   live-feasibility record is informational only (clarification Q2).
2. **S1** (early half f00–f07, no training): on the 412 early width-book trades, the partial
   Spearman correlation of each input with trade pp, given bb_width and the control's p_buy.
   - An input survives only if all of these hold:
     - coverage ≥ 0.90 (below that it is INCONCLUSIVE);
     - |Spearman with bb_width| ≤ 0.5;
     - |pooled partial| ≥ 0.10;
     - the same sign in ≥ 6 of 8 folds.
   - At most 3 survive, ranked by |partial|.
   - The screen first reproduces 412 trades and the control-at-share gap 0.1004 ± 0.002.
3. **S2:** each survivor is added on its own to `NNNC_A_Raw_Open` as a pre-normalised column.
   Everything else is identical, including the control's scaler. Trained on f00–f07, then scored
   with the control sizer's instrument (0.5-share threshold on the validation tail, agree on p_buy
   alone).
   - It passes if its early gap is > 0.1004 with ≤ 2 of 8 folds degenerate.
   - The S2 winner is the passer with the largest gap. Margins are recorded: S2 selects and S3
     decides.
4. **S3** (the S2 winner only; this is the first late-half read): trained on f08–f14, picked on
   all 15 folds, and backtested as the control sizer with only its weights changed.
   - PASS (PROVISIONAL) needs judge_one PASS against `cs_aro` AND against `w_aro`, with
     `wf_aro_g24` reported.
   - A half gain under 1 pp needs a paired seed-2 repeat.

**Decisions:** PASS (PROVISIONAL), FAIL or INCONCLUSIVE. With one era (2023–26), nothing is
recorded as a finding and nothing is adopted. The confirmation route is post-2026-09-07 data or a
dry-run.

**Disclosed:**
- **Scaler exposure:** the control's shared main scaler (fit range not recorded) is read
  identically by the control and every survivor. This is a pre-existing constitution XI exposure,
  outside this comparison.
- **Multiple testing:** with 17 inputs at about 2 standard errors, roughly 0.8 S1 passes are
  expected by chance. S1 filters; S2 and S3 test.
- **The trade set is not new.** The 412 trades, their pp, and the control's split were read in
  033 and 034, and the control sizer has read its backtest. What has not been read is any input's
  relation to them.

## Amendments before this commit (no S1 score had been computed)

1. **FR-013** (planning): new inputs carry no fitted artefact, and the control's scaler is used
   unchanged.
2. **Mixin contract** (implementation): a class with the mixin cannot share its parent's model key
   under feature 030 method hashing. "Default off" means classes without the mixin are unchanged.
3. **Market state** (first build): with gap-filled bars removed, fewer than 6 real pairs remained
   on 22–54% of bars, and MS_VR was never defined.
   - MS_BR is now 026's breadth on the venue frames.
   - MS_VR and MS_RR use each pair's real bars, carried at most 24 h, with a base median of
     min_periods 960.
4. **Snapshot staleness limit** (implementation): an as-of snapshot more than 60 minutes old at the
   bar close is absent, so a missing day is never filled from the day before.

## S0 outcome (recorded before this commit)

- **Fetch:** see `fetch_log.md`. Every archive checksum-verified; 2 metrics days absent (ZEC
  2023-12-13, NEAR 2023-12-18).
- **Causality** (`causality.json`): 200 shared cuts and 1,749 pair-cut checks. Every input of
  every pair equals its full-history value at the cut, so **PASS for all 17**. The planted centred
  control FAILS (135 mismatches over 10 cuts).
- **Coverage on the 412 early trades** (`coverage.json`): 100% for every input except OF_TRU at
  95.9%. All are above 0.90, so none is INCONCLUSIVE.
- **Live sources** (informational):
  - order flow on Binance.US (OF_TBU, OF_TRU) and market state: live from the venue;
  - OF_GUS: the US leg only;
  - global order flow, positioning and basis: no live source reachable from here (Binance global
    returns 451; OKX is the nearest equivalent).
