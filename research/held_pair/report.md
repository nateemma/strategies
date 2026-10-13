# Held-pair position add (DCA) — report (031, 2026-10-13)

**Question.** When a fresh entry signal fires on a pair that already holds a trade, should the
framework add one stake to that trade instead of ignoring the signal?

The 24 h grace studies found that one-trade-per-pair blocking is the main cost of longer holds.
The held-pair screen found DCA was the best response on the early half.

- Pre-registration: `dd39f32` ([preregistration.md](preregistration.md)).
- Verdicts: [verdicts.json](verdicts.json).
- Spec: `specs/031-held-pair-dca/`.

**Labels.**
- NNNC late half (f08–f14): CONFIRMATORY but PROVISIONAL. It is one era, and no NNNC model covers
  2022.
- SimpleTrigger 2022 (e00–e02): CONFIRMATORY.
- Everything else: EXPLORATORY.

## The switch

`HELD_PAIR_ADD` sits on the shared entry framework and is **off by default**. Declared `{}`, it does
the following:
- On a candle whose signal candle has `enter_long == 1` for a pair holding an open long, it adds
  one stake through freqtrade position adjustment.
- The add is sized like a fresh entry: the 900 stake, the liquidity cap and the quote-volume
  floor, bounded by the wallet, with no extra slot.
- At most one add per trade, and the last fill must be at least 1 h old.
- From the add on, every age-based stop and exit rule counts from the last fill
  (`BaseStrategy.trade_clock_start`). These are grace, unclog, max hold, the minimum hold and the
  ATR take-profit's signal bar.
- The stop and take-profit are measured from the averaged entry price.

**It works as specified.**
- **Switch off:** 7 reference arms × 15 folds are byte-identical, and fingerprints and model keys
  are unchanged.
- **Smoke run:** all 10 adds were checked against the archive.
- **Fidelity on ARO early:**
  - 97.5% of the backtest's adds are screen adds, and 99.6% of screen adds are found;
  - no misaligned signal rows, and no exits on the add candle;
  - arm +26.46 pp vs screen +26.17 pp.

## Result: GFM and MLX ADOPTED; ARO, GBR and both SimpleTrigger books NOT ADOPTED

| book | screen early | arm early / late (pp) | paired | Sharpe | maxDD ctl → arm | leverage control | beats leverage | decision |
|---|---|---|---|---|---|---|---|---|
| ARO | +26.17 | +26.46 / **−0.97** | IMPROVED p 0.003 | UNCHANGED | 3.7 → 9.2 **WORSENED** | +21.65 | yes (+25.49) | **NOT ADOPTED** |
| GBR | +42.64 | +39.34 / **−7.74** | IMPROVED p 0.045 | UNCHANGED | 9.7 → 21.5 (UNCHANGED) | +19.21 | yes (+31.60) | **NOT ADOPTED** |
| GFM | +6.98 | +9.14 / +1.75 | UNCHANGED p 0.15 | UNCHANGED | 6.6 → 8.6 (UNCHANGED) | +11.06 | **no** (+10.88) | **ADOPTED** — explained by extra capital |
| MLX | +9.59 | +8.64 / +5.39 | IMPROVED p 0.006 | UNCHANGED | 2.3 → 4.5 (UNCHANGED) | +8.91 | yes (+14.03) | **ADOPTED** |

| SimpleTrigger book | 2022 (confirmatory) | 2022 maxDD ctl → arm | 2023–26 early / late (exploratory) | 2023–26 maxDD | decision |
|---|---|---|---|---|---|
| reversion (`_G24`) | **−4.74** (−6.57 → −11.32), ARM_LOSES | 17.7 → 30.5 **WORSENED** | +45.15 / **−15.56** | 7.6 → 25.1 WORSENED | **NOT ADOPTED** |
| momentum (`_G24`) | **−56.95** (−42.55 → −99.50), ARM_LOSES | 61.8 → 127.8 **WORSENED** | +24.33 / +1.06 | 24.1 → 38.7 WORSENED | **NOT ADOPTED** |

The early half reproduced the screen almost exactly. On ARO it is the fidelity run itself, and
GBR, GFM and MLX land within 3.3 pp of the screen. The late half and 2022 did not.

## What the result says

1. **DCA mostly adds exposure, not edge per unit of risk.**
   - Average committed stake rises 24–74%: ARO 171 → 252, GBR 354 → 491, ST reversion 410 → 703.
   - Peak committed stake rises 38–80%.
   - Sharpe is UNCHANGED on every book, and lower in level on ARO (2.55 → 2.22), GFM (1.72 →
     1.54) and ST reversion (2.24 → 1.79).
   - Max drawdown rises on every book: 1.3× on GFM, 1.6× on ST momentum, 2.0–2.5× on ARO, GBR and MLX, and 3.3× on ST reversion (2023–26).

   Where the base book earns, the add multiplies it, and where it loses, the add multiplies the
   loss. That is the 2022 momentum book (−42.6 → −99.5) and the late half of ARO, GBR and ST
   reversion.
2. **The beyond-leverage gain seen on the early half did not persist.**
   - The screen's "better than extra capital" result held on the early half for ARO, GBR and MLX.
   - Over 15 folds the trade gain still beats the leverage control on ARO, GBR, MLX and ST
     momentum, but on ARO and GBR all of that excess is in the early half.
   - Only MLX gains in both halves and beats leverage: +14.03 vs +8.91, paired IMPROVED p 0.006,
     with the smallest exposure rise (67 → 83 average stake, 89 adds).
3. **GFM passes the rule, but its gain equals the leverage control** (+10.88 vs +11.06). It is
   adopted under the pre-registered rule and stated as explained by extra capital.
4. **The distant era goes against the mechanism.** Both SimpleTrigger books lose in 2022.
   Reversion loses mildly (−4.7), and its control also loses there (as in 028 and the grace
   study). Momentum loses heavily (−57). By constitution II, the NNNC adoptions (GFM, MLX) rest on
   one era. 2022 offers no persistence support for the add as a general framework rule.
5. **Crowd-out is small for NNNC and large for momentum.**
   - Control entries forgone vs new arm entries: ARO 38 vs 42, GBR 121 vs 99, GFM 67 vs 31, MLX 11
     vs 4.
   - Nearly all forgone entries are pair-held: the add keeps a pair longer through the
     clock restart.
   - ST momentum re-sequences heavily (1,018 forgone, 578 new).

## Recommendation (the user's decision; the switch stays off by default)

- Do **not** enable `HELD_PAIR_ADD` framework-wide. It failed the confirmatory test on ARO and GBR
  (late), on ST reversion (2022 and late) and on ST momentum (2022). Every book's drawdown rose,
  by 1.3–3.3×.
- **MLX** is the one book where the add looks like signal rather than leverage: both halves
  positive, paired IMPROVED, beats leverage, and a small exposure increase. Its evidence is one era
  (PROVISIONAL). Enabling it there is defensible but provisional. GFM's adoption is leverage, so
  enabling it there buys exposure, not edge.

Fix-oriented next steps, if the held-pair lever is pursued:

1. **Condition the add on the trade working** (pyramiding). Add only when the held trade is in
   profit, or only after a stop has not been approached. That targets the failure here: adds into
   losing trends (2022 momentum, the late half). The screen priced only `loss`/`all`; an
   `in_profit` condition is a cheap screen on the existing tables.
2. **Size the add for equal risk.** Use a half stake, or scale by the book's drawdown, so the arm
   is judged against the control at matched drawdown instead of at 1.2–1.7× exposure. The
   leverage control then becomes the bar to beat by construction.
3. **Gate the add by regime.** Use the 026 volatility-regime instrument or the BTC regime, so adds
   happen only in the regime where the base book's edge holds (early half / 2023–24-like).
   Constitution VII applies: first check that a trailing regime read beats a constant.

## Process notes

- **Instrument audit.**
  - Fidelity passed before the pre-registration.
  - The judged chain ran entirely after `dd39f32`. The only earlier records are the exempt smoke
    run and ARO f00–f07.
  - A zsh quoting slip left ARO's late folds unrun in the first chain pass. They ran right after it
    under the same tag, still after the commit.
- **Concurrency hazard.** `wf_aro_b030` and `wf_aro_g24` share fold classes, and a running chain
  rewrites `NNNC_ARO_fNN.json` from its own snapshot. A test and a replay that read those files
  mid-chain saw 6.5 h grace; both were re-checked after the chain. See
  `specs/031-held-pair-dca/verification.md`.
- **Gate 5.** No adopted gain is under 1 pp, so no repeat run was needed.

## Tools

- `research/held_pair/build_arms.py`, `screen.py --dump-adds`, `fidelity.py`, `exposure.py`
  (crowd-out split, leverage control) and `judge.py` (refusals with exempt records, zero-add
  override);
- `utils/EntryFramework.py` `HELD_PAIR_ADD`;
- `Framework/BaseStrategy.py` `trade_clock_start` / `_fillable_stake`.
