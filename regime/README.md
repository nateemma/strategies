# Market-Phase Framework and Phase-Adaptive Entry Gate

Study for spec-kit feature `001-market-phase-entry-gate`. See
`specs/001-market-phase-entry-gate/` for the spec, plan, and task list.

Extends the `btc_corr/` regime work (Exp 4) into a three-level phase framework —
asset, market breadth, and BTC — then asks one question: does a phase label
predict which of two entry-aggressiveness settings wins the coming period,
better than always using the better fixed setting?

## Evaluation windows (pinned; research R3)

| Window | Range | Role |
|---|---|---|
| W1 | `20250901-20260831` | Deployment regime — **carries the headline verdict** |
| W2 | `20240601-20250531` | Near-era confirmation |
| W3 | `20230101-20231231` | Temporally distant era |

W1 and W2 are only three months apart and share a regime; they cannot establish
persistence between them. W3 is the window that can.

## Running

All scripts run from the freqtrade root — the system Python has no pandas:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/<script>.py

## Results

_Pending._ Phase A (classify) not yet run.

## Phase A observations (T007-T009, 2026-09-02)

Preliminary distributions from running the instruments on real data. These are
observations, not verdicts - Phase B does the formal characterisation.

**Alts are strongly mean-reverting at these horizons.** `asset_vr24` over full
history: Reverting 56.4% (ETH), 67.2% (SOL), 71.7% (LINK); Trending only
18.5% / 10.8% / 10.7%. This corroborates the project's accumulated evidence that
dip-buying labels work on these pairs while breakout labels do not, and it is
measured here independently of any model.

**`asset_hurst` "Persistent" is a rare state and may not survive Phase B.**
Full-history share: ETH 6.3%, SOL 3.5%, LINK 4.3%. Two of three are already
below the 5% minimum-state-share rule, so that state would be excluded from
verdict-driving comparisons on those pairs. A switch keyed on "be aggressive
when trending" would fire rarely. Flagged now because it bears directly on
actionability - see T015.

**`btc_exp4` at 15m is more Sideways-heavy than the original 1h study**
(35.4% vs 29.8%), with the other four states correspondingly thinner. Expected:
ADX is noisier on 15m bars, and the span here is 5.7 years rather than 2.2. The
taxonomy reproduces bit-identically on the original 1h data, so this is a
timeframe effect and not a transcription difference.

**Defect found and fixed in `market_correlation`.** One bar (2026-08-31 03:45)
had every one of 75 pairs unchanged, so the cross-sectional denominator was
exactly zero and the ratio went NaN - which then propagated through the whole
96-bar averaging window and produced 18 UNKNOWN labels *inside W1*. Fixed with
`min_periods` on the rolling mean and quantiles, so an isolated dead bar is
skipped rather than blanking a day. All three market instruments now emit zero
UNKNOWN inside W1/W2/W3, which is the rule working as designed: UNKNOWN inside
an evaluation window is a defect to investigate, never a state to interpret.

## Universe scoping (T010) — a design correction

Running the classifier over every feather on disk surfaced a scoping error in
the original plan.

**The host trades 11 pairs, not 77.** `config/config.json` uses a StaticPairList
of ZEC, XRP, SOL, LINK, NEAR, AAVE, SUI, AVAX, LTC, BCH, DOT. Asset-level
instruments are only meaningful there, since Phase C aligns phase labels to the
host's realized trades and those occur nowhere else. Classifying all 77 was
wasted work that also polluted the defect check.

**Most of the wider universe is not tradeable on this venue.** Of the 77 pairs
with 15m data, 61 have >50% zero-volume bars in W1 and 16 exceed 90%. OCEAN is
97.1% zero-volume in W1 with a 3,078-bar (32-day) flat run; KAITO and SKL are
similar. These carry stale forward-filled prices, and 480 consecutive flat bars
drive rolling variance to zero, making the estimators genuinely undefined — the
UNKNOWN labels were correct, not a bug. **All 16 problem pairs are outside the
traded universe; the 11 traded pairs produce zero UNKNOWN.**

Consequences, both applied:

- Market-level instruments are built from a LIQUID universe (29 pairs, ≤80%
  zero-volume bars in *all three* windows) rather than all 77. Breadth over
  mostly-dead pairs measures listing artifacts, not market state. The filter is
  applied across all three windows so the universe is FIXED across eras — a
  universe that changed definition per window would manufacture instability and
  corrupt the Principle II comparison. The 29 are a superset of the traded 11.
- Asset-level classification defaults to the traded universe.

**Second defect found and fixed**: the final bar of the dataset
(2026-08-31 17:00) is a PARTIAL candle — the download stopped mid-bar, so only
17 of 29 pairs had posted where every preceding bar had all 29. That produced a
spurious low-coverage UNKNOWN at the right-hand edge of W1. Trailing incomplete
bars are now trimmed from the panel.

Final state: **all 11 instruments emit zero UNKNOWN inside W1/W2/W3.**
