# HGB sizer: report

**Sources:** pre-registration `preregistration.json` / `.md` (commit 0099d1e8; tail fixes 44b3723a,
36fdf5e4, both before any threshold or record); `instrument.json`; `verdicts.json`. Run 2026-10-17.

**Label: PROVISIONAL.** One era (2023–26); the late half was not blind (037's late AUCs were known).
Nothing is adopted or promoted.

## Decision: PASS (PROVISIONAL); does NOT beat the ARO sizer

`cs_hgb` = `cs_aro` with the agree score from the 037 label:buy HGB (794 harvested columns) instead
of ARO's p_buy. Same 930 width-book trades, 1.333× / 0.667× stakes.

| vs | arm pp | comparator pp | early gain | late gain | paired | risk | judge_one |
|---|---|---|---|---|---|---|---|
| width book `w_aro` | 58.67 | 51.44 | +1.40 | +5.83 | **IMPROVED** (p 0.004) | UNCHANGED | PASS |
| control `wf_aro_g24` | 58.67 | 49.97 | +1.96 | +6.74 | UNCHANGED (p 0.51) | UNCHANGED | PASS |
| ARO sizer `cs_aro` (secondary) | 58.67 | 59.95 | −3.46 | +2.18 | UNCHANGED (p 0.53) | UNCHANGED | FAIL |

- The HGB ranks the width book's trades well enough to add +7.2 pp over the width book, so the 037
  AUC gain is real at the trade level too.
- It is no better than ARO overall (−1.3 pp, not separable): worse in the early half, better in the
  late half. The halves disagree, so neither sizer is shown better than the other.

## Instrument (no P&L read before it passed)

Both halves: 0 entry violations, 0 tag mismatches, thresholds loaded at 4 dp, every entry has an HGB
score, capital ratio vs width 0.969 early / 0.978 late; trade set differs from `w_aro` only on f04
(as cs_aro). Every fold's HGB fit ends 48 bars before the strategy's own validation tail.

## Caveats

- Late half not blind; one era; the HGB is deterministic, so no seed repeat applies.
- Live use would need all 794 columns computed at runtime (not assessed).

## Fix-oriented follow-ups (not run)

1. **Combine the two scores** (e.g. the mean rank of ARO p_buy and HGB p_buy, or agree only when
   both agree): the halves split the other way for each, so they may rank different trades well.
   Check the tag agreement share first.
2. **Fewer columns:** refit the HGB on the 037 survivors only (and without tagged price-level /
   calendar columns), which would also make live use practical.
3. **Forward window:** run cs_hgb on the post-2026-09-07 plan (`plan_fwd_*`), as cs_aro was.
