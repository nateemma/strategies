# Donchian ensemble screen -- results (2026-10-22)

Pre-registration `preregistration.json` (3e3a7777); code 954b346a + alignment fix 03de2d48; raw numbers
`results.json`. 10 wf pairs, daily, 30 bp round trip, every pair eligible on every scored day.

## Verdict: FAIL

| half | book | net CAGR | Sharpe | max DD | turnover/yr | mean exposure |
|---|---|---|---|---|---|---|
| early | **band (primary)** | +46.6 % | 1.25 | −27.1 % | 9.3 | 0.40 |
| early | noband | +48.3 % | 1.28 | −28.3 % | 14.8 | 0.40 |
| early | EW monthly | +143.1 % | **1.66** | −41.7 % | 2.0 | 1.00 |
| late | **band (primary)** | −3.8 % | **−0.03** | −27.9 % | 8.1 | 0.31 |
| late | noband | −8.8 % | −0.24 | −34.9 % | 13.7 | 0.30 |
| late | EW monthly | −1.9 % | 0.33 | −56.0 % | 1.7 | 1.00 |

Pooled Sharpe band 0.75 vs EW 1.02; bootstrap one-sided p = 0.87; daily return correlation with EW 0.84.
The band arm fails every condition: below EW in both halves, and negative late.

**Live basket (descriptive, freqtrade, 75 pairs, different universe):** early +252.8 % total (CAGR 89.5 %,
daily-wallet Sharpe 1.20, wallet max DD 61.3 %, 84 trades); late +79.5 % (CAGR 40.7 %, Sharpe 0.50, wallet max
DD 50.8 %, 58 trades; basket data ends 2026-08-31).

## Reading
- Trend-following on these 10 pairs is a de-risked beta book (correlation 0.84, ~0.3-0.4 exposure): it roughly
  halves EW's drawdown but gives up more return than it saves, in both halves. Sharpe never beats holding.
- The no-trade band helps as the literature says (turnover 9 vs 15 per year; +0.2 Sharpe late) but cannot
  rescue the signal.
- The live basket's edge (broad universe, 90-day ranking, BTC regime switch) is not reproduced by a
  per-pair trend rule on the 10 liquid pairs.

## Instrument note
The first run (not used) had `simulate` reading returns by row position from a table starting in 2021, so
every book earned the wrong days' returns (EW showed −24 %/yr in 2023-24 while all 10 pairs rose). Caught by
auditing EW against per-pair buy-and-hold, fixed in 03de2d48 with a regression test; the pre-registration was
not changed.
