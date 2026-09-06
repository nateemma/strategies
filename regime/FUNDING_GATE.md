# Funding as a non-OHLCV feature — gated OUT before any build

The architecture and label studies both landed on the same remaining lever:
information not derivable from OHLCV. Funding was the candidate with prior
evidence in this repo (it was the only signal to survive a temporally distant
test, rejected then on horizon mismatch). Gated per Constitution VII before
spending a scaler regen and retrains.

## Blocker 1 — coverage

`user_data/data/funding/` has all 11 whitelist pairs, 8-hourly, but only from
**2024-04-01 to 2026-07-10**:

    window    span                          covered       %
    training  2022-01-01 .. 2025-08-31         517d   38.6%
    W1        2025-09-01 .. 2026-08-31         312d   85.7%
    W2        2024-06-01 .. 2025-05-31         364d  100.0%
    W3        2023-01-01 .. 2023-12-31           0d    0.0%

**W3 has ZERO coverage**, so the distant-era persistence check that Constitution
II requires could not be run at all. Training is 39% covered, meaning the
feature would be absent or degenerate for most of the fit.

That alone would argue for fetching more history rather than abandoning the
idea — so the second test decides whether the fetch is worth doing.

## Blocker 2 — no signal on the period we DO have

Among the 122,326 trigger bars that have funding data, does the funding rate
predict whether the 2% forward gain arrives?

    quintile         n   funding med   P(gain)   mean gain
    1            24466     -0.000116     56.0%       3.23%
    2            24466     -0.000001     49.1%       2.69%
    3            66939      0.000100     51.1%       2.95%
    4             6455      0.000233     58.4%       3.66%

    rank correlation funding vs forward gain : -0.0040
    rank correlation funding vs hit          : -0.0090

**No monotone relationship and effectively zero rank correlation.** The buckets
are non-monotone (56.0 / 49.1 / 51.1 / 58.4) and the extreme-to-extreme spread
is +2.4pp — against the guard trigger's own +16.2pp, which is itself only worth
26.54% of ceiling.

Caveat: funding is heavily concentrated at the 0.01%/8h base rate, so the
quintile split degenerates to four uneven buckets (66,939 of 122,326 bars in
one). That makes the quintile table coarse — but the rank correlations use the
raw values and are unambiguous at −0.004 and −0.009.

## Verdict

**Funding does not predict the gbb outcome at a 12h horizon.** This is
consistent with the earlier finding that funding lives on an 8–24h cycle and
was horizon-mismatched to gbb — but note the label horizon here is already 12h,
so the mismatch explanation no longer rescues it.

Do not fetch more funding history for THIS purpose. The data gap is real but
irrelevant: the signal is not there on the period already covered.

Funding may still matter for a slow carry strategy, where it is the edge itself
rather than a feature — that line is unaffected by this result.
