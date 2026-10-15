# 035 fetch log (T015)

The fetch ran 2026-10-15 08:00–08:48 PDT. The data is in `user_data/data/external/` (624 MB, not
committed), with `manifest.json` beside it.

| source | files | verified | absent (404) | other |
|---|---|---|---|---|
| global_spot (spot 15m klines, 2021-01..2026-08 monthly + 2026-09-01..07 daily) | 750 | 750 | 0 | 0 |
| um_premium (UM premium-index 15m klines, same span) | 750 | 750 | 0 | 0 |
| um_metrics (UM daily metrics 2021-12-01..2026-09-07) | 17,420 | 17,418 | 2 (ZEC 2023-12-13, NEAR 2023-12-18) | 0 |
| us_spot (Binance.US REST 15m klines, from listing to 2026-09-08) | 10 pairs | 10 | — | — |

- **Binance.US pages** (1,000 bars each): ZEC 164, XRP 147, SOL 210, LINK 163, NEAR 163,
  AAVE 181, AVAX 169, LTC 235, BCH 235, DOT 168. XRP's count reflects its absence from Binance.US
  between 2021-01 and 2023-07.
- **Checksums:** every archive matched its published `.CHECKSUM` (sha256). No mismatches, and no
  refetches were needed.
- **Missing days:** the 2 absent metrics days give at most one day of NaN positioning for that
  pair, at a 60-minute staleness limit. They are flagged as missing, like the pre-2021-12 history.
