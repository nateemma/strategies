# Feasibility: regime gate on the reversion family (026 US1)

*PROVISIONAL -- not a finding under quality gate 2 (no distant era)*

Primary (decides PASS): `st_fam_combo` reversion trades. Cross-check: `st_gbb`.

| instrument | rule (early-chosen) | on-share early | off pnl early / late | on pnl early / late | off trades e/l | off bars e/l | late longest off run | verdict | cross-check |
|---|---|---|---|---|---|---|---|---|---|
| breadth | off if lt p50 (0.3) | 52% | +1.26 / -5.34 | +22.19 / -19.36 | 1665 / 1895 | 34% / 41% | 2% | **STOP** | STOP |
| volreg | off if lt p33 (1.033) | 67% | -14.65 / -11.79 | +38.10 / -12.91 | 1143 / 1114 | 55% / 59% | 3% | **PASS** | PASS |
| persist | off if ge p67 (0.8396) | 67% | +1.07 / -8.41 | +22.37 / -16.29 | 1143 / 863 | 38% / 29% | 3% | **STOP** | STOP |
| btctrend | off if lt p50 (-0.004035) | 50% | +7.98 / -12.98 | +15.46 / -11.72 | 1731 / 1947 | 36% / 44% | 3% | **STOP** | STOP |

Failure reasons:

- `breadth`: off.early >= 0
- `volreg`: none (PASS)
- `persist`: off.early >= 0
- `btctrend`: off.early >= 0

**Overall: PROCEED to US2** (1 of 4 instruments pass).

Bucket splits ignore substitution (023 Addendum 7): a PASS only nominates an arm.
