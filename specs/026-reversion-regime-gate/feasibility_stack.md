# Feasibility: regime gate on the reversion family (026 US1)

*PROVISIONAL -- not a finding under quality gate 2 (no distant era)*

Primary (decides PASS): `st_reg_volreg` reversion trades. Cross-check: `st_gbb`.

Stacked on `volreg:lt:1.033196702052918`: only trades and bars where that gate is ON. Kept 4422/4424 primary and 5942/9056 cross-check trades.

| instrument | rule (early-chosen) | on-share early | off pnl early / late | on pnl early / late | off trades e/l | off bars e/l | late longest off run | verdict | cross-check |
|---|---|---|---|---|---|---|---|---|---|
| breadth | off if lt p33 (0.1) | 68% | +7.67 / +1.75 | +26.01 / -10.95 | 777 / 671 | 10% / 8% | 1% | **STOP** | STOP |
| volreg | off if lt p50 (1.366) | 50% | -4.10 / -14.07 | +37.78 / +4.88 | 1221 / 1125 | 26% / 26% | 0% | **PASS** | STOP |
| persist | off if ge p67 (0.8306) | 67% | +5.87 / +2.03 | +27.81 / -11.23 | 810 / 657 | 15% / 13% | 1% | **STOP** | STOP |
| btctrend | off if lt p50 (-0.006531) | 50% | +4.79 / -8.87 | +28.89 / -0.33 | 1226 / 1315 | 17% / 20% | 2% | **STOP** | STOP |

Failure reasons:

- `breadth`: off.early >= 0, off.late >= 0
- `volreg`: none (PASS)
- `persist`: off.early >= 0, off.late >= 0
- `btctrend`: off.early >= 0

**Overall: PROCEED to US2** (1 of 4 instruments pass).

Bucket splits ignore substitution (023 Addendum 7): a PASS only nominates an arm.
