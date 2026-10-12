# Baseline: 028 Recovery-Aware Reversion Exits (T001)

Commit at baseline: `5f1631b` (2026-10-12).

## Framework fingerprints (`evaluation.fingerprint`, `config/config_wf.json`)

| class | framework_sha |
|---|---|
| `NNNC_MLX_Open (arm D)` | `303456c1fa322b4c` |
| `SimpleTrigger_BreakoutGbb_MomExit (st_bgo_mx)` | `95dd915b90fc6217` |
| `SimpleTrigger_RevOnlyVolreg2 (st_book_rev2)` | `0e552be31732d222` |
| `SimpleTrigger_RevOnlyVolreg2_XC4` | `42b8856d4448fe86` |
| `SimpleTrigger_RevOnlyVolreg2_XC2` | `0e552be31732d222` |
| `SimpleTrigger_RevOnlyVolreg2_XC1` | `0e552be31732d222` |
| `SimpleTrigger_RevOnlyVolreg2_XC42` | `42b8856d4448fe86` |

The fingerprint cannot see method overrides: XC2 and XC1 hash like the control, and XC42 like XC4. Arms are told apart by class name (R5, FR-006).

## Per-fold `trades_hash` (2023–26)

| fold | `st_book_rev2` | `st_xpl_rev_c4` | `st_xpl_rev_c2` | `st_xpl_rev_c1` |
|---|---|---|---|---|
| f00 | 26869046bc547532 | 1d05e54af6a09e72 | 657bfcf2edb6ef6a | a87efbea97585cbc |
| f01 | 1258ca8f31ff5410 | 1239ec4945a9fa36 | 173d59c478e9fd7b | 72bbf8bad015586f |
| f02 | 388a52682d564b6a | a8a743c6a4e6220a | 009f70970e59e871 | b3ef19bef95945f2 |
| f03 | 51b4c195e25adc93 | e68f6679eecc0f9d | d51d47d2bedea17a | 8289ff66abdc4574 |
| f04 | dc4c150b919227d2 | 02241d0b41a2d0d3 | bea779ffcf2adac9 | 043fb5ed96e4bca0 |
| f05 | 2a988ad14da4f912 | caa238942ab7eda3 | b9118e4a374916c3 | 600c1c1f6b04623f |
| f06 | 9b4c5ffe469d1544 | 0c1a2f3000b03cce | 694572c018a5c2ff | b4acde2f879a381a |
| f07 | fb5cbd1d2a05662e | 1880d9cf8d5a1407 | 151b507ded27a5a5 | 90c251abd21dce3e |
| f08 | 9ea7274079158641 | 5a497297d1a42e3b | 59af3bcad75ce936 | 844c695691ad8270 |
| f09 | 1214af2bcba5ccfc | 5d33d6afc9d0ff98 | ae6666a04bbb96fd | 0f5be7b0feb430a7 |
| f10 | eea79a90d928285f | fa4ef7521f8988cb | 95282e4c277af891 | 3977a90deddc10b7 |
| f11 | 4468aefce940827f | ea342c6ca3a07fc4 | 42cee54507d4054c | 835ac5c7fe38f51b |
| f12 | 0dc9944276a3d7ae | 5d8d6ef009815466 | 82d08fa247714982 | f56a260b1bc98b21 |
| f13 | f31689c551ef1d95 | 526e66f9b5fd98a0 | ccd9ef624b742469 | a95832a354d1e0df |
| f14 | 16afd9f84e4a139d | 8646f4e4c52bfbb3 | a50ca94dbe754a84 | 4fd7e52a814b7f7a |

## 2022 freshness audit (R2)

- `regime/arms/*.json` records with a 2022 timerange: **0**
- `backtest_results` SimpleTrigger runs overlapping 2022-03-07…2023-01-01: **0**
- Any strategy's runs overlapping that window: 325 (all NN-era; the shared base exit was settled then, so 2022 is not fresh for the base exit — common to control and candidates).

**PASS:** 2022 is unseen by every SimpleTrigger arm.

## Warm-up probe (T008)

No backtest. Bars before 2022-03-07 only, from the 15m feathers; warm-up = 4096 bars (from 2022-01-23 08:00 UTC).

| pair | first bar | bars in warm-up | note |
|---|---|---|---|
| ZEC | 2022-01-13 | 4095 | full |
| LINK | 2022-01-14 | 4095 | full |
| NEAR | 2022-01-20 | 4095 | full (latest listing) |
| AAVE, AVAX, DOT, SOL, LTC, BCH | ≤ 2021-12-02 | 4095 | full |
| **XRP** | 2021-01-01 | **0** | **no binance.us data 2021-01-13 → 2023-07-14** (the SEC-era delisting). Absent from the whole 2022 window, exactly as in f00–f02 (2023-01 → 2023-07), which ran on 9 pairs. |

- `utils.RegimeState.volreg` on the truncated frames, at the last pre-start bar (2022-03-06 23:45): **0.7666, non-NaN**.
  9 pairs ≥ `MIN_PAIRS` 6.
- **PASS: e00 keeps its 2022-03-07 start.** The universe is the 10 configured pairs; 9 have data in 2022, the same 9 for control and every candidate.
