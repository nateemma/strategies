# Baseline: 027 Target-Aware Exits (T001)

Commit at baseline: `324ce0e` (2026-10-12).

## Framework fingerprints (`evaluation.fingerprint`, `config/config_wf.json`)

| class | framework_sha |
|---|---|
| `NNNC_MLX_Open` (arm D) | `303456c1fa322b4c` |
| `SimpleTrigger_Gbb` | `303456c1fa322b4c` |
| `SimpleTrigger_BreakoutGbb_MomExit` (`st_bgo_mx`) | `95dd915b90fc6217` |
| `SimpleTrigger_RevOnlyVolreg2` (`st_book_rev2`) | `0e552be31732d222` |
| `SimpleTrigger_FamCombined` | `27617f430beca1cd` |
| `SimpleTrigger_RegGateVolreg2` | `f1492b64a9dd9257` |

## Per-fold `trades_hash`

| fold | `st_bgo_mx` | `st_book_rev2` |
|---|---|---|
| f00 | 010350f094a0c19e | 26869046bc547532 |
| f01 | 166b3f0ab21d4c70 | 1258ca8f31ff5410 |
| f02 | 0ef46f15159a3827 | 388a52682d564b6a |
| f03 | f4c1fed61d2dac4a | 51b4c195e25adc93 |
| f04 | 87ac89707f99ac67 | dc4c150b919227d2 |
| f05 | a598024b1e2f51e2 | 2a988ad14da4f912 |
| f06 | 995105814fba6b56 | 9b4c5ffe469d1544 |
| f07 | 2d362bc9aad94118 | fb5cbd1d2a05662e |
| f08 | 690f671a476915e1 | 9ea7274079158641 |
| f09 | fe5d363c38d15a99 | 1214af2bcba5ccfc |
| f10 | 5b476d708128a73c | eea79a90d928285f |
| f11 | caeccb48b8bd96f6 | 4468aefce940827f |
| f12 | 040f3a51b7c426af | 0dc9944276a3d7ae |
| f13 | 0923a8f75b883640 | f31689c551ef1d95 |
| f14 | ec67b3f61fc22390 | 16afd9f84e4a139d |

## Resolved exit values

Read from `class_level_exit_params` + `sell_params` + the family sets, not copied from documents.

| value | `st_bgo_mx` (momentum) | `st_book_rev2` (reversion) |
|---|---|---|
| grace | 6.5 h at −0.27 | same |
| ATR stop | −4 × atr_pct_roll in [−0.06, −0.04], volume confirmation on | same |
| take-profit (net) | 10.0 (off) | 0.022 |
| unclog_12h | deferred to 24 h (`HOLD_WINDOW_HOURS` 24) | 12 h, > 0.5% |
| unclog_1d / max hold | 24 h ≥ 0 / 10 days | same |
| metric_overbought | guard_metric > 0.98 while in profit | same |
| ROI | `minimal_roi` {"0": 0.05}, `ignore_roi_if_entry_signal` | same |

`st_book_rev2`'s momentum family set (take-profit 10, unclog 24 h) is unused: `breakout_gbb` is disabled
in that class, so every trade is reversion.
