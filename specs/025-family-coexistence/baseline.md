# Baseline & evidence log: 025 family coexistence

## T001: pre-change baseline (commit `4d3f2b2`, 2026-10-11)

**Reproduction target:** `st_fam_combo` (024), 15 folds.

```
sha 4d3f2b2
st_fam_combo trades_hash: ['f9e7a4531e21cce3', 'f76e93b35c2c814c', 'ae5f119515b387ca', 'f56a183573aec0b8', 'ff9ba61bbbacaf1b', 'f8b903fbb5fb0bde', '2d8828752ebbc71c', '513445f367ffcb67', 'd0b0f41ddc7e720f', 'b447f987110b7b3a', '42f1b6dd9d79d99f', '5655763187070a4d', '9979cd10228a0ede', 'db2fd5393451066b', '4e206c2c1479c772']
family_trade_counts: {'momentum': 5327, 'reversion': 6400}
family_exit_reason_counts: {'momentum': {'force_exit': 44, 'metric_overbought': 638, 'roi': 824, 'stop_loss': 1, 'trailing_stop_loss': 2918, 'unclog_12h': 662, 'unclog_1d': 240}, 'reversion': {'roi': 20, 'take_profit': 2699, 'trailing_stop_loss': 2485, 'unclog_12h': 1094, 'unclog_1d': 60, 'force_exit': 20, 'metric_overbought': 11, 'stop_loss': 11}}
fingerprint NNNC_A_Raw_Open 303456c1fa322b4c
fingerprint SimpleTrigger_Gbb 303456c1fa322b4c
fingerprint SimpleTrigger_FamCombined 27617f430beca1cd
```

The contention table is in `evaluation/contention_st_fam_combo.md`. Entries lost to the other
family: reversion 1,939 / 1,319 (early / late), momentum 720 / 532.

T024's 023 reference hashes, f00: `st_gbb` `aa058ba815f48627`, `stl_bo` `61bfaa185be5164a`.

## T012 — SC-001 neutral reproduction: PASS

| arm | vs `st_fam_combo` | untagged | refusals logged |
|---|---|---|---|
| `st_coex_capoff` (FAMILY_PAIR_CAP = None) | **15/15** identical | 0 | none |
| `st_coex_winoff` (MOMENTUM_RECENCY_BARS = 0) | **15/15** identical | 0 | none |

The new MRO (`FamilyCoexistenceMixin` first), the rule B code path and the MRO-merged overrides are
behaviour-neutral at the neutral settings.
