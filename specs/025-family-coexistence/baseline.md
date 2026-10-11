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

## T014–T016 — coexistence stage

| arm | trades | per-fold sum | positive folds | early | late | liveness |
|---|---|---|---|---|---|---|
| `st_fam_combo` (024, previous) | 11,727 | +15.39 | 8/15 | +35.76 | −20.37 | — |
| `st_coex_cap3` | 8,128 | +12.16 | 9/15 | +16.62 | **−4.45** | 24,204 pair-cap refusals |
| `st_coex_cap5` | 10,253 | +1.11 | 7/15 | +22.72 | −21.61 | 9,312 |
| `st_coex_cap7` | 11,353 | +9.11 | 6/15 | +34.33 | −25.22 | 2,630 |
| `st_coex_win4` | 11,605 | **+19.24** | 7/15 | +33.24 | −14.00 | 21,956 recency-dropped rows |
| `st_coex_win16` | 11,046 | +8.18 | 7/15 | +27.10 | −18.92 | 74,099 |
| `st_coex_win48` | 9,272 | +4.50 | 7/15 | +23.19 | −18.68 | 128,017 |

The hook was confirmed live in backtests first. `st_coex_cap3` f00 alone logged 2,374 refusals
before the chain ran (G5).

Entries lost to the other family's pair (early / late):

| arm | reversion | momentum |
|---|---|---|
| 024 combo | 1,939 / 1,319 | 720 / 532 |
| cap3 | 1,363 / 970 | 465 / 399 |
| cap5 | 1,702 / 1,202 | 618 / 477 |
| cap7 | 1,863 / 1,296 | 688 / 515 |
| win4 | 1,995 / 1,369 | 605 / 463 |
| win16 | 2,005 / 1,374 | 564 / 417 |
| win48 | 2,077 / 1,437 | 283 / 216 |

**Decision** (`decisions.md`):
- The early winner is `st_coex_cap7` (+34.33).
- Its late verdict against `st_fam_combo` is INCONCLUSIVE_ARM_LOSES (−4.85pp, CI [−11.3, +5.4],
  p 0.50). **Not adopted.**
- No candidate beat `st_fam_combo`'s early sum (+35.76).
- `BASE_ARM` = `st_fam_combo` (class `SimpleTrigger_FamCombined`).
- Post-hoc only, NOT a decision: cap3's late half (−4.45) and win4's total (+19.24) look better,
  but choosing them would read the late folds.

## T017–T020 — tuning stages (each on `st_fam_combo`, none adopted)

See `decisions.md` for the stage blocks and `report.md` for the all-arms table. The resolved
reversion settings were checked before every step (G1). Each candidate carried only its own new
key, because no earlier step was adopted.

## T024 — 023/024 safety re-check: PASS

Fresh `run_arm` f00 re-runs under the 025 MRO (`FamilyCoexistenceMixin` first), with new tags and
result files timestamped 2026-10-11 18:54:

| arm f00 | recorded | now |
|---|---|---|
| `st_fam_combo` (024) | `f9e7a4531e21cce3` | `f9e7a4531e21cce3` ✓ |
| `st_gbb` (023 twin) | `aa058ba815f48627` | `aa058ba815f48627` ✓ |
| `stl_bo` (023 legacy preset) | `61bfaa185be5164a` | `61bfaa185be5164a` ✓ |

Fingerprints are unchanged: arm D and `SimpleTrigger_Gbb` are `303456c1fa322b4c`;
`SimpleTrigger_FamCombined` is `27617f430beca1cd`.
