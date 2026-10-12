# Baseline: 029 NNNC / SimpleTrigger merge (T004–T008)

Code at baseline: `928168d` (before any NNNC, SimpleTrigger or shared-mixin change; T002's report-only stop-check mode is the only edit, proven inert by T003). Baseline commit: `6c978e8`.

## Stop gate (T008)

| check | result |
|---|---|
| stop-invariant violations in arm D (probe, report mode) | **0**  |
| probe trades = arm D baseline | 15/15 ✓ |
| arm D f00 re-run = first run (determinism) | ✓ |
| twin = `st_gbb_b029` | 15/15 ✓ |
| probe check was live | `NNNC_SIP_fNN.custom_stoploss` resolves to `StopInvariantMixin` (ahead of `BaseStrategy` in the MRO); report mode records a planted violation (`utils/tests/test_stop_invariants.py`); the twin runs the same mixin in HALT mode and completed 15/15 |
| no training (SC-005) | `run_folds`: 46 × "model already present — skipping training", 0 × "training on" |
| **verdict** | **PROCEED** to the refactor |

## Arm D and the twin (per-fold `trades_hash`)

| fold | arm D baseline | arm D record (7 Oct) | twin baseline | twin record | twin = st_gbb baseline | twin = st_gbb record |
|---|---|---|---|---|---|---|
| f00 | 5c53c833e423eae4 | 5c53c833e423eae4 | aa058ba815f48627 | aa058ba815f48627 | ✓ | ✓ |
| f01 | ba30d7ee37144509 | ba30d7ee37144509 | 55166215f9ac31aa | 55166215f9ac31aa | ✓ | ✓ |
| f02 | d30b4c4d8424b9bb | d30b4c4d8424b9bb | 1f690367a7690e40 | 1f690367a7690e40 | ✓ | ✓ |
| f03 | 0fd8e37cf8cb2608 | 0fd8e37cf8cb2608 | b4d7d37b1d204f1a | b4d7d37b1d204f1a | ✓ | ✓ |
| f04 | 589a8a98faef6389 | 589a8a98faef6389 | b16ebf1dd6505067 | b16ebf1dd6505067 | ✓ | ✓ |
| f05 | d35467f2f7f51e94 | d35467f2f7f51e94 | 9419a8736467df24 | 9419a8736467df24 | ✓ | ✓ |
| f06 | d41b99b494482847 | d41b99b494482847 | 28254aea3c90a257 | 28254aea3c90a257 | ✓ | ✓ |
| f07 | 13d84f2c7a3be2b9 | 13d84f2c7a3be2b9 | 6b1a542b6682bf2f | 6b1a542b6682bf2f | ✓ | ✓ |
| f08 | aae5a44310f5e173 | aae5a44310f5e173 | 758b4d2cc4e314cf | 758b4d2cc4e314cf | ✓ | ✓ |
| f09 | 6e8b99b5da07525d | 6e8b99b5da07525d | 09e8027c786ba9d5 | 09e8027c786ba9d5 | ✓ | ✓ |
| f10 | 5d4ac5826f7f800a | 5d4ac5826f7f800a | 7c2d55c184801fcc | 7c2d55c184801fcc | ✓ | ✓ |
| f11 | 54d3293558d51208 | 54d3293558d51208 | e3053d309ffa5bc5 | e3053d309ffa5bc5 | ✓ | ✓ |
| f12 | 6918556795bf97fe | 6918556795bf97fe | 3dc89f0a8724f854 | 3dc89f0a8724f854 | ✓ | ✓ |
| f13 | 4ef2a6e041ab2d96 | 4ef2a6e041ab2d96 | e9d0bf579c58755f | e9d0bf579c58755f | ✓ | ✓ |
| f14 | c3a819f6b1316ba5 | c3a819f6b1316ba5 | 8b4d0c18696801fc | 8b4d0c18696801fc | ✓ | ✓ |

Arm D baseline vs its 7 Oct records: 15/15 equal.

## SimpleTrigger reference arms (same-day baseline vs record)

| arm | folds equal to record |
|---|---|
| `st_gbb` | 15/15 |
| `st_bgo_mx` | 15/15 |
| `st_book_rev2` | 15/15 |
| `st_reg_volreg2` | 15/15 |
| `st_coex_win4` | 15/15 |
| `st_coex_cap3` | 15/15 |
| `st_pre_volreg2` | 15/15 |
| `st_prom_c2` | 15/15 |

The after-change runs are judged against these baselines, not the records (spec FR-004).

## Other baseline facts

- Borrowed model artefacts hashed: 15 files under `saved_data/NNNC_ARO_f*/`.
- Regenerated per-fold files (`NNNC_ARO_f*`, `NNNC_TGT_f*`) vs git: unchanged.
- Fingerprint scan: 297 strategies; load errors (pre-existing) ['NNNC_DDPM_MLX_LSTM_MT', 'NNNC_DDPM_MLX_LSTM_MT_v2']; not a strategy: 15 files.

