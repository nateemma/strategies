# 031 baseline (T001–T002)

- Commit: 1592764, 2026-10-13.
- `baseline.json` (from `baseline.py --out`):
  - framework_sha of the 8 base classes;
  - for each of the 7 reference arms and 15 folds (105 rows): the plan's `strategy_class`, its
    framework_sha, its 030 model key (NNNC folds), and the recorded `trades_hash` / `n_trades`.

| reference arm | fold classes | trades (15 folds) |
|---|---|---|
| `wf_aro_b030` (arm D, 6.5 h snapshot) | `NNNC_ARO_fNN` | 1,420 |
| `wf_aro_g24` | `NNNC_ARO_fNN` (24 h snapshot) | 1,176 |
| `fs_gbr_gr` | `NNNC_FS_GBR_GR_fNN` | 2,021 |
| `fs_gfm_gr` | `NNNC_FS_GFM_GR_fNN` | 1,879 |
| `fs_mlx_gr` | `NNNC_FS_MLX_GR_fNN` | 360 |
| `st_bgo_mx` | `SimpleTrigger_BreakoutGbb_MomExit` | 6,156 |
| `st_prom_c2` | `SimpleTrigger_RevOnlyVolreg2_P2` | 3,347 |

Test suites (the quickstart §1 command): **816 passed, 5 skipped**.
