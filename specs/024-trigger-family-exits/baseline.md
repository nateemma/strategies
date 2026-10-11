# Baseline & evidence log: 024 trigger-family exits

## T001: pre-change baseline (commit `1afcc08`, 2026-10-11)

**Neutrality reference arms** (recorded `trades_hash`, `regime/arms/*_F00.json`):

| arm f00 | trades_hash |
|---|---|
| `st_gbb` | `aa058ba815f48627` |
| `wf_aro` (arm D) | `5c53c833e423eae4` |
| `st_bbo` | `57c3b2ff32948968` |
| `stl_bo` | `61bfaa185be5164a` |

**Reproduction targets, all 15 folds.** The per-fold hashes are read from
`regime/arms/st_gbb_f*` and `regime/arms/st_bgo_mx_f*` at check time. f00:
- `st_gbb` = `aa058ba815f48627`
- `st_bgo_mx` = `010350f094a0c19e`

**`st_bgo_mx` exit_reason totals:**

| exit reason | count |
|---|---|
| trailing_stop_loss | 3352 |
| roi | 1122 |
| unclog_12h | 737 |
| metric_overbought | 641 |
| unclog_1d | 257 |
| force_exit | 46 |
| stop_loss | 1 |

**Dry-run dump:** `python evaluation/dump_strategy.py --strategy MomentumRegimeBasket15mFast` →
63 lines, sha1 prefix `0ca669b78f7ae87c`, saved in the session scratchpad. It is the same 63-line
dump 023 T032 recorded. The quickstart and T001's `--config` flag did not exist and were
corrected.

## T005 — neutrality gate: PASS

The first attempt via `run_folds --folds f00` was NOT a test. `run_folds` skips any fold whose
record already exists ("already evaluated — skipping (resumable)"), so it re-read the old
records. The arm D plan also fails outside the repo root (relative params path). The gate was
re-run with `regime.run_arm` directly under fresh `t005_*` tags, which forces real backtests.
Result files are timestamped 2026-10-11 15:49.

| arm f00 | recorded | after the C1/R2 edit | n_trades |
|---|---|---|---|
| `st_gbb` (SimpleTrigger_Gbb) | `aa058ba815f48627` | `aa058ba815f48627` ✓ | 626 |
| `wf_aro` (arm D, NNNC_ARO_f00) | `5c53c833e423eae4` | `5c53c833e423eae4` ✓ | 52 |
| `st_bbo` (SimpleTrigger_BreakoutBb) | `57c3b2ff32948968` | `57c3b2ff32948968` ✓ | 588 |
| `stl_bo` (Legacy_Breakout, static stop) | `61bfaa185be5164a` | `61bfaa185be5164a` ✓ | 714 |

The dry-run dump (`MomentumRegimeBasket15mFast`) is byte-identical to T001.

Unit tests:
- `Framework/tests`: 259 passed. The 3 failures are the known pre-existing training-pipeline
  tests.
- `test_exit_policy_single_source`: needed allowlist entries for the 10 committed 023 `_MomExit`
  configs (a 023 oversight: they set `cexit_take_profit`) and the new T002 test. All of them are
  CONFIGURING, none IMPLEMENTING.
