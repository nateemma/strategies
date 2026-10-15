# Control sizer: report

**Sources:**
- Pre-registered in `preregistration.json` / `.md` (commit 5e1fbfbe), before any threshold was
  picked or any arm record was made.
- Verdict: `verdicts.json` (`judge.py`), built on `instrument.json`.
- Run: 2026-10-15.

**Label: PROVISIONAL.** There is one era (2023–26), and the late half was **not blind**: the
simulator split had already been read (findings §5). Nothing is adopted or promoted. The
confirmation route is post-2026-09-07 folds or a dry-run.

## Decision: PASS (PROVISIONAL)

`cs_aro` takes the width book's own trades and sizes them with the current ARO model: 1.333× when
p_buy is above the fold's 0.5-share threshold, 0.667× otherwise.

| vs | arm pp (15 folds) | comparator pp | early gain | late gain | paired | risk (Sharpe / maxDD) | judge_one |
|---|---|---|---|---|---|---|---|
| width book `w_aro` | 59.95 | 51.44 | +4.85 | +3.66 | **IMPROVED** (p 0.0007; early IMPROVED p 0.001, late UNCHANGED p 0.086) | UNCHANGED / UNCHANGED | PASS |
| control `wf_aro_g24` | 59.95 | 49.97 | +5.42 | +4.57 | UNCHANGED (p 0.45) | UNCHANGED / UNCHANGED | PASS |

- Every half gain is above 1 pp, so the small-margin rule does not apply and no seed-2 repeat is
  needed.
- **Against the width book:** the same 930 trades, re-weighted. The gain shows on 11 of 15 folds,
  and no fold supplies more than 31% of it. Per fold, arm minus width: +0.51, +0.24, +0.13, +0.24,
  +1.17, 0.00, +0.58, +1.99 | +0.76, −0.05, −0.07, +2.65, +0.06, +0.76, −0.47.
- **Against the control:** about +10 pp on 930 trades against the control's 1,176. That is the
  best total of any ARO book so far, but the paired test cannot separate it (UNCHANGED).

## Instrument (`instrument.json`; checked before any P&L was read)

| half | entry violations | tag mismatches | thresholds at 4 dp | folds whose trade set differs from `w_aro` | mean capital vs width |
|---|---|---|---|---|---|
| early | 0 | 0 | yes | 1 (f04) | 0.994 |
| late | 0 | 0 | yes | 0 | 0.964 |

- **Capital:** mean capital is at or below the width book's in both halves, so the gain is not
  extra capital at work.
- **Stakes:** the intended agree/disagree stake ratio is 2.0. Realised, it was 1.39–2.32 on 14
  folds, and 0.86 on f10, where wallet and liquidity limits inverted it. The cap absorbs part of
  the sizing, as in 033.
- **Agree share:** 0.5 on the validation tail by construction. In the test folds the realised
  share ranged from 0.15 (f05) to 0.73 (f13).
- **Thresholds:** f00–f07 reproduce 034's control-at-share picks exactly, for example
  f00 0.7233, f02 0.744.

## What this means

- **The S1 ranking survives execution.** The control's within-book ranking, measured in 033 and
  034 as a gap of 0.072 at its own decisions and 0.100 at the share, survives the wallet, the
  liquidity cap and `max_open_trades` as return. It is +8.5 pp over the width book on identical
  trades, in both halves.
- **First measured model value over the width book.** This is the first study in which the model
  adds measurable value over the model-free width book, and it does so as a sizer, not as an entry
  source.

## Caveats

- **Late half not blind.** The simulator had already shown the late agree/disagree split. What
  was untested was whether it would survive real execution.
- **Late half paired test.** On its own the late half is UNCHANGED (p 0.086). The overall IMPROVED
  rests mostly on the early half.
- **One seed, one model.** GBR, the second model, has not been run.

## Fix-oriented follow-ups (not run)

1. **GBR repeat:** run the same arm on `NNNC_GbbRaw_Open` weights against `w_gbr` / `fs_gbr_gr`.
   It is a cheap replication on the second model.
2. **Restore the intended stake ratio:** the realised ratio is about 1.6, not 2. Lower the base
   stake or `max_open_trades` headroom so the agree side is not capped. Measure the realised ratio,
   not the intended one.
3. **Rank-weighted stakes:** stakes from 0.5 + p_buy rank, instead of a binary split.
4. **Forward confirmation:** run the arm on post-2026-09-07 data, or as a dry-run.

## Provenance

- **Arm records:** `regime/arms/cs_aro_fNN_FNN.json` (local).
- **Plan with thresholds:** `evaluation/folds/plan_cs_aro.json`.
- **Pooled results:** `evaluation/folds/results_cs_aro_seed1.json`.
- **Chain:** picks and verifies 05:27–05:40, backtests 05:41–05:44 on 2026-10-15. The first two
  launches of the backtest step failed before any record was written (working directory, then
  PYTHONPATH); `run_folds` must be run from the repo root with `PYTHONPATH=user_data/strategies`.
