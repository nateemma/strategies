# Quickstart: Family Coexistence and Reversion Late-Half Fix (025)

From `user_data/strategies`. Backtests need `set -a; source ~/.freqtrade_env; set +a` and run as
one sequential chain. Every result is PROVISIONAL (gate 2).

## 0. Neutral reproduction (SC-001)

Two configs: `SimpleTrigger_CoexCapOff` (`FAMILY_PAIR_CAP = None`) and `SimpleTrigger_CoexWinOff`
(`MOMENTUM_RECENCY_BARS = 0`). They must be explicit neutral classes that inherit
`SimpleTrigger_FamCombined`.

Derive and run their plans (`foldplan derive --from evaluation/folds/plan_NNNC_A_Raw_Open.json`,
then `run_folds`). **Expected:** 15/15 folds have a `trades_hash` equal to `st_fam_combo`'s.

## 1. Coexistence stage

Six arms: `st_coex_cap3/5/7` and `st_coex_win4/16/48`. Then:

```bash
../../.venv/bin/python -m evaluation.stage --stage coexistence --previous st_fam_combo \
  --candidates st_coex_cap3,st_coex_cap5,st_coex_cap7,st_coex_win4,st_coex_win16,st_coex_win48 \
  --singles reversion=st_gbb,momentum=st_bgo_mx --out specs/025-family-coexistence/decisions.md
```

**Expected:** every active arm has non-zero `coexistence_refusals` or `coexist_recency_refused`.
The decision block names the early winner and its late verdict.

## 2. Tuning stages 1–4 (inside the combined arm with the adopted rule, or `st_fam_combo` if none)

Each step's candidate configs subclass the previous step's arm and set
`FAMILY_EXIT_OVERRIDES = {"reversion": {...}}`. Run the step's arms, then `evaluation.stage
--stage tuneN --previous <previous arm>`.

**Expected:** the governed reversion exit-reason counts move in every candidate, or it is INERT.

## 3. Final report (US3)

`specs/025-family-coexistence/report.md`: the final arm against `st_bgo_mx`, `st_gbb`,
`st_fam_combo` and the sum of singles, by half. It gives paired verdicts, per-family
contributions, contention and correlation, all labelled PROVISIONAL.
