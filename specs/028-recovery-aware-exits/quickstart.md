# Quickstart: Recovery-Aware Reversion Exits (028)

Run from the repo root with `set -a; source ~/.freqtrade_env; set +a`. Use one backtest chain at a
time.

1. **Setup (before pre-registration; touches 2023–26 only)**
   ```sh
   .venv/bin/python -m evaluation.run_folds --plan user_data/strategies/evaluation/folds/plan_st_xpl_rev_c42.json
   ```
   Expect 15/15 records, with trades different from `st_book_rev2` in most folds. Then:
   - fill `early_gain_pp` / `late_gain_pp` for C42 in `preregistration.json`;
   - run the warm-up probe (R1);
   - run `run_folds --dry-run` on each 2022 plan.
2. **Pre-register**: commit `preregistration.{md,json}` and the five `plan_st_e22_*.json` together.
3. **Control on 2022**: `run_folds --plan .../plan_st_e22_rev2.json`, which gives 3 records.
4. **US1**: `python -m research.exit_coupling.confirm feasibility ...`.
   - Expect fidelity ≥ 0.90.
   - Exit 3 stops the feature: write the report.
5. **US2**: run each PASSING candidate's 2022 plan, then `confirm judge ...`.
6. **US3** (only for adopted C1/C2): implement the switches. Then check:
   - `pytest utils/tests Framework/tests research/exit_coupling/tests evaluation` passes;
   - with the switch off, the reference fingerprints are unchanged;
   - with it on, the trades match the XC class on f00–f14 and e00–e02.
7. **Report**: `report.md`, with each candidate's 2023–26 (EXPLORATORY) and 2022 (CONFIRMATORY,
   PROVISIONAL) results side by side.
