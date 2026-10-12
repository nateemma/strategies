# Quickstart: NNNC / SimpleTrigger Merge (029)

Run from the repo root with `set -a; source ~/.freqtrade_env; set +a`. Use one backtest chain at a
time.

1. **Baseline, on unchanged NNNC code** (report-only invariants are the only code added first):
   ```sh
   PYTHONPATH=user_data/strategies .venv/bin/python -m evaluation.run_folds --plan user_data/strategies/evaluation/folds/plan_wf_aro_b029.json
   ... plan_wf_aro_b029r.json      # f00 again: determinism
   ... plan_wf_trgate_b029.json    # twin
   FT_STOP_INVARIANTS=report ... plan_nnnc_si_probe.json
   .venv/bin/python user_data/strategies/scripts/nnnc_fingerprints.py --out user_data/strategies/specs/029-nnnc-simpletrigger-merge/fingerprints_before.json
   ```
   Expect:
   - the f00 re-run equals the first run;
   - the probe's trades equal arm D's, with 0 violations (otherwise STOP);
   - the twin equals `st_gbb`.
2. **Refactor.** Then run `pytest NNNC/tests SimpleTrigger/tests utils/tests evaluation Framework/tests research/outcomes/tests`.
   The only failures should be the 3 known training-pipeline goldens.
3. **Verify:**
   - run `plan_wf_aro_a029`, `plan_wf_trgate_a029` and the 8 `*_a029` SimpleTrigger plans;
   - run `nnnc_fingerprints.py --compare fingerprints_before.json`, which should exit 0;
   - write `verification.md`.
4. **Liveness:** run the `NNNC_Live_*` f00 plans, then write `liveness.md`.
5. **Report:** write `report.md`.
