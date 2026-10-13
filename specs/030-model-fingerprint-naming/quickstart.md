# Quickstart: validating 030

All commands are run from `user_data/strategies` unless noted. Load the environment with
`set -a; source ~/.freqtrade_env; set +a`. Run one chain at a time.

1. **Baseline (before any code change):**
   - snapshot weight hashes:
     `../../.venv/bin/python scripts/model_index.py verify-bytes --write specs/030-model-fingerprint-naming/weights_before.json`
     (written by the baseline script if the CLI doesn't exist yet);
   - same-day runs from the repo root, with `FT_MODEL_EVAL_STRICT=1` exported:
     `PYTHONPATH=user_data/strategies .venv/bin/python -m evaluation.run_folds --plan …` with the
     plans `wf_aro_b030`, `sw_fmr_ctl_b030`, `sw_gbr_ctl_b030`, `sw_gfm_ctl_b030`,
     `sw_mlx_ctl_b030`, `wf_trgate_b030`, `sw_aro_gate_b030` and `sw_mlx_c2_b030`;
   - NNMT, from the repo root:
     `regime/run_arm.py --strategy NNMT_MLX_Base --window W1 --tag nnmt_base_b030`, and the same
     for `NNMT_Ctl_s42`;
   - fingerprints: `scripts/nnnc_fingerprints.py --out specs/030-…/fingerprints_before.json`.
2. **Migrate:** `scripts/model_index.py migrate --dry-run`, then `migrate`. Expect 79 migrated
   entries, about 58 legacy entries and 0 conflicts.
3. **Live key (report mode):** `FT_MODEL_KEY=report` on arm D f00 and on `sw_aro_gate` f00 should
   log a live key equal to the migrated key, with identical trades.
4. **Lookup:** `scripts/model_index.py key NNNC_SW_ARO_C2 --fold f03 --plan sw_aro_c2` should give
   the same hash as `key NNNC_ARO_f03`.
5. **Strict failure:** a fold class with a changed label gain under `FT_MODEL_EVAL_STRICT=1` should
   exit with `ModelMissing`, and should create no directory.
6. **After the change:** re-run step 1's arms under `_a030` tags. Compare `trades_hash` per fold (all
   equal), run `verify-bytes` (all equal), compare fingerprints (all equal), and check the chain
   logs for 0 × `training on`.
7. **Weightless families:** `scripts/model_index.py key` for every NNPredict, NNWavelet, Anomaly
   and Sklearn class should resolve. One strict-mode run per family should halt with
   `ModelMissing`.
8. **Tests:** `pytest Framework utils NNNC/tests evaluation SimpleTrigger/tests`, including the
   new audit, collision and resolution tests.
