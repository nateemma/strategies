# 023 baselines

## T002 — reproducibility on HEAD before any 023 edit (2026-10-10, HEAD 8dd041e)

Run from the outer repo root with the freqtrade env loaded (`set -a; source ~/.freqtrade_env;
set +a` — without it config validation fails on `api_server.jwt_secret_key` and `run_arm`
reports "no new result file"; see memory `reference_jwt_secret_blocks_all_runs`). Command shape
copied from `evaluation/run_folds.py:evaluate_fold`.

```bash
PYTHONPATH=user_data/strategies .venv/bin/python -m regime.run_arm --strategy NNNC_ARO_f00 \
  --timerange 20230101-20230401 --label F00 --tag base023_aro \
  --config user_data/strategies/config/config_wf.json
PYTHONPATH=user_data/strategies .venv/bin/python -m regime.run_arm --strategy GbbOpen \
  --timerange 20230101-20230401 --label F00 --tag base023_gbbo \
  --config user_data/strategies/config/config_wf.json
```

| arm | recorded | rerun | trades | verdict |
|---|---|---|---|---|
| arm D f00 (`wf_aro_f00_F00`) | `5c53c833e423eae4` | `5c53c833e423eae4` | 52 / 52 | MATCH |
| `wf_gbbo` f00 (`wf_gbbo_f00_F00`) | `2d59b026e4cb9797` | `2d59b026e4cb9797` | 544 / 544 | MATCH |
