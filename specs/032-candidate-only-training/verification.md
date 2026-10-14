# 032 verification: switch off (SC-001)

2026-10-13.

| check | evidence | result |
|---|---|---|
| Training-data build, no filter | `Framework/tests/test_row_filter.py::test_no_filter_matches_pre032_capture`, 4 cases (tensor_method 0 and 3, with and without `%pnl_weight`) against `fixtures/prepare_training_data_pre032.npz`, captured at eb72061 BEFORE `TrainingEngine.py` changed | identical |
| Existing model keys | `test_existing_model_keys_unchanged`: `NNNC_A_Raw_Open`, `NNNC_GbbRaw_Open`, `NNNC_FS_GBR_GR`, `NNNC_ARO_f00`, `NNNC_FS_GBR_GR_f00` against `fixtures/model_keys_pre032.json`, captured before any Framework edit | unchanged |
| Pinned fingerprints | the full suite (909 passed, 7 skipped), incl. arm D `ff03db982af38fa6` and add off at 6.5 h `303456c1fa322b4c` | unchanged |
| Re-run of a recorded fold | `wf_aro_hpon` f00 re-run as `wf_aro_hpon_p032` (plan copy, only run_id and arm_tag changed) | trades_hash `846c306531c85348` = record; framework_sha `ff03db982af38fa6` = record; 48 trades and 4 adds = record |

**Reference substitution.** T011 named `wf_aro_g24` f00. That record predates the in-profit
held-pair add becoming the NNNC default (e3a32dc). Re-running it today would differ for that reason
alone, not because of 032. `wf_aro_hpon` is the same model and fold recorded on the current defaults,
with model-driven entries, so it is the faithful spot check.
