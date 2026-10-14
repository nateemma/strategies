# 033 verification: label off (SC-001)

2026-10-14.

| check | evidence | result |
|---|---|---|
| Labels reaching `train_model`, no `POOLED_LABEL` | `Framework/tests/test_pooled_label.py::test_no_spec_labels_match_pre033_capture` against `fixtures/train_keyed_labels_pre033.npz`, captured at 879cab58 BEFORE `TrainingEngine.py` changed; no `%wr_keep` column added | identical |
| Existing model keys | `test_existing_model_keys_unchanged` (Framework and research/width_relative tests): `NNNC_A_Raw_Open`, `NNNC_GbbRaw_Open`, `NNNC_FS_GBR_GR`, `NNNC_CO_ARO`, `NNNC_CO_GBR`, `NNNC_ARO_f00`, `NNNC_FS_GBR_GR_f00` against `fixtures/model_keys_pre033.json`, captured before any Framework/NNNC edit; values equal 032's records | unchanged |
| Training-data build (032 capture) | `test_row_filter.py::test_no_filter_matches_pre032_capture`, 4 cases | identical |
| Re-run of a recorded fold | `wf_aro_hpon` f00 re-run as `wf_aro_hpon_p033` (a plan copy with only `run_id` and `arm_tag` changed) | trades_hash `846c306531c85348` matches the record, framework_sha `ff03db982af38fa6` matches the record, and the 48 trades match the record |
