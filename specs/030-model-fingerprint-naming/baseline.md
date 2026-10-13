# Baseline (030 T004–T008)

Baseline commit: `a62fb23`

The baseline was run on 2026-10-12, on unchanged loader code, as one chain with
`FT_MODEL_EVAL_STRICT=1` exported. The variable is inert on the old loader.
- **Chain log:** 0 × `training on`, 0 tracebacks.
- **Snapshots:**
  - `weights_before.json`: 152 files under `saved_data/`;
  - `fingerprints_before.json`: 511 strategies, 2 load errors, 16 not-a-strategy;
  - `legacy_redirects_before.json`: the definer of `get_model_path` for 260 classes.

| arm (`_b030`) | compared with | folds equal | trades |
|---|---|---|---|
| `wf_aro` (arm D) | `wf_aro_a029` | 15/15 | 1420 / 1420 |
| `sw_fmr_ctl` | `wf_fmr` | 15/15 | 946 / 946 |
| `sw_gbr_ctl` | `sw_gbr_ctl` | 15/15 | 2395 / 2395 |
| `sw_gfm_ctl` | `sw_gfm_ctl` | 15/15 | 2051 / 2051 |
| `sw_mlx_ctl` | `sw_mlx_ctl` | 15/15 | 378 / 378 |
| `wf_trgate` (twin) | `wf_trgate_b029` | 15/15 | 9056 / 9056 |
| `sw_aro_gate` | `sw_aro_gate` | 15/15 | 848 / 848 |
| `sw_mlx_c2` | `sw_mlx_c2` | 15/15 | 374 / 374 |
| `NNMT_MLX_Base` W1 | `NNMT_Ctl_s42_t07` W1 (old record, reported only) | — | 181 / 173 |
| `NNMT_Ctl_s42` W1 | `NNMT_Ctl_s42_t07` W1 (old record, reported only) | — | 179 / 173 |

The NNMT records are stale: they predate later framework fixes, which is why FR-018a requires a
same-day baseline. The two `_b030` W1 runs **are** the NNMT baseline.

**Pre-existing load failures,** recorded so they are not attributed to 030. They come from the
`legacy_redirects_before.json` scan:
- the whole Sklearn family (`utils.ClassifierKeras` is missing);
- `NNMT_CGP_MLX_MultiLSTM`, `NNNC_DDPM_MLX_LSTM_MT` and `NNNC_DDPM_MLX_LSTM_MT_v2`.
