# 040 NNMT parity & head study -- report

## 1. Baseline: NNMT_MLX as it stands (`mt`, seed 1, T012)

15/15 folds recorded (`regime/arms/mt_fNN_FNN.json`), each with a trained model and trades.
Key a97a4b9cdde4b5a5 (f00, after the T014 key hygiene); trained BEFORE the framework port.

| arm | early f00-f07 | late f08-f14 | total | trades |
|---|---|---|---|---|
| mt | +12.43 | +8.57 | +21.00 | 795 |
| cs_aro | +40.70 | +19.25 | +59.95 | 930 |
| w_aro | +35.85 | +15.59 | +51.44 | 930 |
| st_g24_rev | | | 71.5 | |

Per fold, wallet profit % (trades):

| fold | mt | cs_aro | w_aro |
|---|---|---|---|
| f00 | +1.07 (40) | +4.27 (27) | +3.76 (27) |
| f01 | -3.91 (26) | +3.96 (25) | +3.72 (25) |
| f02 | +0.98 (22) | +1.24 (10) | +1.11 (10) |
| f03 | +2.77 (44) | +1.03 (16) | +0.79 (16) |
| f04 | +3.61 (98) | +11.95 (128) | +10.78 (128) |
| f05 | +1.42 (56) | +0.89 (60) | +0.89 (60) |
| f06 | -1.02 (44) | +3.77 (46) | +3.19 (46) |
| f07 | +7.50 (115) | +13.59 (99) | +11.60 (99) |
| f08 | -0.71 (144) | +3.39 (147) | +2.63 (147) |
| f09 | +0.65 (19) | +2.65 (39) | +2.70 (39) |
| f10 | -0.56 (17) | -0.01 (3) | +0.06 (3) |
| f11 | +4.41 (96) | +6.38 (208) | +3.73 (208) |
| f12 | +1.93 (42) | +2.54 (39) | +2.48 (39) |
| f13 | +2.29 (24) | +2.43 (37) | +1.67 (37) |
| f14 | +0.56 (8) | +1.86 (46) | +2.32 (46) |

Descriptive only (no rule is applied to the baseline): NNMT trails cs_aro by ~39 pp, in both halves;
most of the gap is f04/f07/f08, where all three books trade most.

## 2. Framework port is inert (T017-T018)

`mt_fw0` (NNMT with EntryFrameworkMixin, every switch off, same model key -> same weights) is
trade-identical to `mt` on 15/15 folds (`trades_hash`; `verdicts_models.json["identity"]`). Run with
`FT_STOP_INVARIANTS=report` and a sink: 0 stop-invariant violations on all 15 folds.
