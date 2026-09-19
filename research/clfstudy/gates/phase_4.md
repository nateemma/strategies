# Gate decision: phase 4

Classification metrics against fixed gbb labels; NOT evidence of trading edge (Constitution III).

## Rule (FR-028)

- margin_floor = 0.02
- margin_used = 0.020
- std_bound = 0.03
- majority_of = 3
- std_mode = every
- ridge_phase = 2

## RIDGE reference

| window | MCC (seed-mean) | macro-F1 (seed-mean) |
|---|---|---|
| A | 0.533 | 0.656 |
| B | 0.537 | 0.654 |
| C | 0.547 | 0.666 |

pooled_seed_std_mcc = 0.003

## Per-config

| config_id | arch | A | B | C | viable |
|---|---|---|---|---|---|
| 053da6234557 | GRU_FAM | PASS | fail | fail | False |
| 0727d8e81d1b | LSTM_FAM | PASS | fail | fail | False |
| 1260944a8d42 | MLP_FAM | PASS | PASS | PASS | True |
| 128c3f1c31d8 | LSTM_FAM | PASS | fail | fail | False |
| 12af479dfaa3 | TCN_FAM | PASS | PASS | PASS | True |
| 12c3636e7110 | ROCKET | fail | fail | fail | False |
| 1804839ab883 | LSTM_FAM | PASS | fail | fail | False |
| 1f2b114246b7 | GRU_FAM | PASS | fail | fail | False |
| 248fd1539819 | LSTM_FAM | PASS | fail | fail | False |
| 265588001083 | GRU_FAM | PASS | fail | fail | False |
| 318ea175aa05 | TCN_FAM | PASS | PASS | PASS | True |
| 3413d77700b0 | TRANSFORMER_FAM | PASS | PASS | PASS | True |
| 3c120e544d4e | GRU_FAM | PASS | PASS | PASS | True |
| 486f44b1f862 | TRANSFORMER_FAM | PASS | PASS | PASS | True |
| 4a28d1ce4ebd | MLP_FAM | PASS | PASS | PASS | True |
| 4c265676afae | LSTM_FAM | PASS | PASS | PASS | True |
| 51b427ea03a3 | LSTM_FAM | fail | fail | fail | False |
| 551b8bb6dcf2 | GRU_FAM | PASS | PASS | PASS | True |
| 5616ab5d0bd0 | LSTM_FAM | PASS | PASS | PASS | True |
| 5fa2b2e33ecb | GRU_FAM | PASS | fail | fail | False |
| 67ffa1f12fbd | TRANSFORMER_FAM | PASS | PASS | PASS | True |
| 68c1f710246f | GRU_FAM | PASS | PASS | PASS | True |
| 6f0c110ce13c | MAJORITY | fail | fail | fail | False |
| 6fe7246246f5 | TRANSFORMER_FAM | PASS | PASS | PASS | True |
| 702b00fcb20c | GRU_FAM | PASS | PASS | PASS | True |
| 70eb48f9acfb | GRU_FAM | PASS | PASS | PASS | True |
| 7b50342154ef | LSTM_FAM | PASS | PASS | PASS | True |
| 821f46c115ab | TRANSFORMER_FAM | PASS | PASS | PASS | True |
| 88268c05998b | LSTM_FAM | PASS | fail | fail | False |
| 8a9bbc798cf3 | TCN_FAM | PASS | PASS | PASS | True |
| 8e1f1550ad56 | MLP_FAM | PASS | PASS | PASS | True |
| 8f8526a69864 | LSTM_FAM | fail | fail | fail | False |
| 9278e1441b47 | TRANSFORMER_FAM | PASS | PASS | PASS | True |
| 9637450c6593 | GRU_FAM | PASS | fail | fail | False |
| 9c94bdef2a2d | MLP_FAM | PASS | PASS | PASS | True |
| a09a0b24d6fe | TCN_FAM | PASS | PASS | PASS | True |
| a4fb5eb48337 | TRANSFORMER_FAM | PASS | PASS | PASS | True |
| a7457da86656 | GRU_FAM | PASS | fail | fail | False |
| b558515c6d2c | LSTM_FAM | PASS | fail | fail | False |
| b5b58a6c2cc5 | LSTM_FAM | fail | PASS | PASS | False |
| b690297cbbf5 | LSTM_FAM | PASS | PASS | PASS | True |
| b8285305a779 | MLP_FAM | PASS | PASS | PASS | True |
| b85e94f80dc6 | TCN_FAM | PASS | PASS | PASS | True |
| bbbed50c3b5c | TRANSFORMER_FAM | PASS | PASS | PASS | True |
| c1f46501f6a0 | GRU_FAM | PASS | fail | fail | False |
| c43f5e55fe61 | ROCKET | fail | fail | fail | False |
| c6a3d0fd6d2d | GRU_FAM | PASS | fail | fail | False |
| cef2eef58d6b | GRU_FAM | PASS | PASS | PASS | True |
| d1dc0339b10b | RIDGE | fail | fail | fail | False |
| d5a1481bd4c0 | TRANSFORMER_FAM | PASS | PASS | PASS | True |
| dbd824e7736c | TCN_FAM | PASS | PASS | PASS | True |
| e5974774aa5f | MLP_FAM | PASS | PASS | PASS | True |
| ea594d5a1a9e | GRU_FAM | PASS | fail | fail | False |
| ed95eafb5053 | LSTM_FAM | PASS | fail | fail | False |
| f12de4f7b017 | LSTM_FAM | PASS | fail | PASS | True |
| f5be6cc3a0df | LSTM_FAM | PASS | PASS | PASS | True |
| f7d18cb3a30f | TCN_FAM | PASS | PASS | PASS | True |
| fa4e5ad070f7 | TCN_FAM | PASS | PASS | PASS | True |
| ffb5c40e8d6b | PERSISTENCE | PASS | PASS | PASS | False |

## Viable set

- 1260944a8d42
- 12af479dfaa3
- 318ea175aa05
- 3413d77700b0
- 3c120e544d4e
- 486f44b1f862
- 4a28d1ce4ebd
- 4c265676afae
- 551b8bb6dcf2
- 5616ab5d0bd0
- 67ffa1f12fbd
- 68c1f710246f
- 6fe7246246f5
- 702b00fcb20c
- 70eb48f9acfb
- 7b50342154ef
- 821f46c115ab
- 8a9bbc798cf3
- 8e1f1550ad56
- 9278e1441b47
- 9c94bdef2a2d
- a09a0b24d6fe
- a4fb5eb48337
- b690297cbbf5
- b8285305a779
- b85e94f80dc6
- bbbed50c3b5c
- cef2eef58d6b
- d5a1481bd4c0
- dbd824e7736c
- e5974774aa5f
- f12de4f7b017
- f5be6cc3a0df
- f7d18cb3a30f
- fa4e5ad070f7

## Stability-only failures (third-seed candidates)

_none_

## Notes

- RIDGE/MAJORITY/PERSISTENCE/ROCKET are baselines: shown for reference, never in viable_set.
- margin_used = max(margin_floor, 2 x RIDGE's pooled across-seed MCC std); windows where RIDGE itself had <2 seeds are excluded from that pooled std (research R8/FR-028).
- A config with a missing, collapsed or failed window is never viable, regardless of its metrics on the other windows.
- std_mode=every: 'every' (spec FR-028(ii)) requires std_ok on EVERY window independent of the margin-majority vote; 'majority' folds std_ok into window_pass so instability can be outvoted (pre-fix behaviour, kept for comparison only).
- stability_only_failures lists configs that clear the margin on a majority of windows and have no bad status, but fail the stability bound on at least one window -- FR-004 allows these to receive a third seed before a final fail decision.

decided_at: 2026-09-19T15:28:18.544913+00:00
decided_by: gate.py + author review at checkpoint
