# Gate decision: phase 3

Classification metrics against fixed gbb labels; NOT evidence of trading edge (Constitution III).

## Rule (FR-028)

- margin_floor = 0.02
- margin_used = 0.020
- std_bound = 0.03
- majority_of = 3
- std_mode = every

## RIDGE reference

| window | MCC (seed-mean) | macro-F1 (seed-mean) |
|---|---|---|
| A | 0.533 | 0.656 |
| B | 0.307 | 0.517 |
| C | 0.299 | 0.508 |

pooled_seed_std_mcc = 0.003

## Per-config

| config_id | arch | A | B | C | viable |
|---|---|---|---|---|---|
| 076666f04306 | CNN | fail | PASS | PASS | True |
| 0875fd5bd454 | LSTM_INJIT | PASS | PASS | fail | False |
| 098ca2f95a2e | LSTM2 | PASS | PASS | PASS | True |
| 0b31c7bb9404 | Multihead | PASS | PASS | PASS | True |
| 12c3636e7110 | ROCKET | fail | PASS | PASS | False |
| 225a954527c8 | LSTM_NORESIZE | PASS | fail | fail | False |
| 3a4ddb6c4092 | LSTM_NOISY | PASS | PASS | fail | False |
| 3cd40f90604c | LSTM_KAN | PASS | PASS | PASS | True |
| 3d09879bf74b | LSTM_D10 | PASS | PASS | fail | False |
| 470786990c9e | LSTM_NOCONV | PASS | PASS | fail | False |
| 48b806ceba9b | LSTM_K5 | PASS | PASS | PASS | True |
| 52292292ef8b | Wavenet | PASS | PASS | fail | False |
| 558e2f29084d | LSTM_NORES | PASS | PASS | fail | False |
| 59d72ed6d750 | LSTM_PONDER | PASS | fail | PASS | False |
| 60f40876941b | MLP | PASS | PASS | PASS | True |
| 666675f56e77 | Mamba2 | PASS | PASS | fail | False |
| 6f0c110ce13c | MAJORITY | fail | fail | fail | False |
| 8d1446f2a484 | LSTM_RAW | PASS | PASS | fail | False |
| 9c5b1e952784 | Mamba | fail | fail | fail | False |
| c43f5e55fe61 | ROCKET | fail | fail | fail | False |
| cf6b6ccb530a | Transformer | PASS | PASS | PASS | True |
| d1dc0339b10b | RIDGE | fail | fail | fail | False |
| d2bbc0d7fb15 | LSTM | PASS | PASS | fail | False |
| d7fcd89ef3d6 | LSTM_POOL | PASS | fail | PASS | False |
| e645063be7bc | LSTM_W64 | PASS | PASS | PASS | True |
| ea8d90054291 | Attention | PASS | PASS | PASS | True |
| f1f70fadabbb | TSMamba | fail | PASS | PASS | True |
| f37089f602fe | GRU | PASS | PASS | PASS | True |
| ffb5c40e8d6b | PERSISTENCE | PASS | PASS | PASS | False |

## Viable set

- 076666f04306
- 098ca2f95a2e
- 0b31c7bb9404
- 3cd40f90604c
- 48b806ceba9b
- 60f40876941b
- cf6b6ccb530a
- e645063be7bc
- ea8d90054291
- f1f70fadabbb
- f37089f602fe

## Stability-only failures (third-seed candidates)

| config_id | arch | failing windows |
|---|---|---|
| 0875fd5bd454 | LSTM_INJIT | C |
| 225a954527c8 | LSTM_NORESIZE | B, C |
| 3a4ddb6c4092 | LSTM_NOISY | C |
| 3d09879bf74b | LSTM_D10 | C |
| 470786990c9e | LSTM_NOCONV | C |
| 52292292ef8b | Wavenet | C |
| 558e2f29084d | LSTM_NORES | C |
| 59d72ed6d750 | LSTM_PONDER | B |
| 666675f56e77 | Mamba2 | C |
| 8d1446f2a484 | LSTM_RAW | C |
| d2bbc0d7fb15 | LSTM | C |
| d7fcd89ef3d6 | LSTM_POOL | B |

## Notes

- RIDGE/MAJORITY/PERSISTENCE/ROCKET are baselines: shown for reference, never in viable_set.
- margin_used = max(margin_floor, 2 x RIDGE's pooled across-seed MCC std); windows where RIDGE itself had <2 seeds are excluded from that pooled std (research R8/FR-028).
- A config with a missing, collapsed or failed window is never viable, regardless of its metrics on the other windows.
- std_mode=every: 'every' (spec FR-028(ii)) requires std_ok on EVERY window independent of the margin-majority vote; 'majority' folds std_ok into window_pass so instability can be outvoted (pre-fix behaviour, kept for comparison only).
- stability_only_failures lists configs that clear the margin on a majority of windows and have no bad status, but fail the stability bound on at least one window -- FR-004 allows these to receive a third seed before a final fail decision.

decided_at: 2026-09-17T11:20:28.904180+00:00
decided_by: gate.py + author review at checkpoint
