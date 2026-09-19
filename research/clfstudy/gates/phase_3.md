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
| B | 0.535 | 0.652 |
| C | 0.547 | 0.666 |

pooled_seed_std_mcc = 0.002

## Per-config

| config_id | arch | A | B | C | viable |
|---|---|---|---|---|---|
| 076666f04306 | CNN | fail | fail | fail | False |
| 0875fd5bd454 | LSTM_INJIT | PASS | fail | fail | False |
| 098ca2f95a2e | LSTM2 | PASS | PASS | PASS | True |
| 0b31c7bb9404 | Multihead | PASS | PASS | PASS | True |
| 12c3636e7110 | ROCKET | fail | fail | fail | False |
| 225a954527c8 | LSTM_NORESIZE | PASS | fail | fail | False |
| 3a4ddb6c4092 | LSTM_NOISY | PASS | fail | fail | False |
| 3cd40f90604c | LSTM_KAN | PASS | fail | fail | False |
| 3d09879bf74b | LSTM_D10 | PASS | fail | fail | False |
| 470786990c9e | LSTM_NOCONV | PASS | fail | fail | False |
| 48b806ceba9b | LSTM_K5 | PASS | fail | fail | False |
| 52292292ef8b | Wavenet | PASS | PASS | PASS | True |
| 558e2f29084d | LSTM_NORES | PASS | fail | fail | False |
| 59d72ed6d750 | LSTM_PONDER | PASS | fail | fail | False |
| 60f40876941b | MLP | PASS | PASS | PASS | True |
| 666675f56e77 | Mamba2 | PASS | PASS | PASS | True |
| 6f0c110ce13c | MAJORITY | fail | fail | fail | False |
| 8d1446f2a484 | LSTM_RAW | PASS | fail | fail | False |
| 9c5b1e952784 | Mamba | fail | fail | fail | False |
| c43f5e55fe61 | ROCKET | fail | fail | fail | False |
| cf6b6ccb530a | Transformer | PASS | PASS | PASS | True |
| d1dc0339b10b | RIDGE | fail | fail | fail | False |
| d2bbc0d7fb15 | LSTM | PASS | PASS | PASS | True |
| d7fcd89ef3d6 | LSTM_POOL | PASS | fail | fail | False |
| e645063be7bc | LSTM_W64 | PASS | fail | fail | False |
| ea8d90054291 | Attention | PASS | PASS | PASS | True |
| f1f70fadabbb | TSMamba | fail | PASS | PASS | True |
| f37089f602fe | GRU | PASS | PASS | PASS | True |
| ffb5c40e8d6b | PERSISTENCE | PASS | PASS | PASS | False |

## Viable set

- 098ca2f95a2e
- 0b31c7bb9404
- 52292292ef8b
- 60f40876941b
- 666675f56e77
- cf6b6ccb530a
- d2bbc0d7fb15
- ea8d90054291
- f1f70fadabbb
- f37089f602fe

## Stability-only failures (third-seed candidates)

| config_id | arch | failing windows |
|---|---|---|
| 225a954527c8 | LSTM_NORESIZE | B, C |

## Notes

- RIDGE/MAJORITY/PERSISTENCE/ROCKET are baselines: shown for reference, never in viable_set.
- margin_used = max(margin_floor, 2 x RIDGE's pooled across-seed MCC std); windows where RIDGE itself had <2 seeds are excluded from that pooled std (research R8/FR-028).
- A config with a missing, collapsed or failed window is never viable, regardless of its metrics on the other windows.
- std_mode=every: 'every' (spec FR-028(ii)) requires std_ok on EVERY window independent of the margin-majority vote; 'majority' folds std_ok into window_pass so instability can be outvoted (pre-fix behaviour, kept for comparison only).
- stability_only_failures lists configs that clear the margin on a majority of windows and have no bad status, but fail the stability bound on at least one window -- FR-004 allows these to receive a third seed before a final fail decision.

decided_at: 2026-09-19T14:30:44.891920+00:00
decided_by: gate.py + author review at checkpoint

## Author decision (2026-09-19 07:35): stop the B/C re-run after 11 of 23 types

With the learning-free trigger rule established as the ceiling and every completed type at
it (Transformer, LSTM, LSTM2, Multihead, GRU, Wavenet within 0.003 of each other; Mamba2, MLP
0.01 below; Attention, TSMamba, CNN structurally below), the window-B/C rows of the twelve
LSTM ablation variants were not re-run. Their clean window-A rows stand; their B/C rows are
retracted and reported as "not re-run — ceiling established". Mamba: A/B only.
