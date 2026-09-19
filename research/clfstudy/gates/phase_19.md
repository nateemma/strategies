# Gate decision: phase 19

Classification metrics against fixed gbb labels; NOT evidence of trading edge (Constitution III).

## Rule (FR-028)

- margin_floor = 0.02
- margin_used = 0.020
- std_bound = 0.03
- majority_of = 3
- std_mode = every
- ridge_phase = 19

## RIDGE reference

| window | MCC (seed-mean) | macro-F1 (seed-mean) |
|---|---|---|
| A | 0.202 | 0.437 |
| B | 0.102 | 0.397 |
| C | 0.179 | 0.433 |

pooled_seed_std_mcc = 0.001

## Per-config

| config_id | arch | A | B | C | viable |
|---|---|---|---|---|---|
| 01596109699e | LSTM_FAM | fail | fail | fail | False |
| 1e9e0b212d86 | TRANSFORMER_FAM | fail | fail | fail | False |
| 467d23581f7b | Wavenet | fail | fail | fail | False |
| 5a7196f85699 | PERSISTENCE | PASS | PASS | PASS | False |
| ae29f8c1fcd8 | TCN_FAM | fail | fail | fail | False |
| bbae4b588c23 | MAJORITY | fail | fail | fail | False |
| c1aa32b8dcea | MLP_FAM | fail | fail | fail | False |
| e700753b5111 | RIDGE | fail | fail | fail | False |

## Viable set

_none viable_

## Stability-only failures (third-seed candidates)

_none_

## Notes

- RIDGE/MAJORITY/PERSISTENCE/ROCKET are baselines: shown for reference, never in viable_set.
- margin_used = max(margin_floor, 2 x RIDGE's pooled across-seed MCC std); windows where RIDGE itself had <2 seeds are excluded from that pooled std (research R8/FR-028).
- A config with a missing, collapsed or failed window is never viable, regardless of its metrics on the other windows.
- std_mode=every: 'every' (spec FR-028(ii)) requires std_ok on EVERY window independent of the margin-majority vote; 'majority' folds std_ok into window_pass so instability can be outvoted (pre-fix behaviour, kept for comparison only).
- stability_only_failures lists configs that clear the margin on a majority of windows and have no bad status, but fail the stability bound on at least one window -- FR-004 allows these to receive a third seed before a final fail decision.

decided_at: 2026-09-19T17:46:44.047027+00:00
decided_by: gate.py + author review at checkpoint
