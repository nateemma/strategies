# Methods notes

## Label fidelity (T026)

`PYTHONPATH=. .venv/bin/python scripts/check_label_fidelity.py --pair LINK/USDT` → `IDENTICAL (n=163666)`, exit 0 (2026-09-16). The harness label shim reproduces the StrategyResolver-loaded NNNC_MLX `get_training_labels` output byte-for-byte on the full LINK/USDT history.

## ROCKET baseline: seed-instability defect and fix (2026-09-16)

Observed on the pooled data: 4 of 9 phase-2 ROCKET runs (windows B seeds 1-2, C seeds 1-2) never
improved on their epoch-0 validation MCC and early-stopped into all-HOLD or all-BUY (best_epoch 0).
RIDGE on the identical training loop was stable on every window and seed, so the loop was not at fault.

Diagnosis (LINK/USDT, 100k training windows, untrained model, CPU): PPV features are well behaved
(mean 0.50, per-feature std 0.14, bounded [0,1]); MAX-pooled features are heavy-tailed (per-feature
std ~1.0, batch maxima 41 / 60 / 52 for seeds 0/1/2). Through the random head this gives a wildly
seed-dependent initial prediction: seed 0 predicted SELL for 77% of rows, seed 1 BUY for 99.96%,
seed 2 SELL for 95%. From such a start the shared loop (lr 1e-4, focal loss, patience 10) could not
recover before early stopping fired. Also noted: at seq_len 16 with kernel length 9 every kernel has
dilation 1, so the "multi-scale" property of MiniRocket is absent at the production window length.

Fix (NNNClassifierMLX_Rocket, hp `feature_norm`, new default `standardize`): per-feature mean/std
of the frozen PPV+MAX features are calibrated on the first training batches, then frozen alongside
the kernels; `feature_norm: none` reproduces the old behaviour. On LINK/USDT (60k train / 10k val,
8 epochs, CPU) all three seeds now improve monotonically past epoch 0 (best_epoch 6-7). The phase-2
entry was changed to `hp: {feature_norm: standardize}` (config id 12c3636e7110); the pre-fix rows
remain in results.jsonl under the old id as history and are excluded from the leaderboard narrative.

## Multihead phase-3 rows quarantined and re-run (2026-09-17)

During the period when two phase-3 trainer processes ran concurrently (22:20-22:40 UTC 2026-09-16),
four Multihead keys (A/1, B/0, B/1, C/0) were trained twice at once. Both processes wrote to the
same checkpoint file, so each pair reports identical metrics despite different epochs_run /
best_epoch, i.e. both restored the other's checkpoint. All 9 Multihead phase-3 lines were removed
from results.jsonl (backup: report/logs/results_before_multihead_rerun.jsonl) and the 6 keys were
re-run in isolated subprocesses with a single trainer. No other phase-3 run overlapped another.
Re-run result (isolated, single trainer): Multihead pooled MCC 0.515 ± 0.018 (A 0.560, B 0.478, C 0.506),
6/6 ok. The contaminated pairs had reported 0.516/0.467/0.496/0.525 with identical metrics across
duplicates; the clean A/1 is 0.54, so the earlier A gap was the shared-checkpoint artefact.

## Early-stopping variance dominates B/C results under the study protocol (2026-09-17)

Third-seed runs (seed 2 on the failing window) for the 12 stability-only failures of gate 3
showed that runs which trained to the 40-epoch cap (best_epoch 30-39) reach MCC 0.64 on windows
B and C with near-identical prediction counts across architectures (LSTM, LSTM_D10, NOISY,
NORES, Mamba2, Wavenet: ~19.2k SELL / 140.5k HOLD / 31.3k BUY on C), i.e. the same ceiling
every type reaches on window A (0.66). Seed-0/1 runs that early-stopped at epochs 12-17 with
best_epoch 1-6 sit at 0.45-0.55. The only seed-2 run that also stalled (LSTM_NORESIZE C,
best_epoch 6) scored 0.504. Three-seed stds are 0.07-0.12.

Interpretation: with lr 1e-4, cap 40 and patience 10, whether a run converges is a coin flip
decided during an early validation plateau, and the cap is binding for the runs that do
converge. Between-seed and between-architecture differences on B/C in phases 3-4 are therefore
mostly training-protocol variance. Any architecture that converges reaches the same ceiling.

Phase 4 (stride 2) was paused after 51 runs: 23 of 49 scored runs stalled (best_epoch <= 6),
all on B/C, none on A. A protocol calibration (production LSTM, window C, 3 seeds each) of
lr 3e-4 vs patience 20 decides the protocol for phases 4-7; phase-3 rows stand as the
"production protocol" reference. `train` settings are part of the config id, so re-runs under
the new protocol get new ids and the old rows remain as history.
