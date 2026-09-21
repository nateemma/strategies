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

## Pure recurrent family types: per-bar input LayerNorm blocked learning on B/C (2026-09-17)

Under the amended protocol (lr 3e-4, stride 2) LSTM_FAM/GRU_FAM reached the window-A ceiling
(0.64-0.65) but stayed flat at 0.44-0.53 validation MCC on B/C for 40 epochs, for every size,
depth and direction. Discriminating runs on window C, seed 0: LSTM_RAW 0.638, LSTM_NOCONV
0.640, production GRU 0.637 (no input LayerNorm), LSTM_FAM hidden 64 with `input_norm: false`
0.633 and with both norms off 0.630, versus 0.504 with the per-bar feature LayerNorm. The
LayerNorm across the 24 features of each bar erases absolute feature levels (the label-trigger
inputs guard_metric and bb_width among them). Fix: `input_norm` default is now False (commit
after c2c54ca); phase-4 RNN entries carry `input_norm: false` explicitly, so their config ids
differ from the 30 earlier rows, which remain as history. LSTM_FAM hidden 256 still diverged on
window A under the earlier default; it is re-run under the new default and reported as
"diverged under the shared protocol" if it fails again.

## Warm-start defect in re-runs (found 2026-09-17 23:40, fixed in run.py)

`MLXClassifierNary.train()` loads an existing model file at `model_path` before training and
continues from those weights even with `force_train=True`. The harness stores the model and the
best-epoch checkpoint at the same per-(config_id, seed) path, so any re-run of an existing key
warm-started from stale weights: (a) the Multihead re-run (6 keys) started from the
contaminated concurrent-run checkpoints; (b) five phase-3 runs killed mid-training by trainer
restarts were resumed from their partial checkpoints (LSTM2 A/0, CNN A/1, Multihead A/0 and
B/1, Mamba2 B/1); (c) the RNN-ablation `default` arm failed to load a checkpoint from the
input-norm era of the same config id. Fix: run.py now deletes any file at the key's model and
checkpoint paths before training (`stale_files_removed` recorded on the row) and fails a row
whose training ran zero epochs. The keys in (a) and (b) are re-run cleanly (rows replaced;
backup `report/logs/results_before_clean_rerun.jsonl`); everything else trained from scratch
because its key had no prior file.

## RNN ablation (window C, 3 seeds, lr 3e-4, stride 2; 2026-09-18 01:20)

| arm | seeds | mean MCC | converged |
|---|---|---|---|
| LSTM_FAM h64 default | 0.635, 0.632 (seed 0 failed on a stale file, re-run queued) | 0.633 | 2/2 |
| no pool norm | 0.630, 0.632, 0.633 | 0.632 | 3/3 |
| input projection | 0.638, 0.638, 0.639 | 0.638 | 3/3 |
| bottleneck head | 0.637, 0.639, 0.639 | 0.638 | 3/3 |
| projection + bottleneck | 0.638, 0.639, 0.639 | 0.639 | 3/3 |
| batch 1024 | 0.631, 0.631, 0.632 | 0.631 | 3/3 |
| lr 1e-4, patience 20 | 0.629, 0.628, 0.630 | 0.629 | 3/3 |
| weight decay 1e-4 | 0.633, 0.635, 0.632 | 0.633 | 3/3 |
| LSTM_RAW control | 0.638, 0.639, 0.638 | 0.638 | 3/3 |

Every arm converges; none of the structural or optimiser changes is needed. The phase-4 RNN
rows that stalled (e.g. LSTM_FAM h64, C, seed 1: 0.505, val flat at 0.45 from epoch 1 versus
0.635 / val 0.61 from epoch 1 for the identical architecture, seed and data in the ablation)
all predate the fresh-start fix; the ablation rows all follow it. The RNN family sweep is
therefore discarded and re-run from scratch under the current harness (rows backed up in
report/logs/results_before_rnn_sweep_rerun.jsonl). If the re-run converges, the earlier stalls
were a harness artefact; if it stalls again, the difference is real and will be diagnosed
against these ablation rows (identical config ids).

## Conv ablation (window C, 3 seeds, lr 3e-4, stride 2, clean runs; 2026-09-18 02:00)

Production Wavenet: 0.637, 0.637, 0.639 (3/3 converge, best epochs 32-39). Pure TCN_FAM with
the same receptive field (channels 64, depth 8, dilation base 2, no residual): 0.351 (40
epochs, never learned), 0.504 (stalled), 0.447 (stalled). All six rows post-date the
fresh-start fix, so the pure conv family's failure on the hard windows is real, not a harness
artefact, and the phase-4 TCN/Transformer/MLP sweep rows (all first-ever runs of their keys)
stand. Correction: Wavenet has no residual path either — it is eight plain causal convs with
dilations [1, 2, 4, 8, 1, 2, 4, 8] (receptive field 31 bars, kernel 2, ~32 channels), whereas
TCN_FAM depth 8 / dilation base 2 grows dilations to 128 on a 16-bar window, so its deep layers
see mostly padding: a pathological grid point, not a family result. A TCN_FAM with an explicit
Wavenet dilation schedule is queued (3 seeds) as the fair conv-family test.

## ROOT CAUSE of the window-B/C "stalls": cross-window checkpoint reuse (found 2026-09-18 02:30)

The harness named each run's model/checkpoint file `<config_id>_<seed>.safetensors` — without
the window — and `MLXClassifierNary.train()` loads an existing file at that path before
training. Within every phase the runner trains windows A, B, C in turn for each config and
seed, so the window-B run loaded the window-A model and trained from it, and window C loaded
B's (logs: "MLX model not found" for every A run, "MLX model loaded: ..._<seed>.safetensors"
for every B and C run). A warm-started model sits at a foreign optimum, its validation MCC is
flat from epoch 1, and patience 10 ends the run after ~11-17 epochs with best_epoch <= 6 — the
exact signature previously read as "the pure families don't learn on B/C" and "lr 1e-4 stalls".

Proof: the first post-fix runs covering all three windows (run.py removes the stale file
before each window) — TCN_FAM channels 32: 0.639 / 0.627 / 0.633 (A/B/C); MLP_FAM width 128:
0.649 / 0.633 / 0.635 — versus 0.42-0.53 on B/C for every pre-fix sweep row. All C-only
diagnostics (calibration, ablations, third seeds with a fresh seed-2 key) were clean and all
converged, which is why they contradicted the sweeps.

Status of rows: window-A rows and all seed-2 / phase-0 rows are clean; every window-B and
window-C row of phases 2-4 (seeds 0-1) written before the fix is contaminated and is re-run
from scratch. The protocol amendment (lr 3e-4) was decided on contaminated evidence; the clean
calibration shows lr 1e-4 also converges, so phase 3 is re-run at the production protocol for
comparability with its A rows, and phases 4-7 keep lr 3e-4 / stride 2 (calibrated to converge
at half the cost). The checkpoint path now includes the window id.

## After the fix: read-out, patience and clean re-runs (2026-09-18 04:30)

Window C, 3 seeds, lr 3e-4, stride 2, clean: TCN anchor pool=last 0.639/0.640/0.639, patience
40 0.637/0.636/0.639; Transformer d64/L1 pool=last 0.639/0.641/0.640, patience 40
0.639/0.639/0.639. Neither the read-out nor the patience matters once runs start from scratch.
Clean re-runs of the warm-started keys: Multihead A 0.657/0.660, B 0.640/0.641, C 0.640/0.639;
Mamba2 B/1 0.640; LSTM2 A/0 0.663; CNN A/1 0.552 (CNN is weaker, but not the contaminated
0.44). Every architecture that trains from scratch reaches ~0.64 on B/C and ~0.66 on A.

Wavenet-clone TCN_FAM (channels 32, kernel 2, dilations [1,2,4,8,1,2,4,8], 16,227 params =
Wavenet's count), window C, clean: 0.638 / 0.640 / 0.640 versus Wavenet 0.637 / 0.637 / 0.639.
The conv family code reproduces Wavenet; the depth-8 / dilation-base-2 point was a grid error.

## The ceiling is the label's trigger rule (2026-09-18 20:10)

A learning-free rule — BUY if guard_metric < 0.3 and bb_width > 0.035, SELL if guard_metric >
0.3 and bb_width > 0.035, else HOLD — scored against the labels on each window's test rows:
A 0.673 (BUY precision 0.52, recall 1.00), B 0.644 (0.51, 1.00), C 0.642 (0.45, 1.00). Every
trained architecture converges to 0.66 / 0.64 / 0.64: the models learn the trigger and nothing
about the future-gain condition. RIDGE (0.53-0.55) is the linear approximation of the same AND
of two thresholds. Conclusion for the architecture question: no architecture, size or
composition extracts information beyond the trigger from 24 indicators over 16 bars; the
remaining question is conditional on the trigger — whether any INPUT set predicts which
triggered bars pay off better than the ~0.5 base rate.

Addendum (2026-09-18 21:40): in phase 2 (three seeds) the seed-2 window-B/C rows were ALSO
warm-started (A2 preceded B2), so "seed-2 rows are clean" holds only for the phase-3 third
seeds, which ran on a single window with no prior file. The contaminated RIDGE/ROCKET B/C
seed-2 rows are quarantined (backup report/logs/results_before_seed2_baseline_quarantine.jsonl)
and re-run at the end of the queue; gate 4 was recomputed on the two clean RIDGE seeds.

## Mamba excluded on window C (2026-09-19 05:15)

Clean Mamba runs peak at ~48 GB of Metal memory (Wavenet: 3.5 GB) and take 68 min each on
window B (4062 s), driving swap to 15.5 of 16 GB. With A 0.513 (phase 3) and B 0.484 / 0.457
(clean), far below every other type, its two window-C runs were not worth 2.5 h and an
out-of-memory risk to the census; Mamba's pooled figure is over A and B only and is flagged.

## 019 start: label not carried into isolated children (found and fixed 2026-09-19 08:35)

The first phase-19 launch produced RIDGE/PERSISTENCE rows identical to the gbb ones and gbb
class counts: `phase.py`'s isolated child command serialised every config field except
`label`, so the children trained on the default label under triple-barrier config ids. The 26
rows were purged, the payload now carries `label` (regression test in test_run_smoke.py), and
the relaunched rows show the strict triple-barrier balance (window A test: 29% SELL / 55% HOLD
/ 17% BUY; B: 42 / 35 / 22). Lesson: every config field that changes the data must be asserted
on the written row, not assumed from the launcher.

## Triple-barrier widths were floored at +3% / −2% (found 2026-09-19 09:50)

`labels_triple_barrier` defaults `pt=0.03, sl=0.02` and applies `pt = max(pt, min_gain)`; the
sell labeler applies `sl = max(sl, min_loss)` with `pt` at its default. The study shim passed
only `min_gain`/`min_loss` (= up/down), so every triple-barrier row before the fix — phase 19
and the first phase-20 rows — used buy PT +3% / SL −2% and sell SL −2% / PT +3%, not the
configured ±2%, and the ATR-scaled floor of 0.5% could never apply (ATR-1 labels differed
from the "±2%" labels by ~5 bars per window). Fix: the shim now passes `pt=up, sl=down`
explicitly. Phase 19 is re-run at a true ±2% (the +3%/−2% rows are purged; backup in
report/logs) and the screening campaign restarts afterwards.

## quantile_future's threshold was inert behind the min_gain default (found 2026-09-19 18:20)

A free balance scan of `quantile_future` at `top_quantile` 0.65 / 0.70 / 0.75 returned
byte-identical class shares (HOLD 74.2% at every setting), which is the signature of a
parameter that never reaches the labeler's decision. Cause: the generic shim supplies
`min_gain = min_loss = 0.02` as a default, and `labels_quantile_future_return` applies that
absolute floor on top of the quantile, so the floor bound and the quantile did nothing. With
`min_gain = min_loss = 0.001` the quantile binds as intended (HOLD 56.9 / 65.3 / 74.0 / 83.0%
at q = 0.75 / 0.80 / 0.85 / 0.90). This is the fourth instance in this study of a silently
ineffective parameter; caller-supplied label keys now raise if the labeler cannot accept them,
but a key the labeler DOES accept and then overrides with a default is still only detectable
by a balance scan. Scan every new label config before trusting a run.

## Scaling review, and a much larger data finding (2026-09-20)

Question raised: can one pooled scaler serve both BTC and ZEC?

**Indicator features.** Only 6 of the 24 ind24 columns are scaled at all (the rest are
pre-bounded). Five of those six are effectively pair-invariant (medians within 0.01-0.09 and
IQRs within ~15% across BTC/ZEC/LINK/SOL). `bb_width` is the exception: median 0.0108 (BTC) vs
0.0317 (ZEC), IQR 0.0111 vs 0.0323 -- a 3x difference. Under the pooled scaler BTC's median
bb_width lands at -0.43 and ZEC's at +0.48, so the pair's volatility regime is directly legible
to the model. Clipping at +/-10 is negligible (<= 0.07% of rows). Whether that is a defect
depends on the label: gbb's trigger is an ABSOLUTE bb_width > 0.035 and the triple-barrier
labels use ABSOLUTE +/-x% barriers, so a pair's volatility level genuinely predicts its labels
and pooled scaling preserves real information; per-pair scaling would erase it. For a
relative label (ATR-scaled barriers) per-pair scaling would be the better match. Pooled is
defensible for the current labels but remains an untested assumption.

**BUG: `log_oc` is never scaled.** Its pooled IQR is exactly 0, and sklearn's RobustScaler sets
`scale_ = 1.0` for a zero-IQR feature, so the column passes through at its native magnitude
(std 0.005) beside neighbours at std 1.3-5.9 -- a ~1000x mismatch that makes it a dead input.
Every raw8 result in the study (phases 20, 23-28) was produced with one of the eight features
effectively disabled.

**ROOT CAUSE, and the larger finding: most bars are gap-fill.** freqtrade fills missing candles
with open=high=low=close(prev) and volume=0. Zero-volume share of the 15m binanceus data:
ZEC 54.1%, AAVE 55.0%, BCH 53.7%, NEAR 47.6%, DOT 39.4%, SUI 39.4%, LINK 25.3%, LTC 23.0%,
AVAX 22.2%, XRP 6.7%, SOL 3.8%. For ZEC, 100% of zero-volume bars have OHLC all equal, and the
share by year is 65 / 55 / 64 / 63% for 2022-2025, falling to 10% in 2026. So for six of the
eleven production pairs, roughly half of every training window is synthetic: no trades
occurred. Labels computed across those stretches measure the absence of trading, not market
behaviour, and long runs of identical filled bars are the obvious source of the very high
PERSISTENCE scores (0.76-0.87) seen on every forward-looking label.

Consequences to work through: (a) fix or drop `log_oc` and guard every feature against a
degenerate pooled IQR; (b) decide how filled bars are handled -- excluding them outright breaks
the contiguity that windowing and forward-looking labels need, so the likely answer is to
refuse to emit a SAMPLE whose window or forward horizon is mostly filled; (c) re-examine
whether 15m on binanceus is the right timeframe/venue for this universe at all, and whether
earlier pair-level conclusions (e.g. that the edge is structurally ZEC) survive.

## 2026-09-21 — TSMamba excluded, and a restart that briefly doubled the driver

**TSMamba excluded from the campaign census.** The original include/exclude decision
assumed peak Metal memory is roughly flat in `seq_len`, which is what `LSTM_FAM` shows
(19.9 GB at 16, 17.3 GB at 64, 32.8 GB at 256). That does not hold for the Mamba family:
`Mamba2` measured **13,299 MB at seq_len 16 but 46,653 MB at seq_len 64**, a 3.5x rise.
`TSMamba` is bidirectional Mamba2 and ran 1.34x `Mamba2` at seq_len 16, which projects to
**~62 GB at seq_len 64 on a 64 GB machine**. It was excluded before it ran. `Mamba2`
itself is kept — it completed in 149 s with `status=ok` — but it is now the single
largest memory consumer in the census and is worth watching.

**Instrument defect found while doing that.** Restarting the campaign with
`pkill -f campaign.sh` does NOT stop the driver: `campaign.sh` spawns
`python -m research.clfstudy.campaign --run-all`, which survives the pkill and keeps
launching trainers. Relaunching the script therefore produced two drivers and, for about
40 seconds, two concurrent trainers on the same config — precisely the condition behind
the 2026-09-16 out-of-memory. No data was lost (`results.jsonl` parsed clean, 0 corrupt
lines, no partial TSMamba row; duplicate run keys are deduplicated by
`leaderboard.load_results`, which keeps the last row per key), but the machine was
briefly at risk.

Fixed with a single-instance guard at the top of `campaign.sh`: it refuses to start when
another `research.clfstudy.campaign --run-all` is alive and prints the correct two-step
stop command. Verified by launching it a second time and getting exit 3.

## 2026-09-21 — LSTM_FAM at its DEFAULT hp diverges to NaN at seq_len 64

Stage A of the campaign runs every type at its library defaults. `LSTM_FAM` failed all
4 runs with `ValueError: non-finite values in predicted probabilities` (raised by
`run.py`'s own guard, so this is a detected divergence, not a silent bad number), and was
therefore dropped from the stage-A top 8.

It is a real numerical property, not a harness fault. Tabulating every `LSTM_FAM` row
ever run by `(hp, seq_len, status)`:

* `hp={}` (hidden 64, depth 1) at **seq_len 16**: ok 24/24.
* `hp={}` at **seq_len 64**: failed 4/4.
* `{"depth": 1, "hidden": 32, "input_norm": false}` at seq_len 64: ok.
* `{"depth": 2, "hidden": 32, "input_norm": false}`: ok at 16/32/64/128/256 (58 rows).
* `{"hidden": 256}` at seq_len 16: failed 4, ok 8 — i.e. seed-dependent divergence.

So the instability is driven by **hidden size together with sequence length**: hidden 64
is fine at seq_len 16 and diverges at 64; hidden 32 is fine everywhere tested up to 256;
hidden 256 is marginal even at 16. This extends the earlier hidden-256 NaN finding rather
than contradicting it.

**Consequence, and why the campaign was left alone.** The campaign's top 8 still contains
eight recurrent types, led by `LSTM_RAW` (0.2860), which is a plain LSTM over raw features
and is functionally close to what `LSTM_FAM` is, plus `GRU_FAM` (0.2849), its direct
sibling. The census is therefore not blind to this family, so per the unattended operating
rules this was recorded rather than patched mid-run.

**What it costs.** `LSTM_FAM` at the study's own working setting (hidden 32, depth 2)
scored 0.2812 on window A and is the best cell in the study pooled (0.2343). That specific
type will not be carried into stages B-D, because stage B rebuilds configs from arch names
at default hp and would simply fail again. Revisiting it needs a per-arch default override,
which is a code change and a decision for the user, not an unattended fix.
