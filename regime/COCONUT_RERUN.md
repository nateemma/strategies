# COCONUT re-run at the model-driven operating point

Both COCONUT lines were first measured in 2026-08 at the OLD operating point,
where the guard chain selected ~99% of entries and a PERFECT model was worth
only 0.68pp more than the real one. Nothing could have registered there. Re-run
now at guards-open / threshold-0.80 / label-0.02, where headroom is ~25pp and
arms carry 300-430 W1 trades.

Control throughout is `open` (NNNC_MLX_Open): W1 431tr +6.94%, W2 763tr +16.91%,
W3 295tr +10.25%, summed +34.10%.

## Part 1 — noise-voting (training-free): REJECTION CONFIRMED

Reuses NNNC_MLX_Lbl02's weights, so these are backtests only.

    arm                          W1 tr  W1 ret%  W2 tr  W2 ret%  W3 tr  W3 ret%   summed
    CONTROL (sigma 0)              431     6.94    763    16.91    295    10.25    34.10
    latent noise sigma 0.1         430     6.57    760    12.75    292    11.61    30.93
    latent noise sigma 0.2         428     6.27    756    13.39    301     9.10    28.76
    input jitter sigma 0.1         423     6.06    756    15.10    296     8.62    29.77

    delta vs control                W1       W2       W3   summed
    latent sigma 0.1             -0.37    -4.16    +1.36    -3.17
    latent sigma 0.2             -0.67    -3.52    -1.15    -5.34
    input jitter sigma 0.1       -0.88    -1.81    -1.63    -4.32

**The 2026-08 rejection replicates.** Harm is monotone in sigma on the summed
figure (-3.17 at 0.1, -5.34 at 0.2) and W1 degrades monotonically too
(-0.37, -0.67). Input jitter is no better than latent noise.

This is now a MEANINGFUL null rather than an unmeasurable one: it was measured
where a gain of this size could have been seen, and it was not there. The
training-free COCONUT line is closed.

## Part 2 — stage-2 ponder (requires retraining)

Ponder is the line worth re-running: its original +0.9pp on 4/4 seeds was
RETRACTED as a DDPM-base artefact once the predict-time scaling bug was fixed,
and it was never re-measured on the non-GAN base.

`ponder_steps = 0` is the matched control — the loop is skipped so the forward
pass equals production, but the code path, parameter tree and init RNG match the
N>0 arms. Two seeds (42, 7), because the seed study showed a 6.38pp standard
deviation on summed return at this operating point and the original ponder
effect was under 1pp.

Results appended below when the runs complete.
