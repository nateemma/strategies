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

    arm                        W1 tr  W1 ret%  W2 tr  W2 ret%  W3 tr  W3 ret%   summed
    seed42  N0 (control)         349     6.19    624    15.51    239    10.14    31.84
    seed42  N2                   313     6.07    532     7.23    186     7.61    20.91
    seed42  N4                   298     8.47    483     5.06    173     6.08    19.62
    seed 7  N0 (control)         381     3.87    644    12.60    239    13.07    29.54
    seed 7  N2                   345     2.21    615    11.66    223    11.21    25.07
    seed 7  N4                   327     3.43    564    13.11    190     8.37    24.91

    PAIRED deltas vs the matched N0 control at the SAME seed:
      seed42  N2      -0.13    -8.27    -2.53   -10.93
      seed42  N4      +2.28   -10.45    -4.06   -12.22
      seed 7  N2      -1.66    -0.94    -1.86    -4.46
      seed 7  N4      -0.44    +0.52    -4.70    -4.62

**0 of 4 paired arms improve.** Both step counts lose on both seeds, and the
loss is large (-4.5pp to -12.2pp summed) rather than marginal.

**The retracted edge does not reappear.** Ponder's original +0.9pp on 4/4 seeds
was an artefact of the DDPM base, and on the non-GAN base at an operating point
with ~25pp of headroom — where a genuine +0.9pp would have been visible — it is
consistently HARMFUL.

Note seed42 N4 posts the best single W1 figure of any arm here (+8.47, +2.28
against its control) while losing 12.22pp summed. That is the same shape as the
`NoRes` arm in the architecture study: one flattering window inside a clearly
negative result. Not a signal.

## Verdict — the COCONUT programme is CLOSED

Both halves re-measured at an operating point where they could have registered,
and both are negative:

  - training-free noise-voting: monotone harm, -3.17pp to -5.34pp
  - stage-2 ponder: 0/4 paired arms, -4.46pp to -12.22pp

Do not re-run either. The infrastructure (`PonderStrategyMixin`,
`NoisyCoconutStrategyMixin`, the `LSTM_PONDER` / `LSTM_NOISY` / `LSTM_INJIT`
classifier variants) is kept and works; it is the hypothesis that is spent.
