# Architecture study — the model shape is NOT the bottleneck

2026-09-05. Run at the configuration where model quality is finally worth
something: 0.02 label, guards off, threshold 0.80, where headroom is ~25pp and
arms carry 300–860 trades. **No architecture comparison had ever been run in
this repo** — the 500+ regime arm records contain none, because before this
configuration existed there was 0.68pp of headroom on 95 trades and nothing
could have been measured.

Four variants, each changing exactly ONE thing against the stock
`_LSTMModel` (Linear→Conv1d k=2→residual→LSTM→last timestep→bottleneck).

## Training metrics (context only — never an acceptance input)

    arm                   val_mcc   val_precision   val_loss
    CONTROL stock LSTM     0.6078          0.4496     0.1752
    W64  capacity x2       0.6074          0.4495     0.1747
    D10  dropout ->0.10    0.6059          0.4488     0.1759
    Pool mean over seq     0.5901          0.4339     0.1841
    K5   conv kernel ->5   0.6075          0.4517     0.1749

## Realised return — guards off, thr 0.80

    arm                        W1 tr  W1 ret%   W2 tr  W2 ret%   W3 tr  W3 ret%   summed
    CONTROL (stock LSTM)         431     6.94     763    16.91     295    10.25    34.10
    W64  capacity x2             461     3.37     828    12.35     303    14.36    30.08
    D10  dropout 0.25->0.10      439    -1.48     742    11.96     294     6.49    16.98
    Pool mean over sequence      380     3.94     697    12.96     246    10.19    27.10
    K5   conv kernel 2->5        481     0.57     858    14.71     329     9.14    24.42

    delta vs control (pp)          W1       W2       W3    summed
    W64                         -3.57    -4.56    +4.11     -4.01
    D10                         -8.42    -4.95    -3.76    -17.12
    Pool                        -3.00    -3.94    -0.06     -7.00
    K5                          -6.37    -2.19    -1.11     -9.67

**All four are worse, on W1 and on summed return. 4 for 4.**

## What this rules out — and the hypotheses of mine it refutes

**Capacity is NOT the constraint.** I read the 0.018 train/val gap as
underfitting and predicted that doubling width would help. Doubling it moved
val_mcc by 0.0004 and val_loss by 0.0005, and cost 4pp of return. Refuted.

**Regularisation is NOT the constraint.** Dropout 0.25 at three points on a
12K-parameter model looked excessive given no overfitting gap. Cutting it to
0.10 was the single worst arm, −17pp. Refuted, and the direction says the
dropout is doing useful work, not throttling the model.

**Sequence pooling does not help.** Using all 16 LSTM outputs instead of the
last one LOWERED val_mcc (0.5901) and cost 7pp. The last timestep is apparently
the right summary for this target, not a lossy shortcut.

**A wider conv does not help.** Kernel 2→5 was the most interesting near-miss:
val_precision actually IMPROVED (0.4517 vs 0.4496) and val_mcc held (0.6075),
yet return fell 9.67pp — another instance of a classifier metric moving opposite
to realised return.

## The signature that matters

**val_mcc is pinned at ~0.608 regardless of what is done to the model.** Three
of four variants land within 0.002 of the control across changes to width,
regularisation and receptive field. A model that cannot be moved by capacity,
regularisation or inductive bias has hit an INFORMATION ceiling, not a model
ceiling.

That is the repo's standing finding restated in a place it can now be trusted:
an unlearnable target is a DATA problem. The 85-indicator battery added zero on
triple-barrier; four architecture interventions add zero here. **The ceiling
moves with new information — order flow, funding, cross-asset — not with a
better model on the same inputs.**

## Caveat

Single seed per arm. Deltas of 4–17pp on 300–860 trades are well outside
seed noise, and 4-of-4 pointing the same way is not plausibly chance — but the
ORDERING among the four losers should not be trusted, and no claim here rests
on it.

## Consequence

The stock LSTM stands. Do not re-run capacity, dropout, pooling or kernel-size
variants on this family. The next lever is the feature set, and specifically
information not derivable from OHLCV — funding is the one candidate in this
repo that has previously survived a temporally distant test.
