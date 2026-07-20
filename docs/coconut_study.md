# COCONUT / continuous-latent-reasoning on a crypto direction classifier — study findings

**Verdict:** COCONUT-style continuous latent reasoning does **not** improve the NNNC
LSTM direction classifier. Training-free latent/input "noise voting" is inert-to-harmful;
the trained recurrent-refinement ("pondering") variant showed an apparent edge that turned
out to be a scaling-pipeline artifact and did not survive on a correct baseline. The result
is consistent with an information-ceiling argument: adding test-time compute or head capacity
on the same OHLCV inputs does not add tradeable edge.

Dates: 2026-07-18 → 07-20. Prompted by external interest in COCONUT (Chain Of CONtinUous
Thought) applied to time series.

## What was tested

**COCONUT** (Hao et al.) replaces discrete chain-of-thought tokens in an LLM with continuous
hidden states — the model "thinks" in latent space using the same weights plus more test-time
compute. We adapted the idea to **NNNC**, an MLX LSTM that classifies each 15m crypto candle
as Buy/Hold/Sell (gbb labels, horizon 48, threshold 0.007), and evaluated three translations
as a staged, reject-fast experiment:

| Stage | Mechanism | Retrain? |
|---|---|---|
| 0 | **Input-space jitter** — perturb input features K times, decode each, vote (mean-softmax) | no (wraps trained model) |
| 1 | **Latent-space jitter** ("NoisyCoconut") — perturb the LSTM latent K times, decode each, vote | no |
| 2 | **Looped "pondering"** — N shared-weight residual refinement steps on the latent before decode | **yes** |

Backtests on 11 high-volatility alt pairs, pinned window 2024-06-29 → 2026-06-19, seed=42
unless a seed sweep is stated, volume filter on. `σ=0` / `N=0` reproduce the unmodified model
exactly (a built-in identity/sanity check).

## Results

### Stage 0 — input-space jitter: inert

| σ_in | 0.0 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|
| profit % | 10.64 | 10.64 | 10.64 | 10.64 | 10.89 |

Perturbation below σ=0.05 flips no decisions; at 0.1 it changes a single (winning) trade.
No systematic effect. **Rejected.**

### Stage 1 — latent-space jitter: monotonically harmful

| σ | 0.0 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|
| profit % | 10.64 | 10.37 | 9.71 | 9.36 |
| Calmar | 9.04 | 8.41 | 7.82 | 7.52 |
| stop-outs | 47 | 47 | 48 | 49 |

Every metric degrades monotonically with the noise level; voting *adds* losing trades rather
than filtering them. A seed sweep at σ=0.2 gave 9.71–10.74% across seeds — all at or below the
unperturbed 10.64%. **Rejected.**

> **Methodology note (a false positive we caught):** an early single-seed sweep suggested an
> inverted-U with a "+0.6pp peak at σ=0.2". Two checks killed it: (1) the effect was inside
> the training-seed noise band (~±1pp); (2) the apparent peak was actually a **moving backtest
> window** — the harness computed its date range relative to "today", so a run on the next
> calendar day shifted the window one day and changed a couple of boundary trades. Pinning the
> timerange and doing a paired seed sweep collapsed the "peak" to the monotonic decline above.

### Stage 2 — looped pondering: apparent edge was a pipeline artifact

The trained recurrent-refinement head (N shared residual-MLP steps on the latent) *initially*
looked like the one survivor: N=2 beat the N=0 control at all 4 training seeds (+0.67…+1.17pp),
held in both temporal eras, and traced a clean inverted-U peaking at N=2 — passing the
seed-robustness and persistence checks that killed everything else.

**It did not replicate on a correct baseline.** Those runs used a strategy base that carried a
**train/predict scaling mismatch** (a single-task GAN strategy applied a post-augmentation
tensor scaler at prediction time but not at training). Re-running N0-vs-N2 on the plain,
correctly-scaled base:

| seed | 1 | 7 | 13 |
|---|---|---|---|
| N2 − N0 (profit pp) | +1.28 | −0.52 | −0.32 |

1 of 3 seeds — sign-flipping, within noise. The +0.9pp "edge" was an artifact of the buggy
base, not the pondering head. **Rejected.** (`val_mcc` rose slightly with N but P&L did not —
better classification ≠ better trading.)

## Conclusions

1. **None of the three COCONUT translations improves the classifier.** Input jitter is inert,
   latent jitter is harmful, and trained pondering has no robust edge once measured on a
   correct pipeline.
2. **Why:** the mechanism is test-time compute / added head capacity on the *same* OHLCV
   inputs. It cannot exceed the information ceiling of those inputs. Latent voting further acts
   as a smoother, which pulls marginal decisions toward the mean and adds losing entries. This
   matches independent findings on this codebase that the predictive ceiling moves only with
   *new information* (order flow, funding, cross-asset), not with a cleverer head or more
   inference compute.
3. **`val_mcc` ≠ P&L.** Interventions that improved the classification metric (pondering depth;
   also a separately-tested P&L-magnitude-weighted loss) did not improve backtest P&L.

## Methodology takeaways (the transferable part)

- **Paired seed-robustness before believing sub-percent deltas.** A single backtest difference
  below the training-seed noise floor (~±1pp here) is not signal.
- **Pin the backtest window.** Date-relative windows slide day-to-day and silently make runs
  taken on different days non-comparable — this produced a convincing false positive.
- **Cross-check the baseline.** The Stage-2 "edge" was only exposed by comparing two bases that
  *should* have been identical; the discrepancy uncovered a real scaling bug.

## Reproducibility

Strategies (in `NNNC/`): `NNNC_DDPM_MLX_InJit` (Stage 0), `NNNC_DDPM_MLX_Noisy` (Stage 1),
`NNNC_DDPM_MLX_Ponder` + `_N0/_N2/_N4` (Stage 2). Wrapper/predictor infra:
`Predictors/NoisyCoconut.py`, `NNNC/NoisyCoconutStrategyMixin.py`, `NNNC/PonderStrategyMixin.py`,
`NNNClassifierMLX.py` (`_LSTMPonderModel`, `ClassifierTypeMLX.LSTM_{INJIT,NOISY,PONDER}`).
Sweep harness: `scripts/noisycoconut_sweep.py` (`latent|input <sigmas...>`, `--seeds`,
`--timerange`). Design spec: `docs/superpowers/specs/2026-07-18-noisycoconut-nnnc-design.md`.

*Caveat on absolute numbers:* Stage 0/1 baselines (10.64%) were measured on the pre-fix
pipeline; the reported effects are deltas against the same-base control, so the conclusions
hold regardless. The scaling bug found via Stage 2 was subsequently fixed.
