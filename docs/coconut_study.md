# COCONUT / continuous-latent-reasoning on a crypto direction classifier — study findings

**Verdict:** COCONUT-style continuous latent reasoning does **not** improve the NNNC
LSTM direction classifier. On the correct non-GAN base, training-free input "noise voting"
is inert, and latent "noise voting" is inert at the production gate — and when a loosened
gate exposes it, it's **noise-dominated, not edge** (it straddles the no-voting baseline
across the perturbation RNG). The trained recurrent-refinement ("pondering") variant showed
an apparent edge that turned out to be a scaling-pipeline artifact and did not survive on a
correct baseline. The result is consistent with an information-ceiling argument: adding
test-time compute or head capacity on the same OHLCV inputs does not add tradeable edge.
(Methodology lesson from the powered re-run: judge such interventions where marginal
decisions actually trade, not at a tight gate that trades only the confident tail.)

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

**Base model (a correction).** The right control for a classifier-head mechanism is the plain,
non-GAN **`NNNC_MLX`** — no GAN augmentation, no post-GAN scaling. Earlier runs used the
`NNNC_DDPM_MLX` (GAN) base, which (a) adds augmentation as a confound and (b) carried a
train/predict scaling bug — and that buggy base is what manufactured the false Stage-2 "ponder
edge" and inflated the Stage-1 "harm" (see each stage). Stages 0/1 are training-free (they wrap
a trained model), so they were re-run against `NNNC_MLX` for this write-up; the σ=0 identity
check reproduces the `NNNC_MLX` base exactly (13.68%, 166 trades).

## Results

### Stage 0 — input-space jitter: inert

| σ_in | 0.0 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|
| profit % | 13.68 | 13.68 | 13.68 | 13.40 | 13.40 |
| trades | 166 | 166 | 166 | 164 | 164 |

Perturbation below σ=0.05 flips no decisions; at σ≥0.05 it changes two (winning) trades,
costing 0.28pp. No systematic benefit. **Rejected.**

### Stage 1 — latent-space jitter: inert on the correct base

| σ | 0.0 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|
| profit % | 13.68 | 13.68 | 13.68 | 13.68 | 13.68 |
| trades | 166 | 166 | 166 | 166 | 166 |

On the non-GAN `NNNC_MLX` base, latent voting is **near-inert**: flat across the whole σ range
at the default seed, and a seed sweep at σ=0.2 gives 13.61–13.68% (a single-trade flip at one
seed). The correctly-trained model's decisions are margin-separated enough that latent
perturbation barely moves them. No benefit. **Rejected.**

> **The tight entry gate was masking the mechanism (powered re-run, 2026-07-20).** The sweeps
> above run at the production `prediction_threshold=0.6`, which trades only the *confident tail* —
> exactly where voting has no leverage (it acts on marginal, high-entropy decisions). Re-running at
> a **loosened gate** (`prediction_threshold=0.45`, ~599 trades) makes those marginal decisions
> trade, and the mechanism **becomes visible**: input jitter is still inert (21.24% flat to σ=0.05,
> −0.27pp at 0.1), but *latent* jitter now moves P&L — the default seed gave 21.24→**21.80%** flat
> across σ 0.05–0.4 (culling one marginal stop-out). **But it's noise, not edge:** a voting-seed
> sweep at σ=0.2 gives 20.99 / 21.80 / 21.80% (seeds 1/7/13) — it **straddles** the no-voting
> baseline (21.24%), flipping sign on the perturbation RNG. So latent voting adds *variance*, not
> consistent gain. Verdict upgrades from "inert" to **"active but noise-dominated at a powered
> operating point"** — still not a tradeable edge. **Methodology lesson (generalises to GAN aug):
> judge test-time / augmentation interventions at a POWERED operating point — tight guards trade
> only the confident tail, where these interventions have least effect, so they hide both harm and
> benefit.** (Repro: `NNNC_MLX_InJit_P0` / `NNNC_MLX_Noisy_P0`, sweep families `input_mlx_p0` /
> `latent_mlx_p0`.)

> **Why this differs from the earlier write-up (base choice, and what it revealed).** An earlier
> version of this study ran Stage 1 on the `NNNC_DDPM_MLX` (GAN) base and reported *monotonic
> harm* (10.64 → 9.36 as σ: 0 → 0.4). Two problems: (1) that base is the wrong control for a
> classifier-head mechanism — it adds GAN augmentation and, at the time, a train/predict scaling
> bug; (2) the "harm" was largely that base's **fragility**, not the mechanism. Re-run on the
> current GAN base, the same latent noise still flips several trades per seed (σ=0.2 → 11.98–12.79%
> across seeds), confirming the mechanism is live — it just does essentially *nothing* on the
> robust `NNNC_MLX` base. So the correct-base verdict is "inert," and the mechanism is live, not
> broken.

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

1. **None of the three COCONUT translations improves the classifier.** On the correct non-GAN
   base, input jitter is inert and latent jitter is inert (it barely perturbs the well-trained
   model); trained pondering has no robust edge once measured on a correct pipeline. (Latent
   jitter *looked* harmful on the earlier GAN base, but that was the base's fragility.)
2. **Why:** the mechanism is test-time compute / added head capacity on the *same* OHLCV
   inputs. It cannot exceed the information ceiling of those inputs. On a well-trained model the
   decisions are margin-separated, so K-path latent/input voting mostly reproduces the argmax
   and nets to nothing; on a fragile (mis-scaled) model the same voting smears marginal
   decisions and loses trades — which is what the earlier GAN-base run showed. Either way it adds
   no edge. This matches independent findings on this codebase that the predictive ceiling moves
   only with *new information* (order flow, funding, cross-asset), not with a cleverer head or
   more inference compute.
3. **`val_mcc` ≠ P&L.** Interventions that improved the classification metric (pondering depth;
   also a separately-tested P&L-magnitude-weighted loss) did not improve backtest P&L.

## Methodology takeaways (the transferable part)

- **Paired seed-robustness before believing sub-percent deltas.** A single backtest difference
  below the training-seed noise floor (~±1pp here) is not signal.
- **Pin the backtest window.** Date-relative windows slide day-to-day and silently make runs
  taken on different days non-comparable — this produced a convincing false positive.
- **Cross-check the baseline.** The Stage-2 "edge" was only exposed by comparing two bases that
  *should* have been identical; the discrepancy uncovered a real scaling bug.
- **Pick the right control base.** These mechanisms are classifier-head changes, so the control
  is the plain non-GAN model — not a GAN variant that adds augmentation and (here) a scaling bug.
  The wrong base both created a false Stage-2 signal and exaggerated the Stage-1 "harm."

## Reproducibility

Correct-base (non-GAN) strategies for Stage 0/1 (in `NNNC/`): `NNNC_MLX_InJit` (Stage 0),
`NNNC_MLX_Noisy` (Stage 1) — both `reuse_model_from = "NNNC_MLX"`, training-free. The original
GAN-base variants are kept for reference: `NNNC_DDPM_MLX_InJit`, `NNNC_DDPM_MLX_Noisy`,
`NNNC_DDPM_MLX_Ponder` + `_N0/_N2/_N4` (Stage 2). Wrapper/predictor infra:
`Predictors/NoisyCoconut.py`, `NNNC/NoisyCoconutStrategyMixin.py`, `NNNC/PonderStrategyMixin.py`,
`NNNClassifierMLX.py` (`_LSTMPonderModel`, `ClassifierTypeMLX.LSTM_{INJIT,NOISY,PONDER}`).
Sweep harness: `scripts/noisycoconut_sweep.py` (`input_mlx|latent_mlx|input|latent <sigmas...>`,
`--seeds`, `--timerange`). Design spec:
`docs/superpowers/specs/2026-07-18-noisycoconut-nnnc-design.md`.

*Note on absolute numbers:* Stage 0/1 are now reported on the non-GAN `NNNC_MLX` base (σ=0 =
13.68%). Earlier revisions reported them on the pre-fix `NNNC_DDPM_MLX` base (σ=0 = 10.64%); the
verdicts (both rejected) are unchanged. The scaling bug found via Stage 2 was subsequently
fixed.
