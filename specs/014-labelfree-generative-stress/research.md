# Research: labelfree-generative-stress (brief 06)

Phase 0. Every decision carries the measurement or code reading behind it.

---

## R1 — A "path" is a resampled panel, not a resampled price series

**Decision**: the block bootstrap resamples **blocks of the complete per-pair panel** — prices,
indicator columns, cost, *and* the model's three class probabilities — jointly, from feature
013's `es/panel.py` output.

**Why the probabilities must ride along.** The policy under test is `NNNC_MLX`, and running it on
a synthetic path requires `p(buy)` and `p(sell)` on that path. There are only two ways to get
them:

- **re-run the model on synthetic features** — which means a second inference path, and this
  study has already weighed and rejected that. Feature 013 chose ~12 minutes of real backtests
  over offline inference precisely because a second path can silently diverge; the repo has a
  case where predict-time scaling did exactly that (the single-task GAN scaler bug, 1.53 pp,
  undetected until hunted).
- **resample them with the block** — inside a block, prices, features and predictions are
  internally consistent *because they are real*. Only the joins between blocks are synthetic.

The second is both cheaper and more faithful, and it is what a stationary block bootstrap is for.

**What this buys and what it costs.** The synthetic path is a new *ordering* of real market
conditions, with the model's real reaction to each condition carried along. It is not a path from
the true data-generating process — nothing is — and it cannot contain a market state the data has
never shown. That is the honest limit of the method and is stated in the report, not discovered.

**Reused unchanged**: `es/panel.py` already assembles exactly this per fold (`open/high/low/close`,
`volume`, `rvol_stop`, `cost`, `p_sell/p_hold/p_buy`, and the six guard columns), and its
side-cars for all 15 folds already exist on disk.

---

## R2 — The honest problem with SC-001: a gate that is expected to pass

**This is the most important thing in this document, and it is a weakness of the design rather
than a result.**

The block bootstrap resamples real blocks. The incumbent's P&L distribution on resampled real
blocks should therefore resemble its P&L distribution on real blocks *by construction*. SC-001 is
the feature's single load-bearing gate (clarification Q3 removed the alternative generator), and
it is biased heavily toward passing.

**Where it can genuinely fail**, which is what makes it worth running at all:

- **Block joins are artificial.** A join splices two unrelated moments together, creating a price
  discontinuity the real series never had. Those look like gaps, and gaps fire stops — feature
  013 measured that the incumbent's stop exits already fill at the open on gap-through bars.
  Enough spurious gaps and the synthetic loss tail is fatter than real.
- **Tails depend on runs, not on marginals.** Maximum drawdown is a function of *consecutive*
  bad blocks. A bootstrap that reproduces the marginal P&L distribution perfectly can still get
  the drawdown distribution wrong, because resampling destroys the long-range dependence that
  produces sustained drawdowns.
- **Regime ordering is destroyed.** The real 2023–2026 span has a sequence; a resampled path does
  not.

So the calibration is a **plumbing-and-tails check**, and it should be reported as such. The
feature's informative output is the **stress distribution** (US4), not the calibration.

**Consequence for the design**: the tolerance must be pre-registered (SC-002) precisely because a
near-guaranteed pass invites choosing the threshold that yields one, and because the *tail*
comparison is the part that can fail and therefore the part worth stating in advance.

---

## R3 — Calibration is three numbers, and the tail one is the one that matters

**Decision**: compare the incumbent's synthetic and real P&L on

| statistic | what it catches | pre-registered tolerance |
|---|---|---|
| two-sample KS distance on per-trade P&L | the body of the distribution | **D ≤ 0.10** |
| expected-shortfall ratio at 5% (synthetic ÷ real) | the loss tail | **0.7 ≤ ratio ≤ 1.4** |
| maximum-drawdown ratio (synthetic median ÷ real) | sustained runs | **0.6 ≤ ratio ≤ 1.6** |

**Why these numbers.** They are set from what the comparison can resolve, not from what would be
impressive. The real reference is **378 trades**; at that size the KS statistic's own 95% critical
value against an identical distribution is ≈1.36/√378 ≈ **0.070**, so a 0.10 threshold admits
sampling noise without admitting a visibly different shape. The ES and drawdown bands are
deliberately wider because both are tail statistics on a few hundred trades and are correspondingly
noisy — a tight band there would fail on noise and teach nothing. The asymmetry (wider on the
upside than the downside) is deliberate: a synthetic tail **fatter** than real is a conservative
harness and tolerable, while a synthetic tail **thinner** than real is the dangerous direction,
because it would let a fragile policy through.

**Pre-registered here, before any calibration run**, per SC-002.

---

## R4 — The policy evaluator is feature 013's exit surrogate, reused unchanged

**Decision**: score paths with `research/labelfree/es/exits.py` + `es/evaluate.py`.

**Why**: ≥1,000 paths × a 90-day panel is only affordable with a fast evaluator, and this one is
**31 ms per evaluation** with its rank fidelity already measured against real freqtrade backtests
at **Spearman +0.9766** over 21 candidates. Building a second evaluator for this feature would
repeat the exact mistake R1 avoids.

**What it inherits, including the limits.** The surrogate's four fidelity corrections (exit
sequence, per-check price, stop fill, gated `model_exit`) and its known residual divergence
(`roi` 22 against a real 13 on f07) come along. The residual is recorded in the stress report as
a limitation, since a stress verdict inherits its evaluator's error.

**One change is required**: `exits.simulate` currently takes a fold panel. Path banks are panels
of the same shape, so the adapter is a shape check, not new simulation logic — and a test asserts
a bank and a fold panel produce identical results on the same bars.

---

## R5 — Stationary bootstrap, geometric block lengths, and why not fixed blocks

**Decision**: the **stationary bootstrap** of Politis & Romano (1994) — geometric block lengths
with mean `L`, wrapping at the series end — rather than fixed-length blocks.

**Why**: fixed blocks make the resampled series non-stationary, because a bar's probability of
starting a block depends on its position modulo `L`. The geometric variant removes that artefact
for the cost of one line. The brief asks for blocks of 1–3 days; at 15-minute bars that is
**96–288 bars**, and `L` is swept across that range with the result reported per `L`, since
choosing one and reporting it as *the* answer would hide the sensitivity.

**Blocks are drawn per pair and aligned by position**, not across pairs, so a synthetic panel
keeps each pair's own internal coherence. Cross-pair correlation is *not* preserved, which matters
for portfolio drawdown and is recorded as a limitation: the real universe's pairs move together
in a crash, and an independently resampled panel will understate joint drawdown.

---

## R6 — The bank is a file, with the provenance to detect staleness

**Decision**: one bank per (fold set, block length, seed), persisted, carrying its generator,
parameters, source range, feature list and a content hash.

**Why**: FR-002 exists because a stale bank silently stresses the wrong thing. This study has the
precedent — feature 013's panel cache keys on the side-car's content hash for exactly this reason,
after feature 012 recorded that a cache key must cover everything that changes the panel.

Sampling is offline and once (FR-001/SC-005). The bottleneck the brief warns about (diffusion
sampling) does not arise, since no diffusion model is in scope — but the offline rule still holds,
because a bank that is regenerated per run is not reproducible.

---

## R7 — What the sequence-capability finding is, and why it still ships

Clarification Q3 removed every learned generator from scope, so nothing is built on this. It is
reported anyway (US3, SC-008) because it is a durable fact about the repo's own tooling and is the
documented reason the design pivoted.

```
TabDDPM     (n, 1, num_features)   "the trailing seq axis exists so balance_single_task's
                                    _SQUEEZE_SEQ_DIM_TYPES path can squeeze it"
WGAN        (n, 1, F)              "3D with seq_len=1"
CTAB-GAN+   pd.DataFrame           rows
CGAN        (n, seq_len, F)        the only sequence generator in the repo — unmaintained
```

Verified against the running code (`GANs/df_tabddpm_mlx.py:750-753`, `GANs/GANInterface.py:378-395`,
`GANs/df_wgan_gp.py:1374`), not against documentation. **This is not a defect**: they are
class-conditional row samplers built for class balancing, and they do that job. It is a statement
about what they cannot be reused for.

---

## R8 — The expected outcome, stated before the work

The bootstrap passes calibration on the body and the interesting question is the tail. The stress
run then produces the feature's actual deliverable: a distribution over plausible re-orderings of
real market conditions, answering *how much of the incumbent's record is orderings-of-luck?*

Given brief 05 measured the incumbent at roughly **+0.17 pp/yr** on the pooled test folds under
this same surrogate, the prior is that a large fraction of sampled paths lose money. **That would
be a finding, not a failure** — it would say the incumbent's positive record is thin against
resampling noise, which is consistent with a 3.35 pp/yr floor it does not clear.

What would change the conclusion: a loss-path fraction well under half with a tight P&L
distribution, which would say the edge is structural rather than path-specific.
