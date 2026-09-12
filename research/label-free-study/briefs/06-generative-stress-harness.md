# Brief 06 — Generative models as a stress harness, not an alpha source

**Family:** diffusion / GAN market generators (the repo's TabDDPM, WGAN, CTAB-GAN+, MT variants)
**Type:** infrastructure · **Priority:** 6 · **Budget:** 1–2 sessions
**Depends on:** a candidate policy from brief 1/2/3 to stress

## Reframing the "diffusion approach"

The literature (review §D) is unambiguous: agents trained inside learned market simulators
learn the simulator's artefacts (Coletta 2023); the most faithful crypto generator had
negative CAGR (CTBench 2025); recursive self-training collapses tails (Shumailov 2024);
return-conditioned diffusion planners copy luck (Paster 2022). A generator cannot add
information the ~1.5 M real bars lack. So this brief does **not** train a policy on synthetic
data. It uses the generators the repo already has (`GANs/`, `rebuild.sh` steps 4) to answer
a question the real data cannot: *how does the candidate policy behave on plausible paths it
has never seen?* — which is the robustness gate every other brief needs.

## Hypothesis

A candidate that clears the pooled-WF floor on real data will show one of two behaviours on
generator-sampled paths: (a) its P&L distribution and drawdown tails are similar to real → the
edge is structural; (b) it collapses or explodes → it was fitted to path-specific features of
the 15 folds. Tail-GAN's criterion — judge a generator by whether it reproduces the P&L
distribution of the strategies you care about — is the calibration test for the generators
themselves.

## Design

1. **Generator calibration (Tail-GAN criterion).** For each generator type (TabDDPM, WGAN,
   CTAB-GAN+, and the multi-task variants), sample N feature-row sequences and run the
   *production* `NNNC_MLX` policy through the simulator. Compare the P&L distribution, VaR/ES
   and max-drawdown to the real pooled-WF trades (KS distance, ES ratio). Generators that
   fail are dropped — a generator that cannot reproduce the incumbent's P&L tails tells you
   nothing about a new policy's.
   Note: the existing generators produce *rows conditioned on labels* for class balancing;
   sequence coherence (`seq_len` windows) needs checking — if they do not generate coherent
   sequences, use a block bootstrap of real sequences or a GARCH-t path model as the
   generator, which is the required baseline anyway.
2. **Baseline generator:** stationary block bootstrap of real feature/return sequences
   (block = 1–3 days). Any conclusion drawn from the learned generators must also hold against
   the bootstrap; if the bootstrap alone suffices, the learned generators earn nothing.
3. **Stress runs:** the candidate policy on ≥1,000 sampled paths per surviving generator;
   report P&L quantiles, drawdown distribution, fraction of paths with negative net return,
   and the same for the incumbent.
4. **Adversarial variant (optional):** perturb generator conditioning (volatility regime,
   BTC drift) per the Bayesian-robust-trading design, to find the regime that breaks the
   candidate.

## Experiments / deliverables

- `research/labelfree/stress.py` + per-generator calibration report.
- A one-page stress summary for each accepted candidate from briefs 1–5, attached to its
  `briefs/results/` file.
- Explicit note if compute is a constraint: sampling from the DDPM is the bottleneck; sample
  offline into a fixed bank of paths, never inside a training loop.

## Acceptance / kill

This brief is infrastructure; its "acceptance" is that at least one generator (or the
bootstrap) passes the Tail-GAN calibration on the incumbent. Kill: if none passes, record it —
it is a finding about the repo's GANs (see `docs/GAN_SCORECARD.md`) — and use the bootstrap
alone as the stress harness.

**Never** promote a policy on stress-harness results alone; they are a veto, not a gate.

## Workflow (mandatory — see README "How to run a brief")

```
/speckit-specify   <the starter prompt below, verbatim>      → specs/0NN-labelfree-generative-stress/spec.md
/speckit-clarify                                              → resolve markers; stats/windows come from 00-evaluation-protocol.md
/speckit-plan                                                 → plan.md, research.md (cite 01-literature-review.md)
/speckit-tasks                                                → tasks.md: first tasks = calibration/leakage tests, last = §6 report + trial-log audit
/speckit-analyze   (REQUIRED before implement; fix every CRITICAL, re-run until clean; save as analyze.md)
/speckit-implement                                            → sequential runs only
/speckit-converge  (if a session ends mid-feature)
/speckit-analyze   (again on completion)
write briefs/results/06-<date>.md linking the feature dir and both analyze reports
```

The spec's success criteria must be protocol §5 verbatim or stricter. Any brief-specific
kill criteria above become additional SC-lines.

## Starter prompt (pass to `/speckit-specify` verbatim)

```
Feature: labelfree-generative-stress. This is research/label-free-study brief 06; the feature must be
specified, planned, task-listed, analyzed with /speckit-analyze, and only then implemented.
Read AGENT_GUIDE.md ("Adding to / extending the GAN system"), GANs/README.md,
docs/GAN_SCORECARD.md, research/label-free-study/README.md, 00-evaluation-protocol.md and
briefs/06-generative-stress-harness.md. Implement research/labelfree/stress.py: (1) a
stationary block-bootstrap path generator over real feature/return sequences, (2) adapters
that sample coherent sequences from each existing GAN type via GANInterface (report which
types cannot produce coherent sequences), (3) the Tail-GAN calibration of each generator
against the incumbent NNNC_MLX's pooled-WF P&L distribution, (4) a stress report for a
candidate policy path passed on the command line. Sample offline into a bank; never sample
inside a training loop. Write briefs/results/06-<date>.md.
Success criteria: the acceptance and kill gates of research/label-free-study/
00-evaluation-protocol.md §5 plus the brief's own kill criteria, stated as measurable
SC-lines. Non-goals: the adjacent approaches listed in briefs/09-excluded-and-why.md.
```
