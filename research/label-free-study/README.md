# Label-free study: can a model learn to trade by maximising profit directly?

**Status:** research programme, 2026-09-10. No code yet. Each brief in `briefs/` is written so
it can be handed to Claude Code (or turned into a `specs/0NN-*` feature via `/speckit-specify`)
as a self-contained investigative project.

## The question

Every ML strategy in this repo (NNNC, NNMT, NNPredict, Sklearn) is trained the same way:
`Framework/TrainingSignals.py` looks *forward* through historical data, stamps each bar with a
buy / hold / sell label (forward excursion, triple barrier, quantile, …), and a classifier is
trained to reproduce those labels. Trading then happens through thresholds, guards and exits
that were designed separately from the label.

The question this study asks: **can we drop the hand-designed label and let the model optimise
the trading objective itself** — realised P&L net of fees, or a risk-adjusted version of it —
either fully unsupervised (raw features → position → profit) or semi-supervised (labels as a
warm start, profit as the final objective)? Variants the study covers: end-to-end
differentiable policies, reinforcement learning, recursively / online-adapted models, generative
(diffusion / GAN) world models, evolutionary search, self-supervised representations.

## Why this is worth doing *here* — the repo's own findings motivate it

The regime studies in `regime/` measured a set of facts that make the supervised pipeline look
like the bottleneck, not the data. Every approach below is chosen because of one of these:

| Finding (file) | What it says | Why it points to label-free |
|---|---|---|
| **Learnability and tradeability are opposed** (`regime/LABEL_CEILING_VS_LEARNABILITY.md`) | MCC falls 0.78 → 0.28 while EV/signal rises 3.6% → 9.0% as the label threshold rises. The label the model learns best is the one that cannot pay. | A classifier is optimising the wrong thing. An objective that *is* the P&L does not have this conflict by construction. |
| **The model captures <4% of the ceiling** (same file) | Raising the label 0.007 → 0.02 improved the oracle by +45pp; the real model captured +1.75pp. | The gap is a model/objective problem, not a data problem. |
| **Break-even precision ≈ 0.25** (`regime/CLOSING_THE_LEARNABILITY_GAP.md`) | You do not need near-perfect prediction to be profitable. | A weak but *cost-aware* policy can pay; the objective must include costs. |
| **The gap was mostly an operating-point problem** (same file) | Sweeping the confidence threshold without retraining took W1 from −12.7% to +6.9%. | The threshold is a crude, post-hoc stand-in for what a profit objective would learn directly (when to act, how big, when to stay flat). |
| **Information ceiling ρ ≈ 0.15** (`us_spot_market_study.md`) | Hand-crafted indicators, an 85-indicator battery and CNNs on raw OHLCV all converge on the same predictability. | **Expectation management.** A new objective cannot create information. It can only change *how much of the existing ~ρ 0.15 is converted into net P&L* — which is exactly the <4% capture problem above. |
| **Fills and fees decide everything** (`us_spot_market_study.md` walls 2–3; `regime/EXIT_DIAGNOSIS.md`) | Realistic fills removed ~73% of one strategy's trades; the edge lives in illiquid moments. | Costs and liquidity caps must sit *inside* the training objective, not be applied afterwards. Every brief requires this. |
| **The measurement floor** (`regime/WALKFORWARD_POWER.md`) | Single-window noise floor 7.6pp; pooled walk-forward floor 3.35pp/yr. | Only effects ≥ 3pp/yr are resolvable. Every brief has a kill criterion expressed against this floor; nothing smaller is worth building. |
| **Regime instruments failed** (`regime/VERDICT.md`, Phase C: FAIL) | No vol/Hurst/autocorr/breadth gate beat the fixed choice on two distant windows. | Unsupervised *regime gating* is demoted to low priority; only a method with a genuinely different mechanism (jump models with explicit switching cost) is kept, as a cheap negative-control experiment. |

## What the literature says (full review in `01-literature-review.md`)

Four independent surveys (RL; direct policy optimisation; generative / world models;
self-supervised, adaptive and evolutionary methods) were run in September 2026. The honest
picture:

1. **Directly optimising a profit / Sharpe / utility objective by gradient descent works, but
   only with strong structure** — costs inside the loss, a turnover penalty or no-trade band,
   small models, seed ensembles, and a robust (worst-window) objective. This is the
   Moody–Saffell → Deep Momentum Networks → Momentum Transformer → DeePM line (Oxford-Man,
   1998–2026). It is the best-evidenced label-free family and maps almost 1:1 onto the repo's
   problem. Caveat: evidence is on daily liquid futures; the edge in those papers dies at
   2–3 bp of cost unless costs are trained on. Binance.US taker fees are ~10 bp per side.
2. **Deep RL (PPO/SAC/DQN) has a thin honest record.** The one strong positive result
   (Zhang, Zohren & Roberts 2020) is daily futures with costs in the reward. A 300-year OOS
   study found SAC does not beat 1/N and goes negative at 0.1% costs. FreqAI's built-in RL
   module is a framework, not a strategy: its default reward pays +25 per entry (teaches
   churn), a position-aware agent cannot be backtested, and model selection is on shaped
   reward. RL is worth one carefully-designed experiment, *after* the differentiable version,
   and with our own pooled environment rather than FreqAI's.
3. **Generative / diffusion world models have no credible alpha-transfer record.** Agents
   trained inside learned simulators exploit the simulator's flaws (Coletta et al. 2023);
   the most accurate crypto generator in CTBench had *negative* CAGR ("accuracy–alpha gap").
   The defensible use of the repo's existing DDPM / WGAN / CTAB-GAN+ generators is
   *stress-testing* a policy (Tail-GAN style), not training one.
4. **Return-conditioned generative policies (Decision Transformer / Decision Diffuser)
   provably copy luck in stochastic environments** (Paster et al. 2022). Excluded.
5. **Self-supervised representations learn volatility and similarity, not conditional mean.**
   Generic time-series foundation models have negative OOS R² on returns. Cheap to test as
   extra features; expect a null.
6. **Online / continual adaptation results are mostly leakage.** Leak-free re-evaluations
   shrink the gains by 11–40pp. FreqAI's own docs call `continual_learning` experimental
   and overfit-prone. Scheduled full retraining on a rolling window is the safer default.
7. **Evolutionary search (GA/GP/NEAT) is negative after costs** going back to Allen &
   Karjalainen 1999; a 2025 crypto-15m neuroevolution study went from +300% validation APY
   to −70% live. Gradient-free search on backtest P&L is a multiple-testing machine; only
   usable with PBO/DSR controls, and only for non-differentiable pieces (stops, guards).
8. **Meta-labelling** (a secondary model that learns when to trust the primary signal) has
   modest but real OOS support and is the cheapest semi-supervised bridge available.
9. **LLM trading agents:** advantage disappears on long, broad, post-cutoff tests. Excluded.

## The approaches, ranked

Ranking weighs (a) evidence quality, (b) fit to the repo's measured problems, (c) cost to
build given existing infrastructure (MLX classifiers, pooled walk-forward harness, GANs,
backtest archive), (d) expected effect size relative to the 3.35pp/yr floor.

| # | Brief | Family | Type | Evidence | Build cost | Verdict |
|---|---|---|---|---|---|---|
| 1 | [`briefs/01-direct-policy-optimisation.md`](briefs/01-direct-policy-optimisation.md) | Differentiable P&L, no labels | unsupervised | medium-strong (futures) | medium | **Do first.** Directly attacks the learnability/tradeability conflict and the operating-point problem. Reuses features, scalers, MLX, harness. |
| 2 | [`briefs/02-label-warmstart-pnl-finetune.md`](briefs/02-label-warmstart-pnl-finetune.md) | Pretrain on labels, fine-tune on P&L | semi-supervised | medium (blends beat either alone in DFL work) | low (given #1) | **Do second.** Cheapest path from the existing NNNC/NNMT weights; the natural A/B against #1. |
| 3 | [`briefs/03-meta-labelling-from-trade-archive.md`](briefs/03-meta-labelling-from-trade-archive.md) | Secondary model on realised trade outcomes | semi-supervised | medium-weak but consistent | low | **Do in parallel** — can be built by a separate Claude Code session; the labels are the repo's own backtest trades. |
| 4 | [`briefs/04-rl-pooled-environment.md`](briefs/04-rl-pooled-environment.md) | PPO / MaskablePPO in our own gym env | unsupervised | medium (1 strong paper, many failures) | medium-high | Only after #1 establishes a differentiable baseline; RL must beat it to be justified. |
| 5 | [`briefs/05-gradient-free-policy-search.md`](briefs/05-gradient-free-policy-search.md) | CMA-ES / OpenAI-ES on the vectorised backtest | unsupervised | weak | low-medium (given #1's simulator) | For the non-differentiable parts only (stop, grace, guards). High PBO risk; strict trial accounting. |
| 6 | [`briefs/06-generative-stress-harness.md`](briefs/06-generative-stress-harness.md) | Existing DDPM/WGAN/CTAB as robustness oracle | infrastructure | strong (as a *negative* result for training) | low-medium | Not an alpha source. Builds the tool that every other brief's "robustness" gate needs. |
| 7 | [`briefs/07-self-supervised-features.md`](briefs/07-self-supervised-features.md) | TS2Vec / Kronos embeddings as extra inputs | unsupervised | weak; expect null | low | Cheap null test. Kill fast. |
| 8 | [`briefs/08-jump-model-regime-gate.md`](briefs/08-jump-model-regime-gate.md) | Statistical jump model with switching cost | unsupervised | medium (equity indices, drawdown only) | low | Negative control against `regime/VERDICT.md`. Risk-side, not alpha. |
| — | [`briefs/09-excluded-and-why.md`](briefs/09-excluded-and-why.md) | Diffusion planners, Dreamer, LLM agents, NEAT/GP, TSFM fine-tuning, FreqAI continual learning | — | negative | — | Documented so they are not re-proposed. |

## Recommended sequence

```
Phase 0  (1 session)   00-evaluation-protocol.md → freeze the protocol; write the shared
                       differentiable simulator (`research/labelfree/sim.py`) — used by 1, 2, 4, 5, 6.
Phase 1  (2–4 sessions) Brief 1 (direct policy).            Gate: beats NNNC_MLX control by ≥3pp/yr pooled.
         (parallel)     Brief 3 (meta-label).                Gate: same.
Phase 2  (1–2 sessions) Brief 2 (warm-start). A/B vs brief 1 and vs NNNC_MLX. Decide the family.
Phase 3  (optional)     Brief 6 (stress harness) → apply to the winner of phase 2.
                       Brief 7 (SSL features) as a one-session null test.
Phase 4  (only if 1/2 show ≥3pp/yr) Brief 4 (RL) and brief 5 (ES) — must beat phase-2 winner.
Phase 5  (low priority) Brief 8 (jump model gate) as a control against VERDICT.md.
```

Stop the programme if phase 1 and 2 both come back "not resolved": the literature and the
repo's ρ≈0.15 ceiling both say the remaining upside is then structural (venue, information),
not algorithmic — which is the conclusion `us_spot_market_study.md` already reached for the
supervised pipeline.

## Non-negotiables shared by every brief (details in `00-evaluation-protocol.md`)

- Costs, spread and the ≤10%-of-candle-volume fill cap are **inside** the training objective.
- Act on bar *t* close, fill at *t+1*; no other lookahead anywhere in the simulator.
- Evaluation only through the pooled walk-forward harness (`regime/wf/`), quoting the
  3.35pp/yr floor; W1 single-window numbers are smoke tests.
- Every trial (seed, loss variant, γ, architecture, threshold) is logged and counted for the
  Deflated Sharpe Ratio / PBO.
- Mandatory baselines: `NNNC_MLX` control (current production), equal-weight buy-and-hold of
  the whitelist, a TSMOM rule with a no-trade band, and "current classifier + cost-aware
  threshold" (the operating-point fix from `CLOSING_THE_LEARNABILITY_GAP.md`).
- Multi-seed (≥5) with interquartile mean and bootstrap CIs, never a single run.
- No result is admissible until it is reproduced by a fresh process from a saved config.
- Every brief runs as a spec-kit feature (`/speckit-specify` → `clarify` → `plan` → `tasks`
  → **`analyze`** → `implement` → `analyze`); see "How to run a brief" below. No spec, no result.

## How to run a brief with Claude Code — spec-kit is mandatory

Every brief is executed as a spec-kit feature in `user_data/strategies/specs/0NN-<slug>/`
(next number after the latest existing feature; 007 is the pooled walk-forward harness).
**No code is written outside this workflow**, and no result is admissible unless the feature
directory holds a `spec.md`, `plan.md`, `tasks.md` and a passing `/speckit-analyze` report.
The briefs fix the *hypothesis, design constraints, acceptance gate and kill criteria*; the
spec-kit artefacts turn those into verifiable requirements and dependency-ordered work.

Required sequence, in one or more Claude Code sessions from the repo root:

| Step | Command | What it must produce for this study |
|---|---|---|
| 1 | `/speckit-specify` with the brief's starter prompt | `spec.md` whose functional requirements restate the brief's hypothesis, design constraints and **the acceptance/kill gates from `00-evaluation-protocol.md` §5 as success criteria** (SC-lines), and whose non-goals list the excluded approaches from `briefs/09-excluded-and-why.md` that are adjacent. |
| 2 | `/speckit-clarify` | Resolve any `[NEEDS CLARIFICATION]` markers. Questions about *statistics or windows* are answered by the protocol, not by inventing new ones. |
| 3 | `/speckit-plan` | `plan.md` + `research.md` referencing `01-literature-review.md` for the design choices, and the constitution check against `.specify/memory/constitution.md`. |
| 4 | `/speckit-tasks` | `tasks.md` in which the **first tasks are the calibration / leakage tests** named in the brief and the **last tasks are the protocol §6 report and the trial-log audit**. |
| 5 | `/speckit-analyze` (**required, before any implementation**) | Cross-artefact consistency report. Every CRITICAL finding must be fixed and analyze re-run until clean. The final report is saved as `specs/0NN-<slug>/analyze.md` and linked from the brief's results file. |
| 6 | `/speckit-implement` | Executes `tasks.md`. Run everything sequentially (AGENT_GUIDE operational rules). |
| 7 | `/speckit-converge` (if a session ends mid-feature) | Appends the unbuilt remainder as tasks so the next session can continue rather than restart. |
| 8 | `/speckit-analyze` again on completion | Confirms the delivered code, spec and tasks still agree before the result is written up. |
| 9 | Write `briefs/results/<brief>-<date>.md` (protocol §6) | Must link the feature directory and the two analyze reports. |

Rules:
- A brief that reaches a kill criterion is still closed through step 8–9 — the negative result
  is the deliverable, and the spec is what makes it citable later.
- Do not skip `/speckit-analyze` because the feature "is small". The label-free methods
  optimise the evaluation metric directly; the consistency check between the spec's success
  criteria and what the tasks actually measure is the guard against a silently weakened gate.
- `/speckit-checklist` is optional; use it for the protocol's non-negotiables when a feature
  touches the simulator or the fold plan.
- The starter prompts below are written to be passed to `/speckit-specify` verbatim.

## Files

```
research/label-free-study/
├── README.md                          ← this file
├── 00-evaluation-protocol.md          ← shared harness, baselines, statistics, trial accounting
├── 01-literature-review.md            ← consolidated, cited survey (Sept 2026)
└── briefs/
    ├── 01-direct-policy-optimisation.md
    ├── 02-label-warmstart-pnl-finetune.md
    ├── 03-meta-labelling-from-trade-archive.md
    ├── 04-rl-pooled-environment.md
    ├── 05-gradient-free-policy-search.md
    ├── 06-generative-stress-harness.md
    ├── 07-self-supervised-features.md
    ├── 08-jump-model-regime-gate.md
    └── 09-excluded-and-why.md
```
