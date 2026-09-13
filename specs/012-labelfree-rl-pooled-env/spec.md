# Feature Specification: labelfree-rl-pooled-env

**Feature Branch**: `012-labelfree-rl-pooled-env`

**Created**: 2026-09-12

**Status**: Draft — clarified

**Input**: User description: "Feature: labelfree-rl-pooled-env. This is research/label-free-study brief 04; the feature must be specified, planned, task-listed, analyzed with /speckit-analyze, and only then implemented. Read AGENT_GUIDE.md, research/label-free-study/README.md, 00-evaluation-protocol.md, briefs/01-direct-policy-optimisation.md (reuse its simulator as the env step function) and briefs/04-rl-pooled-environment.md. Implement research/labelfree/env.py as a gymnasium environment with a single shared policy over all whitelist pairs, the reward specified in the brief, and action masking for long-only. Use stable-baselines3 PPO/MaskablePPO on CPU. Never select models on the test fold. Log every configuration to research/labelfree/trials.jsonl before running; report with IQM over ≥5 seeds; write briefs/results/04-<date>.md per protocol §6, including the head-to-head with brief 1. Success criteria: the acceptance and kill gates of research/label-free-study/00-evaluation-protocol.md §5 plus the brief's own kill criteria, stated as measurable SC-lines. Non-goals: the adjacent approaches listed in briefs/09-excluded-and-why.md."

## Context

The hypothesis is narrow and worth stating precisely, because it is **not** "RL beats the
existing strategy". It is: *path-dependent decisions the differentiable simulator can only
approximate smoothly — discrete stop placement, stoploss grace, holding limits, "exit now vs
wait one more bar" — may be better learned by a method that samples them.* If a policy-gradient
method cannot beat brief 1's differentiable policy on the same environment, the extra variance
is not worth it and the family closes.

### What the study has already established

This feature is the fourth in a sequence, and the first three constrain it:

| brief | verdict | what it rules out |
|---|---|---|
| 01 direct policy optimisation | **NOT RESOLVED** — no configuration reached a pooled run | descending net-of-cost return **from noise** |
| 02 label warm-start + fine-tune | **NOT RESOLVED** — advantage was 92% turnover, not signal | descending it **from a warm start**; and the shared trainer carries two open defects |
| 03 meta-labelling | **KILL** — nested AUC 0.4978 | a secondary model over the primary's trades; and showed the confidence→outcome map is **non-stationary across 90-day windows** |

Brief 04 descends the same objective as 01 and 02, through a different optimiser. The
non-stationarity finding from 03 applies to any method that fits one window and acts in the
next, which includes a policy trained per fold. **This is recorded so the gates are set with it
in view, not as an argument against running the feature** — the brief's own question (does
sampling beat smoothing on path-dependent decisions?) is genuinely untested.

### Two defects inherited from the shared trainer

Both are open and both reach this feature if it reuses brief 1's simulator:

- a **−0.7707 log regression** on an identical nominal configuration between briefs 01 and 02,
  with turnover tripling — unexplained
- **`band_hold` bimodality**: at a fixed band width, seeds land in one of two attractors
  (churn at ~600 turnover, or frozen at ~0.1), so any turnover setting is a coin flip

## Clarifications

### Session 2026-09-12

- Q: Brief 04 requires beating "brief 1's best pooled result", but brief 1 was NOT RESOLVED and
  no configuration ever reached a pooled run. What is the comparator? → A: **Split the gate.**
  The brief's question — does sampling beat smoothing on path-dependent decisions? — is answered
  by a **screening-fold head-to-head** against brief 1's differentiable policy on identical
  folds, costs and caps. Protocol §5 remains a **separate** acceptance bar. Two questions, two
  gates, rather than one gate referencing a number that does not exist.
- Q: The observation includes position, bars-in-trade and unrealised P&L — the state the brief
  says FreqAI cannot backtest — yet the final evaluation runs through freqtrade's vectorised,
  position-blind entry population. How is a position-aware policy evaluated? → A: **Keep the
  position state and evaluate through `confirm_trade_entry` / `custom_exit`**, which are
  per-candidate and do see the trade. Dropping the state would discard the brief's stated
  advantage over FreqAI. The head-to-head with brief 1 is unaffected because it runs in the
  simulator, where both policies see an identical environment.
- Q: The brief budgets ~10^8 CPU steps sequentially. What is the ceiling and stop rule? → A:
  **Screen on one fold first; fund the full 15-fold run only if the screen clears a
  pre-registered bar.** No wall-clock ceiling — time is not the binding constraint. Because
  time is not binding, the bar is set on scientific grounds and deliberately weak: the policy
  must be **non-degenerate and distinguishable from brief 1's**, not visibly better. A marginal
  screen is funded; a degenerate or indistinguishable one is not.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A pooled environment that cannot be cheated (Priority: P1) 🎯 MVP

A researcher needs an environment whose reward is the realised economics of trading — and
positive evidence that a reward-hacking agent is *visible* in it rather than silently rewarded.

**Why this priority**: every later story trains against this reward. A reward that pays for
churn produces a confident, worthless agent, and the brief exists partly because FreqAI's
shipped reward does exactly that. This story is also standalone: an auditable pooled trading
environment is reusable by briefs 5 and 6 whether or not PPO works.

**Independent Test**: train against the FreqAI-style reward and confirm the agent churns and
that the environment reports it; then train against the specified reward and confirm it does
not.

**Acceptance Scenarios**:

1. **Given** the environment, **When** a step is taken, **Then** the reward is the position's
   log return net of fee, spread and slippage on the executed exposure change, under the
   liquidity cap — with **no entry bonus, no idle penalty and no take-profit multiplier**.
2. **Given** a deliberately churn-rewarding reward, **When** an agent trains on it, **Then**
   its turnover rises sharply and the run is flagged — an environment that cannot exhibit
   reward hacking cannot be trusted to have avoided it.
3. **Given** a long-only mandate, **When** the agent is flat, **Then** the "sell" action is
   masked as invalid rather than penalised.
4. **Given** an episode, **When** drawdown exceeds the configured maximum, **Then** the episode
   terminates.

---

### User Story 2 - One shared policy across all pairs (Priority: P2)

A researcher trains a single policy over episodes sampled as random (pair, start) windows from
a fold's **training** range, so the policy learns a pair-agnostic decision rule.

**Why this priority**: this is the structural difference from FreqAI's per-pair sliding-window
agents, and it is what makes a pooled evaluation coherent.

**Independent Test**: confirm every training episode's window lies inside the fold's training
range, and that one parameter set serves all pairs.

**Acceptance Scenarios**:

1. **Given** a fold, **When** episodes are sampled, **Then** no episode window overlaps the
   fold's test range.
2. **Given** the trained policy, **When** it is evaluated, **Then** the same parameters are
   applied to every pair without per-pair adaptation.
3. **Given** a pair absent from a fold's training range, **When** it appears at evaluation,
   **Then** the policy still acts and the case is counted.

---

### User Story 3 - Model selection that never sees the test fold (Priority: P3)

A researcher selects checkpoints and variants using **training reward only**; the test fold is
untouched until the final pooled evaluation.

**Why this priority**: selecting on the test split is the specific defect the brief cites in
FreqAI's `best_model.zip`, and it is the cheapest way to manufacture a result that evaporates.

**Independent Test**: audit every selection decision and confirm none consumed a test-fold
quantity.

**Acceptance Scenarios**:

1. **Given** checkpointing, **When** a checkpoint is chosen, **Then** the criterion is computed
   on training episodes only.
2. **Given** the screening comparison, **When** a variant is chosen, **Then** it is ranked on a
   held-out range that is not the fold under evaluation.
3. **Given** any selection, **When** it is recorded, **Then** the quantity used is logged so the
   audit can verify it.

---

### User Story 4 - Seeds before variants (Priority: P4)

A researcher spends the compute budget on seeds rather than on algorithm comparison, and reports
an interquartile mean with an interval, never a best seed.

**Why this priority**: the literature the brief cites is consistent that algorithm choice does
not matter and seed sensitivity is severe. A single-seed RL result is not evidence.

**Independent Test**: every reported figure is an IQM over ≥5 seeds with its interval, and the
per-seed spread is reported beside it.

**Acceptance Scenarios**:

1. **Given** a configuration, **When** it is reported, **Then** at least five seeds contribute
   and the figure is an interquartile mean with a bootstrap interval.
2. **Given** a seed spread wider than the effect being claimed, **When** results are reported,
   **Then** the configuration is recorded as unresolved rather than as an effect.

---

### User Story 5 - The head-to-head that decides the family (Priority: P5)

A researcher compares the RL policy against brief 1's differentiable policy **on the identical
environment and folds**, and records whether sampling beat smoothing.

**Why this priority**: this is the brief's actual question. It is gated last because it is only
meaningful once the environment is trustworthy (US1) and the result is seed-stable (US4).

**Independent Test**: both policies evaluated on the same folds through the same simulator, with
the difference reported against the floor.

**Acceptance Scenarios**:

1. **Given** both policies, **When** they are compared, **Then** they run on identical folds,
   identical cost assumptions and identical liquidity caps.
2. **Given** the comparison, **When** RL only matches the differentiable policy, **Then** the
   family is closed and the report records that the differentiable estimator is preferred for
   lower variance and backtestable state.

### Edge Cases

- **The agent learns to never trade.** Zero exposure yields zero reward, which beats losing.
  This is the degenerate solution and must be reported as inconclusive, never as a return.
- **An episode terminates on drawdown immediately.** Its truncated return must not be averaged
  in as though it were a completed episode.
- **A pair has a data gap inside a sampled window.** The episode is rejected and counted.
- **Two seeds produce identical trades.** That is one measurement, not two.
- **The policy is position-aware but the final evaluation is vectorised.** A policy whose action
  depends on current position cannot be expressed by a position-blind entry rule; this must be
  resolved before the final evaluation, not discovered during it.
- **Training diverges.** A non-finite loss or reward must fail loudly rather than produce a
  policy that silently acts on NaN.

## Requirements *(mandatory)*

### Functional Requirements

**Environment**

- **FR-001**: The environment MUST use brief 1's simulator as its step function rather than
  reimplementing the economics.
- **FR-002**: The per-step reward MUST be the position's log return net of fee, spread and
  slippage on the executed exposure change, under the liquidity cap.
- **FR-003**: The reward MUST NOT contain an entry bonus, an idle penalty or a take-profit
  multiplier.
- **FR-004**: Long-only MUST be enforced by masking invalid actions, not by penalising them.
- **FR-005**: An episode MUST terminate when drawdown exceeds the configured maximum.
- **FR-006**: The environment MUST be able to demonstrate reward hacking under a deliberately
  churn-rewarding reward; if it cannot, its reward design is unverified.
- **FR-007**: Non-finite rewards or losses MUST fail the run rather than propagate.

**Policy and training**

- **FR-008**: A single shared policy MUST serve all pairs, with no per-pair parameters.
- **FR-009**: Training episodes MUST be sampled only from the fold's training range.
- **FR-010**: Both action parameterisations named in the brief MUST be screened — discrete with
  masking, and continuous target exposure through a no-trade band.
- **FR-011**: Compute MUST be spent on seeds rather than on algorithm comparison.

**Selection and evaluation**

- **FR-012**: No selection decision — checkpoint, variant or hyperparameter — MAY consume any
  quantity computed on the fold under evaluation.
- **FR-013**: Every configuration MUST be appended to the trial log before it runs.
- **FR-014**: Every reported figure MUST be an interquartile mean over at least five seeds with
  a bootstrap interval, never a best seed.
- **FR-015**: Seeds producing identical trades MUST count as one measurement.
- **FR-016**: Every reported return MUST carry mean exposure, trades per year and mean holding
  time.
- **FR-017**: A policy whose mean exposure falls below the participation floor MUST be reported
  as inconclusive with its return withheld.
- **FR-018**: The RL policy MUST be compared against brief 1's differentiable policy on
  identical folds, costs and liquidity caps.
- **FR-019**: Any swept parameter MUST be proven to change what it controls before its sweep
  runs.

**Process**

- **FR-020**: All runs MUST execute sequentially.
- **FR-021**: Only files this feature creates or modifies MAY be staged.
- **FR-022**: A result is admissible only with a specification, plan, task list and a clean
  analyze report produced before implementation and again after.
- **FR-023**: A delta below the floor MUST be reported as "not resolved", never as "no effect".
- **FR-024**: The result report MUST be written even on a kill.

### Key Entities

- **Episode**: a (pair, start, length) window drawn from a fold's training range.
- **Observation**: the normalised feature window plus the agent's own position state.
- **Action**: flat or long under masking, or a target exposure in [0, 1].
- **Step reward**: realised net-of-cost log return for that bar.
- **Policy**: one parameter set serving every pair.
- **Selection record**: which quantity chose a checkpoint or variant, and from which range.
- **Trial record**: a configuration written before its result exists.

## Success Criteria *(mandatory)*

### Protocol §5 acceptance gates

- **SC-001**: Pooled walk-forward annualised return is at least the control plus 3.35 pp/yr,
  with a bootstrap interval excluding zero.
- **SC-002**: The configuration is positive in at least 10 of 15 folds and in each of W1/W2/W3
  when run single-window.
- **SC-003**: The configuration beats all four mandatory baselines net of full costs.
- **SC-004**: Backtest-overfitting probability is at most 0.2 and the deflated risk-adjusted
  return is above zero, over the complete trial log.
- **SC-005**: The return delta remains positive at 1.5× fees.
- **SC-006**: A separate process reproduces the accepted result from its saved configuration
  and seed.

### Protocol §5 kill gates

- **SC-007**: If no configuration clears SC-001 within the budgeted work, the verdict is
  "not resolved", recorded as such and never as "no effect".
- **SC-008**: If any configuration's in-sample advantage exceeds its held-out advantage by more
  than the floor, it is recorded as overfitting and stopped.

### Brief-specific gates

- **SC-009**: On the screening fold, the RL policy MUST beat brief 1's differentiable policy by
  the floor, on identical folds, costs and liquidity caps, as an interquartile mean over ≥5
  seeds with an interval. This replaces the brief's "beat brief 1's best pooled result", whose
  referent does not exist — brief 1 was NOT RESOLVED and no configuration reached a pooled run.
  SC-001 remains the separate, unchanged acceptance bar.
- **SC-010**: If the RL policy only matches the comparator, the family is closed and the report
  records that the differentiable estimator is preferred for lower variance and backtestable
  state.
- **SC-011**: The reward-hacking probe demonstrates that a churn-rewarding reward produces
  measurably higher turnover in this environment; if it does not, the reward design is
  unverified and no downstream number is admissible.
- **SC-012**: No selection decision consumed a quantity computed on the fold under evaluation,
  and the audit can verify this from the recorded selection criteria.
- **SC-013**: Every reported return carries mean exposure, trades per year and mean holding
  time, so a policy that learned not to trade cannot rank on inaction.
- **SC-014**: Every configuration appears in the trial log, each written before its result.
- **SC-015**: Every swept parameter is proven live before its sweep runs.
- **SC-016**: The feature directory holds a specification, plan, task list and a clean analyze
  report produced before implementation and again after.
- **SC-017**: The policy's position state is preserved, and the final evaluation reaches it
  through per-candidate hooks that see the trade rather than through vectorised, position-blind
  entry population. Any divergence between the hook-driven trades and the simulator's is
  reported, not averaged away.
- **SC-018**: The full 15-fold run begins only after the screening fold shows the policy is
  **non-degenerate** (mean exposure above the participation floor) and **distinguishable from
  brief 1's policy** (their per-seed distributions are not identical). A screen failing either
  condition stops the feature and is reported as such.
- **SC-019**: The screening fold's role is pre-registered before it runs: it selects **only**
  between the two action parameterisations, and its contribution to the pooled verdict is
  reported separately so the selection pressure it carries is visible. Screening on a fold and
  then counting that fold in the verdict is the defect that voided 60 runs in feature 009.

## Assumptions

- Brief 1's simulator is reused unchanged as the step function; its calibration to the arbiter
  (0.00000000 bps across 99 W1 trades) carries over.
- The environment is `gymnasium`-shaped and trained with `stable-baselines3` / `sb3-contrib`
  PPO on CPU — all three are installed (gymnasium 1.3.0, SB3 2.9.0, sb3-contrib 2.9.0, torch
  2.13.0), and the machine has 14 physical cores (10 performance).
- FreqAI's RL module is used only as a documented counter-example, never as an implementation.
- The reward-hacking probe is a **positive control** for the environment, analogous to the
  planted-leak control in feature 011 and the liveness negative control in feature 010: a check
  that can fail is what makes a pass meaningful.
- Brief 02's two open trainer defects (the −0.7707 architecture regression and `band_hold`
  bimodality) are inherited if the continuous-band action space reuses that policy. They are
  not fixed by this feature.
- Non-stationarity across 90-day windows (brief 03) applies to a policy trained per fold; the
  feature does not assume it has been solved.
- Time is not a binding constraint on this feature. The staged budget exists to avoid spending
  ~10^8 steps on a configuration a cheap screen would reject (Principle VII), not to finish by
  a deadline — so the screening bar is set weak on purpose.
- The screening fold is f07, per the brief's experiment 2.

## Non-Goals

The approaches in `briefs/09-excluded-and-why.md` are out of scope, in particular: FreqAI's RL
module as shipped; offline RL (CQL/IQL) — which re-enters only if this feature's PPO fails for a
reason offline RL addresses; Decision-Transformer and diffusion planners; policies trained
inside a learned market simulator; LLM trading agents; and full decision-focused optimisation
layers.

Also out of scope: algorithm comparison beyond the two action parameterisations named;
modifying the primary strategy, its guards, its exits or its stop; and any per-pair parameter.
