# Brief 04 — Reinforcement learning in our own pooled environment

**Family:** deep RL (PPO / MaskablePPO via stable-baselines3), *not* FreqAI's RL module
**Type:** unsupervised · **Priority:** 4 — only after brief 1 has a resolved result
**Budget:** 3–4 sessions · **Depends on:** brief 1 simulator (as the environment's step function)

## Hypothesis

Path-dependent decisions that the differentiable simulator can only approximate smoothly —
discrete stop placement, the `stoploss_grace` logic, holding-time limits, "exit now vs wait
one more bar" — may be better learned by a policy-gradient method that samples them. If RL
cannot beat brief 1's differentiable policy on the same environment, the extra variance is
not worth it and the family is closed.

## Why not FreqAI's RL module (verified against `develop`, Sept 2026)

- Default reward: +25 per entry, −1 per idle candle, `pnl × 100` on exit (duration-scaled,
  boosted above the profit target) → pays for churn; the docs themselves call it a showcase
  that agents "cheat".
- `add_state_info` (position, P&L, duration in the observation) **raises in backtesting**; a
  position-aware agent can only be evaluated in dry-run — incompatible with the protocol.
- `best_model.zip` selected by shaped reward on the test split (selection leak).
- One agent per pair on a sliding window (~2,880 candles for 30 days of 15 m).
- Ignores `custom_exit` / `custom_stoploss`, which are the repo's live levers.
- Runs on CPU only (no MPS path in `BaseReinforcementLearningModel`); fine for an MLP.
Use SB3 directly against our own `gymnasium` environment instead. The freqtrade wrapper from
brief 1 serves for final evaluation.

## Prior evidence

Zhang, Zohren & Roberts 2020 (strong, daily futures, costs in reward, DQN Sharpe 1.29). Against:
Kruthof & Müller 2025 (SAC ≯ 1/N, negative at 0.1% costs); Théate & Ernst 2021 (seed
sensitivity); Gort et al. 2022 (8-week crypto test); Bai 2024 / Hoque 2025 (algorithm choice
does not matter, shaped rewards do, costs mostly ignored). No credible 15-min spot result.

## Design

- **Environment:** `research/labelfree/env.py` wrapping brief 1's simulator step; **one shared
  policy across all pairs**, episodes sampled as random (pair, start) windows of 30–90 days
  from the fold's training range. Observation = normalised feature window + position + bars
  in trade + unrealised P&L (this is the state FreqAI cannot backtest).
- **Action:** discrete {flat, long} with action masking (MaskablePPO) so "sell" is only valid
  when long, or continuous target exposure in [0,1] through the no-trade band. Test both.
- **Reward:** per-step `position × log_return − (fee + spread + slippage)·|Δw_exec|` with the
  liquidity cap, plus an optional differential-Sharpe term. **No entry bonus, no idle penalty,
  no take-profit multiplier.** Episode terminates on `max_training_drawdown_pct` as in FreqAI.
- **Algorithm:** PPO (SB3 ≥ 2.8). Do not spend sessions on algorithm comparison — the
  reviews say it does not matter. Spend them on seeds.
- **Model selection:** by *training* reward only; the test fold is never touched until the
  final pooled evaluation.
- **Compute:** CPU, `cpu_count` ≈ performance cores, vectorised envs. Budget ~1–2 M steps
  per fold; 15 folds × 5 seeds sequentially per AGENT_GUIDE's "run everything sequentially".

## Experiments

1. Reward-hacking probe: train with the FreqAI default reward in our env and confirm the agent
   churns (documents why the reward was replaced).
2. PPO discrete-masked vs continuous-band × 5 seeds on fold f07.
3. Best variant, full 15-fold pooled run; IQM across seeds.
4. Head-to-head with brief 1's differentiable policy on the identical environment and folds.
5. Cost stress.

## Acceptance / kill

Protocol §5, plus: **must beat brief 1's best pooled result by the floor**, not just
`NNNC_MLX`. If it only matches brief 1, close the family and note that the differentiable
estimator is preferred (lower variance, backtestable state).

## Workflow (mandatory — see README "How to run a brief")

```
/speckit-specify   <the starter prompt below, verbatim>      → specs/0NN-labelfree-rl-pooled-env/spec.md
/speckit-clarify                                              → resolve markers; stats/windows come from 00-evaluation-protocol.md
/speckit-plan                                                 → plan.md, research.md (cite 01-literature-review.md)
/speckit-tasks                                                → tasks.md: first tasks = calibration/leakage tests, last = §6 report + trial-log audit
/speckit-analyze   (REQUIRED before implement; fix every CRITICAL, re-run until clean; save as analyze.md)
/speckit-implement                                            → sequential runs only
/speckit-converge  (if a session ends mid-feature)
/speckit-analyze   (again on completion)
write briefs/results/04-<date>.md linking the feature dir and both analyze reports
```

The spec's success criteria must be protocol §5 verbatim or stricter. Any brief-specific
kill criteria above become additional SC-lines.

## Starter prompt (pass to `/speckit-specify` verbatim)

```
Feature: labelfree-rl-pooled-env. This is research/label-free-study brief 04; the feature must be
specified, planned, task-listed, analyzed with /speckit-analyze, and only then implemented.
Read AGENT_GUIDE.md, research/label-free-study/README.md, 00-evaluation-protocol.md,
briefs/01-direct-policy-optimisation.md (reuse its simulator as the env step function)
and briefs/04-rl-pooled-environment.md. Implement research/labelfree/env.py as a gymnasium
environment with a single shared policy over all whitelist pairs, the reward specified in
the brief, and action masking for long-only. Use stable-baselines3 PPO/MaskablePPO on CPU.
Never select models on the test fold. Log every configuration to
research/labelfree/trials.jsonl before running; report with IQM over ≥5 seeds; write
briefs/results/04-<date>.md per protocol §6, including the head-to-head with brief 1.
Success criteria: the acceptance and kill gates of research/label-free-study/
00-evaluation-protocol.md §5 plus the brief's own kill criteria, stated as measurable
SC-lines. Non-goals: the adjacent approaches listed in briefs/09-excluded-and-why.md.
```
