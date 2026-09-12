# Brief 05 — Gradient-free policy search on the real backtest (CMA-ES / OpenAI-ES)

**Family:** evolution strategies over network weights and discrete exit parameters
**Type:** unsupervised (objective = pooled net P&L or robust Sharpe from the simulator)
**Priority:** 5 · **Budget:** 1–2 sessions · **Depends on:** brief 1's simulator (vectorised)

## Hypothesis

The parts of the strategy that gradient descent cannot reach — the ATR stop multiplier,
`stoploss_grace_*`, the guard thresholds, minimum holding time — jointly matter
(`regime/EXIT_DIAGNOSIS.md`, `AGENT_GUIDE.md` "live levers") but were only ever swept one at a
time, on windows that cannot resolve them. Optimising them **jointly with a small policy
network** by evolution strategies directly on the pooled-fold objective may find an operating
point that one-at-a-time sweeps cannot.

This is the "dynamically adjusting model" idea in its most defensible form: the *search* is
over structure and discrete parameters, but the evaluation is the honest simulator.

## Prior evidence — mostly negative, so the design is defensive

Allen & Karjalainen 1999 (GA rules: no excess return after costs); NEAT 2025 (below
buy-and-hold); *Red Queen's Trap* 2025 (15-m crypto neuroevolution: +300% validation → −70%
live); MadEvolve 2026 (gap widens with search-space size). ES/CMA-ES is a valid policy-search
method (Salimans 2017; Hansen CMA-ES) with **no rigorous OOS trading evidence**. The
consistent failure mode is multiple testing: every candidate is a trial.

## Design

- **Objective:** brief 1's utility (or robust Sharpe) computed by the vectorised simulator on
  the fold's *training* range only; evaluation on the test fold happens once, at the end.
- **Search space:** (a) discrete/exit parameters only (≈8 dims: ATR multiplier, floor, cap,
  grace hours/level, band width, min hold, guard thresholds); (b) (a) + a tiny policy head
  (≤ 200 weights) on top of the frozen brief-1 encoder. Do **not** evolve a full network.
- **Algorithm:** CMA-ES (`cma` package) for (a); OpenAI-ES with antithetic sampling for (b).
  Population ≤ 32, generations ≤ 50 → ≤ 1,600 evaluations per fold, all vectorised in MLX.
- **Trial accounting is the whole brief:** every evaluation is written to `trials.jsonl`; PBO
  via CSCV over the *entire* population history; DSR with N = total evaluations. Expect PBO to
  be high — that is the measurement.
- **Regularisation:** penalise turnover and parameter distance from the production incumbent
  (`NNNC_MLX.json`), so the search must *earn* a move away from the current operating point.

## Experiments

1. Search space (a) on fold f07, 3 CMA-ES seeds → nested OOS on the fold's test range.
2. Full 15-fold pooled run for (a).
3. Search space (b) only if (a) clears the floor.
4. **Random-search control** (Chen & Navet 2007): the same evaluation budget spent on uniform
   random parameter draws. If CMA-ES does not beat random search OOS, the optimiser is
   fitting noise.

## Acceptance / kill

Protocol §5 with **PBO ≤ 0.2 as the primary gate**. Kill immediately if the random-search
control matches CMA-ES on pooled OOS, or if the found optimum sits within one sweep-noise band
(4.93pp, `AGENT_GUIDE.md`) of the incumbent.

## Workflow (mandatory — see README "How to run a brief")

```
/speckit-specify   <the starter prompt below, verbatim>      → specs/0NN-labelfree-es-policy-search/spec.md
/speckit-clarify                                              → resolve markers; stats/windows come from 00-evaluation-protocol.md
/speckit-plan                                                 → plan.md, research.md (cite 01-literature-review.md)
/speckit-tasks                                                → tasks.md: first tasks = calibration/leakage tests, last = §6 report + trial-log audit
/speckit-analyze   (REQUIRED before implement; fix every CRITICAL, re-run until clean; save as analyze.md)
/speckit-implement                                            → sequential runs only
/speckit-converge  (if a session ends mid-feature)
/speckit-analyze   (again on completion)
write briefs/results/05-<date>.md linking the feature dir and both analyze reports
```

The spec's success criteria must be protocol §5 verbatim or stricter. Any brief-specific
kill criteria above become additional SC-lines.

## Starter prompt (pass to `/speckit-specify` verbatim)

```
Feature: labelfree-es-policy-search. This is research/label-free-study brief 05; the feature must be
specified, planned, task-listed, analyzed with /speckit-analyze, and only then implemented.
Read AGENT_GUIDE.md ("The noise floor" and "Silent failure modes" first),
research/label-free-study/README.md, 00-evaluation-protocol.md and
briefs/05-gradient-free-policy-search.md. Using the vectorised simulator from brief 1,
implement research/labelfree/es/ with CMA-ES over the exit/guard parameter space, a random-
search control with the identical budget, and CSCV/PBO over the full evaluation history.
Every evaluation must be appended to research/labelfree/trials.jsonl. Evaluate the test
fold exactly once per fold. Write briefs/results/05-<date>.md per protocol §6 with PBO as
the headline number.
Success criteria: the acceptance and kill gates of research/label-free-study/
00-evaluation-protocol.md §5 plus the brief's own kill criteria, stated as measurable
SC-lines. Non-goals: the adjacent approaches listed in briefs/09-excluded-and-why.md.
```
