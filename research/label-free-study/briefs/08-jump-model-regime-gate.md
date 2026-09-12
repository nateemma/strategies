# Brief 08 — Statistical jump-model regime gate (negative control against `regime/VERDICT.md`)

**Family:** unsupervised regime discovery with explicit switching cost (Shu, Yu & Mulvey 2024)
**Type:** unsupervised, risk-side · **Priority:** 8 (low) · **Budget:** 1 session

## Why this is here despite Phase C's FAIL

`regime/VERDICT.md` tested eight regime instruments (BTC vol, variance ratios, Hurst,
autocorrelation, trend, breadth, correlation) and none beat the fixed choice on two distant
windows. The literature agrees that regime gates mostly buy *drawdown*, not return. Two things
were not tested and are cheap to close out:

1. **A jump model** differs mechanically from the tested instruments: it fits regimes with an
   explicit penalty on switching, which the Princeton study found cuts turnover ~3× versus an
   HMM (44% vs 141%) and halves max drawdown on equity indices with online inference, a 1-day
   delay and 10 bp costs.
2. **Regime as a *conditioning input* to the brief-1 policy** rather than a hard gate — the
   Wood et al. result (changepoint features raised Deep Momentum Network Sharpe ~⅓) is
   about features, not filters.

Expected outcome: the gate does not clear the floor on return (consistent with Phase C) but
may reduce the drawdown of whichever policy wins briefs 1–3. That is worth knowing for
deployment, not for alpha.

## Design

- **Model:** sparse jump model (`jumpmodels` package) on causal features (realised vol,
  downside vol, return sign persistence) per pair and for BTC. **Online inference only** —
  never Viterbi/smoothed states; refit every 90 days on an expanding window; map states by
  mean volatility after each refit so labels do not swap.
- **Leakage test (mandatory, first):** run the gate with smoothed states and confirm the
  result is dramatically better — that is the size of the leak you are avoiding, and it
  should be documented next to the honest number.
- **Uses:** (a) hard gate: no new entries in the high-vol / negative-drift state; (b) soft:
  state probability as an input feature to the brief-1 policy.
- **Control:** the incumbent, and Phase C's best fixed choice.

## Acceptance / kill

Protocol §5 for return; additionally report max drawdown and time-underwater with bootstrap
CIs. Accept as a *deployment* modification (not an alpha claim) if return is unchanged within
the floor and drawdown falls with CI excluding zero across ≥10 of 15 folds. Kill otherwise.

## Workflow (mandatory — see README "How to run a brief")

```
/speckit-specify   <the starter prompt below, verbatim>      → specs/0NN-labelfree-jump-model-gate/spec.md
/speckit-clarify                                              → resolve markers; stats/windows come from 00-evaluation-protocol.md
/speckit-plan                                                 → plan.md, research.md (cite 01-literature-review.md)
/speckit-tasks                                                → tasks.md: first tasks = calibration/leakage tests, last = §6 report + trial-log audit
/speckit-analyze   (REQUIRED before implement; fix every CRITICAL, re-run until clean; save as analyze.md)
/speckit-implement                                            → sequential runs only
/speckit-converge  (if a session ends mid-feature)
/speckit-analyze   (again on completion)
write briefs/results/08-<date>.md linking the feature dir and both analyze reports
```

The spec's success criteria must be protocol §5 verbatim or stricter. Any brief-specific
kill criteria above become additional SC-lines.

## Starter prompt (pass to `/speckit-specify` verbatim)

```
Feature: labelfree-jump-model-gate. This is research/label-free-study brief 08; the feature must be
specified, planned, task-listed, analyzed with /speckit-analyze, and only then implemented.
Read regime/VERDICT.md, regime/phase_c_trackability.py, AGENT_GUIDE.md,
research/label-free-study/README.md, 00-evaluation-protocol.md and
briefs/08-jump-model-regime-gate.md. Implement research/labelfree/jump_gate.py using an
online (filtered) sparse jump model refit every 90 days on an expanding window, with the
smoothed-state leak test run first and reported. Evaluate as a hard entry gate on the
incumbent NNNC_MLX through the pooled walk-forward, reporting drawdown metrics with
bootstrap CIs alongside return. Write briefs/results/08-<date>.md.
Success criteria: the acceptance and kill gates of research/label-free-study/
00-evaluation-protocol.md §5 plus the brief's own kill criteria, stated as measurable
SC-lines. Non-goals: the adjacent approaches listed in briefs/09-excluded-and-why.md.
```
