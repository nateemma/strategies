# Brief 02 — Semi-supervised: label warm-start, P&L fine-tune

**Family:** semi-supervised bridge (Deng 2016 pretrain-then-profit; α-blended DFL/MSE loss)
**Type:** semi-supervised · **Priority:** 2 · **Budget:** 1–2 sessions · **Depends on:** brief 1's simulator and wrapper

## Hypothesis

The existing lookahead labels are not useless — they are a noisy but informative *prior* about
where tradeable moves start. A policy initialised from the trained `NNNC_MLX` / `NNMT_MLX`
encoder and then fine-tuned on the net-P&L objective (or trained from scratch on an α-blend
of label cross-entropy and net P&L) will beat both:
- the pure classifier (it fixes the operating point and cost-awareness), and
- the pure P&L policy of brief 1 (the label prior regularises an objective that is otherwise
  easy to overfit — the failure mode Jansen documented for naive Sharpe-LSTMs).

This is the natural A/B that decides whether labels should be dropped or kept as a prior.

## Prior evidence

- Deng et al. 2016: unsupervised pretraining then "task-aware BPTT" on profit.
- DFL-for-Markowitz ([arXiv 2409.09684](https://arxiv.org/html/2409.09684v4)): best results at
  an α-blend of decision loss and MSE, not at either extreme.
- Ong & Herremans 2023: auxiliary supervised heads improve a Sharpe-trained portfolio.
- Return-weighted cross-entropy ([arXiv 2502.17493](https://arxiv.org/html/2502.17493)):
  weighting CE by |return| raised Sharpe 1.00 → 1.18 in one era, tied in another.
- **No rigorous study of "pretrain on hindsight labels, fine-tune on P&L" exists for trading.**
  This brief fills that gap for our data.

## Design

Three arms, all sharing brief 1's simulator, features and freqtrade wrapper:

| Arm | Init | Loss |
|---|---|---|
| **A. Warm-start** | encoder weights from `saved_data/NNNC_MLX/` (same fold's training range — re-train per fold, do not reuse the production model) | brief 1's utility loss; new sigmoid/band head; encoder LR 10× smaller than head |
| **B. Blended** | random | `α·CE(label) + (1−α)·utility(net P&L)`, α ∈ {0.75, 0.5, 0.25}; label = production 0.007 *and* the 0.02 label (both tested — the repo showed they behave very differently) |
| **C. Auxiliary head** | random | utility loss + auxiliary head predicting forward realised volatility (no directional label at all) |

Controls: brief 1's best config (pure P&L) and `NNNC_MLX` (pure label).

Label purge/embargo: the label horizon (`HORIZON` in the strategy config) must be purged at
each fold boundary because the CE term is forward-looking.

## Experiments

1. Arm A on fold f07 × 5 seeds; if it is not at least as good as brief 1 in-fold, note it and
   move to B (warm-starting can hurt if the classifier's features are tuned to the wrong label).
2. Arm B α-sweep × 2 labels × 5 seeds on f07 → best two to the full pooled run.
3. Arm C × 5 seeds, pooled.
4. **Diagnostic:** for the best arm, measure the correlation between the policy's exposure and
   the label it was warm-started from. If it is ≈1 the fine-tune did nothing; if it is ≈0 the
   label prior was discarded — either is informative.

## Acceptance / kill

Protocol §5. Additional:
- If no arm beats *both* controls by the floor → "not resolved"; record whether labels helped
  (arm > brief 1) or hurt, since that decides the direction of the whole programme.
- α-sweep must not be reported as "best α" — it is 3 trials; the DSR accounts for it.

## Workflow (mandatory — see README "How to run a brief")

```
/speckit-specify   <the starter prompt below, verbatim>      → specs/0NN-labelfree-warmstart-finetune/spec.md
/speckit-clarify                                              → resolve markers; stats/windows come from 00-evaluation-protocol.md
/speckit-plan                                                 → plan.md, research.md (cite 01-literature-review.md)
/speckit-tasks                                                → tasks.md: first tasks = calibration/leakage tests, last = §6 report + trial-log audit
/speckit-analyze   (REQUIRED before implement; fix every CRITICAL, re-run until clean; save as analyze.md)
/speckit-implement                                            → sequential runs only
/speckit-converge  (if a session ends mid-feature)
/speckit-analyze   (again on completion)
write briefs/results/02-<date>.md linking the feature dir and both analyze reports
```

The spec's success criteria must be protocol §5 verbatim or stricter. Any brief-specific
kill criteria above become additional SC-lines.

## Starter prompt (pass to `/speckit-specify` verbatim)

```
Feature: labelfree-warmstart-finetune. This is research/label-free-study brief 02; the feature must be
specified, planned, task-listed, analyzed with /speckit-analyze, and only then implemented.
Read AGENT_GUIDE.md, research/label-free-study/README.md, 00-evaluation-protocol.md,
briefs/01-direct-policy-optimisation.md (whose simulator and wrapper you will reuse — do
not fork them) and briefs/02-label-warmstart-pnl-finetune.md. Implement arms A, B and C as
configurations of the brief-1 trainer, with per-fold purge/embargo of the label horizon.
Log each configuration to research/labelfree/trials.jsonl before running. Report against
brief 1's best config and NNNC_MLX using regime/wf/compare_pooled.py, and write
briefs/results/02-<date>.md using the protocol §6 template.
Success criteria: the acceptance and kill gates of research/label-free-study/
00-evaluation-protocol.md §5 plus the brief's own kill criteria, stated as measurable
SC-lines. Non-goals: the adjacent approaches listed in briefs/09-excluded-and-why.md.
```
