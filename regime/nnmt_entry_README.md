# NNMT_MLX entry-gate sweep (2026-09-02)

Run after the exit repair, on NNMT's OWN defaults — no NNNC tuning imposed.
NNMT has a trained model and no params file, so it runs purely on class
defaults. Windows W1 `20250901-20260831`, W2 `20240601-20250531`, W3 `20230101-20231231`.

## rvol — optimum is 0.25, far looser than NNNC's 2.0

| rvol | trades | W1 | W2 | W3 | summed | worst Calmar |
|---|---|---|---|---|---|---|
| 0.0 | 649 | 8.96 | 6.21 | 0.42 | 15.60 | 0.43 |
| **0.25** | 654 | **10.71** | 8.45 | **1.04** | **20.20** | **1.11** |
| 0.5 *(current)* | 634 | 9.64 | **9.79** | 0.05 | 19.48 | 0.04 |
| 1.0 | 551 | 6.67 | 8.64 | 0.48 | 15.79 | 0.50 |
| 1.5 | 457 | 6.06 | 5.55 | 0.12 | 11.73 | 0.12 |
| 2.0 | 375 | 6.08 | 4.85 | 0.92 | 11.85 | 0.95 |

0.25 wins W1 (headline) and W3 (persistence), summed return and worst-window
Calmar; 0.5 wins only W2. Max regret marginally favours 0.5 (1.06 vs 1.34) but
on W2 alone. **Recommend 0.25** — the gain is modest in return (+0.72pp) but W3
goes 0.05% -> 1.04% and its Calmar 0.04 -> 1.11, on the family's weakest window.

rvol 0.0 (guard fully off) is WORSE than 0.25, so a light volume filter does earn
its place — the peak is just much looser than NNNC's.

## guard_metric 0.3 — dominant here too, second family

| guard | trades | W1 | W2 | W3 | summed | worst Calmar |
|---|---|---|---|---|---|---|
| −0.4 | 603 | 10.06 | 7.80 | 0.59 | 18.45 | 0.64 |
| 0.0 | 646 | 10.05 | 7.97 | 0.62 | 18.64 | 0.66 |
| **0.3** | 654 | **10.71** | **8.45** | **1.04** | **20.20** | **1.11** |
| 0.6 | 654 | 10.71 | 8.45 | 1.04 | 20.20 | 1.11 |

Dominant in all three windows on both metrics, and 0.6 byte-identical to 0.3 by
trade hash — the same saturating plateau as NNNC. **`guard_metric = 0.3` is now
validated on two families and is correctly already the `BaseStrategy` default.**
It is structural in the way rvol is not.

## apply_task_filters — strongly HARMFUL at the current heads

Inference-time entry filter, so this needs no retrain. At rvol 0.25 / guard 0.3:

| filter | W1 | W2 | W3 | summed | trades |
|---|---|---|---|---|---|
| OFF *(default)* | 10.71% | 8.45% | 1.04% | **20.20%** | 654 |
| ON | 3.68% | 1.15% | 1.15% | **5.99%** | 408 |

**−14.21pp**, worse in W1 and W2, LIVE (hashes differ in all three windows).

This **reverses** the recorded finding that the 3-of-4 task filter is the
dominant entry-quality lever. It is dominant, but in the harmful direction at
this operating point. The old measurement was taken under the broken exit and at
a much tighter gate (it reported disabling the filter giving 4.5x the trades;
here enabling it removes only 38%), so the two are not directly comparable.

**This is consistent with the task audit**, which found NNMT trains the three
LOWEST-relevance tasks and excludes the highest (NMI 0.130). The filter is gating
entries on heads that do not carry entry information, so it is discarding good
trades. That is evidence about the *current* heads, not about the concept.

## What this says about re-running the task study

The cheap gate proposed before spending retrains has now run, and it argues for a
**targeted** retrain rather than a combinatorial one. The aux heads as currently
trained are not merely uninformative for entries — using them actively destroys
14pp. Re-sweeping combinations OF THOSE HEADS is unlikely to find an edge. The
question worth a retrain is the one the audit already identified: **train the
high-NMI head that is currently excluded**, then re-test the filter.

## Task-head ablation study, RE-RUN on the repaired exit stack

No retraining — all eight models survive in `saved_data/` from 2026-09-01/02,
each with a strategy class and no params file, so every arm runs on class
defaults (rvol 0.25 / guard 0.3). 24 backtests. All arms verified LIVE: distinct
trade hashes from base in every window.

| arm | trades | W1 | W2 | W3 | summed | Δ vs base | better in all 3? |
|---|---|---|---|---|---|---|---|
| base (risk/mom/flow @1.5) | 654 | 10.71 | 8.45 | 1.04 | 20.20 | — | — |
| **noflow** (−flow) | 658 | **11.30** | 9.20 | 1.31 | **21.81** | **+1.62** | **YES** |
| **trprofit** (trading+profit) | 658 | 10.87 | 9.02 | 1.41 | 21.30 | **+1.10** | **YES** |
| nomom (−momentum) | 658 | 10.53 | **9.53** | 1.00 | 21.06 | +0.87 | no |
| ctrl (exit signal ON) | 654 | 10.58 | 9.07 | 0.98 | 20.63 | +0.44 | no |
| profit (+profit, NMI .130) | 652 | 10.08 | 9.29 | 1.11 | 20.47 | +0.28 | no |
| norisk (−risk) | 655 | 10.71 | 8.59 | 0.99 | 20.29 | +0.09 | no |
| tronly (trading only) | 652 | 10.34 | 8.10 | **1.79** | 20.23 | +0.03 | no |

### The null does NOT reproduce

The original study found no combination distinguishable. On the repaired stack
**every arm beats base**, and two of them — `noflow` and `trprofit` — beat it in
**all three windows**, which is the acceptance rule this project uses. Under the
old exit that ranking was invisible.

This is the same mechanism as the entry-gate reversal: the defect cut ~87% of
stopped trades above the configured stop, compressing exactly the between-arm
variation the study was trying to detect. A null measured through a homogenising
mechanism was uninformative.

### What the ordering says

**It is not aux-weight dilution.** Base is the only arm carrying the full 4.5 of
aux loss weight and every other arm carries ≤3.0, which would explain "all arms
beat base" — but `tronly` carries **zero** aux weight and gains only +0.03pp,
losing in both W1 and W2. If dilution were the story, `tronly` would win.

**The consistent signal is `flow`.** Both arms that improve in all three windows
drop it, and `flow` is the weakest head at NMI 0.011. But dropping it is not
sufficient: `tronly` also drops it and gains nothing, so `risk` and `momentum`
are contributing real positive transfer that `tronly` throws away.

Reading: **drop `flow`, keep the other aux heads.** `noflow` is exactly that
configuration and is the best arm on summed return, on worst-window Calmar
(1.41 vs 1.11) and in every individual window.

**This reverses the recorded "tronly was the best NNMT config".** It is now
indistinguishable from base and negative in the two nearer windows.

### Caveat before adopting

Each arm is a SINGLE trained model. The project rule from the COCONUT work is
paired seed-robustness before believing small deltas, and +1.62pp on a 20.20pp
base is not large. **Before promoting `noflow`, retrain base and `noflow` at 2–3
seeds each and compare paired.** That is a targeted retrain of two arms, not a
re-run of the study.

### Defect found in the study apparatus

`NNMT_MLX_ctrl.py` does not do what its docstring says. The docstring describes a
guards-off control ("the only difference is that the entry guard chain is
disabled"), but the class body only sets `enable_exit_signal: True`, and the file
ENDS at line 31 on a dangling `# MERGE from the parent` comment with no code
after it — the guards-off override appears to have been lost.

Its 654 trades exactly match base (entries identical, only exits differ), which
confirms guards are still on. So this arm is not a guards-off control; it is an
**exit-signal measurement on NNMT, worth +0.44pp** with entries held identical —
compare NNNC, where the same switch was +0.03pp. Worth noting that NNMT's model
does earn something on exits where NNNC's does not, but it is not consistent
across windows (W1 and W3 both negative).

## Seed-robustness test of `noflow` — NOT CONFIRMED

Six models trained fresh (base and `noflow` at MLX_SEED 42/1/7) on the pinned
window `20220101-20250831`, which ends the day before W1 begins. **W1 is
therefore out-of-sample and carries the verdict; W2 and W3 are IN-SAMPLE.**
All six model files verified distinct by SHA-256.

| seed | ΔW1 (OOS) | ΔW2 (in-sample) | ΔW3 (in-sample) | Δ summed |
|---|---|---|---|---|
| 42 | **−0.19** | +1.00 | +0.05 | +0.86 |
| 1 | **+0.08** | +0.01 | +0.18 | +0.27 |
| 7 | **0.00** | +0.03 | +0.42 | +0.45 |
| mean | **−0.04** | +0.35 | +0.22 | +0.53 |

**Verdict: do not promote `noflow`.** On the only out-of-sample window the effect
is zero with inconsistent sign. Every positive summed delta comes from in-sample
windows — the signature of a slightly better fit, not an edge. The single-seed
+1.62pp does not reproduce (best seed +0.86pp), and base's own seed spread on W1
(10.71 / 10.44 / 10.67) is wider than the mean treatment effect.

The earlier +1.62pp was single-seed noise plus in-sample fit, on models whose
training window is unrecoverable and probably overlapped the evaluation windows.

Consistent with the validation metrics measured during training: base moved
0.0067 on `val_trading_mcc` from seed alone (0.6057 → 0.5990) while `noflow` sat
0.0054 BELOW base at the same seed — treatment smaller than seed noise there too.

### The instrument has limited resolution here

Several arms produced **byte-identical trade sets from genuinely different
models**: base W3 is hash `840872fdb3` / 126 trades for ALL THREE seeds, and
`noflow` s42 and s1 share hash `cfff1bb13f` on W1 — while all six weight files
differ.

So the **guard chain, not the model, selects most entries** — the recorded
"guards bind, not the z-gate" finding showing up as measurement power. When the
gate picks the bars, a head-weight change moves only a handful of marginal
trades, which cannot be resolved above seed noise.

This reframes the original null: it was not only the broken exit compressing
variation, the backtest is a low-resolution instrument for head weightings in the
first place. **Head-weight tuning is a poor lever for NNMT regardless of exit
state.** Any future attempt needs either a much looser gate (to raise the share
of model-determined entries) or a model-level metric that is known to track P&L —
and this repository has repeatedly found that MCC does not.

### Corrections to the earlier write-up in this file

The task-ablation section above reports single-seed results on models with an
unknown training window. Its ranking should be read as **not established**. What
survives is the negative result: no head weighting is distinguishable once seeds
are accounted for, which AGREES with the original study's null rather than
overturning it. The claim that "the null does not reproduce" was wrong — it does
reproduce, for a better-understood reason.

### Apparatus notes

- `train_seed` is **silently inert for multi-task models** (9th recorded
  instance). `TrainSeedStrategyMixin` sets `clf.train_seed` and
  `MLXClassifierNary.train()` honours it, but `MLXClassifierMultiTask` extends
  `MLXBaseClassifier` and never reads it. Using the documented mixin route would
  have produced identical models at every seed and reported "seed-robust". The
  working route is the `MLX_SEED` env var, read by `MLXBasePredictor` at import.
- **No training metadata is saved** with a model — `saved_data/<cls>/` holds only
  the `.safetensors`, with no window, seed or config. That is why the pre-existing
  models could not be reused as a seed arm, and why their absolute numbers are
  not comparable to anything measured later.

## WHY head-weight studies cannot work here: the gate picks ~99% of entries

The hash collisions above were weak evidence. Measuring the actual entry sets
settles it. Jaccard overlap of `(pair, open_date)` sets across six models
(base / noflow / rmp × seeds 42, 1, 7):

| window | same weights, different seed | different weights, same seed |
|---|---|---|
| W1 | **99.2%** | **99.2%** |
| W3 | **100.0%** | 96.9% |

On W1 those are **identical**: changing the task weighting moves the entry set no
more than changing the random seed does. On W3 same-weight models are byte
identical, which is why base's W3 hash repeated across all three seeds — a fully
gate-determined entry set, not a bug.

Trade-level, base_s42 vs rmp_s42:

| | W1 | W3 |
|---|---|---|
| shared entries | 175 of 176 | 124 |
| shared trades with IDENTICAL `profit_ratio` | **100.0%** | **100.0%** |
| entries only in base | 1 (+0.56%) | 2 (−2.06%) |
| entries only in rmp | 1 (+0.70%) | 4 (+1.96%) |

**The entire W1 difference between two arms is one trade each way.**

Two consequences, both measured rather than inferred:

1. **The guard chain determines ~99% of entries.** Any arm comparison is a
   referendum on 1–6 marginal bars out of 176. No number of seeds fixes this;
   the instrument has ~1% resolution.
2. **The heads have ZERO influence on exits.** Every shared trade exits
   identically — including between base (profit head untrained, MCC 0.000) and
   rmp (profit head trained, MCC 0.182). So the trained profit head changes no
   exit decision, despite `strategy_custom_exit` being documented as a
   model-prediction bailout reading `%profit` and `%regime`. That path is
   effectively dead, consistent with the exit-signal arm being the only thing
   that ever moved exits.

This explains all three prior results at once: the original null, the `noflow`
false positive, and `rmp` landing at +0.01pp on W1 despite having the best
training metrics of any arm (trading MCC 0.6089 vs base 0.6057, profit head
learning at 0.182, risk and momentum both nudged up).

**Prerequisite for any future head-weight work:** make entries model-determined
first, via a materially looser guard chain, so the model chooses among candidates
instead of ratifying the gate's choice. Until then, head-weight arms are
unmeasurable and should not be run.

**Method, to redo this:** read each arm's `result_file` from `regime/arms/*.json`
via `freqtrade.data.btanalysis.load_backtest_data`, key trades on
`(pair, open_date)`, and compare set overlap plus `profit_ratio` on the shared
keys. Comparing SAME-weights/different-seed against DIFFERENT-weights/same-seed
is the essential control — without it, a high overlap looks like determinism
rather than gate dominance.

## Guards-off control: head weights are not a lever AT ANY operating point

The gate-dominance finding above raised an obvious objection — maybe the heads
matter but the gate hides it. Testing that directly: same four models
(base and rmp × seeds 42, 1), W1, at three operating points. The control is
**arm overlap vs seed overlap** — how much a head-weight change moves the entry
set, against how much a mere reseed moves it.

| config | trades | wallet | seed overlap | arm overlap | arm − seed |
|---|---|---|---|---|---|
| guards ON, exits OFF | 176 | +10.64% | 99.4% | 99.2% | −0.3pp |
| guards OFF, exits OFF | 2814 | −24.27% | 80.4% | 82.4% | **+2.0pp** |
| guards OFF, exits ON | 3013 | −13.84% | 79.6% | 81.7% | **+2.0pp** |

Opening the gate does hand the decision back to the model — overlap falls from
~99% to ~80%. But **arm overlap never falls below seed overlap.** At both open-gate
points head weights move the entry set ~2pp LESS than re-initialising the network
does.

**So this was not a measurement-power problem after all.** The earlier
gate-dominance result was true but not the root cause. The root cause is that
**aux-head weighting does not meaningfully change what this model does** — two
networks trained with different task weightings are about as similar to each
other as two trained identically with different seeds. Adding the exit channel
(where `profit` and `regime` are actually read) does not change that.

**The head-weight line is closed.** Not "unmeasurable pending a looser gate" —
measured at a looser gate, with the model controlling both entries and exits, and
still null. This supersedes the "prerequisite for future head-weight work"
recommendation in the section above: that prerequisite was met and the answer is
still no.

### Side finding: exit signals ARE worth a lot at a powered operating point

Enabling `enable_exit_signal` at the open gate is worth **+10.4pp**
(−24.27% → −13.84%). The same switch at the production gate measured +0.03pp.

The exit signal is not inert — it looked inert because the tight gate leaves it
176 already well-selected trades with little to fix. Textbook confirmation of
"evaluate interventions at a powered operating point", now with a number. It does
NOT argue for enabling exit signals in production, where the measurement stands
at +0.03pp; it argues that any future gate-loosening must re-test the exit signal
rather than inheriting the tight-gate null.
