# Phase 0 Research — Tail-Targeted Breakout Labelling

All figures measured in-session 2026-09-04 unless stated. Scripts named are in
`regime/`. Nothing here required training a model.

---

## R1 — Is the payoff distribution actually fat-tailed? RESOLVED: yes, severely

**Decision**: treat mean/median as uninformative for label design on this signal.

Forward MFE from breakout-condition bars (`guard_metric > 0.2 & bb_width > 0.035`),
pooled XRP/SOL/LINK, `breakout_dist.py`:

| H | median | p90 | p99 | skew | excess kurt | top-10% share |
|---|---|---|---|---|---|---|
| 6 | 0.52 | 2.43 | 6.50 | 3.78 | 64.1 | 43.5% |
| 12 | 0.85 | 3.62 | 9.40 | 3.44 | 34.8 | 41.4% |
| 24 | 1.33 | 5.25 | 12.82 | 7.85 | 225.1 | 39.4% |
| 48 | 2.11 | 7.87 | 17.91 | 5.28 | 80.9 | 38.0% |
| 96 | 3.27 | 11.54 | 26.55 | 4.30 | 43.3 | 37.3% |

**Rationale**: skew 3.4–7.9 and excess kurtosis 35–225 against a Gaussian's 0/0.
The top decile of bars carries ~40% of all upside.

**Alternatives considered**: assuming approximate normality and tuning on the
mean — rejected, it describes almost none of the value.

**Caveat carried forward**: `gbb` shows the SAME shape (skew ~3.8, kurtosis ~30,
top-10% ~38%). Fat tails are generic to this feature basis, not a breakout
property, so they cannot by themselves explain why breakout fails and gbb works.

---

## R2 — Is the current label economically viable? RESOLVED: no

**Decision**: the label is mis-specified relative to the risk taken.

At H=48 / 0.005 the median labelled gain is **2.83%** against an operative stop of
**−4% to −6%**. Only **26.0%** of labelled bars ever clear the stop; reward:risk is
**0.57**.

**Rationale**: the label rewards a population whose typical member cannot pay for
the risk of holding it. This is the only structural cause not yet tested — the
seven prior interventions all acted downstream of the label.

**Alternatives considered**: that the entry model is simply weak — not excluded,
and R6 shows it is at least partly true; the two are separable only by test.

---

## R3 — Can the economics be fixed while staying learnable? RESOLVED: yes, narrowly

**Decision**: `H=96 / 0.03` is the best cell above the density floor;
`H=48 / 0.02` is the best cell needing no exit change.

| setting | %bars | med gain | clears 5% | reward:risk | time-to-peak | stopped first |
|---|---|---|---|---|---|---|
| H=48 0.005 (control) | 9.07% | 2.83% | 26.0% | 0.57 | 6.8h | 4.5% |
| **H=48 0.02** | **5.81%** | 4.29% | 40.6% | 0.86 | **8.2h** | 5.5% |
| H=48 0.03 | 4.30% ✗ | 5.34% | 54.9% | 1.07 | — | — |
| **H=96 0.03** | **5.92%** | **6.41%** | **66.8%** | **1.28** | 17.0h ⚠ | 11.2% |
| H=96 0.05 | 3.95% ✗ | 8.43% | 100.0% | 1.69 | 18.2h ⚠ | 12.7% |

✗ below the ~5.5%-of-bars floor where `aug_risk` flags all-Hold collapse.
⚠ exceeds the 12h forced-exit window (see R4).

**Rationale**: density is the binding constraint, not economics. Two of five cells
are excluded by FR-002 before any retrain.

**Alternatives considered**: pushing to `H=96 / 0.05` for reward:risk 1.69 —
rejected on density (3.95%).

---

## R4 — Does the exit machinery permit the tail candidate? RESOLVED: no, and the fix is a framework edit

**Decision**: `H=96/0.03` cannot be tested as a label change alone.

Median time-to-peak rises **6.8h → 17.0h**, while `unclog_12h` force-exits at 12h
whenever profit exceeds 0.5%. The rule is **hardcoded** at
`Framework/BaseStrategy.py:1040` — `if (num_hours >= 12) & (current_profit > 0.005)` —
not a parameter.

**Rationale**: training on a 17h payoff and harvesting at 12h would manufacture a
false negative. FR-011 therefore requires the label and hold-window contributions
to be measured separately.

**Decision on mechanism**: override the containing method on the host strategy
rather than editing `BaseStrategy`. A live dry run (`MomentumRegimeBasket15mFast`)
imports that file, so FR-010's revert requirement is a safety rule, not
bookkeeping. A per-strategy override needs no revert at all and is therefore
strictly safer.

**Alternatives considered**: edit-measure-revert on `BaseStrategy` (used
successfully in feature 002) — viable but leaves a window in which a live process
could import a modified file; rejected when a subclass override is available.

**Note**: `cexit_max_days` IS a sell-space parameter reachable via `--set-sell`,
so only the unclog rule needs code.

---

## R5 — Does path risk invalidate the tail candidate? RESOLVED: no, but it is not free

**Decision**: proceed; retain the path-aware variant as a candidate rather than a
default.

`tail_path.py`, walking to the MFE bar and measuring worst drawdown en route:

| setting | median DD | p10 DD | stopped at −4% | median bars-to-peak |
|---|---|---|---|---|
| H=48 0.005 (control) | −0.62% | −2.77% | 4.5% | 27 |
| H=48 0.02 | −0.67% | −2.99% | 5.5% | 33 |
| H=96 0.03 | −1.06% | −4.24% | **11.2%** | 68 |
| H=96 0.05 | −1.07% | −4.55% | 12.7% | 73 |

**Rationale**: stop-out-before-peak rises 2.5× but leaves 89% of labels reachable,
so the 2.2× better reward:risk survives it.

**Alternatives considered**: making the MAE constraint mandatory — rejected as a
default. Measured at CURRENT settings it is immaterial (`label_path_audit.py`:
4.5% of breakout and 5.4% of gbb labels need a >4% drawdown first), so it would
add a constraint that does nothing. At tail settings it may earn its place, which
is exactly why FR-006 makes it a candidate to evaluate rather than an assumption.

**Retraction recorded**: an earlier claim that breakout labels fail *because* of
adverse paths was REFUTED by this measurement — breakout paths are marginally
BETTER than gbb's. The EV-screen finding stands; that mechanism for it did not.

---

## R6 — Will a label change reach the trades at all? RESOLVED: on this host, probably

**Decision**: the breakout arm is the only viable host; the liveness gate is
mandatory regardless.

Six model-side changes across two families produced **byte-identical**
out-of-sample trades:

| change | family | OOS trades vs control |
|---|---|---|
| aux head weights | NNMT | 99% entry overlap |
| peak-filter fix (+77% labels, +0.155 MCC) | NNNC | IDENTICAL W1 and W3 |
| peak-filter fix | NNMT | IDENTICAL W1 |
| gbb threshold 0.02 | NNNC | IDENTICAL W1 and W3 |
| exit label level→crossing | breakout | changed (all 3) |
| peak-filter retrain | breakout | changed (all 3) |

**Rationale**: the tightly-gated family absorbs model-side change entirely — its
seven-guard chain selects ~99% of entries. The breakout arm has two of seven
guards neutralised and its trades DO differ across all three windows on a model
change. It is the only place a label change can currently express itself.

**Consequence for reporting**: identical trades must be recorded as INCONCLUSIVE
(FR-015). On the base rate above, this is the expected outcome, not an edge case,
and the reporting must not read it as refutation.

---

## R7 — Which metrics may drive acceptance? RESOLVED: none of them

**Decision**: acceptance on realised out-of-sample return only (FR-014).

Three metrics have each failed to predict return in this codebase:

| metric | what it said | what happened |
|---|---|---|
| EV per signal | breakout_gbb 4.50% vs gbb 3.90% | breakout lost by 43pp |
| precision | 0.129 → 0.428 (3.3×) | W1 −5.05% → −12.40% |
| val_mcc | 0.759 → 0.608 (fell) | summed nominally rose, OOS identical |

**Rationale**: the EV screen is MFE-based and path-blind, so it systematically
flatters momentum labels; MCC rises mechanically with horizon; precision improved
while the arm traded worse. None is a safe acceptance criterion.

**Alternatives considered**: a composite score — rejected, three failures with
three different failure modes give no reason to trust a fourth construction.

---

## R8 — What configuration does the control actually run? RESOLVED: class defaults, and it must be pinned

**Decision**: write `NNNC/NNNC_BreakoutOpen.json` pinning the current effective
configuration, then RE-BASELINE the control before any arm is compared to it.

`NNNC_BreakoutOpen` has no params file, so it inherits `BaseStrategy.sell_params`
— notably `cexit_take_profit` **0.032**, not NNNC's tuned 0.024. All existing
breakout arms ran this way, so historical comparisons among them remain internally
valid (`exit_config_hash` bb3239ab7e52df1b throughout).

**Rationale**: FR-016 requires each arm to carry its own params file. Leaving the
host on class defaults means any NEW arm that gains a params file would silently
differ in exit configuration — precisely the confound that produced a spurious
12pp result on the GAN comparison, where `NNNC_DDPM_MLX` ran 0.032 against a
control on 0.024.

**Alternatives considered**:
- Declare class defaults intentional and give no arm a params file — rejected;
  it leaves the trap armed for the next person and violates FR-016.
- Adopt NNNC's tuned 0.024 for the host — rejected as an unmeasured change
  bundled into a labelling feature (Constitution VIII). Pin what is currently
  running; changing it is a separate experiment.

**Re-baseline cost**: 3 backtests, no retrain. The new control must reproduce
`brkfix`'s trade hashes exactly; if it does not, the params file changed
behaviour and that must be resolved before proceeding.

---

## R9 — How much power does the comparison have? RESOLVED: little; state it explicitly

**Decision**: report trade counts and treat sub-1pp W1 deltas as indistinguishable.

The control has **97 trades on W1**. At the arm's observed dispersion, differences
below roughly 1pp on W1 are not separable from reallocation among a handful of
trades — the peak-filter retrain moved W1 by 0.35pp while changing every window's
trade set, and the threshold sweep moved win rate not at all across a 3.5× range.

**Rationale**: FR-017 requires reporting counts; this quantifies what "underpowered"
means so a small positive is not over-read.

**Alternatives considered**: widening the evaluation to more pairs or windows —
rejected, it would break the pinned-window protocol and the out-of-sample split.
